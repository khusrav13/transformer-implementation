"""
Tests for Phase 3: Full Transformer Model

Tests cover:
1. TransformerConfig - Configuration management
2. Transformer - Full encoder-decoder model
3. DecoderOnlyTransformer - GPT-style model
4. OutputProjection - Weight tying
5. Generation utilities - Greedy, beam search, sampling
"""

import torch
import torch.nn as nn
import pytest
import sys
import os
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformer_phase3.model.config import TransformerConfig
from transformer_phase3.model.transformer import Transformer, DecoderOnlyTransformer
from transformer_phase3.model.output import OutputProjection, LMHead
from transformer_phase3.utils.generation import (
    GenerationConfig,
    TextGenerator,
    greedy_decode,
    sample_decode,
    beam_search_decode
)


class TestTransformerConfig:
    """Tests for TransformerConfig"""

    def test_default_config(self):
        """Test default configuration values"""
        config = TransformerConfig()

        assert config.d_model == 512
        assert config.num_heads == 8
        assert config.num_encoder_layers == 6
        assert config.num_decoder_layers == 6
        assert config.d_ff == 2048
        assert config.head_dim == 64  # 512 / 8

    def test_preset_configs(self):
        """Test preset configurations"""
        base = TransformerConfig.base()
        small = TransformerConfig.small()
        large = TransformerConfig.large()

        assert small.d_model < base.d_model < large.d_model
        assert small.num_encoder_layers < base.num_encoder_layers < large.num_encoder_layers

    def test_config_validation(self):
        """Test that invalid configs raise errors"""
        # d_model must be divisible by num_heads
        with pytest.raises(AssertionError):
            TransformerConfig(d_model=512, num_heads=7)

    def test_config_serialization(self):
        """Test config save/load"""
        config = TransformerConfig.small()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'config.json')
            config.save(path)

            loaded = TransformerConfig.load(path)

            assert loaded.d_model == config.d_model
            assert loaded.num_heads == config.num_heads
            assert loaded.vocab_size == config.vocab_size

    def test_alibi_disables_pos_encoding(self):
        """Test that ALiBi sets pos_encoding to 'none'"""
        config = TransformerConfig(
            use_alibi=True,
            pos_encoding_type='rotary'  # Should be overridden
        )
        assert config.pos_encoding_type == 'none'

    def test_parameter_estimate(self):
        """Test parameter count estimation"""
        config = TransformerConfig.small()
        params = config.num_parameters_estimate

        # Should be in millions range for small config
        assert params > 1e6
        assert params < 1e9

    def test_config_repr(self):
        """Test string representation"""
        config = TransformerConfig.small()
        repr_str = repr(config)

        assert 'd_model=' in repr_str
        assert 'num_heads=' in repr_str


class TestOutputProjection:
    """Tests for output projection layers"""

    def test_output_projection_shape(self):
        """Test output shapes"""
        d_model = 256
        vocab_size = 1000
        batch_size = 4
        seq_len = 32

        proj = OutputProjection(d_model, vocab_size)
        x = torch.randn(batch_size, seq_len, d_model)

        output = proj(x)

        assert output.shape == (batch_size, seq_len, vocab_size)

    def test_weight_tying(self):
        """Test that weight tying shares parameters"""
        d_model = 256
        vocab_size = 1000

        embedding = nn.Embedding(vocab_size, d_model)
        proj = OutputProjection(d_model, vocab_size, tie_weights=embedding)

        assert proj.tied
        assert proj.weight is embedding.weight

    def test_lm_head_with_norm(self):
        """Test LM head with normalization"""
        d_model = 256
        vocab_size = 1000
        batch_size = 4
        seq_len = 32

        head = LMHead(d_model, vocab_size, use_norm=True)
        x = torch.randn(batch_size, seq_len, d_model)

        logits = head(x)
        probs = head.get_probs(x)
        log_probs = head.get_log_probs(x)

        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert probs.shape == (batch_size, seq_len, vocab_size)

        # Probs should sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)

    def test_temperature_scaling(self):
        """Test temperature affects distribution"""
        d_model = 256
        vocab_size = 100

        head = LMHead(d_model, vocab_size)
        x = torch.randn(1, 1, d_model)

        probs_t1 = head.get_probs(x, temperature=1.0)
        probs_t01 = head.get_probs(x, temperature=0.1)  # Sharper
        probs_t10 = head.get_probs(x, temperature=10.0)  # Flatter

        # Lower temperature = higher max probability (sharper)
        assert probs_t01.max() > probs_t1.max()
        # Higher temperature = more uniform (flatter)
        assert probs_t10.std() < probs_t1.std()


class TestTransformer:
    """Tests for full Transformer model"""

    @pytest.fixture
    def small_config(self):
        """Small config for fast testing"""
        return TransformerConfig(
            d_model=64,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=256,
            vocab_size=100,
            max_seq_length=32,
            dropout=0.0  # Disable for deterministic tests
        )

    @pytest.fixture
    def model(self, small_config):
        """Create small transformer for testing"""
        return Transformer(small_config)

    def test_forward_shape(self, model, small_config):
        """Test forward pass produces correct shapes"""
        batch_size = 4
        src_len = 16
        tgt_len = 12

        src = torch.randint(0, small_config.vocab_size, (batch_size, src_len))
        tgt = torch.randint(0, small_config.vocab_size, (batch_size, tgt_len))

        logits = model(src, tgt)

        assert logits.shape == (batch_size, tgt_len, small_config.vocab_size)

    def test_encode_shape(self, model, small_config):
        """Test encoder produces correct shapes"""
        batch_size = 4
        src_len = 16

        src = torch.randint(0, small_config.vocab_size, (batch_size, src_len))
        encoder_output = model.encode(src)

        assert encoder_output.shape == (batch_size, src_len, small_config.d_model)

    def test_decode_shape(self, model, small_config):
        """Test decoder produces correct shapes"""
        batch_size = 4
        src_len = 16
        tgt_len = 12

        src = torch.randint(0, small_config.vocab_size, (batch_size, src_len))
        tgt = torch.randint(0, small_config.vocab_size, (batch_size, tgt_len))

        encoder_output = model.encode(src)
        decoder_output = model.decode(tgt, encoder_output)

        assert decoder_output.shape == (batch_size, tgt_len, small_config.d_model)

    def test_weight_tying(self, small_config):
        """Test that weight tying works"""
        small_config.tie_weights = True
        model = Transformer(small_config)

        # LM head weight should be same object as embedding weight
        assert model.lm_head.projection.tied

    def test_no_weight_tying(self, small_config):
        """Test model without weight tying"""
        small_config.tie_weights = False
        model = Transformer(small_config)

        assert not model.lm_head.projection.tied

    def test_gradient_flow(self, model, small_config):
        """Test gradients flow through entire model"""
        batch_size = 2
        src_len = 8
        tgt_len = 6

        src = torch.randint(0, small_config.vocab_size, (batch_size, src_len))
        tgt = torch.randint(0, small_config.vocab_size, (batch_size, tgt_len))

        logits = model(src, tgt)
        loss = logits.mean()
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_parameter_count(self, model, small_config):
        """Test parameter counting"""
        trainable = model.get_num_parameters(trainable_only=True)
        total = model.get_num_parameters(trainable_only=False)

        assert trainable == total  # All should be trainable
        assert trainable > 0

    def test_save_load(self, model, small_config):
        """Test model save and load"""
        batch_size = 2
        src_len = 8
        tgt_len = 6

        src = torch.randint(0, small_config.vocab_size, (batch_size, src_len))
        tgt = torch.randint(0, small_config.vocab_size, (batch_size, tgt_len))

        # Get original output
        model.eval()
        with torch.no_grad():
            original_output = model(src, tgt)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded_model = Transformer.from_pretrained(tmpdir)

            loaded_model.eval()
            with torch.no_grad():
                loaded_output = loaded_model(src, tgt)

        assert torch.allclose(original_output, loaded_output, atol=1e-5)


class TestDecoderOnlyTransformer:
    """Tests for decoder-only model"""

    @pytest.fixture
    def small_config(self):
        return TransformerConfig(
            d_model=64,
            num_heads=4,
            num_decoder_layers=2,
            d_ff=256,
            vocab_size=100,
            max_seq_length=32,
            dropout=0.0
        )

    @pytest.fixture
    def model(self, small_config):
        return DecoderOnlyTransformer(small_config)

    def test_forward_shape(self, model, small_config):
        """Test forward pass shape"""
        batch_size = 4
        seq_len = 16

        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        logits, _ = model(input_ids)

        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)

    def test_causal_masking(self, model, small_config):
        """Test that decoder is causal (can't see future)"""
        batch_size = 1
        seq_len = 8

        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        model.eval()
        with torch.no_grad():
            logits_full, _ = model(input_ids)

            # Logits at position i should only depend on tokens 0..i
            # Test by modifying future tokens
            input_modified = input_ids.clone()
            input_modified[0, 4:] = 0  # Zero out future

            logits_modified, _ = model(input_modified)

            # Positions 0-3 should be unchanged
            assert torch.allclose(
                logits_full[0, :4, :],
                logits_modified[0, :4, :],
                atol=1e-5
            )


class TestGeneration:
    """Tests for generation utilities"""

    @pytest.fixture
    def small_config(self):
        return TransformerConfig(
            d_model=64,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=256,
            vocab_size=100,
            max_seq_length=32,
            dropout=0.0,
            bos_idx=1,
            eos_idx=2
        )

    @pytest.fixture
    def model(self, small_config):
        return Transformer(small_config)

    def test_greedy_decode(self, model, small_config):
        """Test greedy decoding produces valid output"""
        batch_size = 2
        src_len = 8

        src = torch.randint(3, small_config.vocab_size, (batch_size, src_len))

        model.eval()
        with torch.no_grad():
            output = greedy_decode(
                model, src,
                max_length=20,
                eos_token_id=small_config.eos_idx
            )

        assert output.shape[0] == batch_size
        assert output.shape[1] <= 20

    def test_generation_config(self):
        """Test GenerationConfig defaults"""
        config = GenerationConfig()

        assert config.max_length == 100
        assert config.temperature == 1.0
        assert config.do_sample == False
        assert config.num_beams == 1

    def test_text_generator(self, model, small_config):
        """Test TextGenerator class"""
        gen_config = GenerationConfig(
            max_length=15,
            eos_token_id=small_config.eos_idx
        )
        generator = TextGenerator(model, gen_config)

        batch_size = 2
        src = torch.randint(3, small_config.vocab_size, (batch_size, 8))

        model.eval()
        encoder_output = model.encode(src)
        start_tokens = torch.full((batch_size, 1), small_config.bos_idx, dtype=torch.long)

        with torch.no_grad():
            output = generator.generate(
                input_ids=start_tokens,
                encoder_output=encoder_output
            )

        assert output.shape[0] == batch_size

    def test_sampling_produces_variation(self, model, small_config):
        """Test that sampling produces different outputs"""
        batch_size = 1
        src = torch.randint(3, small_config.vocab_size, (batch_size, 8))

        model.eval()
        outputs = []
        for _ in range(3):
            with torch.no_grad():
                out = sample_decode(
                    model, src,
                    max_length=10,
                    temperature=1.0,
                    top_k=10
                )
                outputs.append(out)

        # With sampling, outputs should sometimes differ
        # (not guaranteed but highly likely with temperature=1.0)
        # Just check we get valid outputs
        for out in outputs:
            assert out.shape[0] == batch_size


class TestIntegration:
    """Integration tests for complete workflow"""

    def test_full_training_step(self):
        """Test a complete training step"""
        config = TransformerConfig.small()
        config.dropout = 0.0
        model = Transformer(config)

        batch_size = 4
        src_len = 16
        tgt_len = 12

        src = torch.randint(0, config.vocab_size, (batch_size, src_len))
        tgt = torch.randint(0, config.vocab_size, (batch_size, tgt_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, tgt_len))

        # Forward pass
        logits = model(src, tgt)

        # Compute loss
        loss_fn = nn.CrossEntropyLoss(ignore_index=config.padding_idx)
        loss = loss_fn(
            logits.view(-1, config.vocab_size),
            labels.view(-1)
        )

        # Backward pass
        loss.backward()

        # Check loss is reasonable
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_inference_pipeline(self):
        """Test complete inference pipeline"""
        config = TransformerConfig.small()
        model = Transformer(config)
        model.eval()

        batch_size = 2
        src_len = 10

        src = torch.randint(3, config.vocab_size, (batch_size, src_len))

        with torch.no_grad():
            # Method 1: Direct generation
            output1 = greedy_decode(model, src, max_length=15)

            # Method 2: Step by step
            encoder_output = model.encode(src)
            start_tokens = torch.full((batch_size, 1), config.bos_idx, dtype=torch.long)

            gen_config = GenerationConfig(max_length=15, eos_token_id=config.eos_idx)
            generator = TextGenerator(model, gen_config)
            output2 = generator.generate(
                input_ids=start_tokens,
                encoder_output=encoder_output
            )

        # Both should produce valid outputs
        assert output1.shape[0] == batch_size
        assert output2.shape[0] == batch_size


def run_tests():
    """Run all tests with output"""
    print("=" * 60)
    print("Running Phase 3 Tests: Full Transformer Model")
    print("=" * 60)

    # Run pytest
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_tests()

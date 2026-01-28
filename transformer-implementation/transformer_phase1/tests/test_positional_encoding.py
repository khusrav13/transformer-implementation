"""
Comprehensive Test Suite for Positional Encoding
Tests all implementations for correctness and properties.
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
import math
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    RotaryPositionalEncoding,
    RelativePositionalEncoding,
    PositionalEncodingFactory
)


class TestSinusoidalPositionalEncoding:
    """Test suite for Sinusoidal Positional Encoding."""
    
    def test_output_shape(self):
        """Test that output shape matches input shape."""
        d_model = 64
        seq_len = 20
        batch_size = 4
        
        pe = SinusoidalPositionalEncoding(d_model, dropout=0.0)
        x = torch.randn(batch_size, seq_len, d_model)
        output = pe(x)
        
        assert output.shape == x.shape, \
            f"Output shape {output.shape} doesn't match input {x.shape}"
    
    def test_deterministic(self):
        """Test that sinusoidal encoding is deterministic (no randomness)."""
        d_model = 32
        pe = SinusoidalPositionalEncoding(d_model, dropout=0.0)
        
        x = torch.zeros(2, 10, d_model)
        output1 = pe(x.clone())
        output2 = pe(x.clone())
        
        # Remove the input (zeros) to get just the encoding
        encoding1 = output1 - x * math.sqrt(d_model)
        encoding2 = output2 - x * math.sqrt(d_model)
        
        assert torch.allclose(encoding1, encoding2), \
            "Sinusoidal encoding should be deterministic"
    
    def test_position_uniqueness(self):
        """Test that each position has a unique encoding."""
        d_model = 64
        seq_len = 50
        
        pe = SinusoidalPositionalEncoding(d_model, dropout=0.0)
        encoding = pe.get_encoding(seq_len)[0]  # Remove batch dimension
        
        # Check that no two positions have identical encodings
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                similarity = torch.cosine_similarity(
                    encoding[i].unsqueeze(0),
                    encoding[j].unsqueeze(0)
                ).item()
                assert similarity < 0.99, \
                    f"Positions {i} and {j} have too similar encodings: {similarity}"
    
    def test_periodicity_property(self):
        """Test the periodicity properties of sinusoidal encoding."""
        d_model = 64
        pe = SinusoidalPositionalEncoding(d_model, dropout=0.0)
        
        # Get encoding for a longer sequence
        encoding = pe.get_encoding(100)[0]
        
        # Check that different dimensions have different frequencies
        # Higher dimensions should have lower frequencies (longer periods)
        dim_0_encoding = encoding[:, 0].numpy()  # Highest frequency
        dim_last_encoding = encoding[:, -2].numpy()  # Lower frequency
        
        # Compute autocorrelation to find periodicity
        def autocorrelation(x):
            result = np.correlate(x, x, mode='full')
            return result[result.size // 2:]
        
        auto_0 = autocorrelation(dim_0_encoding)
        auto_last = autocorrelation(dim_last_encoding)
        
        # Find first peak (period) - higher dims should have longer periods
        # This is a simplified test - just checking they're different
        assert not np.allclose(auto_0[:20], auto_last[:20], rtol=0.1), \
            "Different dimensions should have different periodicities"
    
    def test_relative_position_property(self):
        """Test that relative positions can be recovered through dot products."""
        d_model = 128
        pe = SinusoidalPositionalEncoding(d_model, dropout=0.0)
        
        encoding = pe.get_encoding(50)[0]
        
        # For sinusoidal encoding, the dot product between positions
        # should depend mainly on their distance
        pos_5 = encoding[5]
        pos_10 = encoding[10]
        pos_15 = encoding[15]
        
        # Same distance (5) from pos_10
        dist_5_10 = torch.dot(pos_5, pos_10)
        dist_10_15 = torch.dot(pos_10, pos_15)
        
        # These should be similar (not exactly equal due to absolute position)
        assert abs(dist_5_10 - dist_10_15) < abs(dist_5_10) * 0.5, \
            "Similar relative distances should have similar dot products"
    
    def test_extrapolation_capability(self):
        """Test that sinusoidal encoding can handle longer sequences."""
        d_model = 64
        max_len = 100
        pe = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=0.0)
        
        # Should work for sequences up to max_len
        x_short = torch.randn(1, 50, d_model)
        x_long = torch.randn(1, max_len, d_model)
        
        output_short = pe(x_short)
        output_long = pe(x_long)
        
        assert output_short.shape == x_short.shape
        assert output_long.shape == x_long.shape
        
        # Should fail for sequences longer than max_len
        x_too_long = torch.randn(1, max_len + 10, d_model)
        with pytest.raises(Exception):
            _ = pe(x_too_long)


class TestLearnedPositionalEncoding:
    """Test suite for Learned Positional Encoding."""
    
    def test_output_shape(self):
        """Test output shape matches input."""
        d_model = 64
        seq_len = 20
        batch_size = 4
        
        pe = LearnedPositionalEncoding(d_model, dropout=0.0)
        x = torch.randn(batch_size, seq_len, d_model)
        output = pe(x)
        
        assert output.shape == x.shape
    
    def test_learnable_parameters(self):
        """Test that learned encoding has trainable parameters."""
        d_model = 32
        max_len = 100
        pe = LearnedPositionalEncoding(d_model, max_len=max_len)
        
        # Check that it has parameters
        params = list(pe.parameters())
        assert len(params) > 0, "Learned encoding should have parameters"
        
        # Check parameter shape
        assert params[0].shape == (max_len, d_model), \
            f"Parameter shape {params[0].shape} != expected {(max_len, d_model)}"
        
        # Check that parameters require gradients
        assert params[0].requires_grad, "Parameters should be trainable"
    
    def test_gradient_flow(self):
        """Test that gradients flow through learned encoding."""
        d_model = 32
        pe = LearnedPositionalEncoding(d_model, dropout=0.0)
        
        x = torch.randn(2, 10, d_model, requires_grad=True)
        output = pe(x)
        loss = output.sum()
        loss.backward()
        
        # Check input gradient
        assert x.grad is not None, "Input should have gradients"
        assert x.grad.abs().sum() > 0, "Input gradients should be non-zero"
        
        # Check parameter gradient
        for param in pe.parameters():
            assert param.grad is not None, "Parameters should have gradients"
            assert param.grad.abs().sum() > 0, "Parameter gradients should be non-zero"
    
    def test_different_positions_different_embeddings(self):
        """Test that different positions get different embeddings."""
        d_model = 64
        pe = LearnedPositionalEncoding(d_model, dropout=0.0)
        
        # Get position embeddings
        encoding = pe.get_encoding(10)[0]
        
        # Check all positions are different
        for i in range(10):
            for j in range(i + 1, 10):
                assert not torch.allclose(encoding[i], encoding[j]), \
                    f"Positions {i} and {j} have identical embeddings"


class TestRotaryPositionalEncoding:
    """Test suite for Rotary Positional Encoding (RoPE)."""
    
    def test_rotation_preserves_norm(self):
        """Test that RoPE preserves vector norms (rotation property)."""
        d_model = 64
        rope = RotaryPositionalEncoding(d_model)
        
        # Create random query and key
        batch_size, num_heads, seq_len = 2, 8, 10
        head_dim = d_model // num_heads
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        # Apply RoPE
        q_rot, k_rot = rope(q, k)
        
        # Check norm preservation
        assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), rtol=1e-5), \
            "RoPE should preserve vector norms"
        assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), rtol=1e-5), \
            "RoPE should preserve vector norms"
    
    def test_relative_position_property(self):
        """Test that dot product depends on relative position."""
        d_model = 64
        rope = RotaryPositionalEncoding(d_model)
        
        # Create position-aware vectors
        batch_size, num_heads, seq_len = 1, 1, 20
        head_dim = d_model
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = q.clone()  # Same content, different positions
        
        q_rot, k_rot = rope(q, k)
        
        # Compute attention scores
        scores = torch.matmul(q_rot, k_rot.transpose(-2, -1))[0, 0]
        
        # Check that diagonal (same position) has highest values
        diag_mean = scores.diag().mean()
        off_diag_mean = (scores.sum() - scores.diag().sum()) / (seq_len * (seq_len - 1))
        
        assert diag_mean > off_diag_mean, \
            "Same positions should have higher attention scores"
    
    def test_rotation_consistency(self):
        """Test that rotation is consistent across positions."""
        d_model = 64
        rope = RotaryPositionalEncoding(d_model)
        
        # Test with different sequence lengths
        q1 = torch.randn(1, 1, 10, d_model)
        k1 = torch.randn(1, 1, 10, d_model)
        
        q1_rot, k1_rot = rope(q1, k1)
        
        # Take a subset and rotate
        q2 = q1[:, :, :5, :]
        k2 = k1[:, :, :5, :]
        q2_rot, k2_rot = rope(q2, k2)
        
        # First 5 positions should be identical
        assert torch.allclose(q1_rot[:, :, :5, :], q2_rot, rtol=1e-5), \
            "Rotation should be consistent for same positions"


class TestRelativePositionalEncoding:
    """Test suite for Relative Positional Encoding."""
    
    def test_output_shape(self):
        """Test output shape for relative encoding."""
        d_model = 64
        max_rel_pos = 16
        pe = RelativePositionalEncoding(d_model, max_rel_pos)
        
        seq_len = 20
        output = pe(seq_len)
        
        assert output.shape == (seq_len, seq_len, d_model), \
            f"Output shape {output.shape} != expected {(seq_len, seq_len, d_model)}"
    
    def test_symmetry_property(self):
        """Test that relative positions are symmetric."""
        d_model = 32
        pe = RelativePositionalEncoding(d_model, max_relative_position=10)
        
        seq_len = 15
        rel_pos = pe(seq_len)
        
        # Check that position (i,j) and (j,i) have related encodings
        # They should represent opposite relative positions
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                # These represent opposite relative positions
                # So they should be different but related
                assert not torch.allclose(rel_pos[i, j], rel_pos[j, i]), \
                    f"Relative positions ({i},{j}) and ({j},{i}) should differ"
    
    def test_clipping_behavior(self):
        """Test that relative positions are clipped correctly."""
        d_model = 32
        max_rel = 5
        pe = RelativePositionalEncoding(d_model, max_relative_position=max_rel)
        
        seq_len = 20  # Much longer than max_rel
        rel_pos = pe(seq_len)
        
        # Positions far apart should have the same encoding as max_rel
        # Check that position (0, max_rel+5) has same encoding as (0, max_rel+10)
        if seq_len > max_rel + 10:
            pos1 = rel_pos[0, max_rel + 5]
            pos2 = rel_pos[0, max_rel + 10]
            assert torch.allclose(pos1, pos2), \
                "Positions beyond max_relative should be clipped"


class TestPositionalEncodingFactory:
    """Test the factory pattern for creating encodings."""
    
    def test_create_all_types(self):
        """Test that factory can create all encoding types."""
        d_model = 64
        
        # Test sinusoidal
        sin_pe = PositionalEncodingFactory.create_positional_encoding(
            'sinusoidal', d_model
        )
        assert isinstance(sin_pe, SinusoidalPositionalEncoding)
        
        # Test learned
        learned_pe = PositionalEncodingFactory.create_positional_encoding(
            'learned', d_model
        )
        assert isinstance(learned_pe, LearnedPositionalEncoding)
        
        # Test rotary
        rope = PositionalEncodingFactory.create_positional_encoding(
            'rotary', d_model
        )
        assert isinstance(rope, RotaryPositionalEncoding)
        
        # Test relative
        rel_pe = PositionalEncodingFactory.create_positional_encoding(
            'relative', d_model
        )
        assert isinstance(rel_pe, RelativePositionalEncoding)
    
    def test_invalid_type_raises_error(self):
        """Test that invalid encoding type raises error."""
        with pytest.raises(ValueError):
            PositionalEncodingFactory.create_positional_encoding(
                'invalid_type', 64
            )


class TestIntegrationWithAttention:
    """Test positional encodings integrated with attention mechanism."""
    
    def test_sinusoidal_with_attention(self):
        """Test that sinusoidal encoding works with attention."""
        from src.multi_head import MultiHeadAttention
        
        d_model = 64
        num_heads = 8
        seq_len = 20
        batch_size = 2
        
        # Create modules
        attention = MultiHeadAttention(d_model, num_heads, dropout=0.0)
        pe = SinusoidalPositionalEncoding(d_model, dropout=0.0)
        
        # Create input
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Apply positional encoding then attention
        x_with_pe = pe(x)
        output, _ = attention(x_with_pe, x_with_pe, x_with_pe)
        
        assert output.shape == x.shape, "Output shape should match input"
        assert not torch.isnan(output).any(), "Output should not contain NaN"
        assert not torch.isinf(output).any(), "Output should not contain Inf"
    
    def test_learned_with_attention(self):
        """Test that learned encoding works with attention."""
        from src.multi_head import MultiHeadAttention
        
        d_model = 64
        num_heads = 8
        seq_len = 20
        batch_size = 2
        
        # Create modules
        attention = MultiHeadAttention(d_model, num_heads, dropout=0.0)
        pe = LearnedPositionalEncoding(d_model, dropout=0.0)
        
        # Create input
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Apply positional encoding then attention
        x_with_pe = pe(x)
        output, _ = attention(x_with_pe, x_with_pe, x_with_pe)
        
        assert output.shape == x.shape, "Output shape should match input"
        assert not torch.isnan(output).any(), "Output should not contain NaN"


def run_all_tests():
    """Run all positional encoding tests."""
    print("Running Positional Encoding Tests")
    print("="*60)
    
    # Test Sinusoidal
    print("\nTesting Sinusoidal Positional Encoding...")
    sin_tests = TestSinusoidalPositionalEncoding()
    sin_tests.test_output_shape()
    print("✓ Output shape correct")
    sin_tests.test_deterministic()
    print("✓ Encoding is deterministic")
    sin_tests.test_position_uniqueness()
    print("✓ Each position has unique encoding")
    sin_tests.test_periodicity_property()
    print("✓ Periodicity properties correct")
    sin_tests.test_relative_position_property()
    print("✓ Relative position property holds")
    
    # Test Learned
    print("\nTesting Learned Positional Encoding...")
    learned_tests = TestLearnedPositionalEncoding()
    learned_tests.test_output_shape()
    print("✓ Output shape correct")
    learned_tests.test_learnable_parameters()
    print("✓ Has learnable parameters")
    learned_tests.test_gradient_flow()
    print("✓ Gradients flow correctly")
    learned_tests.test_different_positions_different_embeddings()
    print("✓ Different positions get different embeddings")
    
    # Test RoPE
    print("\nTesting Rotary Positional Encoding...")
    rope_tests = TestRotaryPositionalEncoding()
    rope_tests.test_rotation_preserves_norm()
    print("✓ Rotation preserves norm")
    rope_tests.test_relative_position_property()
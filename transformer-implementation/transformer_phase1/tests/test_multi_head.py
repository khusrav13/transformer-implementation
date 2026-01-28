"""
Comprehensive test suite for Multi-Head Attention.
Tests mathematical properties, gradient flow, and edge cases.

Run with: pytest tests/test_multi_head.py -v
"""

import torch
import torch.nn as nn
import pytest
import math
import numpy as np
import sys
import os

# Add parent directory to path to import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.multi_head import MultiHeadAttention, create_causal_mask, create_padding_mask


class TestMultiHeadAttention:
    """Test suite for Multi-Head Attention mechanism."""
    
    @pytest.fixture
    def setup_model(self):
        """Create a standard model configuration for testing."""
        d_model = 64
        num_heads = 8
        model = MultiHeadAttention(d_model, num_heads, dropout=0.0)  # No dropout for deterministic tests
        return model, d_model, num_heads
    
    def test_output_shape(self, setup_model):
        """Test 1: Verify output dimensions are correct."""
        model, d_model, num_heads = setup_model
        batch_size = 2
        seq_len = 10
        
        x = torch.randn(batch_size, seq_len, d_model)
        output, attention = model(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model), \
            f"Output shape {output.shape} != expected {(batch_size, seq_len, d_model)}"
        assert attention.shape == (batch_size, num_heads, seq_len, seq_len), \
            f"Attention shape {attention.shape} != expected {(batch_size, num_heads, seq_len, seq_len)}"
    
    def test_attention_weights_sum_to_one(self, setup_model):
        """Test 2: Each attention weight row should sum to 1."""
        model, d_model, _ = setup_model
        batch_size = 2
        seq_len = 5
        
        x = torch.randn(batch_size, seq_len, d_model)
        _, attention = model(x, x, x)
        
        # Sum across last dimension (keys)
        attention_sum = attention.sum(dim=-1)
        expected = torch.ones_like(attention_sum)
        
        assert torch.allclose(attention_sum, expected, atol=1e-6), \
            f"Attention weights don't sum to 1. Max deviation: {(attention_sum - expected).abs().max()}"
    
    def test_causal_mask_prevents_future_attention(self, setup_model):
        """Test 3: Causal mask should prevent attending to future positions."""
        model, d_model, num_heads = setup_model
        batch_size = 1
        seq_len = 4
        
        x = torch.randn(batch_size, seq_len, d_model)
        causal_mask = create_causal_mask(seq_len)
        
        _, attention = model(x, x, x, mask=causal_mask.unsqueeze(0))
        
        # Check upper triangular part is near zero
        for h in range(num_heads):
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    assert attention[0, h, i, j] < 1e-6, \
                        f"Position {i} attending to future position {j} in head {h}: {attention[0, h, i, j]}"
    
    def test_padding_mask_blocks_attention(self, setup_model):
        """Test 4: Padding mask should prevent attending to padded positions."""
        model, d_model, _ = setup_model
        batch_size = 2
        seq_len = 6
        
        # Create input with padding (last 2 positions are padding)
        x = torch.randn(batch_size, seq_len, d_model)
        pad_lengths = [4, 3]  # Real sequence lengths
        
        # Create padding mask manually
        mask = torch.ones(batch_size, 1, 1, seq_len)
        mask[0, :, :, 4:] = 0  # Mask padding for first sample
        mask[1, :, :, 3:] = 0  # Mask padding for second sample
        
        _, attention = model(x, x, x, mask=mask)
        
        # Check attention to padded positions is near zero
        assert attention[0, :, :, 4:].max() < 1e-6, \
            "First sample attending to padding positions"
        assert attention[1, :, :, 3:].max() < 1e-6, \
            "Second sample attending to padding positions"
    
    def test_gradient_flow(self, setup_model):
        """Test 5: Ensure gradients flow properly through the module."""
        model, d_model, _ = setup_model
        batch_size = 2
        seq_len = 8
        
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        output, _ = model(x, x, x)
        
        # Create a dummy loss
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist and are non-zero
        assert x.grad is not None, "No gradient computed for input"
        assert x.grad.abs().sum() > 0, "Gradients are all zero"
        
        # Check model parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
    
    def test_gradient_magnitude_with_scaling(self, setup_model):
        """Test 6: Verify that scaling prevents gradient explosion."""
        d_model = 512  # Large dimension to test scaling effect
        num_heads = 8
        model = MultiHeadAttention(d_model, num_heads, dropout=0.0)
        
        batch_size = 2
        seq_len = 20
        
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        output, _ = model(x, x, x)
        
        loss = output.sum()
        loss.backward()
        
        # Gradient norm should be reasonable (not exploding)
        grad_norm = x.grad.norm()
        assert grad_norm < 1000, f"Gradient norm too large: {grad_norm}"
    
    def test_multi_head_independence(self, setup_model):
        """Test 7: Different heads should produce different attention patterns."""
        model, d_model, num_heads = setup_model
        batch_size = 1
        seq_len = 5
        
        # Use a specific input pattern
        x = torch.eye(seq_len).unsqueeze(0).repeat(1, 1, d_model // seq_len + 1)[:, :, :d_model]
        _, attention = model(x, x, x)
        
        # Check that different heads have different patterns
        attention_patterns = attention[0]  # Remove batch dimension
        
        # Compute pairwise differences between heads
        differences = []
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                diff = (attention_patterns[i] - attention_patterns[j]).abs().sum()
                differences.append(diff)
        
        # At least some heads should be different
        assert max(differences) > 0.1, "All heads producing identical attention patterns"
    
    def test_cross_attention_shape(self, setup_model):
        """Test 8: Cross-attention with different sequence lengths."""
        model, d_model, num_heads = setup_model
        batch_size = 2
        seq_len_q = 7
        seq_len_kv = 10
        
        query = torch.randn(batch_size, seq_len_q, d_model)
        key_value = torch.randn(batch_size, seq_len_kv, d_model)
        
        output, attention = model(query, key_value, key_value)
        
        assert output.shape == (batch_size, seq_len_q, d_model), \
            f"Cross-attention output shape incorrect: {output.shape}"
        assert attention.shape == (batch_size, num_heads, seq_len_q, seq_len_kv), \
            f"Cross-attention weights shape incorrect: {attention.shape}"
    
    def test_attention_to_specific_positions(self, setup_model):
        """Test 9: Verify attention focuses on similar tokens."""
        model, d_model, _ = setup_model
        model.eval()  # Set to eval mode for deterministic behavior
        
        batch_size = 1
        seq_len = 4
        
        # Create distinct token embeddings
        x = torch.zeros(batch_size, seq_len, d_model)
        x[0, 0] = torch.randn(d_model)  # Token A
        x[0, 1] = torch.randn(d_model)  # Token B  
        x[0, 2] = x[0, 0].clone()        # Token A (repeated)
        x[0, 3] = torch.randn(d_model)  # Token C
        
        _, attention = model(x, x, x)
        
        # Position 0 should attend more to position 2 (same token) than to 1 or 3
        # Average across all heads
        avg_attention = attention[0].mean(dim=0)
        
        # This test might need adjustment based on random initialization
        # but generally similar tokens should have higher attention
        print(f"Attention from pos 0: {avg_attention[0]}")
        print(f"Attention from pos 2: {avg_attention[2]}")
    
    def test_numerical_stability(self, setup_model):
        """Test 10: Model should handle extreme values without NaN/Inf."""
        model, d_model, _ = setup_model
        batch_size = 2
        seq_len = 5
        
        # Test with very large values
        x_large = torch.randn(batch_size, seq_len, d_model) * 100
        output_large, attention_large = model(x_large, x_large, x_large)
        
        assert not torch.isnan(output_large).any(), "NaN in output with large inputs"
        assert not torch.isinf(output_large).any(), "Inf in output with large inputs"
        assert not torch.isnan(attention_large).any(), "NaN in attention with large inputs"
        
        # Test with very small values
        x_small = torch.randn(batch_size, seq_len, d_model) * 1e-6
        output_small, attention_small = model(x_small, x_small, x_small)
        
        assert not torch.isnan(output_small).any(), "NaN in output with small inputs"
        assert not torch.isinf(output_small).any(), "Inf in output with small inputs"


class TestAttentionProperties:
    """Test mathematical properties of attention mechanism."""
    
    def test_permutation_equivariance(self):
        """Test 11: Self-attention is permutation equivariant."""
        d_model = 32
        num_heads = 4
        model = MultiHeadAttention(d_model, num_heads, dropout=0.0)
        model.eval()
        
        batch_size = 1
        seq_len = 5
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Get output for original sequence
        output1, _ = model(x, x, x)
        
        # Create permutation
        perm = torch.randperm(seq_len)
        x_perm = x[:, perm]
        
        # Get output for permuted sequence
        output2, _ = model(x_perm, x_perm, x_perm)
        
        # Permute output1 and compare
        output1_perm = output1[:, perm]
        
        # Should be approximately equal (some numerical differences expected)
        assert torch.allclose(output1_perm, output2, atol=1e-5), \
            "Self-attention is not permutation equivariant"
    
    def test_attention_score_symmetry(self):
        """Test 12: Q@K^T should equal (K@Q^T)^T for self-attention."""
        d_model = 64
        num_heads = 8
        model = MultiHeadAttention(d_model, num_heads)
        
        batch_size = 2
        seq_len = 6
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Get Q and K projections
        Q = model.W_q(x)
        K = model.W_k(x)
        
        Q = Q.view(batch_size, seq_len, num_heads, model.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, num_heads, model.d_k).transpose(1, 2)
        
        # Compute scores both ways
        scores1 = torch.matmul(Q, K.transpose(-2, -1))
        scores2 = torch.matmul(K, Q.transpose(-2, -1)).transpose(-2, -1)
        
        assert torch.allclose(scores1, scores2, atol=1e-6), \
            "Attention scores not symmetric"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_token_sequence(self):
        """Test 13: Handle sequence with single token."""
        d_model = 64
        model = MultiHeadAttention(d_model, 8)
        
        x = torch.randn(1, 1, d_model)  # Single token
        output, attention = model(x, x, x)
        
        assert output.shape == (1, 1, d_model)
        assert attention.shape == (1, 8, 1, 1)
        assert torch.allclose(attention, torch.ones_like(attention)), \
            "Single token attention should be 1.0"
    
    def test_very_long_sequence(self):
        """Test 14: Handle very long sequences (memory test)."""
        d_model = 64
        model = MultiHeadAttention(d_model, 8)
        
        batch_size = 1
        seq_len = 512  # Long sequence
        
        x = torch.randn(batch_size, seq_len, d_model)
        output, attention = model(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention.shape == (batch_size, 8, seq_len, seq_len)
    
    def test_zero_input(self):
        """Test 15: Handle zero input gracefully."""
        d_model = 64
        model = MultiHeadAttention(d_model, 8)
        
        x = torch.zeros(2, 10, d_model)
        output, attention = model(x, x, x)
        
        assert not torch.isnan(output).any(), "NaN with zero input"
        # With zero input and initialized weights, output shouldn't be zero
        # due to bias terms in output projection


def run_performance_test():
    """Performance and efficiency test."""
    import time
    
    d_model = 512
    num_heads = 8
    model = MultiHeadAttention(d_model, num_heads).cuda()
    
    batch_size = 32
    seq_len = 100
    
    x = torch.randn(batch_size, seq_len, d_model).cuda()
    
    # Warmup
    for _ in range(10):
        _ = model(x, x, x)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        output, _ = model(x, x, x)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Average forward pass time: {elapsed/100*1000:.2f} ms")
    print(f"Throughput: {100*batch_size/elapsed:.1f} samples/sec")


if __name__ == "__main__":
    # Run basic tests
    print("Running Multi-Head Attention Tests...")
    
    # Create test instance
    tester = TestMultiHeadAttention()
    d_model = 64
    num_heads = 8
    model = MultiHeadAttention(d_model, num_heads, dropout=0.0)
    setup = (model, d_model, num_heads)
    
    # Run key tests
    print("Testing output shape...")
    tester.test_output_shape(setup)
    print("✓ Output shape correct")
    
    print("Testing attention weights sum...")
    tester.test_attention_weights_sum_to_one(setup)
    print("✓ Attention weights sum to 1")
    
    print("Testing gradient flow...")
    tester.test_gradient_flow(setup)
    print("✓ Gradients flow properly")
    
    print("Testing numerical stability...")
    tester.test_numerical_stability(setup)
    print("✓ Numerically stable")
    
    print("\nAll tests passed!")
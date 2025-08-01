"""
Test suite for attention mechanism.
Run with: pytest tests/test_attention.py -v
"""

import torch
import pytest
import math
from src.attention import scaled_dot_product_attention


def test_attention_output_shape():
    """Test 1: Output has correct shape"""
    seq_len = 4
    d_k = 8
    d_v = 8
    
    q = torch.randn(seq_len, d_k)
    k = torch.randn(seq_len, d_k)
    v = torch.randn(seq_len, d_v)
    
    output, weights = scaled_dot_product_attention(q, k, v)
    
    assert output.shape == (seq_len, d_v), f"Output shape is {output.shape}, expected {(seq_len, d_v)}"
    assert weights.shape == (seq_len, seq_len), f"Weights shape is {weights.shape}, expected {(seq_len, seq_len)}"


def test_attention_weights_sum_to_one():
    """Test 2: Each row of attention weights sums to 1"""
    seq_len = 5
    d_k = 4
    
    q = torch.randn(seq_len, d_k)
    k = torch.randn(seq_len, d_k)
    v = torch.randn(seq_len, d_k)
    
    output, weights = scaled_dot_product_attention(q, k, v)
    
    # Check that each row sums to 1 (with small tolerance for floating point errors)
    row_sums = weights.sum(dim=1)
    expected = torch.ones(seq_len)
    
    assert torch.allclose(row_sums, expected, atol=1e-6), \
        f"Row sums are {row_sums}, expected all 1.0"


def test_attention_with_identity():
    """Test 3: When Q=K=V=I, check expected behavior"""
    seq_len = 4
    
    # Identity matrix
    identity = torch.eye(seq_len)
    
    output, weights = scaled_dot_product_attention(identity, identity, identity)
    
    # With identity inputs, the attention should be somewhat uniform
    # but slightly favor the diagonal due to the dot product
    
    # Check output shape
    assert output.shape == (seq_len, seq_len)
    
    # TODO: Add your own assertion about what you expect to happen
    # Hint: What happens when you compute I @ I^T?


def test_attention_with_mask():
    """Test 4: Masking works correctly"""
    seq_len = 3
    d_k = 2
    
    q = torch.ones(seq_len, d_k)
    k = torch.ones(seq_len, d_k)
    v = torch.eye(seq_len)  # Identity to track where attention goes
    
    # Create a mask that prevents attending to position 1
    mask = torch.tensor([[0., -float('inf'), 0.],
                        [0., -float('inf'), 0.],
                        [0., -float('inf'), 0.]])
    
    output, weights = scaled_dot_product_attention(q, k, v, mask=mask)
    
    # Check that attention weights for position 1 are near 0
    assert torch.allclose(weights[:, 1], torch.zeros(seq_len), atol=1e-6), \
        "Masked positions should have near-zero attention"
"""
Masking Utilities for Transformer
This module provides essential masking functions for attention mechanisms.

Key concepts:
1. Padding Mask: Prevents attention to padding tokens
2. Look-ahead Mask: Prevents decoder from attending to future positions
3. Combined Mask: Combines padding and look-ahead masks for decoder
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


def create_padding_mask(
    seq: torch.Tensor, 
    pad_idx: int = 0
) -> torch.Tensor:
    """
    Create a padding mask to ignore padding tokens in attention.
    
    The padding mask ensures that the attention mechanism doesn't attend to
    padding tokens, which are added to make all sequences in a batch equal length.
    
    Args:
        seq: Input sequence tensor of shape [batch_size, seq_len] containing token IDs
        pad_idx: The index used for padding tokens (default: 0)
    
    Returns:
        mask: Boolean mask of shape [batch_size, 1, 1, seq_len]
              where True = valid token, False = padding token
    
    Example:
        >>> seq = torch.tensor([[1, 2, 3, 0, 0],
        ...                     [1, 2, 0, 0, 0]])
        >>> mask = create_padding_mask(seq, pad_idx=0)
        >>> # mask[0] will have True for positions 0,1,2 and False for 3,4
        >>> # mask[1] will have True for positions 0,1 and False for 2,3,4
    
    Mathematical insight:
        In attention: scores.masked_fill(mask == 0, -1e9)
        After softmax, -1e9 becomes ~0, effectively ignoring padding
    """
    # Create boolean mask: True where tokens are NOT padding
    # Shape: [batch_size, seq_len]
    mask = (seq != pad_idx)
    
    # Add dimensions for broadcasting with attention scores
    # Shape: [batch_size, 1, 1, seq_len]
    # The middle dimensions will broadcast across num_heads and seq_len (query dimension)
    mask = mask.unsqueeze(1).unsqueeze(2)
    
    return mask


def create_look_ahead_mask(
    size: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create a look-ahead (causal) mask to prevent attending to future positions.
    
    This mask is crucial for decoder self-attention during training. It ensures
    that position i can only attend to positions j <= i, maintaining the
    autoregressive property of the decoder.
    
    Args:
        size: Sequence length
        device: Device to create the mask on (CPU/CUDA)
    
    Returns:
        mask: Boolean mask of shape [1, 1, size, size]
              Lower triangular matrix where True = can attend, False = cannot attend
    
    Example:
        >>> mask = create_look_ahead_mask(4)
        >>> # Creates a 4x4 matrix:
        >>> # [[1, 0, 0, 0],
        >>> #  [1, 1, 0, 0],
        >>> #  [1, 1, 1, 0],
        >>> #  [1, 1, 1, 1]]
    
    Mathematical insight:
        This creates an upper triangular matrix of -inf values (after applying).
        When added to attention scores before softmax, future positions get
        probability ~0, enforcing causality.
    """
    # Create a lower triangular matrix
    # torch.tril creates lower triangular matrix (1s on and below diagonal)
    mask = torch.tril(torch.ones((size, size), device=device)).bool()
    
    # Add batch and head dimensions for broadcasting
    # Shape: [1, 1, size, size]
    mask = mask.unsqueeze(0).unsqueeze(0)
    
    return mask


def create_combined_mask(
    seq: torch.Tensor,
    pad_idx: int = 0
) -> torch.Tensor:
    """
    Combine padding mask and look-ahead mask for decoder self-attention.
    
    The decoder needs both masks:
    1. Don't attend to padding tokens (padding mask)
    2. Don't attend to future positions (look-ahead mask)
    
    Args:
        seq: Target sequence tensor of shape [batch_size, seq_len]
        pad_idx: Padding token index
    
    Returns:
        mask: Combined mask of shape [batch_size, 1, seq_len, seq_len]
    
    Example:
        >>> seq = torch.tensor([[1, 2, 3, 0, 0]])  # Last 2 are padding
        >>> mask = create_combined_mask(seq)
        >>> # Position 0 can only see position 0 (and it's not padding)
        >>> # Position 1 can see positions 0,1 (and they're not padding)
        >>> # Position 2 can see positions 0,1,2 (and they're not padding)
        >>> # Positions 3,4 are padding, so nobody can attend to them
    """
    batch_size, seq_len = seq.shape
    
    # Create padding mask
    # Shape: [batch_size, 1, 1, seq_len]
    padding_mask = create_padding_mask(seq, pad_idx)
    
    # Create look-ahead mask
    # Shape: [1, 1, seq_len, seq_len]
    look_ahead_mask = create_look_ahead_mask(seq_len, device=seq.device)
    
    # Combine masks using logical AND
    # Both conditions must be true for attention to be allowed
    # Broadcasting: [batch_size, 1, 1, seq_len] & [1, 1, seq_len, seq_len]
    # Result: [batch_size, 1, seq_len, seq_len]
    combined_mask = padding_mask & look_ahead_mask
    
    return combined_mask


def create_encoder_decoder_mask(
    encoder_seq: torch.Tensor,
    decoder_seq: torch.Tensor,
    pad_idx: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create all masks needed for encoder-decoder attention.
    
    Args:
        encoder_seq: Source sequence [batch_size, src_len]
        decoder_seq: Target sequence [batch_size, tgt_len]
        pad_idx: Padding token index
    
    Returns:
        encoder_mask: Padding mask for encoder self-attention
        decoder_mask: Combined mask for decoder self-attention
        memory_mask: Padding mask for encoder-decoder cross-attention
    
    This function creates all three masks needed in a transformer:
    1. Encoder self-attention: only mask padding
    2. Decoder self-attention: mask padding AND future positions
    3. Cross-attention: decoder queries attend to encoder, mask encoder padding
    """
    # Encoder just needs padding mask
    encoder_mask = create_padding_mask(encoder_seq, pad_idx)
    
    # Decoder needs combined mask (padding + look-ahead)
    decoder_mask = create_combined_mask(decoder_seq, pad_idx)
    
    # For cross-attention, decoder attends to encoder
    # We need to mask encoder padding positions
    memory_mask = create_padding_mask(encoder_seq, pad_idx)
    
    return encoder_mask, decoder_mask, memory_mask


def apply_mask(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    mask_value: float = -1e9
) -> torch.Tensor:
    """
    Apply mask to attention scores.
    
    Args:
        scores: Attention scores [batch_size, num_heads, seq_len, seq_len]
        mask: Boolean mask where True = keep, False = mask out
        mask_value: Value to use for masked positions (default: -1e9)
    
    Returns:
        Masked scores ready for softmax
    
    Note:
        Using -1e9 instead of -inf prevents NaN during training.
        After softmax, e^(-1e9) ≈ 0, effectively zeroing attention.
    """
    if mask is not None:
        # masked_fill replaces values where mask is False with mask_value
        scores = scores.masked_fill(mask == 0, mask_value)
    return scores


def visualize_mask(
    mask: torch.Tensor,
    title: str = "Mask Visualization"
) -> None:
    """
    Utility function to visualize masks (useful for debugging).
    
    Args:
        mask: Mask tensor to visualize
        title: Title for the visualization
    
    Example:
        >>> mask = create_look_ahead_mask(5)
        >>> visualize_mask(mask[0, 0], "Causal Mask")
    """
    import matplotlib.pyplot as plt
    
    # Handle different mask dimensions
    if mask.dim() == 4:
        # Take first batch and head
        mask_2d = mask[0, 0].cpu().numpy()
    elif mask.dim() == 2:
        mask_2d = mask.cpu().numpy()
    else:
        raise ValueError(f"Unexpected mask dimension: {mask.dim()}")
    
    plt.figure(figsize=(8, 6))
    plt.imshow(mask_2d, cmap='Blues', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.show()


# Utility functions for testing
def test_padding_mask():
    """Test padding mask creation."""
    print("Testing Padding Mask...")
    
    # Create sample sequence with padding
    seq = torch.tensor([[1, 2, 3, 0, 0],
                        [1, 2, 0, 0, 0]])
    
    mask = create_padding_mask(seq)
    print(f"Input shape: {seq.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask for first sequence:\n{mask[0, 0, 0]}")
    print(f"Mask for second sequence:\n{mask[1, 0, 0]}")
    print("✓ Padding mask test passed\n")


def test_look_ahead_mask():
    """Test look-ahead mask creation."""
    print("Testing Look-ahead Mask...")
    
    mask = create_look_ahead_mask(5)
    print(f"Mask shape: {mask.shape}")
    print(f"5x5 Causal mask:\n{mask[0, 0].int()}")
    print("✓ Look-ahead mask test passed\n")


def test_combined_mask():
    """Test combined mask creation."""
    print("Testing Combined Mask...")
    
    # Sequence with padding
    seq = torch.tensor([[1, 2, 3, 0, 0]])
    
    mask = create_combined_mask(seq)
    print(f"Input shape: {seq.shape}")
    print(f"Combined mask shape: {mask.shape}")
    print(f"Combined mask:\n{mask[0, 0].int()}")
    print("✓ Combined mask test passed\n")


if __name__ == "__main__":
    print("="*50)
    print("Running Masking Module Tests")
    print("="*50)
    
    test_padding_mask()
    test_look_ahead_mask()
    test_combined_mask()
    
    print("All masking tests passed successfully!")
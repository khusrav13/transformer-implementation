"""
Utils Module Initialization
This module provides utility functions for the transformer implementation.
"""

# Import all masking functions
from .masking import (
    create_padding_mask,
    create_look_ahead_mask,
    create_combined_mask,
    create_encoder_decoder_mask,
    apply_mask,
    visualize_mask
)

# Define what's available when someone does "from utils import *"
__all__ = [
    # Masking utilities
    'create_padding_mask',
    'create_look_ahead_mask', 
    'create_combined_mask',
    'create_encoder_decoder_mask',
    'apply_mask',
    'visualize_mask',
]

# Optional: Module version
__version__ = '0.1.0'

# Optional: Module level docstring for help()
__doc__ = """
Transformer Utility Functions

This module provides essential utilities for transformer models:

Masking Functions:
-----------------
- create_padding_mask: Mask padding tokens in sequences
- create_look_ahead_mask: Causal mask for decoder self-attention  
- create_combined_mask: Combine padding and causal masks
- create_encoder_decoder_mask: Generate all masks for encoder-decoder model
- apply_mask: Apply mask to attention scores
- visualize_mask: Visualize mask matrices for debugging

Usage Example:
-------------
>>> from utils import create_padding_mask, create_look_ahead_mask
>>> 
>>> # Create padding mask for a batch
>>> sequences = torch.tensor([[1, 2, 3, 0, 0]])
>>> mask = create_padding_mask(sequences)
>>> 
>>> # Create causal mask
>>> causal_mask = create_look_ahead_mask(5)
"""
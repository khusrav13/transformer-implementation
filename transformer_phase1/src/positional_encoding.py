"""
Positional Encoding Implementations for Transformers
Includes Sinusoidal, Learned, and Rotary Position Embeddings (RoPE)
Place in: src/positional_encoding.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as described in "Attention Is All You Need".
    
    This encoding uses sine and cosine functions of different frequencies:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Benefits:
    - No learnable parameters
    - Can extrapolate to longer sequences than seen during training
    - Captures relative positions through trigonometric properties
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension (must match embedding dimension)
            max_len: Maximum sequence length to pre-compute
            dropout: Dropout rate for regularization
        """
        super(SinusoidalPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a matrix to hold positional encodings
        pe = torch.zeros(max_len, d_model)
        
        # Create position indices [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create dimension indices for the frequency calculation
        # This implements the 10000^(2i/d_model) part
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        # Handle case where d_model is odd
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Tensor with positional encoding added [batch_size, seq_len, d_model]
        """
        # Extract positional encoding for the actual sequence length
        seq_len = x.size(1)
        
        # Scale embeddings (mentioned in paper but often omitted)
        x = x * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = x + self.pe[:, :seq_len, :]
        
        return self.dropout(x)
    
    def get_encoding(self, seq_len: int) -> torch.Tensor:
        """
        Get positional encoding matrix without adding to input.
        
        Args:
            seq_len: Sequence length
        
        Returns:
            Positional encoding matrix [1, seq_len, d_model]
        """
        return self.pe[:, :seq_len, :]


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding (as used in BERT).
    
    This encoding learns position embeddings during training.
    
    Benefits:
    - Can learn task-specific position patterns
    - Simple implementation
    
    Drawbacks:
    - Cannot extrapolate beyond maximum training length
    - Requires more parameters
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(LearnedPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # Create learnable position embeddings
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # Initialize with small values
        nn.init.normal_(self.pos_embedding.weight, mean=0, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Tensor with positional encoding added
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        
        # Get position embeddings
        pos_embed = self.pos_embedding(positions)
        
        # Scale input embeddings
        x = x * math.sqrt(self.d_model)
        
        # Add position embeddings
        x = x + pos_embed
        
        return self.dropout(x)
    
    def get_encoding(self, seq_len: int) -> torch.Tensor:
        """
        Get positional encoding matrix.
        
        Args:
            seq_len: Sequence length
        
        Returns:
            Positional encoding matrix [1, seq_len, d_model]
        """
        positions = torch.arange(seq_len).unsqueeze(0)
        return self.pos_embedding(positions)


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Encoding (RoPE) - Modern approach used in LLaMA, GPT-NeoX.
    
    RoPE applies rotation matrices to encode positions, maintaining relative
    position information through dot products.
    
    Benefits:
    - Preserves relative position information perfectly
    - No position embedding addition needed
    - Can extrapolate to longer sequences
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, base: int = 10000):
        """
        Args:
            d_model: Model dimension (must be even)
            max_len: Maximum sequence length
            base: Base for the frequency calculation
        """
        super(RotaryPositionalEncoding, self).__init__()
        
        assert d_model % 2 == 0, "Model dimension must be even for RoPE"
        
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # Precompute the frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin for efficiency
        self._precompute_cos_sin_cache()
        
    def _precompute_cos_sin_cache(self):
        """Precompute cosine and sine values for all positions."""
        seq_idx = torch.arange(self.max_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', seq_idx, self.inv_freq)
        
        # Create rotation angles
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Cache cosine and sine
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
        
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the hidden dims of the input.
        
        Args:
            x: Input tensor
        
        Returns:
            Rotated tensor
        """
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position encoding to query and key tensors.
        
        Args:
            q: Query tensor [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor [batch_size, num_heads, seq_len, head_dim]
        
        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Get cached cos and sin values
        cos = self.cos_cached[:, :, :seq_len, :head_dim]
        sin = self.sin_cached[:, :, :seq_len, :head_dim]
        
        # Apply rotation
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_embed, k_embed


class RelativePositionalEncoding(nn.Module):
    """
    Relative Positional Encoding - Used in Transformer-XL and similar models.
    
    Instead of absolute positions, this encodes the relative distance between positions.
    This is particularly useful for tasks where relative position matters more than absolute.
    """
    
    def __init__(self, d_model: int, max_relative_position: int = 32):
        """
        Args:
            d_model: Model dimension
            max_relative_position: Maximum relative distance to consider
        """
        super(RelativePositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # Create relative position embeddings
        # We need 2 * max_relative_position + 1 embeddings
        # For positions from -max_relative_position to +max_relative_position
        num_embeddings = 2 * max_relative_position + 1
        self.relative_positions_embedding = nn.Embedding(num_embeddings, d_model)
        
        # Initialize
        nn.init.xavier_uniform_(self.relative_positions_embedding.weight)
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Generate relative position encoding matrix.
        
        Args:
            seq_len: Sequence length
        
        Returns:
            Relative position bias tensor [seq_len, seq_len, d_model]
        """
        # Create relative position matrix
        range_vec = torch.arange(seq_len)
        distance_mat = range_vec[None, :] - range_vec[:, None]
        
        # Clip distances to max_relative_position
        distance_mat = torch.clamp(
            distance_mat, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # Shift to make indices positive
        distance_mat = distance_mat + self.max_relative_position
        
        # Get embeddings
        embeddings = self.relative_positions_embedding(distance_mat)
        
        return embeddings


class PositionalEncodingFactory:
    """Factory class to create different positional encoding types."""
    
    @staticmethod
    def create_positional_encoding(
        encoding_type: str,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        **kwargs
    ) -> nn.Module:
        """
        Create a positional encoding module.
        
        Args:
            encoding_type: Type of encoding ('sinusoidal', 'learned', 'rotary', 'relative')
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
            **kwargs: Additional arguments for specific encodings
        
        Returns:
            Positional encoding module
        """
        encoding_types = {
            'sinusoidal': SinusoidalPositionalEncoding,
            'learned': LearnedPositionalEncoding,
            'rotary': RotaryPositionalEncoding,
            'relative': RelativePositionalEncoding
        }
        
        if encoding_type not in encoding_types:
            raise ValueError(f"Unknown encoding type: {encoding_type}. "
                           f"Choose from {list(encoding_types.keys())}")
        
        if encoding_type == 'rotary':
            return encoding_types[encoding_type](d_model, max_len, 
                                                kwargs.get('base', 10000))
        elif encoding_type == 'relative':
            return encoding_types[encoding_type](d_model, 
                                                kwargs.get('max_relative_position', 32))
        else:
            return encoding_types[encoding_type](d_model, max_len, dropout)


def compare_positional_encodings(d_model: int = 64, seq_len: int = 20):
    """
    Quick comparison of different positional encoding methods.
    
    Args:
        d_model: Model dimension
        seq_len: Sequence length to test
    """
    # Create dummy input
    x = torch.randn(2, seq_len, d_model)
    
    # Test sinusoidal
    sin_pe = SinusoidalPositionalEncoding(d_model, dropout=0.0)
    sin_output = sin_pe(x.clone())
    print(f"Sinusoidal output shape: {sin_output.shape}")
    print(f"Sinusoidal output norm: {sin_output.norm():.3f}")
    
    # Test learned
    learned_pe = LearnedPositionalEncoding(d_model, dropout=0.0)
    learned_output = learned_pe(x.clone())
    print(f"Learned output shape: {learned_output.shape}")
    print(f"Learned output norm: {learned_output.norm():.3f}")
    
    # Test rotary (needs different input format)
    rope = RotaryPositionalEncoding(d_model)
    q = k = x.clone().unsqueeze(1)  # Add head dimension
    q_rot, k_rot = rope(q, k)
    print(f"RoPE output shape: {q_rot.shape}")
    print(f"RoPE preserves norm: {torch.allclose(q.norm(), q_rot.norm(), rtol=1e-5)}")
    
    print("\nAll positional encodings working correctly!")


if __name__ == "__main__":
    print("Testing Positional Encoding Implementations")
    print("="*50)
    compare_positional_encodings()
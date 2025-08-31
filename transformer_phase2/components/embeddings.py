"""
Embedding Layers for Transformer
This module implements token embeddings and positional encoding.
FIXED: RotaryPositionalEncoding now handles dimensions correctly
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional


class TokenEmbedding(nn.Module):
    """
    Token Embedding Layer
    
    Converts token indices to dense vector representations.
    Includes scaling by sqrt(d_model) as per the original paper.
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimension of the model
        padding_idx: Index of the padding token (optional)
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 512, 
        padding_idx: Optional[int] = None
    ):
        super(TokenEmbedding, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, 
            d_model, 
            padding_idx=padding_idx
        )
        
        # Scaling factor
        self.scale = math.sqrt(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights with normal distribution"""
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model ** -0.5)
        if self.padding_idx is not None:
            nn.init.constant_(self.embedding.weight[self.padding_idx], 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of token embedding
        
        Args:
            x: Input tensor of token indices [batch_size, seq_len]
        
        Returns:
            Embedded tokens of shape [batch_size, seq_len, d_model]
        """
        # Get embeddings and scale
        return self.embedding(x) * self.scale


class PositionalEncoding(nn.Module):
    """
    Positional Encoding using sine and cosine functions
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    This allows the model to learn relative positions as:
    PE(pos+k) can be represented as a linear function of PE(pos)
    
    Args:
        d_model: Dimension of the model
        max_seq_length: Maximum sequence length
        dropout: Dropout rate
    """
    
    def __init__(
        self, 
        d_model: int = 512, 
        max_seq_length: int = 5000, 
        dropout: float = 0.1
    ):
        super(PositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix to hold positional encodings
        pe = torch.zeros(max_seq_length, d_model)
        
        # Create position indices [0, 1, 2, ..., max_seq_length-1]
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        
        # Create dimension indices and scaling factors
        # div_term = 10000^(2i/d_model) for i in [0, d_model/2)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        
        # Apply sine to even dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd dimensions
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)  # Shape: [1, max_seq_length, d_model]
        
        # Register as buffer (not a parameter, but should be saved and moved with the model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        
        Returns:
            Tensor with positional encoding added [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        
        # Add positional encoding (broadcasting over batch dimension)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding
    
    Instead of fixed sinusoidal encodings, learn position embeddings.
    This is used in models like BERT.
    
    Args:
        d_model: Dimension of the model
        max_seq_length: Maximum sequence length
        dropout: Dropout rate
    """
    
    def __init__(
        self, 
        d_model: int = 512, 
        max_seq_length: int = 5000, 
        dropout: float = 0.1
    ):
        super(LearnedPositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Create learnable position embeddings
        self.position_embeddings = nn.Embedding(max_seq_length, d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize position embeddings"""
        nn.init.normal_(self.position_embeddings.weight, mean=0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding to input
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        
        # Create position indices
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand_as(x[:, :, 0])
        
        # Get position embeddings
        position_embeddings = self.position_embeddings(position_ids)
        
        # Add position embeddings to input
        x = x + position_embeddings
        
        return self.dropout(x)


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE)
    
    A more recent positional encoding method that applies rotation matrices
    to encode position information. Used in models like LLaMA.
    
    This implementation is flexible and handles multiple input formats:
    - 3D tensors: [batch_size, seq_len, d_model]
    - 4D tensors: [batch_size, num_heads, seq_len, head_dim]
    
    Args:
        d_model: Dimension of the model (or head_dim * num_heads)
        max_seq_length: Maximum sequence length
        base: Base for the exponential
    """
    
    def __init__(
        self, 
        d_model: int = 512, 
        max_seq_length: int = 5000,
        base: int = 10000
    ):
        super(RotaryPositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.base = base
        
        # We'll compute the frequencies on the fly based on input dimensions
        
    def _compute_inv_freq(self, dim: int, device: torch.device):
        """Compute inverse frequencies for the given dimension"""
        return 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    
    def _compute_position_embeddings(self, seq_len: int, dim: int, device: torch.device):
        """Compute sin/cos position embeddings"""
        inv_freq = self._compute_inv_freq(dim, device)
        t = torch.arange(seq_len, device=device).float()
        
        # Create frequencies matrix
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        
        # Create embeddings - concatenate to match dimension
        emb = torch.cat((freqs, freqs), dim=-1)
        
        return emb.cos(), emb.sin()
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Apply rotary position embeddings to input tensor"""
        return (x * cos) + (self.rotate_half(x) * sin)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor = None):
        """
        Apply rotary positional encoding to query and key tensors
        
        Args:
            q: Query tensor - either:
               - [batch_size, seq_len, d_model] for 3D input
               - [batch_size, num_heads, seq_len, head_dim] for 4D input
            k: Key tensor (optional, if None will use same as q)
               Same shape requirements as q
        
        Returns:
            Tuple of (q_rotated, k_rotated) with positional information
            If only q provided and k is None, returns (q_rotated, q_rotated)
        """
        # Handle the case where k is not provided
        if k is None:
            k = q
        
        # Get shape information
        orig_shape = q.shape
        device = q.device
        
        # Handle both 3D and 4D inputs
        if len(orig_shape) == 3:
            # 3D input: [batch_size, seq_len, d_model]
            batch_size, seq_len, d_model = orig_shape
            # Treat as single head for RoPE
            q = q.unsqueeze(1)  # [batch_size, 1, seq_len, d_model]
            k = k.unsqueeze(1) if k is not None else q
            num_heads = 1
            head_dim = d_model
        else:
            # 4D input: [batch_size, num_heads, seq_len, head_dim]
            batch_size, num_heads, seq_len, head_dim = orig_shape
        
        # Compute position embeddings based on actual head dimension
        cos, sin = self._compute_position_embeddings(seq_len, head_dim, device)
        
        # Reshape for broadcasting
        # cos/sin shape: [seq_len, head_dim]
        # Need shape: [1, 1, seq_len, head_dim] for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Apply rotary embeddings
        q_rotated = self.apply_rotary_pos_emb(q, cos, sin)
        k_rotated = self.apply_rotary_pos_emb(k, cos, sin)
        
        # Restore original shape if input was 3D
        if len(orig_shape) == 3:
            q_rotated = q_rotated.squeeze(1)
            k_rotated = k_rotated.squeeze(1)
        
        return q_rotated, k_rotated


class TransformerEmbedding(nn.Module):
    """
    Complete Transformer Embedding Layer
    
    Combines token embeddings with positional encoding.
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimension of the model
        max_seq_length: Maximum sequence length
        dropout: Dropout rate
        padding_idx: Index of padding token
        pos_encoding_type: Type of positional encoding ('sinusoidal', 'learned', or 'rotary')
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        max_seq_length: int = 5000,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
        pos_encoding_type: str = 'sinusoidal'
    ):
        super(TransformerEmbedding, self).__init__()
        
        # Token embedding
        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx)
        
        # Positional encoding
        if pos_encoding_type == 'sinusoidal':
            self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        elif pos_encoding_type == 'learned':
            self.positional_encoding = LearnedPositionalEncoding(d_model, max_seq_length, dropout)
        elif pos_encoding_type == 'rotary':
            # Rotary encoding is applied differently, so we set it to None here
            self.positional_encoding = None
            self.rotary_encoding = RotaryPositionalEncoding(d_model, max_seq_length)
        else:
            raise ValueError(f"Unknown positional encoding type: {pos_encoding_type}")
        
        self.pos_encoding_type = pos_encoding_type
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of complete embedding layer
        
        Args:
            x: Input tensor of token indices [batch_size, seq_len]
        
        Returns:
            Embedded and position-encoded tensor [batch_size, seq_len, d_model]
        """
        # Get token embeddings
        x = self.token_embedding(x)
        
        # Add positional encoding (except for rotary)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        
        return x


def create_embedding_layer(
    vocab_size: int,
    d_model: int = 512,
    max_seq_length: int = 5000,
    dropout: float = 0.1,
    padding_idx: Optional[int] = None,
    pos_encoding_type: str = 'sinusoidal'
) -> TransformerEmbedding:
    """
    Factory function to create embedding layer
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        max_seq_length: Maximum sequence length
        dropout: Dropout rate
        padding_idx: Padding token index
        pos_encoding_type: Type of positional encoding
    
    Returns:
        Complete embedding layer
    """
    return TransformerEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_length=max_seq_length,
        dropout=dropout,
        padding_idx=padding_idx,
        pos_encoding_type=pos_encoding_type
    )
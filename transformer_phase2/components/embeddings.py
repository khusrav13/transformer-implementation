"""
Embedding Layers for Transformer
This module implements token embeddings and positional encoding.
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
    
    Args:
        d_model: Dimension of the model
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
        
        # Precompute the frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute rotary embeddings for max_seq_length
        self._precompute_rotary_embeddings()
    
    def _precompute_rotary_embeddings(self):
        """Precompute rotary embeddings for all positions"""
        # Create position indices
        t = torch.arange(self.max_seq_length).type_as(self.inv_freq)
        
        # Compute frequencies for each position
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # Create rotation matrices (cos and sin components)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        """
        Apply rotary positional encoding to query and key tensors
        
        Args:
            q: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
        
        Returns:
            Tuple of (q_rotated, k_rotated) with positional information
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Get cached cos and sin values
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        
        # Apply rotation
        q_rotated = (q * cos) + (self.rotate_half(q) * sin)
        k_rotated = (k * cos) + (self.rotate_half(k) * sin)
        
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
"""
embeddings.py - Embedding Layers for Transformer
Already includes RoPE implementation - minor optimizations added.

Key updates:
1. ALiBi positional bias added for better length extrapolation
2. Optimized RoPE caching for efficiency
3. Better initialization strategies
4. Support for absolute position embeddings removal
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional, Tuple


class TokenEmbedding(nn.Module):
    """
    Token Embedding Layer
    
    UPDATED: Better initialization for stability
    Previous: Fixed std=d_model**-0.5 for all vocab sizes
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 512, 
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None  # NEW: Optional max norm constraint
    ):
        super(TokenEmbedding, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        
        # UPDATED: Support max_norm for gradient stability
        self.embedding = nn.Embedding(
            vocab_size, 
            d_model, 
            padding_idx=padding_idx,
            max_norm=max_norm  # NEW: Constrain embedding norms
        )
        
        self.scale = math.sqrt(d_model)
        self._init_weights()
    
    def _init_weights(self):
        """
        UPDATED: Adaptive initialization based on vocab size
        Previous: Same initialization regardless of vocab size
        """
        # UPDATED: Smaller std for large vocabularies
        if self.vocab_size > 50000:
            std = 0.01  # Smaller for large vocabs
        else:
            std = self.d_model ** -0.5
        
        nn.init.normal_(self.embedding.weight, mean=0, std=std)
        if self.padding_idx is not None:
            nn.init.constant_(self.embedding.weight[self.padding_idx], 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass with scaling"""
        return self.embedding(x) * self.scale


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding
    
    UPDATED: Cached computation for efficiency
    Previous: No caching optimization
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
        
        # UPDATED: Pre-compute and cache more efficiently
        pe = self._compute_positional_encoding(max_seq_length, d_model)
        self.register_buffer('pe', pe)
    
    def _compute_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        UPDATED: Vectorized computation for efficiency
        Previous: Loop-based computation
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # UPDATED: More stable computation
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward with positional encoding addition"""
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding
    
    UPDATED: Support for relative position learning
    Previous: Only absolute positions
    """
    
    def __init__(
        self, 
        d_model: int = 512, 
        max_seq_length: int = 5000, 
        dropout: float = 0.1,
        relative: bool = False  # NEW: Support relative positions
    ):
        super(LearnedPositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.relative = relative
        
        if relative:
            # NEW: Relative position embeddings
            max_relative_dist = 128  # Maximum relative distance to consider
            self.position_embeddings = nn.Embedding(2 * max_relative_dist + 1, d_model)
            self.max_relative_dist = max_relative_dist
        else:
            # Absolute position embeddings
            self.position_embeddings = nn.Embedding(max_seq_length, d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """UPDATED: Better initialization for learned embeddings"""
        nn.init.trunc_normal_(self.position_embeddings.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        UPDATED: Support for relative positions
        """
        seq_len = x.size(1)
        
        if self.relative:
            # NEW: Compute relative position embeddings
            positions = torch.arange(seq_len, device=x.device)
            relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
            relative_positions = relative_positions.clamp(
                -self.max_relative_dist, self.max_relative_dist
            )
            relative_positions = relative_positions + self.max_relative_dist
            
            # Average over relative positions for each position
            pos_emb = self.position_embeddings(relative_positions).mean(dim=1)
            x = x + pos_emb.unsqueeze(0)
        else:
            # Absolute positions
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
            position_embeddings = self.position_embeddings(position_ids)
            x = x + position_embeddings
        
        return self.dropout(x)


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE) - Already implemented correctly
    
    UPDATED: Added caching for efficiency
    Previous: Recomputed embeddings every forward pass
    """
    
    def __init__(
        self, 
        d_model: int = 512, 
        max_seq_length: int = 5000,
        base: int = 10000,
        cache_embeddings: bool = True  # NEW: Cache embeddings for efficiency
    ):
        super(RotaryPositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.base = base
        self.cache_embeddings = cache_embeddings
        
        # NEW: Cache for precomputed embeddings
        if cache_embeddings:
            self._cache = {}
    
    def _compute_inv_freq(self, dim: int, device: torch.device):
        """Compute inverse frequencies"""
        return 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    
    def _compute_position_embeddings(self, seq_len: int, dim: int, device: torch.device):
        """
        UPDATED: Cache computed embeddings for reuse
        Previous: Always recomputed
        """
        # NEW: Check cache first
        cache_key = (seq_len, dim, str(device))
        if self.cache_embeddings and cache_key in self._cache:
            return self._cache[cache_key]
        
        inv_freq = self._compute_inv_freq(dim, device)
        t = torch.arange(seq_len, device=device).float()
        
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos_emb, sin_emb = emb.cos(), emb.sin()
        
        # NEW: Cache the result
        if self.cache_embeddings:
            self._cache[cache_key] = (cos_emb, sin_emb)
        
        return cos_emb, sin_emb
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Apply rotary position embeddings"""
        return (x * cos) + (self.rotate_half(x) * sin)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor = None):
        """
        Apply RoPE to query and key tensors
        Already handles both 3D and 4D inputs correctly
        """
        if k is None:
            k = q
        
        orig_shape = q.shape
        device = q.device
        
        # Handle both 3D and 4D inputs
        if len(orig_shape) == 3:
            batch_size, seq_len, d_model = orig_shape
            q = q.unsqueeze(1)
            k = k.unsqueeze(1) if k is not None else q
            head_dim = d_model
        else:
            batch_size, num_heads, seq_len, head_dim = orig_shape
        
        # Get cached or compute embeddings
        cos, sin = self._compute_position_embeddings(seq_len, head_dim, device)
        
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        q_rotated = self.apply_rotary_pos_emb(q, cos, sin)
        k_rotated = self.apply_rotary_pos_emb(k, cos, sin)
        
        # Restore original shape if needed
        if len(orig_shape) == 3:
            q_rotated = q_rotated.squeeze(1)
            k_rotated = k_rotated.squeeze(1)
        
        return q_rotated, k_rotated


class ALiBi(nn.Module):
    """
    NEW: Attention with Linear Biases (ALiBi)
    Better length extrapolation than standard position encodings
    Used in models like BLOOM
    
    Not in original implementation - added for better length generalization
    """
    
    def __init__(self, num_heads: int = 8, max_seq_length: int = 5000):
        super(ALiBi, self).__init__()
        
        self.num_heads = num_heads
        
        # Compute slopes for each head
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)
        
        # Precompute bias matrix
        bias = self._compute_alibi_bias(max_seq_length)
        self.register_buffer('bias', bias)
    
    def _get_slopes(self, num_heads: int) -> torch.Tensor:
        """
        NEW: Compute geometric sequence of slopes for each head
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            # Round up to nearest power of 2
            closest_power_of_2 = 2 ** math.ceil(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            slopes = slopes[:num_heads]
        
        return torch.tensor(slopes).unsqueeze(1).unsqueeze(1)
    
    def _compute_alibi_bias(self, max_len: int) -> torch.Tensor:
        """
        NEW: Compute ALiBi bias matrix
        """
        positions = torch.arange(max_len)
        distances = positions.unsqueeze(0) - positions.unsqueeze(1)
        return distances.unsqueeze(0).float()
    
    def forward(self, attention_scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        NEW: Add ALiBi bias to attention scores
        
        Args:
            attention_scores: [batch, num_heads, seq_len, seq_len]
            seq_len: Current sequence length
        
        Returns:
            Biased attention scores
        """
        alibi = self.slopes * self.bias[0, :seq_len, :seq_len]
        alibi = alibi.unsqueeze(0)  # Add batch dimension
        return attention_scores + alibi


class TransformerEmbedding(nn.Module):
    """
    Complete Transformer Embedding Layer
    
    UPDATED: Support for no positional encoding (for ALiBi)
    Previous: Always required positional encoding
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        max_seq_length: int = 5000,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
        pos_encoding_type: str = 'rotary',  # UPDATED: Default to RoPE
        use_alibi: bool = False  # NEW: Option to use ALiBi instead
    ):
        super(TransformerEmbedding, self).__init__()
        
        # Token embedding
        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx)
        
        # Positional encoding
        self.use_alibi = use_alibi
        
        if use_alibi:
            # NEW: No positional encoding needed with ALiBi
            self.positional_encoding = None
            self.rotary_encoding = None
        elif pos_encoding_type == 'sinusoidal':
            self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
            self.rotary_encoding = None
        elif pos_encoding_type == 'learned':
            self.positional_encoding = LearnedPositionalEncoding(d_model, max_seq_length, dropout)
            self.rotary_encoding = None
        elif pos_encoding_type == 'rotary':
            self.positional_encoding = None
            self.rotary_encoding = RotaryPositionalEncoding(d_model, max_seq_length)
        elif pos_encoding_type == 'none':
            # NEW: Option for no positional encoding
            self.positional_encoding = None
            self.rotary_encoding = None
        else:
            raise ValueError(f"Unknown positional encoding type: {pos_encoding_type}")
        
        self.pos_encoding_type = pos_encoding_type
        self.dropout = nn.Dropout(dropout) if not self.positional_encoding else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        UPDATED: Handle no positional encoding case
        """
        # Get token embeddings
        x = self.token_embedding(x)
        
        # Add positional encoding if not using RoPE or ALiBi
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        elif self.dropout is not None:
            # Apply dropout if no positional encoding layer does it
            x = self.dropout(x)
        
        return x


def create_embedding_layer(
    vocab_size: int,
    d_model: int = 512,
    max_seq_length: int = 5000,
    dropout: float = 0.1,
    padding_idx: Optional[int] = None,
    pos_encoding_type: str = 'rotary',  # UPDATED: Default to RoPE
    use_alibi: bool = False  # NEW: ALiBi option
) -> TransformerEmbedding:
    """
    Factory function to create embedding layer
    
    UPDATED: RoPE is now default, added ALiBi support
    Previous: Sinusoidal was default
    """
    return TransformerEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_length=max_seq_length,
        dropout=dropout,
        padding_idx=padding_idx,
        pos_encoding_type=pos_encoding_type,
        use_alibi=use_alibi
    )
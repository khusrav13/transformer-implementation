"""
Attention Mechanisms for Transformer
This module implements scaled dot-product attention and multi-head attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism
    
    The attention function can be described as mapping a query and a set of key-value pairs
    to an output, where the query, keys, values, and output are all vectors.
    
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    
    Args:
        dropout: Dropout rate for attention weights
    """
    
    def __init__(self, dropout: float = 0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of scaled dot-product attention
        
        Args:
            query: Query tensor of shape [batch_size, num_heads, seq_len_q, d_k]
            key: Key tensor of shape [batch_size, num_heads, seq_len_k, d_k]
            value: Value tensor of shape [batch_size, num_heads, seq_len_v, d_v]
            mask: Optional mask tensor
        
        Returns:
            output: Attention output of shape [batch_size, num_heads, seq_len_q, d_v]
            attention_weights: Attention weights of shape [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size, num_heads, seq_len_q, d_k = query.shape
        seq_len_k = key.shape[2]
        
        # Calculate attention scores
        # Shape: [batch_size, num_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale scores by square root of dimension
        # This prevents the dot products from growing too large in magnitude
        scores = scores / math.sqrt(d_k)
        
        # Apply mask if provided (for padding or causal attention)
        if mask is not None:
            # Set masked positions to very large negative value
            # After softmax, these positions will have near-zero attention
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights (probabilities)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        # Shape: [batch_size, num_heads, seq_len_q, d_v]
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism
    
    Multi-head attention allows the model to jointly attend to information
    from different representation subspaces at different positions.
    
    MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    Args:
        d_model: Dimension of the model (embedding dimension)
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        self.d_v = d_model // num_heads  # Usually d_k == d_v
        
        # Linear projections for queries, keys, and values
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Attention and dropout
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform distribution"""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention
        
        Now properly handles different sequence lengths for Q, K, V
        
        Args:
            query: Query tensor of shape [batch_size, seq_len_q, d_model]
            key: Key tensor of shape [batch_size, seq_len_k, d_model]
            value: Value tensor of shape [batch_size, seq_len_v, d_model]
            mask: Optional mask tensor
        
        Returns:
            output: Attention output of shape [batch_size, seq_len_q, d_model]
            attention_weights: Attention weights of shape [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.shape[0]
        
        # Get sequence lengths for each input
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]
        seq_len_v = value.shape[1]
        
        # seq_len_k should equal seq_len_v (keys and values come from same source)
        assert seq_len_k == seq_len_v, \
            f"Key sequence length ({seq_len_k}) must equal value sequence length ({seq_len_v})"
        
        # Step 1: Linear projections in batch from d_model => num_heads x d_k
        # Shape: [batch_size, seq_len_*, d_model] -> [batch_size, seq_len_*, num_heads, d_k]
        Q = self.W_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k)
        K = self.W_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k)
        V = self.W_v(value).view(batch_size, seq_len_v, self.num_heads, self.d_v)
        
        # Step 2: Transpose for attention calculation
        # Shape: [batch_size, num_heads, seq_len_*, d_k]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Step 3: Apply scaled dot-product attention
        # Output shape: [batch_size, num_heads, seq_len_q, d_v]
        # Attention weights shape: [batch_size, num_heads, seq_len_q, seq_len_k]
        attention_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Step 4: Concatenate heads
        # Shape: [batch_size, seq_len_q, num_heads * d_v]
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        # Step 5: Final linear projection
        output = self.W_o(attention_output)
        output = self.dropout(output)
        
        return output, attention_weights


class SelfAttention(MultiHeadAttention):
    """
    Self-Attention is a special case of Multi-Head Attention
    where query, key, and value all come from the same source
    """
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of self-attention
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional mask tensor
        
        Returns:
            output: Attention output of shape [batch_size, seq_len, d_model]
            attention_weights: Attention weights
        """
        # In self-attention, Q, K, V all come from the same input
        return super().forward(x, x, x, mask)


class CrossAttention(MultiHeadAttention):
    """
    Cross-Attention (used in decoder)
    Query comes from decoder, Key and Value come from encoder output
    """
    
    def forward(
        self, 
        query: torch.Tensor, 
        encoder_output: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of cross-attention
        
        Args:
            query: Query from decoder of shape [batch_size, seq_len_q, d_model]
            encoder_output: Output from encoder of shape [batch_size, seq_len_k, d_model]
            mask: Optional mask tensor
        
        Returns:
            output: Attention output of shape [batch_size, seq_len_q, d_model]
            attention_weights: Attention weights of shape [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        # Query from decoder, Key and Value from encoder
        return super().forward(query, encoder_output, encoder_output, mask)


def create_attention_layer(
    attention_type: str = "self", 
    d_model: int = 512, 
    num_heads: int = 8, 
    dropout: float = 0.1
) -> nn.Module:
    """
    Factory function to create different types of attention layers
    
    Args:
        attention_type: Type of attention ('self', 'cross', or 'multi')
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    
    Returns:
        Attention layer module
    """
    attention_types = {
        "self": SelfAttention,
        "cross": CrossAttention,
        "multi": MultiHeadAttention
    }
    
    if attention_type not in attention_types:
        raise ValueError(f"Unknown attention type: {attention_type}")
    
    return attention_types[attention_type](d_model, num_heads, dropout)
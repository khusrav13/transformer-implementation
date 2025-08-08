"""
Multi-Head Attention Implementation
Educational implementation with detailed explanations of the mathematical concepts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism as described in "Attention Is All You Need".
    
    The key insight: Instead of performing a single attention function with 
    d_model-dimensional keys, values and queries, we project them h times 
    with different, learned linear projections to d_k dimensions.
    
    Mathematical formulation:
    MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    Args:
        d_model: The dimension of the model (typically 512 or 768)
        num_heads: Number of attention heads (typically 8 or 12)
        dropout: Dropout probability for regularization
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        # Ensure d_model is divisible by num_heads for even split
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        # These learn to project input into query, key, and value spaces
        # Weight matrices are d_model x d_model, enabling parallel computation
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection to combine information from all heads
        # This allows the model to jointly attend to information from different representation subspaces
        self.W_o = nn.Linear(d_model, d_model)
        
        # Dropout for regularization - applied after attention
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor for dot product attention
        # Pre-compute for efficiency
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """
        Initialize parameters using Xavier uniform initialization.
        This helps with gradient flow at the beginning of training.
        
        Xavier initialization sets weights such that the variance of outputs
        equals the variance of inputs, preventing gradient vanishing/explosion.
        """
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor of shape [batch_size, seq_len_q, d_model]
            key: Key tensor of shape [batch_size, seq_len_k, d_model]
            value: Value tensor of shape [batch_size, seq_len_v, d_model]
                   Note: seq_len_k == seq_len_v always
            mask: Optional mask tensor. Can be either:
                  - Padding mask: [batch_size, 1, 1, seq_len_k]
                  - Attention mask (e.g., causal): [batch_size, 1, seq_len_q, seq_len_k]
                  - Combined mask: [batch_size, num_heads, seq_len_q, seq_len_k]
            return_attention: Whether to return attention weights (useful for visualization)
        
        Returns:
            output: Attention output [batch_size, seq_len_q, d_model]
            attention_weights: Optional attention weights [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # Step 1: Linear projections
        # Project input to query, key, value spaces
        # Shape: [batch_size, seq_len, d_model]
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Step 2: Reshape to separate heads
        # We reshape to [batch_size, seq_len, num_heads, d_k] then transpose
        # This allows each head to operate independently
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        # Shape after transpose: [batch_size, num_heads, seq_len, d_k]
        
        # Step 3: Scaled dot-product attention for all heads in parallel
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask, self.scale
        )
        # attention_output shape: [batch_size, num_heads, seq_len_q, d_k]
        
        # Step 4: Concatenate heads
        # Transpose back and reshape to concatenate all heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        # Shape: [batch_size, seq_len_q, num_heads, d_k]
        
        attention_output = attention_output.view(batch_size, seq_len_q, self.d_model)
        # Shape: [batch_size, seq_len_q, d_model]
        
        # Step 5: Final linear projection
        # Project concatenated heads to output space
        output = self.W_o(attention_output)
        output = self.dropout(output)
        
        if return_attention:
            return output, attention_weights
        else:
            return output, None
    
    @staticmethod
    def scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        scale: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scaled Dot-Product Attention.
        
        Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
        
        The scaling factor 1/sqrt(d_k) is crucial:
        - Without scaling, for large d_k, the dot products grow large in magnitude
        - This pushes softmax into regions with extremely small gradients
        - The scaling maintains gradient flow during backpropagation
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len_q, d_k]
            key: Key tensor [batch_size, num_heads, seq_len_k, d_k]
            value: Value tensor [batch_size, num_heads, seq_len_v, d_k]
            mask: Optional mask tensor
            scale: Optional scaling factor (if None, uses 1/sqrt(d_k))
        
        Returns:
            output: Weighted sum of values [batch_size, num_heads, seq_len_q, d_k]
            attention_weights: Attention weights [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        d_k = query.size(-1)
        
        if scale is None:
            scale = 1.0 / math.sqrt(d_k)
        
        # Compute attention scores
        # QK^T gives us a matrix of dot products between all query-key pairs
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        # Shape: [batch_size, num_heads, seq_len_q, seq_len_k]
        
        # Apply mask if provided
        # Mask should contain -inf for positions to ignore
        if mask is not None:
            # Expand mask if necessary
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # Add head dimension
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        # Softmax is applied over the key dimension (last dimension)
        # This ensures each query position has a probability distribution over keys
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        # This computes a weighted sum of values for each query position
        output = torch.matmul(attention_weights, value)
        # Shape: [batch_size, num_heads, seq_len_q, d_k]
        
        return output, attention_weights
    
    def get_attention_maps(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Helper method to extract attention patterns for visualization.
        
        Returns:
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        _, attention_weights = self.forward(query, key, value, mask, return_attention=True)
        return attention_weights


class MultiHeadCrossAttention(MultiHeadAttention):
    """
    Cross-attention variant where queries come from one source (e.g., decoder)
    and keys/values come from another source (e.g., encoder output).
    
    This is used in the decoder to attend to encoder outputs.
    The implementation is identical to self-attention, but we explicitly
    separate it for clarity in transformer architectures.
    """
    
    def forward(
        self,
        query: torch.Tensor,
        encoder_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for cross-attention.
        
        Args:
            query: Query from decoder [batch_size, seq_len_q, d_model]
            encoder_output: Key and Value from encoder [batch_size, seq_len_k, d_model]
            mask: Optional attention mask
            return_attention: Whether to return attention weights
        
        Returns:
            output: Attention output
            attention_weights: Optional attention weights
        """
        # In cross-attention, keys and values come from the same source (encoder)
        return super().forward(
            query=query,
            key=encoder_output,
            value=encoder_output,
            mask=mask,
            return_attention=return_attention
        )


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal mask for autoregressive attention.
    
    This prevents positions from attending to subsequent positions,
    preserving the autoregressive property during training.
    
    Args:
        seq_len: Sequence length
        device: Device to create tensor on
    
    Returns:
        mask: Lower triangular matrix of shape [seq_len, seq_len]
              with 1s below diagonal and 0s above
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    mask = mask.masked_fill(mask == 0, float(0.0))
    return mask


def create_padding_mask(
    seq: torch.Tensor,
    pad_idx: int = 0
) -> torch.Tensor:
    """
    Create padding mask for sequences with padding tokens.
    
    Args:
        seq: Input sequence tensor [batch_size, seq_len]
        pad_idx: Index of padding token (usually 0)
    
    Returns:
        mask: Boolean mask [batch_size, 1, 1, seq_len]
              with False for padding positions
    """
    # Create boolean mask: True for real tokens, False for padding
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


if __name__ == "__main__":
    # Example usage and testing
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    # Initialize model
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Self-attention
    output, attention = mha(x, x, x)
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attention.shape}")
    
    # Verify attention weights sum to 1
    attention_sum = attention.sum(dim=-1)
    print(f"Attention weights sum (should be close to 1): {attention_sum[0, 0, :5]}")
    
    # Test with causal mask
    causal_mask = create_causal_mask(seq_len)
    output_masked, _ = mha(x, x, x, mask=causal_mask.unsqueeze(0))
    print(f"Masked output shape: {output_masked.shape}")
"""
Decoder Layer for Transformer
This module implements a complete decoder layer with masked self-attention,
cross-attention to encoder output, feed-forward network, and layer normalization.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.attention import MultiHeadAttention, SelfAttention, CrossAttention
from components.feedforward import PositionwiseFeedForward
from components.normalization import LayerNormalization


class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer
    
    Each decoder layer consists of:
    1. Masked multi-head self-attention sublayer
    2. Multi-head cross-attention sublayer (attending to encoder output)
    3. Position-wise feed-forward sublayer
    4. Residual connections around each sublayer
    5. Layer normalization after each sublayer
    
    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward layer
        dropout: Dropout rate
        activation: Activation function for feed-forward layer
        norm_type: Type of normalization ('pre' or 'post')
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        norm_type: str = 'post'
    ):
        super(DecoderLayer, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.norm_type = norm_type
        
        # Masked self-attention sublayer
        self.self_attention = SelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Cross-attention sublayer (decoder attending to encoder)
        self.cross_attention = CrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward sublayer
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )
        
        # Layer normalization
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass of decoder layer
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            encoder_output: Encoder output [batch_size, enc_seq_len, d_model]
            self_mask: Mask for self-attention (causal mask)
            cross_mask: Mask for cross-attention (padding mask)
            cache: Cache for fast decoding (stores key-value pairs)
            return_attention: Whether to return attention weights
        
        Returns:
            output: Output tensor [batch_size, seq_len, d_model]
            cache: Updated cache if provided
            attention_weights: Optional tuple of (self_attn_weights, cross_attn_weights)
        """
        # Store attention weights if needed
        attention_weights = {}
        
        if self.norm_type == 'pre':
            # Pre-normalization path
            
            # Masked self-attention with pre-norm
            residual = x
            x_norm = self.norm1(x)
            
            # Use cache for incremental decoding if provided
            if cache is not None and 'self_k' in cache and 'self_v' in cache:
                # Concatenate cached keys and values
                self_attn_output, self_attn_weights = self._cached_self_attention(
                    x_norm, cache, self_mask
                )
            else:
                self_attn_output, self_attn_weights = self.self_attention(
                    x_norm, mask=self_mask
                )
            
            x = residual + self.dropout(self_attn_output)
            if return_attention:
                attention_weights['self'] = self_attn_weights
            
            # Cross-attention with pre-norm
            residual = x
            x_norm = self.norm2(x)
            cross_attn_output, cross_attn_weights = self.cross_attention(
                query=x_norm,
                encoder_output=encoder_output,
                mask=cross_mask
            )
            x = residual + self.dropout(cross_attn_output)
            if return_attention:
                attention_weights['cross'] = cross_attn_weights
            
            # Feed-forward with pre-norm
            residual = x
            x_norm = self.norm3(x)
            ff_output = self.feed_forward(x_norm)
            x = residual + self.dropout(ff_output)
            
        else:  # post-normalization
            # Post-normalization path (original transformer)
            
            # Masked self-attention with post-norm
            residual = x
            
            # Use cache for incremental decoding if provided
            if cache is not None and 'self_k' in cache and 'self_v' in cache:
                self_attn_output, self_attn_weights = self._cached_self_attention(
                    x, cache, self_mask
                )
            else:
                self_attn_output, self_attn_weights = self.self_attention(
                    x, mask=self_mask
                )
            
            x = self.norm1(residual + self.dropout(self_attn_output))
            if return_attention:
                attention_weights['self'] = self_attn_weights
            
            # Cross-attention with post-norm
            residual = x
            cross_attn_output, cross_attn_weights = self.cross_attention(
                query=x,
                encoder_output=encoder_output,
                mask=cross_mask
            )
            x = self.norm2(residual + self.dropout(cross_attn_output))
            if return_attention:
                attention_weights['cross'] = cross_attn_weights
            
            # Feed-forward with post-norm
            residual = x
            ff_output = self.feed_forward(x)
            x = self.norm3(residual + self.dropout(ff_output))
        
        # Update cache if provided
        if cache is not None:
            cache = self._update_cache(cache, x)
        
        if return_attention:
            return x, cache, attention_weights
        return x, cache
    
    def _cached_self_attention(
        self,
        x: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform self-attention using cached keys and values
        
        Args:
            x: Current input [batch_size, 1, d_model] (single position)
            cache: Dictionary containing cached keys and values
            mask: Optional attention mask
        
        Returns:
            Attention output and weights
        """
        # This is a simplified version - full implementation would handle
        # the incremental key/value computation more efficiently
        return self.self_attention(x, mask=mask)
    
    def _update_cache(
        self,
        cache: Dict[str, torch.Tensor],
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Update cache with new keys and values
        
        Args:
            cache: Current cache dictionary
            x: Current decoder output
        
        Returns:
            Updated cache
        """
        # This would store computed keys and values for next step
        # Implementation depends on specific caching strategy
        return cache


class ParallelDecoderLayer(nn.Module):
    """
    Parallel Decoder Layer
    
    A variant where self-attention and cross-attention are computed in parallel
    rather than sequentially. This can be more efficient but may affect quality.
    
    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward layer
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super(ParallelDecoderLayer, self).__init__()
        
        # Attention layers
        self.self_attention = SelfAttention(d_model, num_heads, dropout)
        self.cross_attention = CrossAttention(d_model, num_heads, dropout)
        
        # Combination layer for parallel outputs
        self.combine = nn.Linear(d_model * 2, d_model)
        
        # Feed-forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Normalization
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with parallel attention
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            encoder_output: Encoder output
            self_mask: Self-attention mask
            cross_mask: Cross-attention mask
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        residual = x
        
        # Compute self-attention and cross-attention in parallel
        self_attn_output, _ = self.self_attention(x, mask=self_mask)
        cross_attn_output, _ = self.cross_attention(
            query=x,
            encoder_output=encoder_output,
            mask=cross_mask
        )
        
        # Combine parallel outputs
        combined = torch.cat([self_attn_output, cross_attn_output], dim=-1)
        combined = self.combine(combined)
        
        # Add residual and normalize
        x = self.norm1(residual + self.dropout(combined))
        
        # Feed-forward
        residual = x
        ff_output = self.feed_forward(x)
        x = self.norm2(residual + self.dropout(ff_output))
        
        return x


class MemoryDecoderLayer(nn.Module):
    """
    Decoder Layer with External Memory
    
    Decoder that can attend to external memory banks in addition to encoder output.
    Useful for models that need to incorporate external knowledge.
    
    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward layer
        dropout: Dropout rate
        num_memory_slots: Number of external memory slots
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        num_memory_slots: int = 100
    ):
        super(MemoryDecoderLayer, self).__init__()
        
        self.num_memory_slots = num_memory_slots
        
        # Standard components
        self.self_attention = SelfAttention(d_model, num_heads, dropout)
        self.cross_attention = CrossAttention(d_model, num_heads, dropout)
        
        # Memory attention
        self.memory_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Learnable memory bank
        self.memory_bank = nn.Parameter(
            torch.randn(1, num_memory_slots, d_model)
        )
        
        # Feed-forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Normalization
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)
        self.norm4 = LayerNormalization(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with memory attention
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            encoder_output: Encoder output
            self_mask: Self-attention mask
            cross_mask: Cross-attention mask
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size = x.size(0)
        
        # Self-attention
        residual = x
        self_attn_output, _ = self.self_attention(x, mask=self_mask)
        x = self.norm1(residual + self.dropout(self_attn_output))
        
        # Cross-attention to encoder
        residual = x
        cross_attn_output, _ = self.cross_attention(
            query=x,
            encoder_output=encoder_output,
            mask=cross_mask
        )
        x = self.norm2(residual + self.dropout(cross_attn_output))
        
        # Memory attention
        residual = x
        memory = self.memory_bank.expand(batch_size, -1, -1)
        memory_attn_output, _ = self.memory_attention(
            query=x,
            key=memory,
            value=memory
        )
        x = self.norm3(residual + self.dropout(memory_attn_output))
        
        # Feed-forward
        residual = x
        ff_output = self.feed_forward(x)
        x = self.norm4(residual + self.dropout(ff_output))
        
        return x


class Decoder(nn.Module):
    """
    Complete Transformer Decoder
    
    Stack of decoder layers.
    
    Args:
        num_layers: Number of decoder layers
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward layer
        dropout: Dropout rate
        activation: Activation function
        norm_type: Type of normalization
    """
    
    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        norm_type: str = 'post'
    ):
        super(Decoder, self).__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                norm_type=norm_type
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = LayerNormalization(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        return_all_layers: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through all decoder layers
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            encoder_output: Encoder output
            self_mask: Self-attention mask
            cross_mask: Cross-attention mask
            cache: Cache for fast decoding
            return_all_layers: Whether to return outputs from all layers
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
            If return_all_layers=True, returns list of all layer outputs
        """
        all_layers = []
        
        for i, layer in enumerate(self.layers):
            # Get cache for this layer if available
            layer_cache = None
            if cache is not None and f'layer_{i}' in cache:
                layer_cache = cache[f'layer_{i}']
            
            x, layer_cache = layer(
                x=x,
                encoder_output=encoder_output,
                self_mask=self_mask,
                cross_mask=cross_mask,
                cache=layer_cache
            )
            
            # Update cache
            if cache is not None:
                cache[f'layer_{i}'] = layer_cache
            
            if return_all_layers:
                all_layers.append(x)
        
        # Apply final normalization
        x = self.final_norm(x)
        
        if return_all_layers:
            all_layers.append(x)
            return all_layers
        
        return x
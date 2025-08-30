"""
Encoder Layer for Transformer
This module implements a complete encoder layer with self-attention,
feed-forward network, residual connections, and layer normalization.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.attention import MultiHeadAttention, SelfAttention
from components.feedforward import PositionwiseFeedForward
from components.normalization import LayerNormalization, PreNorm, PostNorm


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    
    Each encoder layer consists of:
    1. Multi-head self-attention sublayer
    2. Position-wise feed-forward sublayer
    3. Residual connections around each sublayer
    4. Layer normalization after each sublayer
    
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
        super(EncoderLayer, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.norm_type = norm_type
        
        # Self-attention sublayer
        self.self_attention = SelfAttention(
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
        
        # Normalization and residual connections
        if norm_type == 'pre':
            # Pre-normalization (used in GPT-style models)
            self.norm1 = LayerNormalization(d_model)
            self.norm2 = LayerNormalization(d_model)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
        elif norm_type == 'post':
            # Post-normalization (original transformer)
            self.norm1 = LayerNormalization(d_model)
            self.norm2 = LayerNormalization(d_model)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of encoder layer
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional attention mask of shape [batch_size, 1, seq_len, seq_len]
            return_attention: Whether to return attention weights
        
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
            Optionally returns attention weights if return_attention=True
        """
        if self.norm_type == 'pre':
            # Pre-normalization path
            # Self-attention with pre-norm
            residual = x
            x = self.norm1(x)
            attn_output, attn_weights = self.self_attention(x, mask=mask)
            x = residual + self.dropout1(attn_output)
            
            # Feed-forward with pre-norm
            residual = x
            x = self.norm2(x)
            ff_output = self.feed_forward(x)
            x = residual + self.dropout2(ff_output)
            
        else:  # post-normalization
            # Post-normalization path (original transformer)
            # Self-attention with post-norm
            residual = x
            attn_output, attn_weights = self.self_attention(x, mask=mask)
            x = residual + self.dropout1(attn_output)
            x = self.norm1(x)
            
            # Feed-forward with post-norm
            residual = x
            ff_output = self.feed_forward(x)
            x = residual + self.dropout2(ff_output)
            x = self.norm2(x)
        
        if return_attention:
            return x, attn_weights
        return x


class EncoderLayerWithCrossAttention(nn.Module):
    """
    Encoder Layer with Cross-Attention capability
    
    This variant can attend to external memory or context,
    useful for models that need to incorporate external information.
    
    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward layer
        dropout: Dropout rate
        activation: Activation function
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super(EncoderLayerWithCrossAttention, self).__init__()
        
        # Self-attention
        self.self_attention = SelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Cross-attention (attending to external context)
        self.cross_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward
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
        context: Optional[torch.Tensor] = None,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional cross-attention
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            context: External context tensor [batch_size, context_len, d_model]
            self_mask: Mask for self-attention
            cross_mask: Mask for cross-attention
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention
        residual = x
        self_attn_output, _ = self.self_attention(x, mask=self_mask)
        x = self.norm1(residual + self.dropout(self_attn_output))
        
        # Cross-attention (if context is provided)
        if context is not None:
            residual = x
            cross_attn_output, _ = self.cross_attention(
                query=x,
                key=context,
                value=context,
                mask=cross_mask
            )
            x = self.norm2(residual + self.dropout(cross_attn_output))
        
        # Feed-forward
        residual = x
        ff_output = self.feed_forward(x)
        x = self.norm3(residual + self.dropout(ff_output))
        
        return x


class ConditionalEncoderLayer(nn.Module):
    """
    Conditional Encoder Layer
    
    Encoder layer that can be conditioned on external inputs,
    useful for controllable generation or style transfer.
    
    Args:
        d_model: Dimension of the model
        d_condition: Dimension of conditioning vector
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward layer
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 512,
        d_condition: int = 256,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super(ConditionalEncoderLayer, self).__init__()
        
        self.d_model = d_model
        self.d_condition = d_condition
        
        # Self-attention
        self.self_attention = SelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Conditional layers
        self.condition_projection = nn.Linear(d_condition, d_model)
        self.gate = nn.Linear(d_model * 2, d_model)
        
        # Normalization
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with conditioning
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            condition: Conditioning tensor [batch_size, d_condition]
            mask: Optional attention mask
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Project condition to model dimension
        condition_proj = self.condition_projection(condition)  # [batch_size, d_model]
        condition_proj = condition_proj.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Self-attention with conditioning
        residual = x
        attn_output, _ = self.self_attention(x, mask=mask)
        
        # Gate attention output with condition
        gate_input = torch.cat([
            attn_output,
            condition_proj.expand_as(attn_output)
        ], dim=-1)
        gate_values = torch.sigmoid(self.gate(gate_input))
        attn_output = attn_output * gate_values
        
        x = self.norm1(residual + self.dropout(attn_output))
        
        # Feed-forward
        residual = x
        ff_output = self.feed_forward(x)
        x = self.norm2(residual + self.dropout(ff_output))
        
        return x


class Encoder(nn.Module):
    """
    Complete Transformer Encoder
    
    Stack of encoder layers with optional positional encoding.
    
    Args:
        num_layers: Number of encoder layers
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
        super(Encoder, self).__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                norm_type=norm_type
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization (optional)
        self.final_norm = LayerNormalization(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_all_layers: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through all encoder layers
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            return_all_layers: Whether to return outputs from all layers
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
            If return_all_layers=True, returns list of all layer outputs
        """
        all_layers = []
        
        for layer in self.layers:
            x = layer(x, mask=mask)
            if return_all_layers:
                all_layers.append(x)
        
        # Apply final normalization
        x = self.final_norm(x)
        
        if return_all_layers:
            all_layers.append(x)
            return all_layers
        
        return x
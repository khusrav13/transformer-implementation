"""
Normalization Layers for Transformer
This module implements layer normalization and its variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LayerNormalization(nn.Module):
    """
    Layer Normalization
    
    Normalizes the input across the feature dimension.
    LayerNorm(x) = gamma * (x - mean) / sqrt(variance + eps) + beta
    
    Unlike batch normalization, layer norm normalizes across the feature dimension
    rather than the batch dimension, making it more suitable for transformers.
    
    Args:
        d_model: Dimension of the model (number of features)
        eps: Small value to prevent division by zero
    """
    
    def __init__(self, d_model: int = 512, eps: float = 1e-6):
        super(LayerNormalization, self).__init__()
        
        self.d_model = d_model
        self.eps = eps
        
        # Learnable parameters for scaling and shifting
        self.gamma = nn.Parameter(torch.ones(d_model))  # Scale parameter
        self.beta = nn.Parameter(torch.zeros(d_model))  # Shift parameter
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of layer normalization
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        
        Returns:
            Normalized tensor of shape [batch_size, seq_len, d_model]
        """
        # Calculate mean and variance across the last dimension (features)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        output = self.gamma * x_norm + self.beta
        
        return output


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    A simpler version of layer normalization that normalizes by the RMS statistic.
    Used in models like LLaMA for improved efficiency.
    
    RMSNorm(x) = gamma * x / RMS(x)
    where RMS(x) = sqrt(mean(x^2) + eps)
    
    Args:
        d_model: Dimension of the model
        eps: Small value to prevent division by zero
    """
    
    def __init__(self, d_model: int = 512, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        
        self.d_model = d_model
        self.eps = eps
        
        # Only scale parameter, no shift
        self.gamma = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RMS normalization
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        
        Returns:
            Normalized tensor of shape [batch_size, seq_len, d_model]
        """
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        output = self.gamma * (x / rms)
        
        return output


class PreNorm(nn.Module):
    """
    Pre-Layer Normalization wrapper
    
    Applies layer normalization before the sublayer (attention or FFN).
    This is the configuration used in models like GPT.
    
    Args:
        d_model: Dimension of the model
        sublayer: The sublayer module (attention or feed-forward)
        norm_type: Type of normalization ('layer' or 'rms')
    """
    
    def __init__(
        self, 
        d_model: int, 
        sublayer: nn.Module,
        norm_type: str = 'layer'
    ):
        super(PreNorm, self).__init__()
        
        if norm_type == 'layer':
            self.norm = LayerNormalization(d_model)
        elif norm_type == 'rms':
            self.norm = RMSNorm(d_model)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")
        
        self.sublayer = sublayer
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with pre-normalization
        
        Args:
            x: Input tensor
            *args, **kwargs: Additional arguments for the sublayer
        
        Returns:
            Output tensor with residual connection
        """
        # Normalize first, then apply sublayer
        normalized = self.norm(x)
        output = self.sublayer(normalized, *args, **kwargs)
        
        # Handle output that might be a tuple (e.g., attention with weights)
        if isinstance(output, tuple):
            output, *extras = output
            return x + output, *extras
        else:
            return x + output


class PostNorm(nn.Module):
    """
    Post-Layer Normalization wrapper
    
    Applies layer normalization after the sublayer (attention or FFN).
    This is the original configuration from the "Attention is All You Need" paper.
    
    Args:
        d_model: Dimension of the model
        sublayer: The sublayer module
        norm_type: Type of normalization ('layer' or 'rms')
        dropout: Dropout rate for residual connection
    """
    
    def __init__(
        self, 
        d_model: int, 
        sublayer: nn.Module,
        norm_type: str = 'layer',
        dropout: float = 0.1
    ):
        super(PostNorm, self).__init__()
        
        if norm_type == 'layer':
            self.norm = LayerNormalization(d_model)
        elif norm_type == 'rms':
            self.norm = RMSNorm(d_model)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")
        
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with post-normalization
        
        Args:
            x: Input tensor
            *args, **kwargs: Additional arguments for the sublayer
        
        Returns:
            Output tensor with residual connection and normalization
        """
        # Apply sublayer
        output = self.sublayer(x, *args, **kwargs)
        
        # Handle output that might be a tuple
        if isinstance(output, tuple):
            output, *extras = output
            # Residual connection, dropout, then normalize
            output = self.norm(x + self.dropout(output))
            return output, *extras
        else:
            # Residual connection, dropout, then normalize
            return self.norm(x + self.dropout(output))


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization
    
    Layer normalization with adaptive gain and bias parameters.
    Used in some conditional transformer models.
    
    Args:
        d_model: Dimension of the model
        d_condition: Dimension of the conditioning vector
        eps: Small value for numerical stability
    """
    
    def __init__(
        self, 
        d_model: int = 512, 
        d_condition: int = 512,
        eps: float = 1e-6
    ):
        super(AdaptiveLayerNorm, self).__init__()
        
        self.d_model = d_model
        self.eps = eps
        
        # Linear layers to generate adaptive parameters from condition
        self.gamma_linear = nn.Linear(d_condition, d_model)
        self.beta_linear = nn.Linear(d_condition, d_model)
        
        # Initialize to approximate identity function
        nn.init.ones_(self.gamma_linear.weight)
        nn.init.zeros_(self.gamma_linear.bias)
        nn.init.zeros_(self.beta_linear.weight)
        nn.init.zeros_(self.beta_linear.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with adaptive normalization
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            condition: Conditioning tensor of shape [batch_size, d_condition]
        
        Returns:
            Normalized tensor of shape [batch_size, seq_len, d_model]
        """
        # Generate adaptive parameters
        gamma = self.gamma_linear(condition).unsqueeze(1)  # [batch_size, 1, d_model]
        beta = self.beta_linear(condition).unsqueeze(1)    # [batch_size, 1, d_model]
        
        # Standard layer normalization
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply adaptive scaling and shifting
        output = gamma * x_norm + beta
        
        return output


class GroupNorm(nn.Module):
    """
    Group Normalization for Transformers
    
    Divides channels into groups and normalizes within each group.
    Can be more stable than layer norm in some cases.
    
    Args:
        num_groups: Number of groups to divide channels into
        d_model: Dimension of the model
        eps: Small value for numerical stability
    """
    
    def __init__(
        self, 
        num_groups: int = 32, 
        d_model: int = 512,
        eps: float = 1e-6
    ):
        super(GroupNorm, self).__init__()
        
        assert d_model % num_groups == 0, \
            f"d_model ({d_model}) must be divisible by num_groups ({num_groups})"
        
        self.num_groups = num_groups
        self.d_model = d_model
        self.eps = eps
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of group normalization
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        
        Returns:
            Normalized tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Reshape to separate groups
        x = x.view(batch_size, seq_len, self.num_groups, d_model // self.num_groups)
        
        # Calculate mean and variance per group
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Reshape back
        x_norm = x_norm.view(batch_size, seq_len, d_model)
        
        # Scale and shift
        output = self.gamma * x_norm + self.beta
        
        return output


def create_normalization_layer(
    norm_type: str = 'layer',
    d_model: int = 512,
    **kwargs
) -> nn.Module:
    """
    Factory function to create different types of normalization layers
    
    Args:
        norm_type: Type of normalization ('layer', 'rms', 'group', or 'adaptive')
        d_model: Model dimension
        **kwargs: Additional arguments for specific normalization types
    
    Returns:
        Normalization layer module
    """
    norm_types = {
        'layer': LayerNormalization,
        'rms': RMSNorm,
        'group': GroupNorm,
        'adaptive': AdaptiveLayerNorm
    }
    
    if norm_type not in norm_types:
        raise ValueError(f"Unknown normalization type: {norm_type}")
    
    return norm_types[norm_type](d_model=d_model, **kwargs)
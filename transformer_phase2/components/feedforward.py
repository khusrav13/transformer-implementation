"""
Position-wise Feed-Forward Network for Transformer
This module implements the feed-forward network used in transformer layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    Implements FFN(x) = max(0, xW1 + b1)W2 + b2
    
    This network is applied to each position separately and identically.
    It consists of two linear transformations with a ReLU activation in between.
    The dimensionality typically increases in the hidden layer (d_model -> d_ff)
    and then projects back to the model dimension (d_ff -> d_model).
    
    Args:
        d_model: Dimension of the model (input and output dimension)
        d_ff: Dimension of the feed-forward hidden layer (typically 4 * d_model)
        dropout: Dropout rate
        activation: Activation function to use ('relu' or 'gelu')
    """
    
    def __init__(
        self, 
        d_model: int = 512, 
        d_ff: int = 2048, 
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super(PositionwiseFeedForward, self).__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Two linear transformations
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        self.activation_type = activation.lower()
        if self.activation_type == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_type == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform distribution"""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward network
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        # First linear transformation: d_model -> d_ff
        output = self.linear1(x)
        
        # Apply activation function
        output = self.activation(output)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Second linear transformation: d_ff -> d_model
        output = self.linear2(output)
        
        # Apply dropout again
        output = self.dropout(output)
        
        return output


class GatedFeedForward(nn.Module):
    """
    Gated Feed-Forward Network (variant used in some transformer models)
    
    Uses a gating mechanism similar to GLU (Gated Linear Unit)
    FFN(x) = (xW1) * σ(xW_gate) W2
    
    Args:
        d_model: Dimension of the model
        d_ff: Dimension of the feed-forward hidden layer
        dropout: Dropout rate
    """
    
    def __init__(
        self, 
        d_model: int = 512, 
        d_ff: int = 2048, 
        dropout: float = 0.1
    ):
        super(GatedFeedForward, self).__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Linear transformations
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear_gate = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in [self.linear1, self.linear_gate, self.linear2]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the gated feed-forward network
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Compute gate values
        gate = torch.sigmoid(self.linear_gate(x))
        
        # Apply gating mechanism
        output = self.linear1(x) * gate
        
        # Apply dropout
        output = self.dropout(output)
        
        # Final linear transformation
        output = self.linear2(output)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output


class ExpertFeedForward(nn.Module):
    """
    Expert Feed-Forward Network for Mixture of Experts (MoE) transformers
    
    Each expert is a separate feed-forward network.
    This is used in sparse transformers where tokens are routed to different experts.
    
    Args:
        d_model: Dimension of the model
        d_ff: Dimension of the feed-forward hidden layer
        num_experts: Number of expert networks
        dropout: Dropout rate
    """
    
    def __init__(
        self, 
        d_model: int = 512, 
        d_ff: int = 2048, 
        num_experts: int = 8,
        dropout: float = 0.1
    ):
        super(ExpertFeedForward, self).__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        
        # Create multiple expert networks
        self.experts = nn.ModuleList([
            PositionwiseFeedForward(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])
        
        # Gating network to select experts
        self.gate = nn.Linear(d_model, num_experts)
    
    def forward(
        self, 
        x: torch.Tensor, 
        num_experts_per_token: int = 2
    ) -> torch.Tensor:
        """
        Forward pass routing tokens to top-k experts
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            num_experts_per_token: Number of experts to use per token
        
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Compute gating scores
        gate_scores = self.gate(x)  # [batch_size, seq_len, num_experts]
        
        # Select top-k experts per token
        top_k_scores, top_k_indices = torch.topk(
            gate_scores, k=num_experts_per_token, dim=-1
        )
        
        # Normalize scores with softmax
        top_k_scores = F.softmax(top_k_scores, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Route tokens to selected experts
        for i in range(num_experts_per_token):
            expert_indices = top_k_indices[:, :, i]  # [batch_size, seq_len]
            expert_scores = top_k_scores[:, :, i:i+1]  # [batch_size, seq_len, 1]
            
            # Process tokens through selected experts
            for expert_idx in range(self.num_experts):
                # Find tokens assigned to this expert
                mask = (expert_indices == expert_idx).unsqueeze(-1)
                
                if mask.any():
                    # Process tokens through expert
                    expert_input = x * mask.float()
                    expert_output = self.experts[expert_idx](expert_input)
                    
                    # Weighted sum of expert outputs
                    output += expert_output * expert_scores * mask.float()
        
        return output


def create_feedforward_layer(
    ff_type: str = "standard",
    d_model: int = 512,
    d_ff: int = 2048,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create different types of feed-forward layers
    
    Args:
        ff_type: Type of feed-forward network ('standard', 'gated', or 'expert')
        d_model: Model dimension
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        **kwargs: Additional arguments for specific layer types
    
    Returns:
        Feed-forward layer module
    """
    ff_types = {
        "standard": PositionwiseFeedForward,
        "gated": GatedFeedForward,
        "expert": ExpertFeedForward
    }
    
    if ff_type not in ff_types:
        raise ValueError(f"Unknown feed-forward type: {ff_type}")
    
    return ff_types[ff_type](d_model, d_ff, dropout, **kwargs)
"""
Output Projection Layer for Transformer

This module implements the output projection layer that converts
decoder hidden states to vocabulary logits, with optional weight tying
to the input embedding layer.

Weight tying reduces parameters and often improves performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class OutputProjection(nn.Module):
    """
    Output Projection Layer

    Projects decoder output from d_model dimensions to vocabulary logits.
    Supports weight tying with input embeddings for parameter efficiency.

    Args:
        d_model: Dimension of model hidden states
        vocab_size: Size of target vocabulary
        bias: Whether to use bias in projection
        tie_weights: If provided, ties weights with this embedding layer
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        bias: bool = False,
        tie_weights: Optional[nn.Module] = None
    ):
        super(OutputProjection, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        if tie_weights is not None:
            # Weight tying: share weights with embedding layer
            self.weight = tie_weights.weight
            self.tied = True
        else:
            # Separate output projection weights
            self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
            self._init_weights()
            self.tied = False

        if bias:
            self.bias = nn.Parameter(torch.zeros(vocab_size))
        else:
            self.register_parameter('bias', None)

    def _init_weights(self):
        """Initialize weights if not tied"""
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to vocabulary logits

        Args:
            x: Hidden states [batch_size, seq_len, d_model]

        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        # Linear projection: [batch, seq, d_model] @ [d_model, vocab] -> [batch, seq, vocab]
        logits = F.linear(x, self.weight, self.bias)
        return logits

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, vocab_size={self.vocab_size}, tied={self.tied}'


class LMHead(nn.Module):
    """
    Language Model Head with optional normalization

    Enhanced output projection that includes:
    - Optional layer normalization before projection
    - Optional temperature scaling
    - Weight tying support

    Args:
        d_model: Dimension of model hidden states
        vocab_size: Size of vocabulary
        tie_weights: Optional embedding layer to tie weights with
        use_norm: Whether to apply layer norm before projection
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        tie_weights: Optional[nn.Module] = None,
        use_norm: bool = True
    ):
        super(LMHead, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Optional pre-projection normalization
        if use_norm:
            self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = None

        # Output projection
        self.projection = OutputProjection(
            d_model=d_model,
            vocab_size=vocab_size,
            bias=False,
            tie_weights=tie_weights
        )

    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute vocabulary logits

        Args:
            x: Hidden states [batch_size, seq_len, d_model]
            temperature: Temperature for scaling logits (default 1.0)

        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        # Apply normalization if present
        if self.norm is not None:
            x = self.norm(x)

        # Project to vocabulary
        logits = self.projection(x)

        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        return logits

    def get_probs(
        self,
        x: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute probability distribution over vocabulary

        Args:
            x: Hidden states [batch_size, seq_len, d_model]
            temperature: Temperature for softmax

        Returns:
            Probabilities [batch_size, seq_len, vocab_size]
        """
        logits = self.forward(x, temperature)
        return F.softmax(logits, dim=-1)

    def get_log_probs(
        self,
        x: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute log probabilities over vocabulary

        Args:
            x: Hidden states [batch_size, seq_len, d_model]
            temperature: Temperature for log softmax

        Returns:
            Log probabilities [batch_size, seq_len, vocab_size]
        """
        logits = self.forward(x, temperature)
        return F.log_softmax(logits, dim=-1)

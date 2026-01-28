"""
TransformerConfig - Centralized configuration for Transformer models

This module provides a dataclass-based configuration system that:
1. Centralizes all hyperparameters in one place
2. Provides sensible defaults matching modern practices
3. Supports serialization for checkpoint saving/loading
4. Validates configuration consistency
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
import json
import math


@dataclass
class TransformerConfig:
    """
    Configuration for Transformer model architecture.

    Architecture Parameters:
        d_model: Dimension of embeddings and hidden states
        num_heads: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        d_ff: Dimension of feed-forward hidden layer

    Vocabulary Parameters:
        vocab_size: Size of vocabulary
        src_vocab_size: Source vocabulary size (if different from vocab_size)
        tgt_vocab_size: Target vocabulary size (if different from vocab_size)
        padding_idx: Index of padding token

    Sequence Parameters:
        max_seq_length: Maximum sequence length

    Regularization Parameters:
        dropout: Dropout rate
        attention_dropout: Dropout for attention weights (if different)

    Architecture Choices:
        activation: Activation function ('relu', 'gelu', 'swiglu')
        norm_type: Normalization placement ('pre' or 'post')
        pos_encoding_type: Positional encoding ('sinusoidal', 'learned', 'rotary', 'none')
        use_alibi: Whether to use ALiBi instead of positional encoding
        tie_weights: Whether to tie input/output embeddings

    Optimization:
        use_flash_attention: Whether to use flash attention for long sequences
        gradient_checkpointing: Whether to use gradient checkpointing
    """

    # Architecture
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048

    # Vocabulary
    vocab_size: int = 32000
    src_vocab_size: Optional[int] = None  # Uses vocab_size if None
    tgt_vocab_size: Optional[int] = None  # Uses vocab_size if None
    padding_idx: int = 0
    bos_idx: int = 1
    eos_idx: int = 2

    # Sequence
    max_seq_length: int = 512

    # Regularization
    dropout: float = 0.1
    attention_dropout: Optional[float] = None  # Uses dropout if None

    # Architecture choices
    activation: Literal['relu', 'gelu', 'swiglu'] = 'swiglu'
    norm_type: Literal['pre', 'post'] = 'pre'
    pos_encoding_type: Literal['sinusoidal', 'learned', 'rotary', 'none'] = 'rotary'
    use_alibi: bool = False
    tie_weights: bool = True

    # Optimization
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False

    # Initialization
    init_std: float = 0.02

    def __post_init__(self):
        """Validate and set derived values"""
        # Set source/target vocab sizes
        if self.src_vocab_size is None:
            self.src_vocab_size = self.vocab_size
        if self.tgt_vocab_size is None:
            self.tgt_vocab_size = self.vocab_size

        # Set attention dropout
        if self.attention_dropout is None:
            self.attention_dropout = self.dropout

        # Validate head dimension
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"

        # Compute head dimension
        self.head_dim = self.d_model // self.num_heads

        # Validate ALiBi/positional encoding compatibility
        if self.use_alibi and self.pos_encoding_type != 'none':
            # ALiBi replaces traditional positional encoding
            self.pos_encoding_type = 'none'

    @property
    def num_parameters_estimate(self) -> int:
        """Estimate total number of parameters"""
        # Embeddings
        embed_params = self.src_vocab_size * self.d_model
        if not self.tie_weights:
            embed_params += self.tgt_vocab_size * self.d_model

        # Encoder layers
        encoder_attn = 4 * self.d_model * self.d_model  # Q, K, V, O projections
        encoder_ff = 2 * self.d_model * self.d_ff
        if self.activation == 'swiglu':
            encoder_ff = 3 * self.d_model * self.d_ff  # Extra gate projection
        encoder_params = (encoder_attn + encoder_ff) * self.num_encoder_layers

        # Decoder layers (similar but with cross-attention)
        decoder_attn = 4 * self.d_model * self.d_model  # Self-attention
        decoder_cross = 4 * self.d_model * self.d_model  # Cross-attention
        decoder_ff = 2 * self.d_model * self.d_ff
        if self.activation == 'swiglu':
            decoder_ff = 3 * self.d_model * self.d_ff
        decoder_params = (decoder_attn + decoder_cross + decoder_ff) * self.num_decoder_layers

        # Output projection
        output_params = 0 if self.tie_weights else self.d_model * self.tgt_vocab_size

        return embed_params + encoder_params + decoder_params + output_params

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return asdict(self)

    def save(self, path: str):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TransformerConfig':
        """Create config from dictionary"""
        return cls(**config_dict)

    @classmethod
    def load(cls, path: str) -> 'TransformerConfig':
        """Load config from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def base(cls) -> 'TransformerConfig':
        """Base configuration (65M parameters)"""
        return cls(
            d_model=512,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            d_ff=2048
        )

    @classmethod
    def small(cls) -> 'TransformerConfig':
        """Small configuration for testing (15M parameters)"""
        return cls(
            d_model=256,
            num_heads=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            d_ff=1024,
            vocab_size=10000
        )

    @classmethod
    def large(cls) -> 'TransformerConfig':
        """Large configuration (300M parameters)"""
        return cls(
            d_model=1024,
            num_heads=16,
            num_encoder_layers=12,
            num_decoder_layers=12,
            d_ff=4096
        )

    def __repr__(self) -> str:
        params = self.num_parameters_estimate
        if params >= 1e9:
            param_str = f"{params/1e9:.1f}B"
        elif params >= 1e6:
            param_str = f"{params/1e6:.1f}M"
        else:
            param_str = f"{params/1e3:.1f}K"

        return (
            f"TransformerConfig(\n"
            f"  d_model={self.d_model}, num_heads={self.num_heads},\n"
            f"  encoder_layers={self.num_encoder_layers}, decoder_layers={self.num_decoder_layers},\n"
            f"  d_ff={self.d_ff}, vocab_size={self.vocab_size},\n"
            f"  activation={self.activation}, norm_type={self.norm_type},\n"
            f"  pos_encoding={self.pos_encoding_type}, tie_weights={self.tie_weights},\n"
            f"  estimated_params={param_str}\n"
            f")"
        )

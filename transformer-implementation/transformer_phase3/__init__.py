"""
Transformer Phase 3: Full Architecture

This phase assembles the complete Transformer model by combining:
- Encoder and Decoder stacks from Phase 2
- Embeddings with positional encoding
- Output projection with weight tying
- Generation utilities (greedy, beam search, sampling)

Key Components:
- TransformerConfig: Centralized configuration management
- Transformer: Full encoder-decoder model
- DecoderOnlyTransformer: GPT-style decoder-only model
- TextGenerator: Generation utilities for inference
"""

from .model.config import TransformerConfig
from .model.transformer import Transformer, DecoderOnlyTransformer
from .model.output import OutputProjection, LMHead
from .utils.generation import (
    GenerationConfig,
    TextGenerator,
    greedy_decode,
    sample_decode,
    beam_search_decode
)

__all__ = [
    # Config
    'TransformerConfig',

    # Models
    'Transformer',
    'DecoderOnlyTransformer',

    # Output layers
    'OutputProjection',
    'LMHead',

    # Generation
    'GenerationConfig',
    'TextGenerator',
    'greedy_decode',
    'sample_decode',
    'beam_search_decode',
]

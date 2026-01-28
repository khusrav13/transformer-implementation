"""
Core Components Module - Building blocks of the transformer
"""

from .attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    SelfAttention,
    CrossAttention,
    create_attention_layer
)

# Note: Import these only after you create the corresponding files
# Commented out for now to avoid import errors

# from .embeddings import (
#     TokenEmbedding,
#     PositionalEncoding,
#     TransformerEmbedding
# )

# from .feedforward import (
#     PositionwiseFeedForward,
#     FeedForward
# )

# from .normalization import (
#     LayerNormalization,
#     PreNorm,
#     PostNorm
# )

__all__ = [
    # Attention mechanisms
    'ScaledDotProductAttention',
    'MultiHeadAttention',
    'SelfAttention',
    'CrossAttention',
    'create_attention_layer',
    
    # Will add these as we create the files
    # 'TokenEmbedding',
    # 'PositionalEncoding',
    # 'TransformerEmbedding',
    # 'PositionwiseFeedForward',
    # 'FeedForward',
    # 'LayerNormalization',
    # 'PreNorm',
    # 'PostNorm',
]
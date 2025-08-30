"""
Layers package for Transformer implementation.
Exports encoder and decoder layers.
"""

from .encoder_layer import (
    EncoderLayer,
    Encoder,
    EncoderStack
)

from .decoder_layer import (
    DecoderLayer,
    Decoder,
    DecoderStack
)

# Define public API
__all__ = [
    # Encoder components
    'EncoderLayer',
    'Encoder',
    'EncoderStack',
    
    # Decoder components
    'DecoderLayer',
    'Decoder',
    'DecoderStack'
]

# Version info
__version__ = '0.2.0'
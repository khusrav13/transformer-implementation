"""
Layers package for Transformer implementation.
Exports encoder and decoder layers.
"""

from .encoder_layer import (
    EncoderLayer,
    Encoder
)

from .decoder_layer import (
    DecoderLayer,
    Decoder
)

# Define public API
__all__ = [
    # Encoder components
    'EncoderLayer',
    'Encoder',
    
    # Decoder components
    'DecoderLayer',
    'Decoder'
]

# Version info
__version__ = '0.2.0'
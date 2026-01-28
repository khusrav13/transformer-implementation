"""
Full Transformer Model

This module implements the complete Transformer architecture by combining:
- Encoder with self-attention
- Decoder with masked self-attention and cross-attention
- Embeddings with positional encoding
- Output projection with optional weight tying

Supports both encoder-decoder (translation) and decoder-only (generation) modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformer_phase2.layers.encoder_layer import Encoder
from transformer_phase2.layers.decoder_layer import Decoder
from transformer_phase2.components.embeddings import TransformerEmbedding, TokenEmbedding
from transformer_phase2.utils.masking import (
    create_padding_mask,
    create_look_ahead_mask,
    create_combined_mask
)

from .config import TransformerConfig
from .output import LMHead


class Transformer(nn.Module):
    """
    Full Transformer Model

    Implements the complete encoder-decoder Transformer architecture from
    "Attention Is All You Need" with modern optimizations.

    Args:
        config: TransformerConfig object with all hyperparameters

    Architecture:
        Source -> Encoder Embedding -> Encoder Stack -> Encoder Output
        Target -> Decoder Embedding -> Decoder Stack (+ Encoder Output) -> LM Head -> Logits
    """

    def __init__(self, config: TransformerConfig):
        super(Transformer, self).__init__()

        self.config = config

        # Source embedding (encoder input)
        self.encoder_embedding = TransformerEmbedding(
            vocab_size=config.src_vocab_size,
            d_model=config.d_model,
            max_seq_length=config.max_seq_length,
            dropout=config.dropout,
            padding_idx=config.padding_idx,
            pos_encoding_type=config.pos_encoding_type,
            use_alibi=config.use_alibi
        )

        # Target embedding (decoder input)
        if config.src_vocab_size == config.tgt_vocab_size and config.tie_weights:
            # Share embeddings between encoder and decoder
            self.decoder_embedding = self.encoder_embedding
            self.shared_embedding = True
        else:
            self.decoder_embedding = TransformerEmbedding(
                vocab_size=config.tgt_vocab_size,
                d_model=config.d_model,
                max_seq_length=config.max_seq_length,
                dropout=config.dropout,
                padding_idx=config.padding_idx,
                pos_encoding_type=config.pos_encoding_type,
                use_alibi=config.use_alibi
            )
            self.shared_embedding = False

        # Encoder stack
        self.encoder = Encoder(
            num_layers=config.num_encoder_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            norm_type=config.norm_type
        )

        # Decoder stack
        self.decoder = Decoder(
            num_layers=config.num_decoder_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            norm_type=config.norm_type
        )

        # Output projection (LM Head)
        tie_embedding = None
        if config.tie_weights:
            # Tie output weights to decoder token embedding
            tie_embedding = self.decoder_embedding.token_embedding.embedding

        self.lm_head = LMHead(
            d_model=config.d_model,
            vocab_size=config.tgt_vocab_size,
            tie_weights=tie_embedding,
            use_norm=True  # Pre-norm before output projection
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Store config values for quick access
        self.d_model = config.d_model
        self.padding_idx = config.padding_idx

    def _init_weights(self, module: nn.Module):
        """Initialize weights with appropriate scheme"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode source sequence

        Args:
            src: Source token IDs [batch_size, src_len]
            src_mask: Source padding mask [batch_size, 1, 1, src_len]

        Returns:
            Encoder output [batch_size, src_len, d_model]
        """
        # Create mask if not provided
        if src_mask is None:
            src_mask = create_padding_mask(src, self.padding_idx)

        # Embed source tokens
        src_embedded = self.encoder_embedding(src)

        # Pass through encoder
        encoder_output = self.encoder(src_embedded, mask=src_mask)

        return encoder_output

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Decode target sequence

        Args:
            tgt: Target token IDs [batch_size, tgt_len]
            encoder_output: Encoder output [batch_size, src_len, d_model]
            tgt_mask: Combined causal and padding mask for target
            memory_mask: Padding mask for encoder output
            cache: KV cache for incremental decoding

        Returns:
            Decoder output [batch_size, tgt_len, d_model]
            Updated cache
        """
        # Create masks if not provided
        if tgt_mask is None:
            tgt_mask = create_combined_mask(tgt, self.padding_idx)

        # Embed target tokens
        tgt_embedded = self.decoder_embedding(tgt)

        # Pass through decoder
        decoder_output = self.decoder(
            x=tgt_embedded,
            encoder_output=encoder_output,
            self_mask=tgt_mask,
            cross_mask=memory_mask,
            cache=cache
        )

        return decoder_output

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        return_encoder_output: bool = False
    ) -> torch.Tensor:
        """
        Full forward pass

        Args:
            src: Source token IDs [batch_size, src_len]
            tgt: Target token IDs [batch_size, tgt_len]
            src_mask: Source padding mask
            tgt_mask: Target combined mask (causal + padding)
            memory_mask: Memory (encoder output) padding mask
            return_encoder_output: Whether to also return encoder output

        Returns:
            Logits [batch_size, tgt_len, vocab_size]
            Optionally (logits, encoder_output) if return_encoder_output=True
        """
        # Create masks if not provided
        if src_mask is None:
            src_mask = create_padding_mask(src, self.padding_idx)
        if tgt_mask is None:
            tgt_mask = create_combined_mask(tgt, self.padding_idx)
        if memory_mask is None:
            memory_mask = src_mask  # Use source padding mask for memory

        # Encode source
        encoder_output = self.encode(src, src_mask)

        # Decode target
        decoder_output = self.decode(
            tgt=tgt,
            encoder_output=encoder_output,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )

        # Project to vocabulary
        logits = self.lm_head(decoder_output)

        if return_encoder_output:
            return logits, encoder_output
        return logits

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """Count number of parameters"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, path: str):
        """Save model checkpoint and config"""
        import os
        os.makedirs(path, exist_ok=True)

        # Save config
        self.config.save(os.path.join(path, 'config.json'))

        # Save model weights
        torch.save(self.state_dict(), os.path.join(path, 'model.pt'))

    @classmethod
    def from_pretrained(cls, path: str, device: str = 'cpu') -> 'Transformer':
        """Load model from checkpoint"""
        import os

        # Load config
        config = TransformerConfig.load(os.path.join(path, 'config.json'))

        # Create model
        model = cls(config)

        # Load weights
        state_dict = torch.load(
            os.path.join(path, 'model.pt'),
            map_location=device
        )
        model.load_state_dict(state_dict)

        return model.to(device)


class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-Only Transformer (GPT-style)

    For language modeling and text generation tasks where
    we don't have a separate encoder.

    Args:
        config: TransformerConfig object
    """

    def __init__(self, config: TransformerConfig):
        super(DecoderOnlyTransformer, self).__init__()

        self.config = config

        # Token embedding
        self.embedding = TransformerEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_length=config.max_seq_length,
            dropout=config.dropout,
            padding_idx=config.padding_idx,
            pos_encoding_type=config.pos_encoding_type,
            use_alibi=config.use_alibi
        )

        # Decoder stack (modified to work without encoder)
        self.layers = nn.ModuleList([
            DecoderOnlyLayer(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                activation=config.activation,
                norm_type=config.norm_type
            )
            for _ in range(config.num_decoder_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)

        # Output projection
        tie_embedding = self.embedding.token_embedding.embedding if config.tie_weights else None
        self.lm_head = LMHead(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            tie_weights=tie_embedding,
            use_norm=False  # We have final_norm before lm_head
        )

        self.padding_idx = config.padding_idx

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[List[Dict[str, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
        """
        Forward pass

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Causal attention mask
            cache: List of KV caches for each layer
            use_cache: Whether to return updated cache

        Returns:
            Logits [batch_size, seq_len, vocab_size]
            Updated cache (if use_cache=True)
        """
        # Create causal mask
        if attention_mask is None:
            attention_mask = create_look_ahead_mask(input_ids.size(1), input_ids.device)

        # Embed tokens
        x = self.embedding(input_ids)

        # Initialize cache if needed
        new_cache = [] if use_cache else None

        # Pass through decoder layers
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, updated_cache = layer(x, attention_mask, cache=layer_cache)
            if use_cache:
                new_cache.append(updated_cache)

        # Final norm
        x = self.final_norm(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        if use_cache:
            return logits, new_cache
        return logits, None


class DecoderOnlyLayer(nn.Module):
    """
    Single layer for decoder-only transformer

    Contains only:
    - Masked self-attention
    - Feed-forward network
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'swiglu',
        norm_type: str = 'pre'
    ):
        super(DecoderOnlyLayer, self).__init__()

        from transformer_phase2.components.attention import SelfAttention
        from transformer_phase2.components.feedforward import PositionwiseFeedForward
        from transformer_phase2.components.normalization import LayerNormalization

        self.norm_type = norm_type

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
            dropout=dropout,
            activation=activation
        )

        # Layer norms
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass with optional caching"""
        if self.norm_type == 'pre':
            # Pre-norm
            residual = x
            x = self.norm1(x)
            attn_out, _ = self.self_attention(x, mask=mask)
            x = residual + self.dropout(attn_out)

            residual = x
            x = self.norm2(x)
            ff_out = self.feed_forward(x)
            x = residual + self.dropout(ff_out)
        else:
            # Post-norm
            residual = x
            attn_out, _ = self.self_attention(x, mask=mask)
            x = self.norm1(residual + self.dropout(attn_out))

            residual = x
            ff_out = self.feed_forward(x)
            x = self.norm2(residual + self.dropout(ff_out))

        return x, cache

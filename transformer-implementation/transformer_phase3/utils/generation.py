"""
Generation Utilities for Transformer Models

Implements various decoding strategies:
1. Greedy decoding - Select most probable token at each step
2. Beam search - Explore multiple hypotheses in parallel
3. Sampling - Stochastic generation with temperature
4. Top-k sampling - Sample from top k most probable tokens
5. Top-p (nucleus) sampling - Sample from smallest set exceeding probability p
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_length: int = 100
    min_length: int = 1

    # Stopping criteria
    eos_token_id: int = 2
    pad_token_id: int = 0

    # Decoding strategy
    do_sample: bool = False
    num_beams: int = 1

    # Sampling parameters
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0

    # Repetition handling
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0

    # Length control
    length_penalty: float = 1.0
    early_stopping: bool = False


class TextGenerator:
    """
    Text generation utilities for Transformer models

    Supports both encoder-decoder and decoder-only models.
    """

    def __init__(self, model: nn.Module, config: GenerationConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate sequences using the configured strategy

        Args:
            input_ids: Input token IDs for decoder (or encoder for enc-dec)
            encoder_output: Pre-computed encoder output (for enc-dec models)
            attention_mask: Attention mask for input
            **kwargs: Override generation config parameters

        Returns:
            Generated token IDs [batch_size, max_length]
        """
        # Update config with any overrides
        config = GenerationConfig(
            **{**self.config.__dict__, **kwargs}
        )

        if config.num_beams > 1:
            return self._beam_search(
                input_ids=input_ids,
                encoder_output=encoder_output,
                attention_mask=attention_mask,
                config=config
            )
        elif config.do_sample:
            return self._sample(
                input_ids=input_ids,
                encoder_output=encoder_output,
                attention_mask=attention_mask,
                config=config
            )
        else:
            return self._greedy_decode(
                input_ids=input_ids,
                encoder_output=encoder_output,
                attention_mask=attention_mask,
                config=config
            )

    def _greedy_decode(
        self,
        input_ids: Optional[torch.Tensor],
        encoder_output: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        config: GenerationConfig
    ) -> torch.Tensor:
        """
        Greedy decoding: select most probable token at each step

        Simple and fast but may miss better sequences.
        """
        batch_size = input_ids.size(0) if input_ids is not None else encoder_output.size(0)

        # Initialize output with BOS token or input
        if input_ids is None:
            generated = torch.full(
                (batch_size, 1),
                config.pad_token_id,
                dtype=torch.long,
                device=self.device
            )
        else:
            generated = input_ids.clone()

        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(config.max_length - generated.size(1)):
            # Get model predictions
            if encoder_output is not None:
                # Encoder-decoder model
                logits = self.model.decode(
                    tgt=generated,
                    encoder_output=encoder_output,
                    memory_mask=attention_mask
                )
                if isinstance(logits, tuple):
                    logits = logits[0]
                logits = self.model.lm_head(logits)
            else:
                # Decoder-only model
                logits, _ = self.model(generated)

            # Get next token logits (last position)
            next_token_logits = logits[:, -1, :]

            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated, config.repetition_penalty
                )

            # Select most probable token
            next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)

            # Don't update finished sequences
            next_tokens = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_tokens, config.pad_token_id),
                next_tokens
            )

            # Append to generated
            generated = torch.cat([generated, next_tokens], dim=1)

            # Update finished status
            finished = finished | (next_tokens.squeeze(-1) == config.eos_token_id)

            # Stop if all sequences finished
            if finished.all():
                break

        return generated

    def _sample(
        self,
        input_ids: Optional[torch.Tensor],
        encoder_output: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        config: GenerationConfig
    ) -> torch.Tensor:
        """
        Sampling with temperature, top-k, and top-p filtering

        More diverse outputs than greedy, good for creative tasks.
        """
        batch_size = input_ids.size(0) if input_ids is not None else encoder_output.size(0)

        # Initialize output
        if input_ids is None:
            generated = torch.full(
                (batch_size, 1),
                config.pad_token_id,
                dtype=torch.long,
                device=self.device
            )
        else:
            generated = input_ids.clone()

        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(config.max_length - generated.size(1)):
            # Get model predictions
            if encoder_output is not None:
                logits = self.model.decode(
                    tgt=generated,
                    encoder_output=encoder_output,
                    memory_mask=attention_mask
                )
                if isinstance(logits, tuple):
                    logits = logits[0]
                logits = self.model.lm_head(logits)
            else:
                logits, _ = self.model(generated)

            next_token_logits = logits[:, -1, :]

            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated, config.repetition_penalty
                )

            # Apply temperature
            if config.temperature != 1.0:
                next_token_logits = next_token_logits / config.temperature

            # Apply top-k filtering
            if config.top_k > 0:
                next_token_logits = self._top_k_filtering(
                    next_token_logits, config.top_k
                )

            # Apply top-p (nucleus) filtering
            if config.top_p < 1.0:
                next_token_logits = self._top_p_filtering(
                    next_token_logits, config.top_p
                )

            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            # Handle finished sequences
            next_tokens = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_tokens, config.pad_token_id),
                next_tokens
            )

            generated = torch.cat([generated, next_tokens], dim=1)
            finished = finished | (next_tokens.squeeze(-1) == config.eos_token_id)

            if finished.all():
                break

        return generated

    def _beam_search(
        self,
        input_ids: Optional[torch.Tensor],
        encoder_output: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        config: GenerationConfig
    ) -> torch.Tensor:
        """
        Beam search: explore multiple hypotheses in parallel

        Better quality than greedy, widely used in translation.
        """
        batch_size = input_ids.size(0) if input_ids is not None else encoder_output.size(0)
        num_beams = config.num_beams
        vocab_size = self.model.config.vocab_size if hasattr(self.model, 'config') else self.model.lm_head.vocab_size

        # Initialize beams
        if input_ids is None:
            beam_tokens = torch.full(
                (batch_size * num_beams, 1),
                config.pad_token_id,
                dtype=torch.long,
                device=self.device
            )
        else:
            # Expand input for beams
            beam_tokens = input_ids.unsqueeze(1).repeat(1, num_beams, 1)
            beam_tokens = beam_tokens.view(batch_size * num_beams, -1)

        # Expand encoder output for beams
        if encoder_output is not None:
            encoder_output = encoder_output.unsqueeze(1).repeat(1, num_beams, 1, 1)
            encoder_output = encoder_output.view(
                batch_size * num_beams, -1, encoder_output.size(-1)
            )

        # Beam scores (log probabilities)
        beam_scores = torch.zeros(batch_size * num_beams, device=self.device)
        beam_scores[1::num_beams] = -1e9  # Only first beam is active initially

        # Track finished beams
        done = [False] * batch_size

        for step in range(config.max_length - beam_tokens.size(1)):
            # Get model predictions
            if encoder_output is not None:
                logits = self.model.decode(
                    tgt=beam_tokens,
                    encoder_output=encoder_output
                )
                if isinstance(logits, tuple):
                    logits = logits[0]
                logits = self.model.lm_head(logits)
            else:
                logits, _ = self.model(beam_tokens)

            next_token_logits = logits[:, -1, :]
            log_probs = F.log_softmax(next_token_logits, dim=-1)

            # Add beam scores
            next_scores = beam_scores.unsqueeze(-1) + log_probs

            # Reshape for beam selection
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)

            # Get top beams
            next_scores, next_tokens = next_scores.topk(
                num_beams, dim=-1, largest=True, sorted=True
            )

            # Compute beam indices and token indices
            beam_indices = next_tokens // vocab_size
            token_indices = next_tokens % vocab_size

            # Build next beam tokens
            next_beam_tokens = []
            next_beam_scores = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # Keep existing beams for done batches
                    for beam_idx in range(num_beams):
                        flat_idx = batch_idx * num_beams + beam_idx
                        next_beam_tokens.append(beam_tokens[flat_idx])
                        next_beam_scores.append(beam_scores[flat_idx])
                    continue

                for beam_rank, (beam_idx, token_idx) in enumerate(
                    zip(beam_indices[batch_idx], token_indices[batch_idx])
                ):
                    flat_idx = batch_idx * num_beams + beam_idx.item()
                    new_tokens = torch.cat([
                        beam_tokens[flat_idx],
                        token_idx.unsqueeze(0)
                    ])
                    next_beam_tokens.append(new_tokens)
                    next_beam_scores.append(next_scores[batch_idx, beam_rank])

                    # Check if beam is done
                    if token_idx.item() == config.eos_token_id:
                        done[batch_idx] = True

            beam_tokens = torch.stack(next_beam_tokens)
            beam_scores = torch.stack(next_beam_scores)

            if all(done):
                break

        # Select best beam for each batch
        beam_scores = beam_scores.view(batch_size, num_beams)
        best_beam_idx = beam_scores.argmax(dim=-1)

        beam_tokens = beam_tokens.view(batch_size, num_beams, -1)
        best_sequences = beam_tokens[
            torch.arange(batch_size, device=self.device),
            best_beam_idx
        ]

        return best_sequences

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to discourage repeated tokens"""
        for batch_idx in range(logits.size(0)):
            for token_id in generated[batch_idx].unique():
                if logits[batch_idx, token_id] > 0:
                    logits[batch_idx, token_id] /= penalty
                else:
                    logits[batch_idx, token_id] *= penalty
        return logits

    def _top_k_filtering(
        self,
        logits: torch.Tensor,
        top_k: int
    ) -> torch.Tensor:
        """Keep only top-k tokens, set rest to -inf"""
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits

    def _top_p_filtering(
        self,
        logits: torch.Tensor,
        top_p: float
    ) -> torch.Tensor:
        """Keep smallest set of tokens with cumulative probability >= top_p"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        return logits


def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    max_length: int = 100,
    eos_token_id: int = 2,
    pad_token_id: int = 0
) -> torch.Tensor:
    """
    Convenience function for greedy decoding

    Args:
        model: Transformer model
        src: Source sequence [batch_size, src_len]
        max_length: Maximum generation length
        eos_token_id: End of sequence token ID
        pad_token_id: Padding token ID

    Returns:
        Generated sequence [batch_size, gen_len]
    """
    config = GenerationConfig(
        max_length=max_length,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        do_sample=False,
        num_beams=1
    )
    generator = TextGenerator(model, config)

    # Encode source
    encoder_output = model.encode(src)

    # Start with BOS token
    batch_size = src.size(0)
    start_tokens = torch.full(
        (batch_size, 1),
        model.config.bos_idx if hasattr(model, 'config') else 1,
        dtype=torch.long,
        device=src.device
    )

    return generator.generate(
        input_ids=start_tokens,
        encoder_output=encoder_output
    )


def sample_decode(
    model: nn.Module,
    src: torch.Tensor,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    eos_token_id: int = 2
) -> torch.Tensor:
    """
    Convenience function for sampling-based decoding

    Args:
        model: Transformer model
        src: Source sequence
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus (top-p) filtering
        eos_token_id: End of sequence token ID

    Returns:
        Generated sequence
    """
    config = GenerationConfig(
        max_length=max_length,
        eos_token_id=eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    generator = TextGenerator(model, config)

    encoder_output = model.encode(src)
    batch_size = src.size(0)
    start_tokens = torch.full(
        (batch_size, 1),
        model.config.bos_idx if hasattr(model, 'config') else 1,
        dtype=torch.long,
        device=src.device
    )

    return generator.generate(
        input_ids=start_tokens,
        encoder_output=encoder_output
    )


def beam_search_decode(
    model: nn.Module,
    src: torch.Tensor,
    max_length: int = 100,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    eos_token_id: int = 2
) -> torch.Tensor:
    """
    Convenience function for beam search decoding

    Args:
        model: Transformer model
        src: Source sequence
        max_length: Maximum generation length
        num_beams: Number of beams
        length_penalty: Length penalty (>1 encourages longer, <1 shorter)
        eos_token_id: End of sequence token ID

    Returns:
        Generated sequence
    """
    config = GenerationConfig(
        max_length=max_length,
        eos_token_id=eos_token_id,
        num_beams=num_beams,
        length_penalty=length_penalty
    )
    generator = TextGenerator(model, config)

    encoder_output = model.encode(src)
    batch_size = src.size(0)
    start_tokens = torch.full(
        (batch_size, 1),
        model.config.bos_idx if hasattr(model, 'config') else 1,
        dtype=torch.long,
        device=src.device
    )

    return generator.generate(
        input_ids=start_tokens,
        encoder_output=encoder_output
    )

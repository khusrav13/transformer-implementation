"""
Trainer for Transformer Models

Complete training pipeline with:
- Training loop with gradient accumulation
- Validation
- Checkpointing
- Logging and metrics tracking
- Early stopping
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Any
import time
import os
from dataclasses import dataclass, field
from collections import defaultdict
import json

from .scheduler import create_scheduler
from .losses import create_loss_function


@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Schedule
    scheduler_type: str = 'cosine'  # 'transformer', 'cosine', 'linear'
    warmup_steps: int = 1000
    total_steps: int = 100000

    # Loss
    label_smoothing: float = 0.1

    # Training
    num_epochs: int = 10
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_metric: str = 'val_loss'

    # Checkpointing
    output_dir: str = './checkpoints'
    save_best_only: bool = True

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class TrainingMetrics:
    """Track and aggregate training metrics"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = defaultdict(list)
        self.step_metrics = defaultdict(float)
        self.step_count = 0

    def update(self, **kwargs):
        """Update metrics for current step"""
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.step_metrics[key] += value
        self.step_count += 1

    def get_average(self) -> Dict[str, float]:
        """Get average of step metrics"""
        if self.step_count == 0:
            return {}
        return {
            key: value / self.step_count
            for key, value in self.step_metrics.items()
        }

    def log_step(self, step: int):
        """Log step metrics and reset"""
        avg_metrics = self.get_average()
        for key, value in avg_metrics.items():
            self.metrics[key].append((step, value))
        self.step_metrics = defaultdict(float)
        self.step_count = 0
        return avg_metrics

    def get_history(self) -> Dict[str, List]:
        """Get full training history"""
        return dict(self.metrics)


class Trainer:
    """
    Trainer for Transformer models

    Handles the complete training loop including:
    - Forward/backward passes
    - Gradient accumulation and clipping
    - Learning rate scheduling
    - Validation
    - Checkpointing
    - Metrics tracking

    Args:
        model: Transformer model to train
        config: TrainingConfig object
        train_dataloader: Training data loader
        val_dataloader: Validation data loader (optional)
        tokenizer: Tokenizer for decoding (optional, for generation)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        tokenizer: Optional[Any] = None
    ):
        self.model = model.to(config.device)
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9
        )

        # Setup scheduler
        d_model = model.config.d_model if hasattr(model, 'config') else 512
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type=config.scheduler_type,
            d_model=d_model,
            warmup_steps=config.warmup_steps,
            total_steps=config.total_steps
        )

        # Setup loss
        padding_idx = model.config.padding_idx if hasattr(model, 'config') else 0
        self.loss_fn = create_loss_function(
            loss_type='label_smoothing',
            smoothing=config.label_smoothing,
            ignore_index=padding_idx
        )

        # Metrics tracking
        self.train_metrics = TrainingMetrics()
        self.val_metrics = TrainingMetrics()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'perplexity': []
        }

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def train(self) -> Dict[str, List]:
        """
        Run full training loop

        Returns:
            Training history dictionary
        """
        print(f"Starting training on {self.config.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(self.train_dataloader.dataset)}")
        if self.val_dataloader:
            print(f"Validation samples: {len(self.val_dataloader.dataset)}")
        print("-" * 60)

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()

            # Training epoch
            train_loss = self._train_epoch(epoch)

            # Validation
            if self.val_dataloader:
                val_loss = self._validate()
                self.history['val_loss'].append((self.global_step, val_loss))

                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    if self.config.save_best_only:
                        self._save_checkpoint('best')
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs} completed in {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.4f}")
            if self.val_dataloader:
                print(f"  Val Loss: {val_loss:.4f} (best: {self.best_val_loss:.4f})")
            print("-" * 60)

        # Save final model
        self._save_checkpoint('final')
        self._save_history()

        return self.history

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch"""
        self.model.train()
        self.train_metrics.reset()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_dataloader):
            loss = self._train_step(batch)
            epoch_loss += loss
            num_batches += 1

            # Logging
            if self.global_step % self.config.log_interval == 0:
                avg_metrics = self.train_metrics.log_step(self.global_step)
                lr = self.scheduler.get_last_lr()[0]
                self.history['train_loss'].append((self.global_step, avg_metrics.get('loss', 0)))
                self.history['learning_rate'].append((self.global_step, lr))

                print(f"  Step {self.global_step}: loss={avg_metrics.get('loss', 0):.4f}, "
                      f"ppl={avg_metrics.get('perplexity', 0):.2f}, lr={lr:.2e}")

            # Validation during training
            if self.val_dataloader and self.global_step % self.config.eval_interval == 0:
                val_loss = self._validate()
                self.history['val_loss'].append((self.global_step, val_loss))
                self.model.train()

            # Save checkpoint
            if self.global_step % self.config.save_interval == 0:
                self._save_checkpoint(f'step_{self.global_step}')

        return epoch_loss / max(1, num_batches)

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step with gradient accumulation"""
        # Move batch to device
        src = batch['src'].to(self.config.device)
        tgt = batch['tgt'].to(self.config.device)
        labels = batch.get('labels', tgt[:, 1:]).to(self.config.device)

        # Forward pass
        # For training, we typically use teacher forcing:
        # Input to decoder: [BOS, tok1, tok2, ..., tokN]
        # Labels:           [tok1, tok2, ..., tokN, EOS]
        decoder_input = tgt[:, :-1] if tgt.size(1) > 1 else tgt

        logits = self.model(src, decoder_input)

        # Adjust labels to match logits length
        if logits.size(1) != labels.size(1):
            labels = labels[:, :logits.size(1)]

        # Compute loss
        loss = self.loss_fn(logits, labels)
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Update metrics
        with torch.no_grad():
            perplexity = torch.exp(loss * self.config.gradient_accumulation_steps)
            self.train_metrics.update(
                loss=loss.item() * self.config.gradient_accumulation_steps,
                perplexity=perplexity.item()
            )

        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        self.global_step += 1
        return loss.item() * self.config.gradient_accumulation_steps

    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation"""
        self.model.eval()
        self.val_metrics.reset()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_dataloader:
            src = batch['src'].to(self.config.device)
            tgt = batch['tgt'].to(self.config.device)
            labels = batch.get('labels', tgt[:, 1:]).to(self.config.device)

            decoder_input = tgt[:, :-1] if tgt.size(1) > 1 else tgt
            logits = self.model(src, decoder_input)

            if logits.size(1) != labels.size(1):
                labels = labels[:, :logits.size(1)]

            loss = self.loss_fn(logits, labels)
            total_loss += loss.item()
            num_batches += 1

            self.val_metrics.update(
                loss=loss.item(),
                perplexity=torch.exp(loss).item()
            )

        avg_loss = total_loss / max(1, num_batches)
        self.history['perplexity'].append((self.global_step, torch.exp(torch.tensor(avg_loss)).item()))
        return avg_loss

    def _save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.config.output_dir, f'checkpoint_{name}.pt')

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {}
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")

    def _save_history(self):
        """Save training history"""
        history_path = os.path.join(self.config.output_dir, 'training_history.json')

        # Convert to serializable format
        serializable_history = {}
        for key, value in self.history.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], tuple):
                    serializable_history[key] = [
                        {'step': step, 'value': val} for step, val in value
                    ]
                else:
                    serializable_history[key] = value

        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)

    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from step {self.global_step}")

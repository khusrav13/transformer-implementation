"""
Learning Rate Schedulers for Transformer Training

Implements:
1. Warmup + Inverse Square Root decay (original Transformer)
2. Warmup + Cosine Annealing (modern alternative)
3. Warmup + Linear decay
"""

import math
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional


class TransformerLRScheduler(_LRScheduler):
    """
    Original Transformer learning rate schedule from "Attention Is All You Need"

    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    This increases linearly during warmup, then decays proportionally to
    the inverse square root of the step number.

    Args:
        optimizer: Optimizer to schedule
        d_model: Model dimension (used for scaling)
        warmup_steps: Number of warmup steps
        last_epoch: Last epoch for resuming
    """

    def __init__(
        self,
        optimizer,
        d_model: int = 512,
        warmup_steps: int = 4000,
        last_epoch: int = -1
    ):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.scale = d_model ** (-0.5)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)

        # Warmup phase: linear increase
        # Decay phase: inverse square root
        lr_scale = self.scale * min(
            step ** (-0.5),
            step * (self.warmup_steps ** (-1.5))
        )

        return [base_lr * lr_scale for base_lr in self.base_lrs]


class WarmupCosineScheduler(_LRScheduler):
    """
    Warmup + Cosine Annealing scheduler

    Popular in modern transformer training (GPT, BERT fine-tuning).

    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr: Minimum learning rate at end
        last_epoch: Last epoch for resuming
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int = 1000,
        total_steps: int = 100000,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            # Linear warmup
            lr_scale = step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        return [
            self.min_lr + (base_lr - self.min_lr) * lr_scale
            for base_lr in self.base_lrs
        ]


class WarmupLinearScheduler(_LRScheduler):
    """
    Warmup + Linear decay scheduler

    Simple and effective, used in many BERT-style models.

    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        last_epoch: Last epoch for resuming
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int = 1000,
        total_steps: int = 100000,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            # Linear warmup
            lr_scale = step / max(1, self.warmup_steps)
        else:
            # Linear decay
            remaining = self.total_steps - step
            total_decay_steps = self.total_steps - self.warmup_steps
            lr_scale = max(0.0, remaining / max(1, total_decay_steps))

        return [base_lr * lr_scale for base_lr in self.base_lrs]


def create_scheduler(
    optimizer,
    scheduler_type: str = 'transformer',
    d_model: int = 512,
    warmup_steps: int = 4000,
    total_steps: int = 100000,
    min_lr: float = 1e-6
):
    """
    Factory function to create learning rate scheduler

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: 'transformer', 'cosine', or 'linear'
        d_model: Model dimension (for transformer scheduler)
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr: Minimum learning rate

    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'transformer':
        return TransformerLRScheduler(
            optimizer,
            d_model=d_model,
            warmup_steps=warmup_steps
        )
    elif scheduler_type == 'cosine':
        return WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr
        )
    elif scheduler_type == 'linear':
        return WarmupLinearScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

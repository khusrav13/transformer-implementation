"""
Transformer Phase 4 & 5: Training Pipeline and Applications

This phase implements complete training infrastructure:
- Learning rate schedulers (warmup + decay)
- Label smoothing loss
- Training loop with gradient accumulation
- Evaluation metrics (BLEU, perplexity, accuracy)
- Visualization utilities
- Synthetic datasets for testing

Key Components:
- Trainer: Complete training loop
- TrainingConfig: Training hyperparameters
- Datasets: Copy, Reversal, Sorting, Arithmetic, Translation
- Metrics: Perplexity, BLEU, token/sequence accuracy
- Visualization: Training curves, attention plots
"""

from .training.trainer import Trainer, TrainingConfig
from .training.scheduler import (
    TransformerLRScheduler,
    WarmupCosineScheduler,
    WarmupLinearScheduler,
    create_scheduler
)
from .training.losses import (
    LabelSmoothingLoss,
    CrossEntropyLoss,
    create_loss_function
)
from .training.metrics import (
    compute_perplexity,
    compute_token_accuracy,
    compute_sequence_accuracy,
    compute_bleu,
    evaluate_model
)
from .data.datasets import (
    CopyTaskDataset,
    ReversalTaskDataset,
    SortingTaskDataset,
    ArithmeticDataset,
    SimpleTranslationDataset,
    create_dataloader
)

__all__ = [
    # Training
    'Trainer',
    'TrainingConfig',

    # Schedulers
    'TransformerLRScheduler',
    'WarmupCosineScheduler',
    'WarmupLinearScheduler',
    'create_scheduler',

    # Losses
    'LabelSmoothingLoss',
    'CrossEntropyLoss',
    'create_loss_function',

    # Metrics
    'compute_perplexity',
    'compute_token_accuracy',
    'compute_sequence_accuracy',
    'compute_bleu',
    'evaluate_model',

    # Datasets
    'CopyTaskDataset',
    'ReversalTaskDataset',
    'SortingTaskDataset',
    'ArithmeticDataset',
    'SimpleTranslationDataset',
    'create_dataloader',
]

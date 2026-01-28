"""
Loss Functions for Transformer Training

Implements:
1. Label Smoothing Cross-Entropy Loss
2. Focal Loss (for imbalanced data)
3. KL Divergence Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross-Entropy Loss

    Instead of one-hot targets, distributes some probability mass to other classes.
    This helps prevent overconfidence and improves generalization.

    Original paper: "Rethinking the Inception Architecture for Computer Vision"
    Used in original Transformer paper with smoothing=0.1

    Args:
        smoothing: Label smoothing factor (0.0 = no smoothing, 1.0 = uniform)
        ignore_index: Index to ignore in loss computation (padding)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        ignore_index: int = 0,
        reduction: str = 'mean'
    ):
        super().__init__()
        assert 0.0 <= smoothing <= 1.0, "Smoothing must be in [0, 1]"

        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.confidence = 1.0 - smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label smoothing loss

        Args:
            logits: Model predictions [batch_size, seq_len, vocab_size] or [N, vocab_size]
            targets: Target indices [batch_size, seq_len] or [N]

        Returns:
            Loss value
        """
        # Flatten if needed
        if logits.dim() == 3:
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(-1, vocab_size)
            targets = targets.view(-1)
        else:
            vocab_size = logits.size(-1)

        # Create mask for ignored tokens
        non_pad_mask = targets.ne(self.ignore_index)

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed target distribution
        # Instead of one-hot, distribute (smoothing / (vocab_size - 1)) to other classes
        # and (1 - smoothing) to the correct class
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (vocab_size - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)

        # Zero out padding positions in targets
        smooth_targets = smooth_targets * non_pad_mask.unsqueeze(-1)

        # Compute KL divergence (equivalent to cross-entropy with soft targets)
        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == 'mean':
            # Mean over non-padding tokens
            return loss.sum() / non_pad_mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CrossEntropyLoss(nn.Module):
    """
    Standard Cross-Entropy Loss with padding support

    Simple wrapper for cleaner interface.

    Args:
        ignore_index: Index to ignore (padding)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        ignore_index: int = 0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss

        Args:
            logits: [batch_size, seq_len, vocab_size] or [N, vocab_size]
            targets: [batch_size, seq_len] or [N]

        Returns:
            Loss value
        """
        if logits.dim() == 3:
            # Flatten for loss computation
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(-1, vocab_size)
            targets = targets.view(-1)

        return self.loss_fn(logits, targets)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance

    Down-weights easy examples and focuses on hard ones.
    Useful when vocabulary has very uneven frequency distribution.

    Args:
        gamma: Focusing parameter (0 = CE, higher = more focus on hard examples)
        alpha: Class balancing weight
        ignore_index: Index to ignore
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        ignore_index: int = 0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss"""
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

        # Create mask
        non_pad_mask = targets.ne(self.ignore_index)

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Get probability of correct class
        targets_one_hot = F.one_hot(
            targets.clamp(min=0),
            num_classes=logits.size(-1)
        ).float()
        pt = (probs * targets_one_hot).sum(dim=-1)

        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            focal_weight = focal_weight * self.alpha

        # Compute loss
        loss = focal_weight * ce_loss

        # Mask padding
        loss = loss * non_pad_mask

        if self.reduction == 'mean':
            return loss.sum() / non_pad_mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def create_loss_function(
    loss_type: str = 'label_smoothing',
    smoothing: float = 0.1,
    ignore_index: int = 0,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss function

    Args:
        loss_type: 'label_smoothing', 'cross_entropy', or 'focal'
        smoothing: Label smoothing factor
        ignore_index: Padding index to ignore
        **kwargs: Additional arguments for specific loss functions

    Returns:
        Loss function module
    """
    if loss_type == 'label_smoothing':
        return LabelSmoothingLoss(
            smoothing=smoothing,
            ignore_index=ignore_index
        )
    elif loss_type == 'cross_entropy':
        return CrossEntropyLoss(ignore_index=ignore_index)
    elif loss_type == 'focal':
        return FocalLoss(
            ignore_index=ignore_index,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

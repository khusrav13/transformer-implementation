"""
Evaluation Metrics for Transformer Models

Implements:
1. Perplexity
2. BLEU Score (simplified implementation)
3. Accuracy (token-level and sequence-level)
4. Edit Distance
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from collections import Counter
import math


def compute_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 0
) -> float:
    """
    Compute perplexity from logits and targets

    Perplexity = exp(cross_entropy_loss)
    Lower is better.

    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        targets: Target token IDs [batch_size, seq_len]
        ignore_index: Index to ignore (padding)

    Returns:
        Perplexity value
    """
    # Flatten
    if logits.dim() == 3:
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)

    # Compute cross-entropy
    loss = F.cross_entropy(
        logits,
        targets,
        ignore_index=ignore_index,
        reduction='mean'
    )

    return math.exp(loss.item())


def compute_token_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 0
) -> float:
    """
    Compute token-level accuracy

    Args:
        predictions: Predicted token IDs [batch_size, seq_len]
        targets: Target token IDs [batch_size, seq_len]
        ignore_index: Index to ignore (padding)

    Returns:
        Accuracy (0.0 to 1.0)
    """
    # Create mask for valid positions
    mask = targets != ignore_index

    # Compare predictions to targets
    correct = (predictions == targets) & mask

    # Compute accuracy
    accuracy = correct.sum().float() / mask.sum().float()

    return accuracy.item()


def compute_sequence_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 0
) -> float:
    """
    Compute sequence-level accuracy (exact match)

    Args:
        predictions: Predicted token IDs [batch_size, seq_len]
        targets: Target token IDs [batch_size, seq_len]
        ignore_index: Index to ignore (padding)

    Returns:
        Sequence accuracy (0.0 to 1.0)
    """
    batch_size = predictions.size(0)

    # Create mask for valid positions
    mask = targets != ignore_index

    # Check if entire sequence matches
    correct_sequences = 0
    for i in range(batch_size):
        seq_mask = mask[i]
        if torch.all(predictions[i][seq_mask] == targets[i][seq_mask]):
            correct_sequences += 1

    return correct_sequences / batch_size


def compute_bleu(
    predictions: List[List[int]],
    references: List[List[int]],
    max_n: int = 4,
    smooth: bool = True
) -> Dict[str, float]:
    """
    Compute BLEU score (simplified implementation)

    BLEU measures n-gram overlap between prediction and reference.

    Args:
        predictions: List of predicted token sequences
        references: List of reference token sequences
        max_n: Maximum n-gram order (typically 4)
        smooth: Whether to apply smoothing for short sequences

    Returns:
        Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4, and BLEU
    """
    def get_ngrams(sequence: List[int], n: int) -> Counter:
        """Extract n-grams from sequence"""
        return Counter(
            tuple(sequence[i:i + n])
            for i in range(len(sequence) - n + 1)
        )

    def compute_precision(pred_ngrams: Counter, ref_ngrams: Counter) -> float:
        """Compute clipped precision"""
        clipped_count = 0
        total_count = 0

        for ngram, count in pred_ngrams.items():
            clipped_count += min(count, ref_ngrams.get(ngram, 0))
            total_count += count

        if total_count == 0:
            return 0.0
        return clipped_count / total_count

    # Compute n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        precision_sum = 0.0
        count = 0

        for pred, ref in zip(predictions, references):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)

            if len(pred) >= n:
                precision = compute_precision(pred_ngrams, ref_ngrams)
                if smooth and precision == 0:
                    precision = 1.0 / (len(pred) - n + 1 + 1)  # Add-1 smoothing
                precision_sum += precision
                count += 1

        precisions.append(precision_sum / max(count, 1))

    # Compute brevity penalty
    pred_len = sum(len(p) for p in predictions)
    ref_len = sum(len(r) for r in references)

    if pred_len > ref_len:
        bp = 1.0
    elif pred_len == 0:
        bp = 0.0
    else:
        bp = math.exp(1 - ref_len / pred_len)

    # Compute BLEU scores
    results = {}
    for n in range(1, max_n + 1):
        # BLEU-n is geometric mean of precisions 1 to n
        if all(p > 0 for p in precisions[:n]):
            log_precision = sum(math.log(p) for p in precisions[:n]) / n
            results[f'bleu_{n}'] = bp * math.exp(log_precision)
        else:
            results[f'bleu_{n}'] = 0.0

    # Standard BLEU is BLEU-4
    results['bleu'] = results.get('bleu_4', 0.0)

    return results


def compute_edit_distance(
    pred: List[int],
    ref: List[int]
) -> int:
    """
    Compute Levenshtein edit distance between two sequences

    Args:
        pred: Predicted sequence
        ref: Reference sequence

    Returns:
        Edit distance (number of insertions, deletions, substitutions)
    """
    m, n = len(pred), len(ref)

    # Create distance matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill in the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1],      # Insertion
                    dp[i - 1][j - 1]   # Substitution
                )

    return dp[m][n]


def compute_character_error_rate(
    predictions: List[List[int]],
    references: List[List[int]]
) -> float:
    """
    Compute Character Error Rate (CER)

    CER = (insertions + deletions + substitutions) / reference_length

    Args:
        predictions: List of predicted sequences
        references: List of reference sequences

    Returns:
        CER (lower is better)
    """
    total_distance = 0
    total_length = 0

    for pred, ref in zip(predictions, references):
        total_distance += compute_edit_distance(pred, ref)
        total_length += len(ref)

    if total_length == 0:
        return 0.0
    return total_distance / total_length


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cpu',
    ignore_index: int = 0,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Comprehensive evaluation of a model

    Args:
        model: Model to evaluate
        dataloader: Evaluation data loader
        device: Device to use
        ignore_index: Padding index
        max_batches: Maximum number of batches (for quick evaluation)

    Returns:
        Dictionary with all metrics
    """
    model.eval()

    total_loss = 0.0
    total_token_acc = 0.0
    total_seq_acc = 0.0
    all_predictions = []
    all_references = []
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break

        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        labels = batch.get('labels', tgt[:, 1:]).to(device)

        # Forward pass
        decoder_input = tgt[:, :-1] if tgt.size(1) > 1 else tgt
        logits = model(src, decoder_input)

        # Adjust labels
        if logits.size(1) != labels.size(1):
            labels = labels[:, :logits.size(1)]

        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=ignore_index,
            reduction='mean'
        )
        total_loss += loss.item()

        # Get predictions
        predictions = logits.argmax(dim=-1)

        # Token accuracy
        total_token_acc += compute_token_accuracy(predictions, labels, ignore_index)

        # Sequence accuracy
        total_seq_acc += compute_sequence_accuracy(predictions, labels, ignore_index)

        # Collect for BLEU
        for i in range(predictions.size(0)):
            mask = labels[i] != ignore_index
            pred_seq = predictions[i][mask].tolist()
            ref_seq = labels[i][mask].tolist()
            if pred_seq and ref_seq:
                all_predictions.append(pred_seq)
                all_references.append(ref_seq)

        num_batches += 1

    # Compute averages
    avg_loss = total_loss / max(num_batches, 1)
    perplexity = math.exp(avg_loss)
    token_accuracy = total_token_acc / max(num_batches, 1)
    sequence_accuracy = total_seq_acc / max(num_batches, 1)

    # Compute BLEU
    bleu_scores = compute_bleu(all_predictions, all_references)

    # Compute CER
    cer = compute_character_error_rate(all_predictions, all_references)

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'token_accuracy': token_accuracy,
        'sequence_accuracy': sequence_accuracy,
        'cer': cer,
        **bleu_scores
    }

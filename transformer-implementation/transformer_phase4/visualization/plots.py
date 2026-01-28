"""
Visualization Utilities for Transformer Training

Implements:
1. Training curves (loss, accuracy, perplexity)
2. Learning rate schedule visualization
3. Attention pattern visualization
4. Model comparison plots
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import os


def setup_plot_style():
    """Set up consistent plot styling"""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })


def plot_training_curves(
    history: Dict[str, List],
    save_path: Optional[str] = None,
    title: str = "Training Progress"
):
    """
    Plot training curves (loss, accuracy, etc.)

    Args:
        history: Dictionary with training history
            Keys like 'train_loss', 'val_loss' with values as list of (step, value) tuples
        save_path: Path to save figure (optional)
        title: Plot title
    """
    setup_plot_style()

    # Determine number of subplots needed
    metrics = ['loss', 'perplexity', 'learning_rate']
    available_metrics = [m for m in metrics if f'train_{m}' in history or m in history]

    if not available_metrics:
        # Try alternative keys
        available_metrics = [k.replace('train_', '').replace('val_', '')
                           for k in history.keys()]
        available_metrics = list(set(available_metrics))[:4]

    n_plots = len(available_metrics)
    if n_plots == 0:
        print("No metrics to plot")
        return

    fig, axes = plt.subplots(1, min(n_plots, 3), figsize=(5 * min(n_plots, 3), 4))
    if n_plots == 1:
        axes = [axes]

    colors = {'train': '#2196F3', 'val': '#FF5722', 'default': '#4CAF50'}

    for idx, metric in enumerate(available_metrics[:3]):
        ax = axes[idx]

        # Plot training metric
        train_key = f'train_{metric}' if f'train_{metric}' in history else metric
        if train_key in history and history[train_key]:
            data = history[train_key]
            if isinstance(data[0], (tuple, list)):
                steps, values = zip(*data)
            elif isinstance(data[0], dict):
                steps = [d['step'] for d in data]
                values = [d['value'] for d in data]
            else:
                steps = range(len(data))
                values = data

            ax.plot(steps, values, color=colors['train'], label='Train', linewidth=2)

        # Plot validation metric
        val_key = f'val_{metric}'
        if val_key in history and history[val_key]:
            data = history[val_key]
            if isinstance(data[0], (tuple, list)):
                steps, values = zip(*data)
            elif isinstance(data[0], dict):
                steps = [d['step'] for d in data]
                values = [d['value'] for d in data]
            else:
                steps = range(len(data))
                values = data

            ax.plot(steps, values, color=colors['val'], label='Validation',
                   linewidth=2, linestyle='--')

        ax.set_xlabel('Step')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.legend()

        # Log scale for perplexity
        if 'perplexity' in metric:
            ax.set_yscale('log')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    return fig


def plot_learning_rate_schedule(
    scheduler,
    total_steps: int = 10000,
    save_path: Optional[str] = None
):
    """
    Visualize learning rate schedule

    Args:
        scheduler: Learning rate scheduler
        total_steps: Number of steps to simulate
        save_path: Path to save figure
    """
    setup_plot_style()

    # Simulate learning rate over steps
    lrs = []
    for step in range(total_steps):
        scheduler.last_epoch = step
        lr = scheduler.get_lr()[0]
        lrs.append(lr)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(range(total_steps), lrs, color='#2196F3', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')

    # Mark warmup phase if applicable
    if hasattr(scheduler, 'warmup_steps'):
        warmup = scheduler.warmup_steps
        ax.axvline(x=warmup, color='#FF5722', linestyle='--', label=f'Warmup ends ({warmup})')
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved LR schedule to {save_path}")

    return fig


def plot_attention_weights(
    attention_weights: torch.Tensor,
    src_tokens: Optional[List[str]] = None,
    tgt_tokens: Optional[List[str]] = None,
    layer: int = 0,
    head: int = 0,
    save_path: Optional[str] = None,
    title: str = "Attention Weights"
):
    """
    Visualize attention weights as heatmap

    Args:
        attention_weights: Attention weights [batch, heads, tgt_len, src_len] or [heads, tgt_len, src_len]
        src_tokens: Source token labels (optional)
        tgt_tokens: Target token labels (optional)
        layer: Which layer to visualize
        head: Which head to visualize
        save_path: Path to save figure
        title: Plot title
    """
    setup_plot_style()

    # Handle different input shapes
    if attention_weights.dim() == 4:
        # [batch, heads, tgt, src]
        weights = attention_weights[0, head].detach().cpu().numpy()
    elif attention_weights.dim() == 3:
        # [heads, tgt, src]
        weights = attention_weights[head].detach().cpu().numpy()
    else:
        weights = attention_weights.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(weights, cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax, label='Attention Weight')

    # Add labels if provided
    if tgt_tokens:
        ax.set_yticks(range(len(tgt_tokens)))
        ax.set_yticklabels(tgt_tokens)
    if src_tokens:
        ax.set_xticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens, rotation=45, ha='right')

    ax.set_xlabel('Source Position')
    ax.set_ylabel('Target Position')
    ax.set_title(f'{title} (Head {head})')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention plot to {save_path}")

    return fig


def plot_multi_head_attention(
    attention_weights: torch.Tensor,
    num_heads: int = 8,
    save_path: Optional[str] = None
):
    """
    Visualize all attention heads

    Args:
        attention_weights: [heads, tgt_len, src_len] or [batch, heads, tgt, src]
        num_heads: Number of heads to plot
        save_path: Path to save figure
    """
    setup_plot_style()

    if attention_weights.dim() == 4:
        weights = attention_weights[0].detach().cpu().numpy()
    else:
        weights = attention_weights.detach().cpu().numpy()

    num_heads = min(num_heads, weights.shape[0])
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).flatten()

    for head in range(num_heads):
        ax = axes[head]
        im = ax.imshow(weights[head], cmap='Blues', aspect='auto')
        ax.set_title(f'Head {head}')
        ax.set_xlabel('Src')
        ax.set_ylabel('Tgt')

    # Hide unused subplots
    for i in range(num_heads, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Multi-Head Attention Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved multi-head attention to {save_path}")

    return fig


def plot_prediction_comparison(
    inputs: List[List[int]],
    targets: List[List[int]],
    predictions: List[List[int]],
    vocab: Optional[Dict[int, str]] = None,
    num_examples: int = 5,
    save_path: Optional[str] = None
):
    """
    Visualize prediction vs target comparison

    Args:
        inputs: Input sequences
        targets: Target sequences
        predictions: Predicted sequences
        vocab: Optional vocabulary mapping
        num_examples: Number of examples to show
        save_path: Path to save figure
    """
    setup_plot_style()

    num_examples = min(num_examples, len(inputs))

    fig, axes = plt.subplots(num_examples, 1, figsize=(12, 2 * num_examples))
    if num_examples == 1:
        axes = [axes]

    for idx in range(num_examples):
        ax = axes[idx]
        ax.axis('off')

        # Convert to strings if vocab provided
        if vocab:
            inp_str = ' '.join(vocab.get(t, str(t)) for t in inputs[idx] if t > 2)
            tgt_str = ' '.join(vocab.get(t, str(t)) for t in targets[idx] if t > 2)
            pred_str = ' '.join(vocab.get(t, str(t)) for t in predictions[idx] if t > 2)
        else:
            inp_str = str(inputs[idx])
            tgt_str = str(targets[idx])
            pred_str = str(predictions[idx])

        # Check if correct
        is_correct = targets[idx] == predictions[idx]
        color = '#4CAF50' if is_correct else '#F44336'

        text = f"Input:    {inp_str}\n"
        text += f"Target:   {tgt_str}\n"
        text += f"Predict:  {pred_str}"

        ax.text(0.02, 0.5, text, transform=ax.transAxes, fontsize=10,
               verticalalignment='center', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

    plt.suptitle('Prediction Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved predictions to {save_path}")

    return fig


def plot_metrics_summary(
    metrics: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Evaluation Metrics"
):
    """
    Create a summary visualization of all metrics

    Args:
        metrics: Dictionary of metric names and values
        save_path: Path to save figure
        title: Plot title
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter metrics for visualization
    plot_metrics = {k: v for k, v in metrics.items()
                   if isinstance(v, (int, float)) and not np.isnan(v)}

    names = list(plot_metrics.keys())
    values = list(plot_metrics.values())

    # Color based on metric type
    colors = []
    for name in names:
        if 'loss' in name.lower() or 'error' in name.lower():
            colors.append('#F44336')  # Red for errors
        elif 'accuracy' in name.lower() or 'bleu' in name.lower():
            colors.append('#4CAF50')  # Green for accuracy
        else:
            colors.append('#2196F3')  # Blue for others

    # Create bar chart
    bars = ax.bar(names, values, color=colors, alpha=0.8)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
               f'{value:.4f}',
               ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Value')
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')

    # Add legend
    legend_patches = [
        mpatches.Patch(color='#F44336', label='Error metrics (lower is better)'),
        mpatches.Patch(color='#4CAF50', label='Accuracy metrics (higher is better)'),
        mpatches.Patch(color='#2196F3', label='Other metrics'),
    ]
    ax.legend(handles=legend_patches, loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics summary to {save_path}")

    return fig


def create_training_report(
    history: Dict[str, List],
    final_metrics: Dict[str, float],
    output_dir: str,
    model_name: str = "Transformer"
):
    """
    Create a complete training report with all visualizations

    Args:
        history: Training history dictionary
        final_metrics: Final evaluation metrics
        output_dir: Directory to save report
        model_name: Name of the model
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating training report in {output_dir}")
    print("=" * 50)

    # 1. Training curves
    plot_training_curves(
        history,
        save_path=os.path.join(output_dir, 'training_curves.png'),
        title=f"{model_name} Training Progress"
    )

    # 2. Metrics summary
    plot_metrics_summary(
        final_metrics,
        save_path=os.path.join(output_dir, 'metrics_summary.png'),
        title=f"{model_name} Final Metrics"
    )

    # 3. Print summary
    print("\nFinal Metrics:")
    print("-" * 30)
    for name, value in final_metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")

    print("\nReport saved successfully!")
    plt.close('all')

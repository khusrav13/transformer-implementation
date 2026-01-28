"""
Main Training Script for Transformer

Trains a transformer model on synthetic tasks (copy, reversal, sorting)
and generates visualizations of training progress.

Usage:
    python train.py --task copy --epochs 5
    python train.py --task reversal --epochs 10
    python train.py --task arithmetic --epochs 20
"""

import torch
import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformer_phase3.model.config import TransformerConfig
from transformer_phase3.model.transformer import Transformer
from transformer_phase4.training.trainer import Trainer, TrainingConfig
from transformer_phase4.data.datasets import create_dataloader
from transformer_phase4.training.metrics import evaluate_model
from transformer_phase4.visualization.plots import (
    plot_training_curves,
    plot_metrics_summary,
    create_training_report
)


def train_copy_task(
    num_epochs: int = 5,
    batch_size: int = 64,
    seq_length: int = 10,
    vocab_size: int = 50,
    device: str = None
):
    """
    Train on copy task - the simplest test to verify the model works

    The model learns to copy input sequence to output.
    """
    print("=" * 60)
    print("TRAINING ON COPY TASK")
    print("=" * 60)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model config (small for fast training)
    model_config = TransformerConfig(
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        vocab_size=vocab_size,
        max_seq_length=seq_length + 10,
        dropout=0.1,
        activation='relu',  # Simpler for small task
        norm_type='post'
    )

    print(f"\nModel Configuration:")
    print(model_config)

    # Create model
    model = Transformer(model_config)
    print(f"\nTotal parameters: {model.get_num_parameters():,}")

    # Create datasets
    print("\nCreating datasets...")
    train_loader, train_dataset = create_dataloader(
        dataset_type='copy',
        num_samples=5000,
        batch_size=batch_size,
        seq_length=seq_length,
        vocab_size=vocab_size,
        shuffle=True
    )

    val_loader, val_dataset = create_dataloader(
        dataset_type='copy',
        num_samples=500,
        batch_size=batch_size,
        seq_length=seq_length,
        vocab_size=vocab_size,
        shuffle=False
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Training config
    training_config = TrainingConfig(
        learning_rate=1e-3,
        weight_decay=0.01,
        max_grad_norm=1.0,
        scheduler_type='cosine',
        warmup_steps=100,
        total_steps=num_epochs * len(train_loader),
        label_smoothing=0.1,
        num_epochs=num_epochs,
        log_interval=50,
        eval_interval=200,
        save_interval=500,
        early_stopping_patience=3,
        output_dir='./checkpoints/copy_task',
        device=device
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        config=training_config,
        train_dataloader=train_loader,
        val_dataloader=val_loader
    )

    # Train!
    print("\nStarting training...")
    start_time = time.time()
    history = trainer.train()
    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time:.1f}s")

    # Final evaluation
    print("\nRunning final evaluation...")
    model.eval()
    final_metrics = evaluate_model(
        model,
        val_loader,
        device=device,
        ignore_index=model_config.padding_idx
    )

    print("\nFinal Metrics:")
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    os.makedirs('./visualizations', exist_ok=True)

    plot_training_curves(
        history,
        save_path='./visualizations/copy_task_training.png',
        title='Copy Task Training Progress'
    )

    plot_metrics_summary(
        final_metrics,
        save_path='./visualizations/copy_task_metrics.png',
        title='Copy Task Final Metrics'
    )

    # Test some examples
    print("\nTesting examples:")
    print("-" * 40)
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        src = batch['src'][:3].to(device)
        tgt = batch['tgt'][:3].to(device)

        # Get predictions
        logits = model(src, tgt[:, :-1])
        predictions = logits.argmax(dim=-1)

        for i in range(3):
            src_seq = src[i].tolist()
            tgt_seq = tgt[i, 1:].tolist()  # Remove BOS
            pred_seq = predictions[i].tolist()

            # Remove padding and special tokens for display
            src_clean = [t for t in src_seq if t > 2]
            tgt_clean = [t for t in tgt_seq if t > 2]
            pred_clean = [t for t in pred_seq[:len(tgt_clean)] if t > 2]

            match = "OK" if tgt_clean == pred_clean else "MISMATCH"
            print(f"Input:    {src_clean}")
            print(f"Target:   {tgt_clean}")
            print(f"Predict:  {pred_clean} [{match}]")
            print()

    return model, history, final_metrics


def train_reversal_task(
    num_epochs: int = 10,
    batch_size: int = 64,
    seq_length: int = 8,
    vocab_size: int = 30,
    device: str = None
):
    """
    Train on reversal task - slightly harder than copy
    """
    print("=" * 60)
    print("TRAINING ON REVERSAL TASK")
    print("=" * 60)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model config
    model_config = TransformerConfig(
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        vocab_size=vocab_size,
        max_seq_length=seq_length + 10,
        dropout=0.1
    )

    print(f"\nModel: {model_config.d_model}d, {model_config.num_heads}h")
    model = Transformer(model_config)
    print(f"Parameters: {model.get_num_parameters():,}")

    # Datasets
    train_loader, _ = create_dataloader(
        dataset_type='reversal',
        num_samples=8000,
        batch_size=batch_size,
        seq_length=seq_length,
        vocab_size=vocab_size
    )

    val_loader, _ = create_dataloader(
        dataset_type='reversal',
        num_samples=1000,
        batch_size=batch_size,
        seq_length=seq_length,
        vocab_size=vocab_size,
        shuffle=False
    )

    # Training
    training_config = TrainingConfig(
        learning_rate=1e-3,
        scheduler_type='cosine',
        warmup_steps=200,
        total_steps=num_epochs * len(train_loader),
        num_epochs=num_epochs,
        log_interval=100,
        eval_interval=500,
        output_dir='./checkpoints/reversal_task',
        device=device
    )

    trainer = Trainer(model, training_config, train_loader, val_loader)

    print("\nTraining...")
    history = trainer.train()

    # Evaluate
    model.eval()
    final_metrics = evaluate_model(model, val_loader, device=device)

    print("\nFinal Metrics:")
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # Visualize
    os.makedirs('./visualizations', exist_ok=True)
    plot_training_curves(
        history,
        save_path='./visualizations/reversal_task_training.png',
        title='Reversal Task Training'
    )

    # Examples
    print("\nExamples:")
    with torch.no_grad():
        batch = next(iter(val_loader))
        src = batch['src'][:3].to(device)
        tgt = batch['tgt'][:3].to(device)
        logits = model(src, tgt[:, :-1])
        preds = logits.argmax(dim=-1)

        for i in range(3):
            src_clean = [t for t in src[i].tolist() if t > 2]
            tgt_clean = [t for t in tgt[i, 1:].tolist() if t > 2]
            pred_clean = [t for t in preds[i].tolist()[:len(tgt_clean)] if t > 2]
            print(f"In:  {src_clean} -> Out: {pred_clean} (expect: {tgt_clean})")

    return model, history, final_metrics


def train_arithmetic_task(
    num_epochs: int = 20,
    batch_size: int = 64,
    device: str = None
):
    """
    Train on simple addition task
    """
    print("=" * 60)
    print("TRAINING ON ARITHMETIC TASK (Addition)")
    print("=" * 60)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model - small vocab (just digits + operators)
    model_config = TransformerConfig(
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        vocab_size=15,  # 0-9 + special tokens + operators
        max_seq_length=20,
        dropout=0.1
    )

    print(f"\nModel: {model_config.d_model}d, vocab={model_config.vocab_size}")
    model = Transformer(model_config)
    print(f"Parameters: {model.get_num_parameters():,}")

    # Datasets
    train_loader, _ = create_dataloader(
        dataset_type='arithmetic',
        num_samples=10000,
        batch_size=batch_size,
        max_digits=2
    )

    val_loader, _ = create_dataloader(
        dataset_type='arithmetic',
        num_samples=1000,
        batch_size=batch_size,
        max_digits=2,
        shuffle=False
    )

    # Training
    training_config = TrainingConfig(
        learning_rate=5e-4,
        scheduler_type='cosine',
        warmup_steps=300,
        total_steps=num_epochs * len(train_loader),
        num_epochs=num_epochs,
        log_interval=100,
        eval_interval=500,
        output_dir='./checkpoints/arithmetic_task',
        device=device
    )

    trainer = Trainer(model, training_config, train_loader, val_loader)

    print("\nTraining...")
    history = trainer.train()

    # Evaluate
    model.eval()
    final_metrics = evaluate_model(model, val_loader, device=device)

    print("\nFinal Metrics:")
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # Visualize
    os.makedirs('./visualizations', exist_ok=True)
    plot_training_curves(
        history,
        save_path='./visualizations/arithmetic_task_training.png',
        title='Arithmetic Task Training'
    )

    return model, history, final_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Transformer on synthetic tasks')
    parser.add_argument('--task', type=str, default='copy',
                       choices=['copy', 'reversal', 'arithmetic', 'all'],
                       help='Task to train on')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("TRANSFORMER TRAINING PIPELINE")
    print("Phase 4 & 5: Training and Applications")
    print("=" * 60)
    print(f"\nDevice: {args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Task: {args.task}")
    print(f"Epochs: {args.epochs}")

    if args.task == 'copy' or args.task == 'all':
        train_copy_task(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )

    if args.task == 'reversal' or args.task == 'all':
        train_reversal_task(
            num_epochs=args.epochs * 2,  # Harder task
            batch_size=args.batch_size,
            device=args.device
        )

    if args.task == 'arithmetic' or args.task == 'all':
        train_arithmetic_task(
            num_epochs=args.epochs * 4,  # Hardest task
            batch_size=args.batch_size,
            device=args.device
        )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("Visualizations saved to ./visualizations/")
    print("Checkpoints saved to ./checkpoints/")
    print("=" * 60)


if __name__ == '__main__':
    main()

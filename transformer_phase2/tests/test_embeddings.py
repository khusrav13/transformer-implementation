"""
Comprehensive test suite for Embedding components.
Tests token embeddings, positional encodings, and their properties.
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.embeddings import (
    TokenEmbedding,
    PositionalEncoding,
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding
)

# Test configuration
BATCH_SIZE = 2
SEQ_LEN = 100
D_MODEL = 512
VOCAB_SIZE = 10000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Visualization directory
VIZ_DIR = Path(__file__).parent / 'visualizations' / 'embeddings'
VIZ_DIR.mkdir(parents=True, exist_ok=True)

class TestTokenEmbedding:
    """Test token embedding layer."""
    
    def test_initialization(self):
        """Test token embedding initialization."""
        print("\n" + "="*50)
        print("Testing Token Embedding Initialization...")
        
        embed = TokenEmbedding(VOCAB_SIZE, D_MODEL)
        
        # Check weight shape
        assert embed.embedding.weight.shape == (VOCAB_SIZE, D_MODEL)
        
        # Check scale factor
        expected_scale = np.sqrt(D_MODEL)
        assert abs(embed.scale - expected_scale) < 1e-5
        
        print("✓ Token embedding initialized correctly")
        print(f"  Vocabulary size: {VOCAB_SIZE}")
        print(f"  Embedding dimension: {D_MODEL}")
        print(f"  Scale factor: {embed.scale:.2f}")
    
    def test_forward_pass(self):
        """Test token embedding forward pass."""
        print("\n" + "="*50)
        print("Testing Token Embedding Forward Pass...")
        
        embed = TokenEmbedding(VOCAB_SIZE, D_MODEL).to(DEVICE)
        
        # Create random token indices
        tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(DEVICE)
        
        # Forward pass
        output = embed(tokens)
        
        # Check output shape
        assert output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
        
        # Check scaling
        unscaled = embed.embedding(tokens)
        torch.testing.assert_close(output, unscaled * embed.scale)
        
        print("✓ Forward pass successful")
        print(f"  Input shape: {tokens.shape}")
        print(f"  Output shape: {output.shape}")
    
    def test_gradient_flow(self):
        """Test gradient flow through token embedding."""
        print("\n" + "="*50)
        print("Testing Token Embedding Gradient Flow...")
        
        embed = TokenEmbedding(VOCAB_SIZE, D_MODEL).to(DEVICE)
        tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(DEVICE)
        
        # Forward pass
        output = embed(tokens)
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist
        assert embed.embedding.weight.grad is not None
        assert not torch.isnan(embed.embedding.weight.grad).any()
        
        grad_norm = embed.embedding.weight.grad.norm().item()
        print(f"✓ Gradients flow correctly")
        print(f"  Gradient norm: {grad_norm:.6f}")

class TestPositionalEncoding:
    """Test sinusoidal positional encoding."""
    
    def test_initialization(self):
        """Test positional encoding initialization."""
        print("\n" + "="*50)
        print("Testing Positional Encoding Initialization...")
        
        pos_enc = PositionalEncoding(D_MODEL, max_len=5000)
        
        # Check encoding shape
        assert pos_enc.pe.shape == (1, 5000, D_MODEL)
        
        print("✓ Positional encoding initialized")
        print(f"  Max length: 5000")
        print(f"  Model dimension: {D_MODEL}")
    
    def test_sinusoidal_properties(self):
        """Test mathematical properties of sinusoidal encoding."""
        print("\n" + "="*50)
        print("Testing Sinusoidal Properties...")
        
        pos_enc = PositionalEncoding(D_MODEL, max_len=1000)
        encoding = pos_enc.pe[0, :100, :]  # First 100 positions
        
        # Test 1: Values bounded between -1 and 1
        assert encoding.abs().max() <= 1.0
        
        # Test 2: Even dimensions use sin, odd use cos
        position = 10
        for i in range(0, min(10, D_MODEL), 2):
            div_term = 10000 ** (i / D_MODEL)
            expected_sin = np.sin(position / div_term)
            actual_sin = encoding[position, i].item()
            assert abs(actual_sin - expected_sin) < 1e-4
        
        print("✓ Sinusoidal properties verified")
        print(f"  Max absolute value: {encoding.abs().max():.4f}")
        print(f"  Min value: {encoding.min():.4f}")
        print(f"  Max value: {encoding.max():.4f}")
    
    def test_relative_positions(self):
        """Test that model can learn relative positions."""
        print("\n" + "="*50)
        print("Testing Relative Position Properties...")
        
        pos_enc = PositionalEncoding(D_MODEL)
        encoding = pos_enc.pe[0, :100, :]
        
        # Compute dot products between positions
        dots = torch.matmul(encoding, encoding.T)
        
        # Positions should have decreasing similarity with distance
        pos1 = 10
        similarities = dots[pos1, :].numpy()
        
        # Check that nearby positions are more similar
        assert similarities[pos1] > similarities[pos1 + 10]
        assert similarities[pos1 + 1] > similarities[pos1 + 20]
        
        print("✓ Relative position properties verified")
        print(f"  Self-similarity: {similarities[pos1]:.4f}")
        print(f"  Similarity at +1: {similarities[pos1+1]:.4f}")
        print(f"  Similarity at +10: {similarities[pos1+10]:.4f}")
    
    def test_visualization(self):
        """Visualize positional encoding patterns."""
        print("\n" + "="*50)
        print("Creating Positional Encoding Visualizations...")
        
        pos_enc = PositionalEncoding(D_MODEL)
        encoding = pos_enc.pe[0, :100, :128].numpy()  # First 100 positions, 128 dims
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Heatmap of encoding
        ax = axes[0, 0]
        im = ax.imshow(encoding.T, cmap='RdBu_r', aspect='auto')
        ax.set_xlabel('Position')
        ax.set_ylabel('Dimension')
        ax.set_title('Positional Encoding Heatmap')
        plt.colorbar(im, ax=ax)
        
        # 2. Specific dimensions over positions
        ax = axes[0, 1]
        dims_to_plot = [0, 1, 4, 5, 10, 11, 20, 21]
        for dim in dims_to_plot:
            ax.plot(encoding[:, dim], label=f'Dim {dim}', alpha=0.7)
        ax.set_xlabel('Position')
        ax.set_ylabel('Encoding Value')
        ax.set_title('Encoding Values for Selected Dimensions')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 3. Frequency analysis
        ax = axes[1, 0]
        positions = np.arange(100)
        for i in range(0, 8, 2):
            wavelength = 10000 ** (i / D_MODEL)
            ax.plot(positions, np.sin(positions / wavelength), 
                   label=f'Dim {i} (λ={wavelength:.1f})', alpha=0.7)
        ax.set_xlabel('Position')
        ax.set_ylabel('Sine Value')
        ax.set_title('Sine Wavelengths for Different Dimensions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Distance matrix
        ax = axes[1, 1]
        distances = np.sqrt(((encoding[:50, None, :] - encoding[None, :50, :]) ** 2).sum(axis=2))
        im = ax.imshow(distances, cmap='viridis')
        ax.set_xlabel('Position')
        ax.set_ylabel('Position')
        ax.set_title('L2 Distance Between Position Encodings')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        save_path = VIZ_DIR / 'positional_encoding_analysis.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")
        plt.close()

class TestLearnedPositionalEmbedding:
    """Test learned positional embeddings."""
    
    def test_initialization(self):
        """Test learned positional embedding initialization."""
        print("\n" + "="*50)
        print("Testing Learned Positional Embedding...")
        
        learned_pe = LearnedPositionalEmbedding(D_MODEL, max_len=1000)
        
        # Check shape
        assert learned_pe.pe.weight.shape == (1000, D_MODEL)
        
        # Check that embeddings are learnable
        assert learned_pe.pe.weight.requires_grad
        
        print("✓ Learned positional embedding initialized")
        print(f"  Parameters: {learned_pe.pe.weight.numel():,}")
    
    def test_forward_pass(self):
        """Test learned embedding forward pass."""
        print("\n" + "="*50)
        print("Testing Learned Embedding Forward Pass...")
        
        learned_pe = LearnedPositionalEmbedding(D_MODEL, max_len=1000).to(DEVICE)
        
        # Create input
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL).to(DEVICE)
        
        # Forward pass
        output = learned_pe(x)
        
        # Check shape preserved
        assert output.shape == x.shape
        
        # Check that positional info was added
        assert not torch.allclose(output, x)
        
        print("✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")

class TestRotaryPositionalEmbedding:
    """Test Rotary Position Embedding (RoPE)."""
    
    def test_rotation_properties(self):
        """Test rotation properties of RoPE."""
        print("\n" + "="*50)
        print("Testing Rotary Position Embedding...")
        
        rope = RotaryPositionalEmbedding(D_MODEL)
        
        # Create query and key
        q = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL).to(DEVICE)
        k = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL).to(DEVICE)
        
        # Apply RoPE
        q_rot = rope(q, seq_len=SEQ_LEN)
        k_rot = rope(k, seq_len=SEQ_LEN)
        
        # Check shape preservation
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        
        # Check that rotation was applied
        assert not torch.allclose(q_rot, q)
        assert not torch.allclose(k_rot, k)
        
        print("✓ RoPE properties verified")
        print(f"  Rotation applied successfully")
    
    def test_relative_position_preservation(self):
        """Test that RoPE preserves relative positions."""
        print("\n" + "="*50)
        print("Testing RoPE Relative Position Preservation...")
        
        rope = RotaryPositionalEmbedding(D_MODEL)
        
        # Create simple embeddings
        x = torch.ones(1, 10, D_MODEL)
        
        # Apply RoPE
        x_rot = rope(x, seq_len=10)
        
        # Compute attention scores between positions
        scores = torch.matmul(x_rot, x_rot.transpose(-2, -1))
        
        # Check that relative distances are preserved
        # Positions with same distance should have similar scores
        pos0_to_pos1 = scores[0, 0, 1].item()
        pos1_to_pos2 = scores[0, 1, 2].item()
        
        assert abs(pos0_to_pos1 - pos1_to_pos2) < 0.1
        
        print("✓ Relative positions preserved")
        print(f"  Score(0→1): {pos0_to_pos1:.4f}")
        print(f"  Score(1→2): {pos1_to_pos2:.4f}")

def compare_positional_encodings():
    """Compare different positional encoding methods."""
    print("\n" + "="*50)
    print("Comparing Positional Encoding Methods...")
    
    # Initialize encodings
    sinusoidal = PositionalEncoding(D_MODEL, max_len=100)
    learned = LearnedPositionalEmbedding(D_MODEL, max_len=100)
    
    # Get encodings
    sin_enc = sinusoidal.pe[0, :100, :].detach().numpy()
    learn_enc = learned.pe.weight[:100, :].detach().numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Sinusoidal encoding
    axes[0, 0].imshow(sin_enc[:50, :128].T, cmap='RdBu_r', aspect='auto')
    axes[0, 0].set_title('Sinusoidal Encoding')
    axes[0, 0].set_xlabel('Position')
    axes[0, 0].set_ylabel('Dimension')
    
    # Learned encoding (random init)
    axes[0, 1].imshow(learn_enc[:50, :128].T, cmap='RdBu_r', aspect='auto')
    axes[0, 1].set_title('Learned Encoding (Random Init)')
    axes[0, 1].set_xlabel('Position')
    axes[0, 1].set_ylabel('Dimension')
    
    # Difference
    diff = sin_enc[:50, :128] - learn_enc[:50, :128]
    im = axes[0, 2].imshow(diff.T, cmap='RdBu_r', aspect='auto')
    axes[0, 2].set_title('Difference (Sinusoidal - Learned)')
    axes[0, 2].set_xlabel('Position')
    axes[0, 2].set_ylabel('Dimension')
    plt.colorbar(im, ax=axes[0, 2])
    
    # Compute statistics
    # Rank analysis
    sin_rank = np.linalg.matrix_rank(sin_enc)
    learn_rank = np.linalg.matrix_rank(learn_enc)
    
    axes[1, 0].bar(['Sinusoidal', 'Learned'], [sin_rank, learn_rank])
    axes[1, 0].set_title('Rank of Encoding Matrix')
    axes[1, 0].set_ylabel('Rank')
    
    # Norm distribution
    sin_norms = np.linalg.norm(sin_enc, axis=1)
    learn_norms = np.linalg.norm(learn_enc, axis=1)
    
    axes[1, 1].plot(sin_norms, label='Sinusoidal', alpha=0.7)
    axes[1, 1].plot(learn_norms, label='Learned', alpha=0.7)
    axes[1, 1].set_title('L2 Norm per Position')
    axes[1, 1].set_xlabel('Position')
    axes[1, 1].set_ylabel('L2 Norm')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Correlation matrix
    sin_corr = np.corrcoef(sin_enc.T[:20])
    im = axes[1, 2].imshow(sin_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 2].set_title('Dimension Correlation (First 20 dims)')
    axes[1, 2].set_xlabel('Dimension')
    axes[1, 2].set_ylabel('Dimension')
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.suptitle('Comparison of Positional Encoding Methods', fontsize=16)
    plt.tight_layout()
    
    save_path = VIZ_DIR / 'positional_encoding_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison saved to {save_path}")
    plt.close()
    
    print(f"  Sinusoidal rank: {sin_rank}")
    print(f"  Learned rank: {learn_rank}")
    print(f"  Sinusoidal mean norm: {sin_norms.mean():.4f}")
    print(f"  Learned mean norm: {learn_norms.mean():.4f}")

# Run all tests
if __name__ == "__main__":
    print("\n" + "="*60)
    print("EMBEDDINGS TEST SUITE")
    print("="*60)
    
    # Token Embedding tests
    test_token = TestTokenEmbedding()
    test_token.test_initialization()
    test_token.test_forward_pass()
    test_token.test_gradient_flow()
    
    # Positional Encoding tests
    test_pos = TestPositionalEncoding()
    test_pos.test_initialization()
    test_pos.test_sinusoidal_properties()
    test_pos.test_relative_positions()
    test_pos.test_visualization()
    
    # Learned Positional tests
    test_learned = TestLearnedPositionalEmbedding()
    test_learned.test_initialization()
    test_learned.test_forward_pass()
    
    # RoPE tests
    test_rope = TestRotaryPositionalEmbedding()
    test_rope.test_rotation_properties()
    test_rope.test_relative_position_preservation()
    
    # Comparison
    compare_positional_encodings()
    
    print("\n" + "="*60)
    print(" ALL EMBEDDING TESTS PASSED!")
    print("="*60)
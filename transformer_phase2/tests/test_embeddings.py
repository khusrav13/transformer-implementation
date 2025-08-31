"""
Comprehensive test suite for Embedding components.
Tests token embeddings, positional encodings, and their properties.
FIXED: Now handles both 3D and 4D tensor inputs for RoPE
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
current_dir = Path(__file__).resolve()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from components.embeddings import (
    TokenEmbedding,
    PositionalEncoding,
    LearnedPositionalEncoding,
    RotaryPositionalEncoding,
    TransformerEmbedding,
    create_embedding_layer
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
    
    def test_padding(self):
        """Test padding token handling."""
        print("\n" + "="*50)
        print("Testing Padding Token Handling...")
        
        padding_idx = 0
        embed = TokenEmbedding(VOCAB_SIZE, D_MODEL, padding_idx=padding_idx).to(DEVICE)
        
        # Check padding embedding is zero - need to move zeros to same device
        assert torch.allclose(embed.embedding.weight[padding_idx], torch.zeros(D_MODEL).to(DEVICE))
        
        # Create tokens with padding
        tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(DEVICE)
        tokens[:, :10] = padding_idx  # First 10 tokens are padding
        
        output = embed(tokens)
        
        # Check padding positions have zero embeddings (before scaling)
        padding_output = output[:, :10, :] / embed.scale
        assert torch.allclose(padding_output, torch.zeros_like(padding_output))
        
        print("✓ Padding handled correctly")


class TestPositionalEncoding:
    """Test sinusoidal positional encoding."""
    
    def test_initialization(self):
        """Test positional encoding initialization."""
        print("\n" + "="*50)
        print("Testing Positional Encoding Initialization...")
        
        pos_enc = PositionalEncoding(D_MODEL, max_seq_length=5000)
        
        # Check encoding shape
        assert pos_enc.pe.shape == (1, 5000, D_MODEL)
        
        print("✓ Positional encoding initialized")
        print(f"  Max length: 5000")
        print(f"  Model dimension: {D_MODEL}")
    
    def test_sinusoidal_properties(self):
        """Test mathematical properties of sinusoidal encoding."""
        print("\n" + "="*50)
        print("Testing Sinusoidal Properties...")
        
        pos_enc = PositionalEncoding(D_MODEL, max_seq_length=1000)
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


class TestLearnedPositionalEncoding:
    """Test learned positional encodings."""
    
    def test_initialization(self):
        """Test learned positional embedding initialization."""
        print("\n" + "="*50)
        print("Testing Learned Positional Encoding...")
        
        learned_pe = LearnedPositionalEncoding(D_MODEL, max_seq_length=1000)
        
        # Check shape
        assert learned_pe.position_embeddings.weight.shape == (1000, D_MODEL)
        assert learned_pe.position_embeddings.weight.requires_grad
        
        print("✓ Learned positional encoding initialized")
        print(f"  Parameters: {learned_pe.position_embeddings.weight.numel():,}")
    
    def test_forward_pass(self):
        """Test learned embedding forward pass."""
        print("\n" + "="*50)
        print("Testing Learned Encoding Forward Pass...")
        
        learned_pe = LearnedPositionalEncoding(D_MODEL, max_seq_length=1000).to(DEVICE)
        
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
    
    def test_gradient_flow(self):
        """Test gradient flow through learned positional encoding."""
        print("\n" + "="*50)
        print("Testing Learned Encoding Gradient Flow...")
        
        learned_pe = LearnedPositionalEncoding(D_MODEL).to(DEVICE)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, requires_grad=True).to(DEVICE)
        
        output = learned_pe(x)
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist
        assert learned_pe.position_embeddings.weight.grad is not None
        assert not torch.isnan(learned_pe.position_embeddings.weight.grad).any()
        
        grad_norm = learned_pe.position_embeddings.weight.grad.norm().item()
        print(f"✓ Gradients flow correctly")
        print(f"  Position embedding gradient norm: {grad_norm:.6f}")


class TestRotaryPositionalEncoding:
    """Test Rotary Position Encoding (RoPE)."""
    
    def test_3d_input(self):
        """Test RoPE with 3D input tensors."""
        print("\n" + "="*50)
        print("Testing RoPE with 3D Input...")
        
        rope = RotaryPositionalEncoding(D_MODEL)
        
        # Create 3D tensors [batch_size, seq_len, d_model]
        q = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL).to(DEVICE)
        k = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL).to(DEVICE)
        
        # Apply RoPE
        q_rot, k_rot = rope(q, k)
        
        # Check shape preservation
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        
        # Check that rotation was applied
        assert not torch.allclose(q_rot, q)
        assert not torch.allclose(k_rot, k)
        
        print("✓ RoPE with 3D input successful")
        print(f"  Input shape: {q.shape}")
        print(f"  Output shape: {q_rot.shape}")
    
    def test_4d_input(self):
        """Test RoPE with 4D input tensors."""
        print("\n" + "="*50)
        print("Testing RoPE with 4D Input...")
        
        rope = RotaryPositionalEncoding(D_MODEL)
        
        # Create 4D tensors [batch_size, num_heads, seq_len, head_dim]
        num_heads = 8
        head_dim = D_MODEL // num_heads
        q = torch.randn(BATCH_SIZE, num_heads, SEQ_LEN, head_dim).to(DEVICE)
        k = torch.randn(BATCH_SIZE, num_heads, SEQ_LEN, head_dim).to(DEVICE)
        
        # Apply RoPE
        q_rot, k_rot = rope(q, k)
        
        # Check shape preservation
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        
        # Check that rotation was applied
        assert not torch.allclose(q_rot, q)
        assert not torch.allclose(k_rot, k)
        
        print("✓ RoPE with 4D input successful")
        print(f"  Input shape: {q.shape}")
        print(f"  Output shape: {q_rot.shape}")
    
    def test_single_input(self):
        """Test RoPE with single input (q only)."""
        print("\n" + "="*50)
        print("Testing RoPE with Single Input...")
        
        rope = RotaryPositionalEncoding(D_MODEL)
        
        # Create single tensor
        q = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL).to(DEVICE)
        
        # Apply RoPE with only q
        q_rot, k_rot = rope(q)  # k should be same as q
        
        # Check that both outputs are identical when only q is provided
        assert torch.allclose(q_rot, k_rot)
        
        print("✓ RoPE with single input successful")
    
    def test_relative_position_preservation(self):
        """Test that RoPE preserves relative positions."""
        print("\n" + "="*50)
        print("Testing RoPE Relative Position Preservation...")
        
        rope = RotaryPositionalEncoding(D_MODEL)
        
        # Create simple embeddings
        x = torch.ones(1, 10, D_MODEL).to(DEVICE)
        
        # Apply RoPE
        x_rot_q, x_rot_k = rope(x, x)
        
        # Compute attention scores between positions
        scores = torch.matmul(x_rot_q, x_rot_k.transpose(-2, -1))
        
        # Check diagonal dominance (self-attention should be strongest)
        diagonal = torch.diagonal(scores[0])
        off_diagonal = scores[0] - torch.diag(diagonal)
        
        assert diagonal.mean() > off_diagonal.abs().mean()
        
        print("✓ Relative positions preserved")
        print(f"  Mean diagonal score: {diagonal.mean():.4f}")
        print(f"  Mean off-diagonal score: {off_diagonal.abs().mean():.4f}")
    
    def test_rotation_properties(self):
        """Test mathematical properties of rotation."""
        print("\n" + "="*50)
        print("Testing RoPE Rotation Properties...")
        
        rope = RotaryPositionalEncoding(D_MODEL)
        
        # Test rotate_half function
        x = torch.randn(2, 10, D_MODEL).to(DEVICE)
        x_rotated = rope.rotate_half(x)
        
        # Check that rotating twice gives negative of original
        x_rotated_twice = rope.rotate_half(x_rotated)
        assert torch.allclose(x_rotated_twice, -x, atol=1e-5)
        
        print("✓ Rotation properties verified")


class TestTransformerEmbedding:
    """Test complete transformer embedding layer."""
    
    def test_sinusoidal_embedding(self):
        """Test transformer embedding with sinusoidal encoding."""
        print("\n" + "="*50)
        print("Testing TransformerEmbedding with Sinusoidal...")
        
        embed = TransformerEmbedding(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            pos_encoding_type='sinusoidal'
        ).to(DEVICE)
        
        tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(DEVICE)
        output = embed(tokens)
        
        assert output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
        print("✓ Sinusoidal transformer embedding works")
    
    def test_learned_embedding(self):
        """Test transformer embedding with learned encoding."""
        print("\n" + "="*50)
        print("Testing TransformerEmbedding with Learned...")
        
        embed = TransformerEmbedding(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            pos_encoding_type='learned'
        ).to(DEVICE)
        
        tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(DEVICE)
        output = embed(tokens)
        
        assert output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
        print("✓ Learned transformer embedding works")
    
    def test_rotary_embedding(self):
        """Test transformer embedding with rotary encoding."""
        print("\n" + "="*50)
        print("Testing TransformerEmbedding with Rotary...")
        
        embed = TransformerEmbedding(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            pos_encoding_type='rotary'
        ).to(DEVICE)
        
        tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(DEVICE)
        output = embed(tokens)
        
        # Note: Rotary encoding is applied later in attention, not here
        assert output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
        assert hasattr(embed, 'rotary_encoding')
        print("✓ Rotary transformer embedding initialized")
    
    def test_factory_function(self):
        """Test the factory function."""
        print("\n" + "="*50)
        print("Testing Factory Function...")
        
        embed = create_embedding_layer(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            pos_encoding_type='sinusoidal'
        )
        
        assert isinstance(embed, TransformerEmbedding)
        print("✓ Factory function works correctly")


def compare_positional_encodings():
    """Compare different positional encoding methods."""
    print("\n" + "="*50)
    print("Comparing Positional Encoding Methods...")
    
    # Initialize encodings
    sinusoidal = PositionalEncoding(D_MODEL, max_seq_length=100)
    learned = LearnedPositionalEncoding(D_MODEL, max_seq_length=100)
    
    # Get encodings
    sin_enc = sinusoidal.pe[0, :100, :].detach().numpy()
    learn_enc = learned.position_embeddings.weight[:100, :].detach().numpy()
    
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
    test_token.test_padding()
    
    # Positional Encoding tests
    test_pos = TestPositionalEncoding()
    test_pos.test_initialization()
    test_pos.test_sinusoidal_properties()
    test_pos.test_relative_positions()
    test_pos.test_visualization()
    
    # Learned Positional tests
    test_learned = TestLearnedPositionalEncoding()
    test_learned.test_initialization()
    test_learned.test_forward_pass()
    test_learned.test_gradient_flow()
    
    # RoPE tests
    test_rope = TestRotaryPositionalEncoding()
    test_rope.test_3d_input()
    test_rope.test_4d_input()
    test_rope.test_single_input()
    test_rope.test_relative_position_preservation()
    test_rope.test_rotation_properties()
    
    # TransformerEmbedding tests
    test_transformer = TestTransformerEmbedding()
    test_transformer.test_sinusoidal_embedding()
    test_transformer.test_learned_embedding()
    test_transformer.test_rotary_embedding()
    test_transformer.test_factory_function()
    
    # Comparison
    compare_positional_encodings()
    
    print("\n" + "="*60)
    print(" ALL EMBEDDING TESTS PASSED!")
    print("="*60)
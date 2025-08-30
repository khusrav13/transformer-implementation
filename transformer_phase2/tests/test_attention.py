"""
Test Suite for Attention Mechanisms
This module tests all attention components and provides visualizations.
"""

import torch
import torch.nn as nn
import numpy as np
import unittest
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from components.attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    SelfAttention,
    CrossAttention,
    create_attention_layer
)
from utils.masking import create_look_ahead_mask, create_padding_mask


class TestAttentionMechanisms(unittest.TestCase):
    """Test cases for attention mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 10
        self.d_model = 512
        self.num_heads = 8
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory for visualizations
        self.viz_dir = Path("test_outputs/attention_visualizations")
        self.viz_dir.mkdir(parents=True, exist_ok=True)
    
    def test_scaled_dot_product_attention(self):
        """Test scaled dot-product attention."""
        print("\n" + "="*50)
        print("Testing Scaled Dot-Product Attention")
        print("="*50)
        
        attention = ScaledDotProductAttention(dropout=0.1)
        attention.eval()  # Set to eval mode for consistent results
        
        # Create sample inputs
        d_k = self.d_model // self.num_heads
        query = torch.randn(self.batch_size, self.num_heads, self.seq_len, d_k)
        key = torch.randn(self.batch_size, self.num_heads, self.seq_len, d_k)
        value = torch.randn(self.batch_size, self.num_heads, self.seq_len, d_k)
        
        # Test without mask
        output, attn_weights = attention(query, key, value)
        
        # Check output shapes
        self.assertEqual(output.shape, value.shape)
        self.assertEqual(attn_weights.shape, 
                        (self.batch_size, self.num_heads, self.seq_len, self.seq_len))
        
        # Check attention weights sum to 1
        weight_sums = attn_weights.sum(dim=-1)
        torch.testing.assert_close(weight_sums, torch.ones_like(weight_sums), rtol=1e-5, atol=1e-8)
        
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Attention weights shape: {attn_weights.shape}")
        print(f"✓ Attention weights sum to 1: {weight_sums[0, 0, :5]}")
        
        # Test with causal mask
        mask = create_look_ahead_mask(self.seq_len, device=query.device)
        output_masked, attn_weights_masked = attention(query, key, value, mask)
        
        # Check that future positions have zero attention
        for i in range(self.seq_len):
            for j in range(i + 1, self.seq_len):
                self.assertTrue(
                    torch.allclose(attn_weights_masked[:, :, i, j], torch.zeros(1)),
                    f"Position {i} should not attend to future position {j}"
                )
        
        print("✓ Causal masking working correctly")
    
    def test_multi_head_attention(self):
        """Test multi-head attention."""
        print("\n" + "="*50)
        print("Testing Multi-Head Attention")
        print("="*50)
        
        mha = MultiHeadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=0.1
        )
        mha.to(self.device)
        mha.eval()
        
        # Create sample inputs
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)
        
        # Test forward pass
        output, attn_weights = mha(x, x, x)
        
        # Check shapes
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(attn_weights.shape,
                        (self.batch_size, self.num_heads, self.seq_len, self.seq_len))
        
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Attention weights shape: {attn_weights.shape}")
        
        # Test gradient flow
        loss = output.mean()
        loss.backward()
        
        # Check that gradients are computed
        for name, param in mha.named_parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.isnan(param.grad).any())
            print(f"✓ Gradient computed for {name}: shape {param.grad.shape}")
    
    def test_self_attention(self):
        """Test self-attention."""
        print("\n" + "="*50)
        print("Testing Self-Attention")
        print("="*50)
        
        self_attn = SelfAttention(
            d_model=self.d_model,
            num_heads=self.num_heads
        )
        self_attn.to(self.device)
        
        # Create input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)
        
        # Test forward pass
        output, attn_weights = self_attn(x)
        
        # Check that it's actually self-attention (Q, K, V from same source)
        self.assertEqual(output.shape, x.shape)
        print(f"✓ Self-attention output shape: {output.shape}")
        
        # Visualize attention pattern
        self._visualize_attention_weights(
            attn_weights[0, 0].detach().cpu().numpy(),
            title="Self-Attention Pattern",
            filename="self_attention_pattern.png"
        )
    
    def test_cross_attention(self):
        """Test cross-attention."""
        print("\n" + "="*50)
        print("Testing Cross-Attention")
        print("="*50)
        
        cross_attn = CrossAttention(
            d_model=self.d_model,
            num_heads=self.num_heads
        )
        cross_attn.to(self.device)
        
        # Create inputs (different sequence lengths for encoder and decoder)
        decoder_len = 8
        encoder_len = 12
        query = torch.randn(self.batch_size, decoder_len, self.d_model, device=self.device)
        encoder_output = torch.randn(self.batch_size, encoder_len, self.d_model, device=self.device)
        
        # Test forward pass
        output, attn_weights = cross_attn(query, encoder_output)
        
        # Check shapes
        self.assertEqual(output.shape, query.shape)
        self.assertEqual(attn_weights.shape,
                        (self.batch_size, self.num_heads, decoder_len, encoder_len))
        
        print(f"✓ Cross-attention output shape: {output.shape}")
        print(f"✓ Cross-attention weights shape: {attn_weights.shape}")
        
        # Visualize cross-attention pattern
        self._visualize_attention_weights(
            attn_weights[0, 0].detach().cpu().numpy(),
            title="Cross-Attention Pattern",
            filename="cross_attention_pattern.png",
            xlabel="Encoder Position",
            ylabel="Decoder Position"
        )
    
    def test_attention_with_padding(self):
        """Test attention with padding mask."""
        print("\n" + "="*50)
        print("Testing Attention with Padding")
        print("="*50)
        
        mha = MultiHeadAttention(self.d_model, self.num_heads)
        
        # Create input with padding (last 3 positions are padding)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Create padding mask
        seq_ids = torch.ones(self.batch_size, self.seq_len, dtype=torch.long)
        seq_ids[:, -3:] = 0  # Mark last 3 as padding
        padding_mask = create_padding_mask(seq_ids, pad_idx=0)
        
        # Apply attention
        output, attn_weights = mha(x, x, x, mask=padding_mask)
        
        # Check that padded positions have near-zero attention
        padded_attention = attn_weights[:, :, :, -3:].mean()
        self.assertLess(padded_attention, 1e-6)
        
        print(f"✓ Padded positions have attention weight: {padded_attention:.6f}")
        print("✓ Padding mask working correctly")
    
    def test_attention_memory_efficiency(self):
        """Test memory efficiency of attention."""
        print("\n" + "="*50)
        print("Testing Attention Memory Efficiency")
        print("="*50)
        
        if not torch.cuda.is_available():
            print("⚠ CUDA not available, skipping memory test")
            return
        
        mha = MultiHeadAttention(self.d_model, self.num_heads).cuda()
        
        # Measure memory usage
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Large batch computation
        large_batch = 16
        x = torch.randn(large_batch, self.seq_len, self.d_model, device='cuda')
        
        initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        output, _ = mha(x, x, x)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        memory_used = peak_memory - initial_memory
        
        print(f"✓ Memory used for batch {large_batch}: {memory_used:.2f} MB")
        print(f"✓ Memory per sample: {memory_used/large_batch:.2f} MB")
    
    def test_attention_factory(self):
        """Test the attention layer factory function."""
        print("\n" + "="*50)
        print("Testing Attention Layer Factory")
        print("="*50)
        
        # Test creating different attention types
        self_attn = create_attention_layer("self", self.d_model, self.num_heads)
        cross_attn = create_attention_layer("cross", self.d_model, self.num_heads)
        multi_attn = create_attention_layer("multi", self.d_model, self.num_heads)
        
        self.assertIsInstance(self_attn, SelfAttention)
        self.assertIsInstance(cross_attn, CrossAttention)
        self.assertIsInstance(multi_attn, MultiHeadAttention)
        
        print("✓ Self-attention layer created successfully")
        print("✓ Cross-attention layer created successfully")
        print("✓ Multi-head attention layer created successfully")
        
        # Test invalid attention type
        with self.assertRaises(ValueError):
            create_attention_layer("invalid_type")
        print("✓ Invalid attention type raises ValueError as expected")
    
    def test_attention_with_different_dimensions(self):
        """Test attention with various dimension configurations."""
        print("\n" + "="*50)
        print("Testing Attention with Different Dimensions")
        print("="*50)
        
        configurations = [
            (256, 4),   # Smaller model
            (512, 8),   # Standard
            (768, 12),  # BERT-base like
            (1024, 16), # Larger model
        ]
        
        for d_model, num_heads in configurations:
            with self.subTest(d_model=d_model, num_heads=num_heads):
                mha = MultiHeadAttention(d_model, num_heads)
                x = torch.randn(self.batch_size, self.seq_len, d_model)
                
                output, attn_weights = mha(x, x, x)
                
                self.assertEqual(output.shape, (self.batch_size, self.seq_len, d_model))
                self.assertEqual(attn_weights.shape,
                               (self.batch_size, num_heads, self.seq_len, self.seq_len))
                
                print(f"✓ d_model={d_model}, heads={num_heads}: shapes correct")
    
    def _visualize_attention_weights(
        self,
        weights: np.ndarray,
        title: str = "Attention Weights",
        filename: str = "attention_weights.png",
        xlabel: str = "Key Position",
        ylabel: str = "Query Position"
    ):
        """Visualize attention weights and save to file."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                weights,
                cmap='Blues',
                cbar=True,
                square=True,
                xticklabels=False,
                yticklabels=False,
                vmin=0,
                vmax=weights.max()
            )
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            
            # Save figure
            save_path = self.viz_dir / filename
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Visualization saved to {save_path}")
            
        except ImportError:
            print("⚠ Matplotlib/Seaborn not available for visualization")


def run_attention_benchmarks():
    """Run performance benchmarks for attention mechanisms."""
    print("\n" + "="*50)
    print("Running Attention Performance Benchmarks")
    print("="*50)
    
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test different configurations
    configs = [
        (32, 128, 512, 8),   # (batch, seq_len, d_model, heads)
        (16, 256, 512, 8),
        (8, 512, 512, 8),
        (4, 1024, 512, 8),
    ]
    
    for batch_size, seq_len, d_model, num_heads in configs:
        mha = MultiHeadAttention(d_model, num_heads).to(device)
        mha.eval()
        
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        # Warmup
        for _ in range(3):
            _ = mha(x, x, x)
        
        # Benchmark
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        num_iterations = 10
        
        for _ in range(num_iterations):
            output, _ = mha(x, x, x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / num_iterations * 1000  # ms
        
        print(f"Config (B={batch_size}, L={seq_len}, D={d_model}, H={num_heads}): "
              f"{avg_time:.2f} ms/forward")


def visualize_attention_patterns():
    """Create various attention pattern visualizations."""
    print("\n" + "="*50)
    print("Creating Attention Pattern Visualizations")
    print("="*50)
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    viz_dir = Path("test_outputs/attention_visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Uniform attention
    uniform = np.ones((10, 10)) / 10
    axes[0, 0].imshow(uniform, cmap='Blues')
    axes[0, 0].set_title("Uniform Attention")
    
    # 2. Diagonal attention (self-focus)
    diagonal = np.eye(10) * 0.8 + np.ones((10, 10)) * 0.02
    axes[0, 1].imshow(diagonal, cmap='Blues')
    axes[0, 1].set_title("Self-Focus Pattern")
    
    # 3. Causal attention
    causal = np.tril(np.ones((10, 10)))
    causal = causal / causal.sum(axis=1, keepdims=True)
    axes[0, 2].imshow(causal, cmap='Blues')
    axes[0, 2].set_title("Causal Attention")
    
    # 4. Local attention (attending to nearby positions)
    local = np.zeros((10, 10))
    for i in range(10):
        for j in range(max(0, i-2), min(10, i+3)):
            local[i, j] = 1
    local = local / local.sum(axis=1, keepdims=True)
    axes[1, 0].imshow(local, cmap='Blues')
    axes[1, 0].set_title("Local Attention")
    
    # 5. Strided attention
    strided = np.zeros((10, 10))
    strided[::2, ::2] = 1
    strided[1::2, 1::2] = 1
    strided = strided / strided.sum(axis=1, keepdims=True)
    axes[1, 1].imshow(strided, cmap='Blues')
    axes[1, 1].set_title("Strided Attention")
    
    # 6. Block attention
    block = np.zeros((10, 10))
    block[:5, :5] = 1
    block[5:, 5:] = 1
    block = block / block.sum(axis=1, keepdims=True)
    axes[1, 2].imshow(block, cmap='Blues')
    axes[1, 2].set_title("Block Attention")
    
    for ax in axes.flat:
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
    
    plt.suptitle("Common Attention Patterns", fontsize=16)
    plt.tight_layout()
    
    save_path = viz_dir / "attention_patterns_overview.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Attention patterns saved to {save_path}")


if __name__ == "__main__":
    # Run unit tests
    print("Starting Attention Module Tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAttentionMechanisms)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run additional benchmarks and visualizations
    if result.wasSuccessful():
        run_attention_benchmarks()
        visualize_attention_patterns()
        
        print("\n" + "="*50)
        print("All Attention Tests Passed Successfully!")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("Some tests failed. Please check the errors above.")
        print("="*50)
"""
Test suite for Transformer Encoder and Decoder Layers
Includes comprehensive tests with visualizations for debugging and understanding.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
import unittest
from typing import Tuple, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.encoder_layer import EncoderLayer, Encoder, EncoderLayerWithCrossAttention
from layers.decoder_layer import DecoderLayer, Decoder, ParallelDecoderLayer

# Create visualization directory
VIZ_DIR = Path(__file__).parent / "layers_visualizations"
VIZ_DIR.mkdir(exist_ok=True)


class TestTransformerLayers(unittest.TestCase):
    """Comprehensive tests for Transformer layers with visualizations"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Model parameters
        self.batch_size = 2
        self.seq_len = 10
        self.d_model = 512
        self.num_heads = 8
        self.d_ff = 2048
        self.dropout = 0.1
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create sample inputs
        self.encoder_input = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.decoder_input = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Create attention masks
        self.self_mask = self._create_causal_mask(self.seq_len)
        self.padding_mask = self._create_padding_mask(self.batch_size, self.seq_len)
    
    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask for decoder self-attention"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    def _create_padding_mask(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """Create padding mask (simulate some padding)"""
        mask = torch.zeros(batch_size, 1, 1, seq_len)
        # Simulate padding in last 2 positions for first batch
        mask[0, :, :, -2:] = float('-inf')
        return mask
    
    def test_encoder_layer_forward(self):
        """Test encoder layer forward pass and visualize attention"""
        print("\n=== Testing Encoder Layer ===")
        
        # Test both normalization types
        for norm_type in ['post', 'pre']:
            encoder_layer = EncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout=0.0,  # Disable dropout for testing
                norm_type=norm_type
            )
            encoder_layer.eval()
            
            # Forward pass with attention weights
            output, attention_weights = encoder_layer(
                self.encoder_input,
                mask=self.padding_mask,
                return_attention=True
            )
            
            # Assertions
            self.assertEqual(output.shape, self.encoder_input.shape)
            self.assertFalse(torch.isnan(output).any())
            self.assertFalse(torch.isinf(output).any())
            
            # Visualize attention for first head
            self._visualize_attention(
                attention_weights[0, 0].detach().numpy(),
                title=f"Encoder Self-Attention ({norm_type}-norm)",
                filename=f"encoder_attention_{norm_type}.png"
            )
            
            print(f"✓ Encoder layer ({norm_type}-norm) - Output shape: {output.shape}")
    
    def test_decoder_layer_forward(self):
        """Test decoder layer forward pass with cross-attention"""
        print("\n=== Testing Decoder Layer ===")
        
        decoder_layer = DecoderLayer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dropout=0.0
        )
        decoder_layer.eval()
        
        # Create encoder output (simulate)
        encoder_output = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        output, cache, attention_weights = decoder_layer(
            self.decoder_input,
            encoder_output=encoder_output,
            self_mask=self.self_mask,
            cross_mask=self.padding_mask,
            return_attention=True
        )
        
        # Assertions
        self.assertEqual(output.shape, self.decoder_input.shape)
        self.assertIn('self', attention_weights)
        self.assertIn('cross', attention_weights)
        
        # Visualize both attention types
        self._visualize_attention(
            attention_weights['self'][0, 0].detach().numpy(),
            title="Decoder Self-Attention (Causal)",
            filename="decoder_self_attention.png"
        )
        
        self._visualize_attention(
            attention_weights['cross'][0, 0].detach().numpy(),
            title="Decoder Cross-Attention",
            filename="decoder_cross_attention.png"
        )
        
        print(f"✓ Decoder layer - Output shape: {output.shape}")
    
    def test_gradient_flow(self):
        """Test gradient flow through layers"""
        print("\n=== Testing Gradient Flow ===")
        
        encoder = Encoder(
            num_layers=6,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff
        )
        
        # Forward pass
        encoder_input = torch.randn(2, 10, self.d_model, requires_grad=True)
        output = encoder(encoder_input)
        
        # Compute dummy loss
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist and are not zero
        self.assertIsNotNone(encoder_input.grad)
        self.assertGreater(encoder_input.grad.abs().mean().item(), 0)
        
        # Visualize gradient magnitudes across layers
        grad_norms = []
        for i, layer in enumerate(encoder.layers):
            layer_grad_norm = 0
            for param in layer.parameters():
                if param.grad is not None:
                    layer_grad_norm += param.grad.norm().item()
            grad_norms.append(layer_grad_norm)
        
        self._plot_gradient_flow(grad_norms, "encoder_gradient_flow.png")
        print("✓ Gradient flow test passed")
    
    def test_attention_pattern_analysis(self):
        """Analyze attention patterns in multi-head attention"""
        print("\n=== Testing Attention Pattern Analysis ===")
        
        encoder_layer = EncoderLayer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dropout=0.0
        )
        encoder_layer.eval()
        
        # Get attention weights
        _, attention_weights = encoder_layer(
            self.encoder_input,
            return_attention=True
        )
        
        # Analyze attention entropy (measure of focus vs. spread)
        attention_probs = attention_weights[0]  # First batch
        entropy = self._compute_attention_entropy(attention_probs)
        
        # Visualize attention patterns for all heads
        self._visualize_multihead_attention(
            attention_weights[0].detach().numpy(),
            filename="multihead_attention_patterns.png"
        )
        
        print(f"✓ Average attention entropy: {entropy:.4f}")
    
    def test_layer_outputs_distribution(self):
        """Test output distributions across layers"""
        print("\n=== Testing Layer Output Distributions ===")
        
        encoder = Encoder(
            num_layers=6,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            norm_type='post'
        )
        encoder.eval()
        
        # Get outputs from all layers
        all_outputs = encoder(self.encoder_input, return_all_layers=True)
        
        # Compute statistics for each layer
        stats = []
        for i, layer_output in enumerate(all_outputs):
            mean = layer_output.mean().item()
            std = layer_output.std().item()
            stats.append((mean, std))
            print(f"  Layer {i}: mean={mean:.4f}, std={std:.4f}")
        
        # Visualize distribution evolution
        self._plot_layer_distributions(all_outputs, "layer_distributions.png")
        
        # Check for vanishing/exploding values
        for i, (mean, std) in enumerate(stats):
            self.assertLess(abs(mean), 10, f"Layer {i} mean too large")
            self.assertGreater(std, 0.1, f"Layer {i} std too small (vanishing)")
            self.assertLess(std, 10, f"Layer {i} std too large (exploding)")
        
        print("✓ Layer distributions are stable")
    
    def test_positional_information_preservation(self):
        """Test if positional information is preserved through layers"""
        print("\n=== Testing Positional Information Preservation ===")
        
        encoder = Encoder(num_layers=6, d_model=self.d_model)
        encoder.eval()
        
        # Create input with clear positional pattern
        position_input = torch.zeros(1, self.seq_len, self.d_model)
        for i in range(self.seq_len):
            position_input[0, i, i % self.d_model] = 1.0
        
        # Pass through encoder
        output = encoder(position_input)
        
        # Compute position-wise similarity matrix
        similarity = torch.cosine_similarity(
            output[0].unsqueeze(0),
            output[0].unsqueeze(1),
            dim=2
        )
        
        self._visualize_similarity_matrix(
            similarity.detach().numpy(),
            "position_similarity.png"
        )
        
        print("✓ Positional information test completed")
    
    def test_parallel_decoder_layer(self):
        """Test parallel decoder layer variant"""
        print("\n=== Testing Parallel Decoder Layer ===")
        
        parallel_decoder = ParallelDecoderLayer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dropout=0.0
        )
        parallel_decoder.eval()
        
        encoder_output = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        output = parallel_decoder(
            self.decoder_input,
            encoder_output=encoder_output,
            self_mask=self.self_mask
        )
        
        self.assertEqual(output.shape, self.decoder_input.shape)
        self.assertFalse(torch.isnan(output).any())
        
        print(f"✓ Parallel decoder - Output shape: {output.shape}")
    
    # === Visualization Helper Methods ===
    
    def _visualize_attention(self, attention: np.ndarray, title: str, filename: str):
        """Visualize attention matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(attention, cmap='Blues', cbar=True, square=True)
        plt.title(title)
        plt.xlabel("Keys")
        plt.ylabel("Queries")
        plt.tight_layout()
        plt.savefig(VIZ_DIR / filename, dpi=100)
        plt.close()
    
    def _visualize_multihead_attention(self, attention: np.ndarray, filename: str):
        """Visualize all attention heads"""
        num_heads = attention.shape[0]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for head in range(min(num_heads, 8)):
            sns.heatmap(attention[head], cmap='Blues', cbar=False, 
                       square=True, ax=axes[head])
            axes[head].set_title(f'Head {head+1}')
            axes[head].set_xlabel("Keys")
            axes[head].set_ylabel("Queries")
        
        plt.suptitle("Multi-Head Attention Patterns")
        plt.tight_layout()
        plt.savefig(VIZ_DIR / filename, dpi=100)
        plt.close()
    
    def _plot_gradient_flow(self, grad_norms: list, filename: str):
        """Plot gradient flow across layers"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(grad_norms)), grad_norms, 'b-o', linewidth=2)
        plt.xlabel("Layer")
        plt.ylabel("Gradient Norm")
        plt.title("Gradient Flow Through Encoder Layers")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(VIZ_DIR / filename, dpi=100)
        plt.close()
    
    def _plot_layer_distributions(self, outputs: list, filename: str):
        """Plot output distributions for each layer"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, output in enumerate(outputs[:6]):
            data = output.flatten().detach().numpy()
            axes[i].hist(data, bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[i].set_title(f'Layer {i} Output Distribution')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.5)
            
            # Add statistics
            mean = data.mean()
            std = data.std()
            axes[i].text(0.7, 0.9, f'μ={mean:.3f}\nσ={std:.3f}', 
                        transform=axes[i].transAxes)
        
        plt.suptitle("Layer Output Distributions")
        plt.tight_layout()
        plt.savefig(VIZ_DIR / filename, dpi=100)
        plt.close()
    
    def _visualize_similarity_matrix(self, similarity: np.ndarray, filename: str):
        """Visualize position similarity matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity, cmap='coolwarm', center=0, cbar=True, square=True)
        plt.title("Position-wise Cosine Similarity")
        plt.xlabel("Position")
        plt.ylabel("Position")
        plt.tight_layout()
        plt.savefig(VIZ_DIR / filename, dpi=100)
        plt.close()
    
    def _compute_attention_entropy(self, attention: torch.Tensor) -> float:
        """Compute entropy of attention distribution"""
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        attention = attention + eps
        entropy = -(attention * torch.log(attention)).sum(dim=-1).mean()
        return entropy.item()


class TestMemoryEfficiency(unittest.TestCase):
    """Test memory efficiency and performance"""
    
    def test_memory_usage(self):
        """Profile memory usage of layers"""
        print("\n=== Testing Memory Efficiency ===")
        
        if not torch.cuda.is_available():
            print("⚠ CUDA not available, skipping GPU memory test")
            return
        
        device = torch.device('cuda')
        
        # Test different sequence lengths
        seq_lengths = [128, 256, 512, 1024]
        memory_usage = []
        
        for seq_len in seq_lengths:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Create model and input
            encoder = Encoder(num_layers=6, d_model=512).to(device)
            input_tensor = torch.randn(1, seq_len, 512).to(device)
            
            # Forward pass
            _ = encoder(input_tensor)
            
            # Record memory
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            memory_usage.append(peak_memory)
            
            print(f"  Seq length {seq_len}: {peak_memory:.2f} MB")
        
        # Plot memory scaling
        plt.figure(figsize=(10, 6))
        plt.plot(seq_lengths, memory_usage, 'b-o', linewidth=2)
        plt.xlabel("Sequence Length")
        plt.ylabel("Peak Memory (MB)")
        plt.title("Memory Usage vs Sequence Length")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "memory_scaling.png", dpi=100)
        plt.close()
        
        print("✓ Memory efficiency test completed")


def run_comprehensive_tests():
    """Run all tests with detailed output"""
    print("="*60)
    print("TRANSFORMER LAYERS COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTransformerLayers))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryEfficiency))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors))/result.testsRun*100:.1f}%")
    print(f"\n✓ Visualizations saved to: {VIZ_DIR}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
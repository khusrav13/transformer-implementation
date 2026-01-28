"""
Test Suite for Feed-Forward Networks
This module tests feed-forward components and provides visualizations.
"""

import torch
import torch.nn as nn
import numpy as np
import unittest
import os
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from components.feedforward import (
    PositionwiseFeedForward,
    GatedFeedForward,
    ExpertFeedForward
)


class TestFeedForwardNetworks(unittest.TestCase):
    """Test cases for feed-forward networks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.seq_len = 20
        self.d_model = 512
        self.d_ff = 2048
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory for visualizations
        self.viz_dir = Path("test_outputs/feedforward_visualizations")
        self.viz_dir.mkdir(parents=True, exist_ok=True)
    
    def test_positionwise_feedforward(self):
        """Test standard position-wise feed-forward network."""
        print("\n" + "="*50)
        print("Testing Position-wise Feed-Forward Network")
        print("="*50)
        
        # Test with ReLU activation
        ffn_relu = PositionwiseFeedForward(
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout=0.1,
            activation='relu'
        )
        ffn_relu.to(self.device)
        
        # Create input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)
        
        # Forward pass
        output = ffn_relu(x)
        
        # Check shape preservation
        self.assertEqual(output.shape, x.shape)
        print(f"✓ ReLU FFN output shape: {output.shape}")
        
        # Test with GELU activation
        ffn_gelu = PositionwiseFeedForward(
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout=0.1,
            activation='gelu'
        )
        ffn_gelu.to(self.device)
        
        output_gelu = ffn_gelu(x)
        self.assertEqual(output_gelu.shape, x.shape)
        print(f"✓ GELU FFN output shape: {output_gelu.shape}")
        
        # Test gradient flow
        loss = output.mean()
        loss.backward()
        
        for name, param in ffn_relu.named_parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.isnan(param.grad).any())
            print(f"✓ Gradient computed for {name}")
    
    def test_gated_feedforward(self):
        """Test gated feed-forward network."""
        print("\n" + "="*50)
        print("Testing Gated Feed-Forward Network")
        print("="*50)
        
        gated_ffn = GatedFeedForward(
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout=0.1
        )
        gated_ffn.to(self.device)
        
        # Create input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)
        
        # Forward pass
        output = gated_ffn(x)
        
        # Check shape
        self.assertEqual(output.shape, x.shape)
        print(f"✓ Gated FFN output shape: {output.shape}")
        
        # Analyze gating behavior
        with torch.no_grad():
            gate_values = torch.sigmoid(gated_ffn.linear_gate(x))
            mean_gate = gate_values.mean().item()
            std_gate = gate_values.std().item()
            
        print(f"✓ Gate statistics - Mean: {mean_gate:.3f}, Std: {std_gate:.3f}")
        
        # Visualize gate values
        self._visualize_gate_values(
            gate_values[0, :, :100].cpu().numpy(),
            title="Gated FFN - Gate Activation Values",
            filename="gated_ffn_gates.png"
        )
    
    def test_expert_feedforward(self):
        """Test mixture of experts feed-forward network."""
        print("\n" + "="*50)
        print("Testing Expert Feed-Forward Network (MoE)")
        print("="*50)
        
        num_experts = 4
        expert_ffn = ExpertFeedForward(
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_experts=num_experts,
            dropout=0.1
        )
        expert_ffn.to(self.device)
        
        # Create input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)
        
        # Forward pass with different number of experts per token
        for k in [1, 2, 3]:
            output = expert_ffn(x, num_experts_per_token=k)
            self.assertEqual(output.shape, x.shape)
            print(f"✓ MoE FFN with top-{k} experts, output shape: {output.shape}")
        
        # Analyze expert selection
        with torch.no_grad():
            gate_scores = expert_ffn.gate(x)
            expert_probs = torch.softmax(gate_scores, dim=-1)
            
            # Check load balancing across experts
            avg_expert_probs = expert_probs.mean(dim=(0, 1))
            
        print(f"✓ Expert load distribution: {avg_expert_probs.cpu().numpy()}")
        
        # Visualize expert selection
        self._visualize_expert_selection(
            expert_probs[0].cpu().numpy(),
            title="Expert Selection Patterns",
            filename="expert_selection.png"
        )
    
    def test_feedforward_capacity(self):
        """Test the capacity and expressiveness of feed-forward networks."""
        print("\n" + "="*50)
        print("Testing Feed-Forward Network Capacity")
        print("="*50)
        
        # Create networks with different capacities
        capacities = [512, 1024, 2048, 4096]
        
        for d_ff in capacities:
            ffn = PositionwiseFeedForward(
                d_model=self.d_model,
                d_ff=d_ff,
                dropout=0.0  # No dropout for capacity test
            )
            
            # Count parameters
            num_params = sum(p.numel() for p in ffn.parameters())
            
            # Test on random input
            x = torch.randn(1, 1, self.d_model)
            with torch.no_grad():
                output = ffn(x)
            
            print(f"✓ FFN with d_ff={d_ff}: {num_params:,} parameters")
    
    def test_feedforward_activations(self):
        """Test and visualize different activation patterns."""
        print("\n" + "="*50)
        print("Testing Activation Functions")
        print("="*50)
        
        # Create input with specific patterns
        x = torch.randn(1, 100, self.d_model)
        
        activations = ['relu', 'gelu']
        activation_outputs = {}
        
        for activation in activations:
            ffn = PositionwiseFeedForward(
                d_model=self.d_model,
                d_ff=self.d_ff,
                dropout=0.0,
                activation=activation
            )
            
            with torch.no_grad():
                # Get intermediate activations
                hidden = ffn.linear1(x)
                if activation == 'relu':
                    activated = torch.relu(hidden)
                else:  # gelu
                    activated = torch.nn.functional.gelu(hidden)
                
                activation_outputs[activation] = activated
                
                # Compute statistics
                sparsity = (activated == 0).float().mean().item()
                mean_activation = activated.mean().item()
                
                print(f"✓ {activation.upper()} - Sparsity: {sparsity:.3f}, "
                      f"Mean activation: {mean_activation:.3f}")
        
        # Visualize activation distributions
        self._visualize_activation_distributions(
            activation_outputs,
            filename="activation_distributions.png"
        )
    
    def test_feedforward_memory_efficiency(self):
        """Test memory efficiency of different FFN variants."""
        print("\n" + "="*50)
        print("Testing Memory Efficiency")
        print("="*50)
        
        if not torch.cuda.is_available():
            print("⚠ CUDA not available, skipping memory test")
            return
        
        configs = [
            ("Standard", PositionwiseFeedForward),
            ("Gated", GatedFeedForward),
        ]
        
        for name, ffn_class in configs:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            ffn = ffn_class(self.d_model, self.d_ff).cuda()
            x = torch.randn(8, 128, self.d_model, device='cuda')
            
            initial_memory = torch.cuda.memory_allocated() / 1024**2
            
            output = ffn(x)
            output.sum().backward()  # Include backward pass
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            memory_used = peak_memory - initial_memory
            
            print(f"✓ {name} FFN - Memory used: {memory_used:.2f} MB")
    
    def test_feedforward_speed_benchmark(self):
        """Benchmark speed of different FFN implementations."""
        print("\n" + "="*50)
        print("Running Speed Benchmarks")
        print("="*50)
        
        device = self.device
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, device=device)
        
        configs = [
            ("Standard ReLU", PositionwiseFeedForward, {'activation': 'relu'}),
            ("Standard GELU", PositionwiseFeedForward, {'activation': 'gelu'}),
            ("Gated", GatedFeedForward, {}),
        ]
        
        for name, ffn_class, kwargs in configs:
            ffn = ffn_class(self.d_model, self.d_ff, **kwargs).to(device)
            ffn.eval()
            
            # Warmup
            for _ in range(3):
                _ = ffn(x)
            
            # Benchmark
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            num_iterations = 100
            
            for _ in range(num_iterations):
                output = ffn(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            
            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / num_iterations * 1000
            
            print(f"✓ {name}: {avg_time:.2f} ms/forward pass")
    
    def _visualize_gate_values(self, gates, title, filename):
        """Visualize gating values."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            plt.imshow(gates.T, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
            plt.colorbar(label='Gate Value')
            plt.xlabel('Sequence Position')
            plt.ylabel('Hidden Dimension')
            plt.title(title)
            
            save_path = self.viz_dir / filename
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Visualization saved to {save_path}")
            
        except ImportError:
            print("⚠ Matplotlib not available for visualization")
    
    def _visualize_expert_selection(self, expert_probs, title, filename):
        """Visualize expert selection probabilities."""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Heatmap of expert probabilities
            im = ax1.imshow(expert_probs.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
            ax1.set_xlabel('Sequence Position')
            ax1.set_ylabel('Expert Index')
            ax1.set_title('Expert Selection Probabilities')
            plt.colorbar(im, ax=ax1)
            
            # Bar plot of average expert usage
            avg_usage = expert_probs.mean(axis=0)
            ax2.bar(range(len(avg_usage)), avg_usage)
            ax2.set_xlabel('Expert Index')
            ax2.set_ylabel('Average Usage')
            ax2.set_title('Expert Load Balancing')
            ax2.set_ylim([0, 1])
            
            plt.suptitle(title)
            
            save_path = self.viz_dir / filename
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Visualization saved to {save_path}")
            
        except ImportError:
            print("⚠ Matplotlib not available for visualization")
    
    def _visualize_activation_distributions(self, activation_outputs, filename):
        """Visualize activation distributions for different functions."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, len(activation_outputs), figsize=(12, 4))
            
            for idx, (name, activations) in enumerate(activation_outputs.items()):
                ax = axes[idx] if len(activation_outputs) > 1 else axes
                
                # Flatten and sample activations
                flat_activations = activations.flatten().numpy()
                sample_size = min(10000, len(flat_activations))
                sampled = np.random.choice(flat_activations, sample_size, replace=False)
                
                ax.hist(sampled, bins=50, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Activation Value')
                ax.set_ylabel('Count')
                ax.set_title(f'{name.upper()} Activation Distribution')
                ax.grid(True, alpha=0.3)
            
            plt.suptitle('Activation Function Distributions')
            plt.tight_layout()
            
            save_path = self.viz_dir / filename
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Visualization saved to {save_path}")
            
        except ImportError:
            print("⚠ Matplotlib not available for visualization")


def visualize_ffn_architecture():
    """Create architectural diagram of FFN."""
    print("\n" + "="*50)
    print("Creating FFN Architecture Visualization")
    print("="*50)
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        viz_dir = Path("test_outputs/feedforward_visualizations")
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Define layer positions
        layers = [
            ("Input\n[batch, seq, 512]", 1, 0.5, 'lightblue'),
            ("Linear 1\n512 → 2048", 3, 0.5, 'lightgreen'),
            ("Activation\n(ReLU/GELU)", 5, 0.5, 'yellow'),
            ("Dropout", 7, 0.5, 'pink'),
            ("Linear 2\n2048 → 512", 9, 0.5, 'lightgreen'),
            ("Dropout", 11, 0.5, 'pink'),
            ("Output\n[batch, seq, 512]", 13, 0.5, 'lightblue'),
        ]
        
        # Draw layers
        for name, x, width, color in layers:
            rect = patches.Rectangle((x-width/2, 2), width, 2, 
                                    linewidth=2, edgecolor='black', 
                                    facecolor=color)
            ax.add_patch(rect)
            ax.text(x, 3, name, ha='center', va='center', fontsize=10)
        
        # Draw connections
        for i in range(len(layers) - 1):
            x1 = layers[i][1] + layers[i][2]/2
            x2 = layers[i+1][1] - layers[i+1][2]/2
            ax.arrow(x1, 3, x2-x1-0.1, 0, head_width=0.2, 
                    head_length=0.1, fc='black', ec='black')
        
        # Add residual connection
        ax.annotate('', xy=(13, 5.5), xytext=(1, 5.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                                 color='red', linewidth=2))
        ax.text(7, 6, 'Residual Connection', ha='center', color='red')
        
        ax.set_xlim(0, 14)
        ax.set_ylim(1, 7)
        ax.axis('off')
        ax.set_title('Position-wise Feed-Forward Network Architecture', fontsize=14)
        
        save_path = viz_dir / "ffn_architecture.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Architecture diagram saved to {save_path}")
        
    except ImportError:
        print("⚠ Matplotlib not available for visualization")


if __name__ == "__main__":
    # Run unit tests
    print("Starting Feed-Forward Network Tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFeedForwardNetworks)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Create architecture visualization
    if result.wasSuccessful():
        visualize_ffn_architecture()
        
        print("\n" + "="*50)
        print("All Feed-Forward Tests Passed Successfully!")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("Some tests failed. Please check the errors above.")
        print("="*50)
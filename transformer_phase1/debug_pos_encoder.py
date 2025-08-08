"""
Visualization Tools for Positional Encoding Analysis
Creates comprehensive visualizations to understand positional encoding behavior.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
from datetime import datetime
from typing import Optional, Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    RotaryPositionalEncoding,
    RelativePositionalEncoding
)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PositionalEncodingVisualizer:
    """Comprehensive visualization tools for positional encodings."""
    
    def __init__(self, output_dir: str = "visualizations/svg"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Saving positional encoding visualizations to: {self.output_dir}")
    
    def visualize_sinusoidal_pattern(
        self,
        d_model: int = 128,
        max_len: int = 100,
        save_name: str = "sinusoidal_pattern"
    ):
        """
        Visualize the sinusoidal positional encoding pattern.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length to visualize
            save_name: Base name for saved file
        """
        pe = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)
        encoding = pe.get_encoding(max_len)[0].numpy()
        
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Full encoding matrix heatmap
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(encoding.T, aspect='auto', cmap='RdBu_r', 
                        vmin=-1, vmax=1, interpolation='nearest')
        ax1.set_xlabel('Position', fontsize=10)
        ax1.set_ylabel('Dimension', fontsize=10)
        ax1.set_title('Sinusoidal Positional Encoding Matrix', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # 2. First few dimensions over positions
        ax2 = plt.subplot(2, 3, 2)
        dims_to_plot = [0, 1, 2, 3, 10, 20]
        for i, dim in enumerate(dims_to_plot):
            ax2.plot(encoding[:, dim], label=f'Dim {dim}', alpha=0.8)
        ax2.set_xlabel('Position', fontsize=10)
        ax2.set_ylabel('Encoding Value', fontsize=10)
        ax2.set_title('Encoding Values Across Positions', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Frequency analysis
        ax3 = plt.subplot(2, 3, 3)
        # Compute frequency spectrum for different dimensions
        for dim in [0, d_model//4, d_model//2, 3*d_model//4]:
            signal = encoding[:, dim]
            fft = np.fft.fft(signal)
            freq = np.fft.fftfreq(len(signal))
            ax3.plot(freq[:len(freq)//2], np.abs(fft)[:len(freq)//2], 
                    label=f'Dim {dim}', alpha=0.7)
        ax3.set_xlabel('Frequency', fontsize=10)
        ax3.set_ylabel('Magnitude', fontsize=10)
        ax3.set_title('Frequency Spectrum of Different Dimensions', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Position similarity matrix
        ax4 = plt.subplot(2, 3, 4)
        # Compute cosine similarity between all position pairs
        similarity_matrix = np.zeros((max_len, max_len))
        for i in range(max_len):
            for j in range(max_len):
                similarity_matrix[i, j] = np.dot(encoding[i], encoding[j]) / (
                    np.linalg.norm(encoding[i]) * np.linalg.norm(encoding[j])
                )
        im4 = ax4.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax4.set_xlabel('Position', fontsize=10)
        ax4.set_ylabel('Position', fontsize=10)
        ax4.set_title('Position Similarity Matrix', fontsize=12, fontweight='bold')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        
        # 5. Relative position encoding
        ax5 = plt.subplot(2, 3, 5)
        # Show how dot product changes with distance
        reference_pos = max_len // 2
        similarities = []
        for i in range(max_len):
            sim = np.dot(encoding[reference_pos], encoding[i]) / (
                np.linalg.norm(encoding[reference_pos]) * np.linalg.norm(encoding[i])
            )
            similarities.append(sim)
        distances = np.abs(np.arange(max_len) - reference_pos)
        ax5.scatter(distances, similarities, alpha=0.6, s=20)
        ax5.set_xlabel('Distance from Reference Position', fontsize=10)
        ax5.set_ylabel('Cosine Similarity', fontsize=10)
        ax5.set_title(f'Similarity vs Distance (Ref Pos={reference_pos})', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. 3D visualization of first 3 dimensions
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        positions = np.arange(min(50, max_len))
        ax6.plot3D(encoding[positions, 0], 
                  encoding[positions, 1], 
                  encoding[positions, 2], 
                  'b-', alpha=0.7, linewidth=2)
        ax6.scatter(encoding[positions, 0], 
                   encoding[positions, 1], 
                   encoding[positions, 2], 
                   c=positions, cmap='viridis', s=30)
        ax6.set_xlabel('Dim 0', fontsize=9)
        ax6.set_ylabel('Dim 1', fontsize=9)
        ax6.set_zlabel('Dim 2', fontsize=9)
        ax6.set_title('3D Trajectory of Positions', fontsize=12, fontweight='bold')
        
        plt.suptitle('Sinusoidal Positional Encoding Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        filepath = os.path.join(self.output_dir, f"{save_name}_{self.timestamp}.svg")
        plt.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    
    def compare_encoding_types(
        self,
        d_model: int = 64,
        seq_len: int = 50,
        save_name: str = "encoding_comparison"
    ):
        """
        Compare different positional encoding types.
        
        Args:
            d_model: Model dimension
            seq_len: Sequence length
            save_name: Base name for saved file
        """
        # Create encodings
        sin_pe = SinusoidalPositionalEncoding(d_model, dropout=0.0)
        learned_pe = LearnedPositionalEncoding(d_model, dropout=0.0)
        
        # Get encodings
        sin_encoding = sin_pe.get_encoding(seq_len)[0].detach().numpy()
        learned_encoding = learned_pe.get_encoding(seq_len)[0].detach().numpy()
        
        fig = plt.figure(figsize=(18, 10))
        
        # Row 1: Encoding matrices
        ax1 = plt.subplot(2, 4, 1)
        im1 = ax1.imshow(sin_encoding.T[:32], aspect='auto', cmap='RdBu_r', 
                        vmin=-1, vmax=1)
        ax1.set_title('Sinusoidal Encoding', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Position', fontsize=9)
        ax1.set_ylabel('Dimension', fontsize=9)
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        ax2 = plt.subplot(2, 4, 2)
        im2 = ax2.imshow(learned_encoding.T[:32], aspect='auto', cmap='RdBu_r')
        ax2.set_title('Learned Encoding', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Position', fontsize=9)
        ax2.set_ylabel('Dimension', fontsize=9)
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # Row 1: Position similarity matrices
        ax3 = plt.subplot(2, 4, 3)
        sin_sim = self._compute_similarity_matrix(sin_encoding)
        im3 = ax3.imshow(sin_sim, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_title('Sinusoidal Similarity', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Position', fontsize=9)
        ax3.set_ylabel('Position', fontsize=9)
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        ax4 = plt.subplot(2, 4, 4)
        learned_sim = self._compute_similarity_matrix(learned_encoding)
        im4 = ax4.imshow(learned_sim, cmap='coolwarm', vmin=-1, vmax=1)
        ax4.set_title('Learned Similarity', fontsize=11, fontweight='bold')
        ax4.set_xlabel('Position', fontsize=9)
        ax4.set_ylabel('Position', fontsize=9)
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        
        # Row 2: Statistical analysis
        ax5 = plt.subplot(2, 4, 5)
        sin_norms = np.linalg.norm(sin_encoding, axis=1)
        learned_norms = np.linalg.norm(learned_encoding, axis=1)
        ax5.plot(sin_norms, label='Sinusoidal', alpha=0.7, linewidth=2)
        ax5.plot(learned_norms, label='Learned', alpha=0.7, linewidth=2)
        ax5.set_xlabel('Position', fontsize=9)
        ax5.set_ylabel('L2 Norm', fontsize=9)
        ax5.set_title('Encoding Norms', fontsize=11, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        ax6 = plt.subplot(2, 4, 6)
        # Compute entropy of attention-like distribution
        def compute_entropy(encoding):
            # Treat each position as a distribution over dimensions
            probs = torch.softmax(torch.tensor(encoding), dim=1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
            return entropy.numpy()
        
        sin_entropy = compute_entropy(sin_encoding)
        learned_entropy = compute_entropy(learned_encoding)
        ax6.plot(sin_entropy, label='Sinusoidal', alpha=0.7, linewidth=2)
        ax6.plot(learned_entropy, label='Learned', alpha=0.7, linewidth=2)
        ax6.set_xlabel('Position', fontsize=9)
        ax6.set_ylabel('Entropy', fontsize=9)
        ax6.set_title('Encoding Entropy', fontsize=11, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Row 2: Distance preservation
        ax7 = plt.subplot(2, 4, 7)
        self._plot_distance_preservation(sin_encoding, ax7, 'Sinusoidal')
        
        ax8 = plt.subplot(2, 4, 8)
        self._plot_distance_preservation(learned_encoding, ax8, 'Learned')
        
        plt.suptitle('Positional Encoding Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        filepath = os.path.join(self.output_dir, f"{save_name}_{self.timestamp}.svg")
        plt.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    
    def visualize_rope_mechanics(
        self,
        d_model: int = 64,
        seq_len: int = 20,
        save_name: str = "rope_mechanics"
    ):
        """
        Visualize how Rotary Position Encoding works.
        
        Args:
            d_model: Model dimension
            seq_len: Sequence length
            save_name: Base name for saved file
        """
        rope = RotaryPositionalEncoding(d_model)
        
        # Create test queries and keys
        batch_size, num_heads = 1, 1
        q = torch.randn(batch_size, num_heads, seq_len, d_model)
        k = q.clone()  # Same content for clarity
        
        # Apply RoPE
        q_rot, k_rot = rope(q, k)
        
        # Extract for visualization
        q_orig = q[0, 0].numpy()
        q_rotated = q_rot[0, 0].detach().numpy()
        
        fig = plt.figure(figsize=(18, 10))
        
        # 1. Original vs Rotated vectors
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(q_orig.T, aspect='auto', cmap='RdBu_r')
        ax1.set_title('Original Query Vectors', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Position', fontsize=9)
        ax1.set_ylabel('Dimension', fontsize=9)
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        ax2 = plt.subplot(2, 3, 2)
        im2 = ax2.imshow(q_rotated.T, aspect='auto', cmap='RdBu_r')
        ax2.set_title('RoPE Rotated Query Vectors', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Position', fontsize=9)
        ax2.set_ylabel('Dimension', fontsize=9)
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # 3. Difference
        ax3 = plt.subplot(2, 3, 3)
        diff = q_rotated - q_orig
        im3 = ax3.imshow(diff.T, aspect='auto', cmap='RdBu_r')
        ax3.set_title('Rotation Difference', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Position', fontsize=9)
        ax3.set_ylabel('Dimension', fontsize=9)
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # 4. Norm preservation
        ax4 = plt.subplot(2, 3, 4)
        orig_norms = np.linalg.norm(q_orig, axis=1)
        rot_norms = np.linalg.norm(q_rotated, axis=1)
        positions = np.arange(seq_len)
        width = 0.35
        ax4.bar(positions - width/2, orig_norms, width, label='Original', alpha=0.7)
        ax4.bar(positions + width/2, rot_norms, width, label='Rotated', alpha=0.7)
        ax4.set_xlabel('Position', fontsize=9)
        ax4.set_ylabel('L2 Norm', fontsize=9)
        ax4.set_title('Norm Preservation Check', fontsize=11, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Attention scores
        ax5 = plt.subplot(2, 3, 5)
        # Compute attention scores with and without RoPE
        scores_orig = np.matmul(q_orig, q_orig.T)
        scores_rot = np.matmul(q_rotated, q_rotated.T)
        
        # Plot difference in attention patterns
        im5 = ax5.imshow(scores_rot - scores_orig, cmap='RdBu_r')
        ax5.set_title('Attention Score Change from RoPE', fontsize=11, fontweight='bold')
        ax5.set_xlabel('Key Position', fontsize=9)
        ax5.set_ylabel('Query Position', fontsize=9)
        plt.colorbar(im5, ax=ax5, fraction=0.046)
        
        # 6. Rotation visualization in 2D
        ax6 = plt.subplot(2, 3, 6)
        # Visualize rotation for first two dimensions
        for i in range(min(10, seq_len)):
            # Original vector (first 2 dims)
            orig = q_orig[i, :2]
            rot = q_rotated[i, :2]
            
            # Plot vectors
            ax6.arrow(0, 0, orig[0], orig[1], head_width=0.05, 
                     head_length=0.05, fc='blue', ec='blue', alpha=0.5)
            ax6.arrow(0, 0, rot[0], rot[1], head_width=0.05, 
                     head_length=0.05, fc='red', ec='red', alpha=0.5)
            
            # Connect them
            ax6.plot([orig[0], rot[0]], [orig[1], rot[1]], 'k--', alpha=0.3)
        
        ax6.set_xlim([-2, 2])
        ax6.set_ylim([-2, 2])
        ax6.set_xlabel('Dimension 0', fontsize=9)
        ax6.set_ylabel('Dimension 1', fontsize=9)
        ax6.set_title('2D Rotation Visualization', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend(['Original', 'Rotated'], loc='upper right')
        
        plt.suptitle('Rotary Position Encoding (RoPE) Mechanics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        filepath = os.path.join(self.output_dir, f"{save_name}_{self.timestamp}.svg")
        plt.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    
    def analyze_encoding_properties(
        self,
        d_model: int = 128,
        max_len: int = 100,
        save_name: str = "encoding_properties"
    ):
        """
        Analyze mathematical properties of positional encodings.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            save_name: Base name for saved file
        """
        # Create encodings
        sin_pe = SinusoidalPositionalEncoding(d_model, dropout=0.0)
        learned_pe = LearnedPositionalEncoding(d_model, dropout=0.0)
        
        sin_enc = sin_pe.get_encoding(max_len)[0].numpy()
        learned_enc = learned_pe.get_encoding(max_len)[0].detach().numpy()
        
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Orthogonality analysis
        ax1 = plt.subplot(3, 3, 1)
        # Compute orthogonality between positions
        sin_orthogonality = []
        learned_orthogonality = []
        for i in range(max_len - 1):
            sin_orth = np.dot(sin_enc[i], sin_enc[i+1]) / (
                np.linalg.norm(sin_enc[i]) * np.linalg.norm(sin_enc[i+1])
            )
            learned_orth = np.dot(learned_enc[i], learned_enc[i+1]) / (
                np.linalg.norm(learned_enc[i]) * np.linalg.norm(learned_enc[i+1])
            )
            sin_orthogonality.append(sin_orth)
            learned_orthogonality.append(learned_orth)
        
        ax1.plot(sin_orthogonality, label='Sinusoidal', alpha=0.7)
        ax1.plot(learned_orthogonality, label='Learned', alpha=0.7)
        ax1.set_xlabel('Position Pair (i, i+1)', fontsize=9)
        ax1.set_ylabel('Cosine Similarity', fontsize=9)
        ax1.set_title('Adjacent Position Orthogonality', fontsize=11, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Eigenvalue spectrum
        ax2 = plt.subplot(3, 3, 2)
        sin_cov = np.cov(sin_enc.T)
        learned_cov = np.cov(learned_enc.T)
        sin_eigenvals = np.linalg.eigvalsh(sin_cov)[::-1]
        learned_eigenvals = np.linalg.eigvalsh(learned_cov)[::-1]
        
        ax2.semilogy(sin_eigenvals[:50], label='Sinusoidal', alpha=0.7)
        ax2.semilogy(learned_eigenvals[:50], label='Learned', alpha=0.7)
        ax2.set_xlabel('Eigenvalue Index', fontsize=9)
        ax2.set_ylabel('Eigenvalue (log scale)', fontsize=9)
        ax2.set_title('Eigenvalue Spectrum', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Rank analysis
        ax3 = plt.subplot(3, 3, 3)
        sin_rank = np.linalg.matrix_rank(sin_enc)
        learned_rank = np.linalg.matrix_rank(learned_enc)
        ax3.bar(['Sinusoidal', 'Learned'], [sin_rank, learned_rank], 
               color=['blue', 'orange'], alpha=0.7)
        ax3.set_ylabel('Rank', fontsize=9)
        ax3.set_title(f'Encoding Matrix Rank (max={min(max_len, d_model)})', 
                     fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Distance preservation
        ax4 = plt.subplot(3, 3, 4)
        # Check if Euclidean distances are preserved
        original_distances = []
        sin_distances = []
        learned_distances = []
        for i in range(0, max_len, 5):
            for j in range(i+1, min(i+20, max_len), 5):
                original_distances.append(abs(j - i))
                sin_distances.append(np.linalg.norm(sin_enc[j] - sin_enc[i]))
                learned_distances.append(np.linalg.norm(learned_enc[j] - learned_enc[i]))
        
        ax4.scatter(original_distances, sin_distances, alpha=0.5, s=20, label='Sinusoidal')
        ax4.scatter(original_distances, learned_distances, alpha=0.5, s=20, label='Learned')
        ax4.set_xlabel('Original Distance', fontsize=9)
        ax4.set_ylabel('Encoding Distance', fontsize=9)
        ax4.set_title('Distance Preservation', fontsize=11, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Variance per dimension
        ax5 = plt.subplot(3, 3, 5)
        sin_var = np.var(sin_enc, axis=0)
        learned_var = np.var(learned_enc, axis=0)
        ax5.plot(sin_var, label='Sinusoidal', alpha=0.7)
        ax5.plot(learned_var, label='Learned', alpha=0.7)
        ax5.set_xlabel('Dimension', fontsize=9)
        ax5.set_ylabel('Variance', fontsize=9)
        ax5.set_title('Variance per Dimension', fontsize=11, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Mutual information proxy
        ax6 = plt.subplot(3, 3, 6)
        # Use correlation as proxy for mutual information
        sin_corr = np.corrcoef(sin_enc.T)
        learned_corr = np.corrcoef(learned_enc.T)
        
        ax6.hist(sin_corr.flatten(), bins=50, alpha=0.5, label='Sinusoidal', density=True)
        ax6.hist(learned_corr.flatten(), bins=50, alpha=0.5, label='Learned', density=True)
        ax6.set_xlabel('Correlation Coefficient', fontsize=9)
        ax6.set_ylabel('Density', fontsize=9)
        ax6.set_title('Dimension Correlation Distribution', fontsize=11, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Isotropy measure
        ax7 = plt.subplot(3, 3, 7)
        # Measure isotropy: variance of norms
        sin_norms = np.linalg.norm(sin_enc, axis=1)
        learned_norms = np.linalg.norm(learned_enc, axis=1)
        
        ax7.boxplot([sin_norms, learned_norms], labels=['Sinusoidal', 'Learned'])
        ax7.set_ylabel('L2 Norm', fontsize=9)
        ax7.set_title('Norm Distribution (Isotropy)', fontsize=11, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # 8. Effective dimensionality
        ax8 = plt.subplot(3, 3, 8)
        # Compute effective dimensionality using eigenvalues
        def effective_dim(eigenvals):
            eigenvals = eigenvals[eigenvals > 1e-10]  # Remove near-zero
            normalized = eigenvals / eigenvals.sum()
            entropy = -np.sum(normalized * np.log(normalized + 1e-10))
            return np.exp(entropy)
        
        sin_eff_dim = effective_dim(sin_eigenvals)
        learned_eff_dim = effective_dim(learned_eigenvals)
        
        ax8.bar(['Sinusoidal', 'Learned'], [sin_eff_dim, learned_eff_dim],
               color=['blue', 'orange'], alpha=0.7)
        ax8.set_ylabel('Effective Dimensionality', fontsize=9)
        ax8.set_title('Effective Dimensionality', fontsize=11, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        
        # 9. Summary statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        summary_text = f"""
        Encoding Properties Summary:
        
        Sinusoidal:
        • Rank: {sin_rank}/{min(max_len, d_model)}
        • Mean Norm: {np.mean(sin_norms):.3f} ± {np.std(sin_norms):.3f}
        • Effective Dim: {sin_eff_dim:.1f}
        • Adjacent Similarity: {np.mean(sin_orthogonality):.3f}
        
        Learned:
        • Rank: {learned_rank}/{min(max_len, d_model)}
        • Mean Norm: {np.mean(learned_norms):.3f} ± {np.std(learned_norms):.3f}
        • Effective Dim: {learned_eff_dim:.1f}
        • Adjacent Similarity: {np.mean(learned_orthogonality):.3f}
        
        Key Differences:
        • Sinusoidal is deterministic
        • Learned can adapt to task
        • Sinusoidal has fixed frequencies
        • Learned has trainable parameters
        """
        
        ax9.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Positional Encoding Mathematical Properties', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        filepath = os.path.join(self.output_dir, f"{save_name}_{self.timestamp}.svg")
        plt.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    
    def _compute_similarity_matrix(self, encoding: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix for positions."""
        n = len(encoding)
        similarity = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                similarity[i, j] = np.dot(encoding[i], encoding[j]) / (
                    np.linalg.norm(encoding[i]) * np.linalg.norm(encoding[j]) + 1e-10
                )
        return similarity
    
    def _plot_distance_preservation(self, encoding: np.ndarray, ax, title: str):
        """Plot how well encoding preserves distances."""
        n = len(encoding)
        original_dists = []
        encoding_dists = []
        
        for i in range(0, n, 5):
            for j in range(i+1, min(i+20, n), 5):
                original_dists.append(abs(j - i))
                encoding_dists.append(np.linalg.norm(encoding[j] - encoding[i]))
        
        ax.scatter(original_dists, encoding_dists, alpha=0.6, s=20)
        ax.set_xlabel('Position Distance', fontsize=9)
        ax.set_ylabel('Encoding Distance', fontsize=9)
        ax.set_title(f'{title} Distance Preservation', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(original_dists, encoding_dists, 1)
        p = np.poly1d(z)
        ax.plot(sorted(original_dists), p(sorted(original_dists)), 
               "r--", alpha=0.5, label=f'Trend: {z[0]:.3f}x + {z[1]:.3f}')
        ax.legend(fontsize=8)


def run_all_visualizations():
    """Run comprehensive positional encoding visualizations."""
    print("Positional Encoding Visualization Suite")
    print("="*60)
    
    visualizer = PositionalEncodingVisualizer()
    
    print("\n1. Visualizing Sinusoidal Patterns...")
    visualizer.visualize_sinusoidal_pattern(d_model=128, max_len=100)
    print("   ✓ Sinusoidal pattern visualization complete")
    
    print("\n2. Comparing Encoding Types...")
    visualizer.compare_encoding_types(d_model=64, seq_len=50)
    print("   ✓ Encoding comparison complete")
    
    print("\n3. Visualizing RoPE Mechanics...")
    visualizer.visualize_rope_mechanics(d_model=64, seq_len=20)
    print("   ✓ RoPE mechanics visualization complete")
    
    print("\n4. Analyzing Mathematical Properties...")
    visualizer.analyze_encoding_properties(d_model=128, max_len=100)
    print("   ✓ Properties analysis complete")
    
    print("\n" + "="*60)
    print(" All positional encoding visualizations complete!")
    print(f" Check '{visualizer.output_dir}' for SVG files")
    print("="*60)


if __name__ == "__main__":
    run_all_visualizations()
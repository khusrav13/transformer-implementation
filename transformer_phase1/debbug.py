"""
Debug script to visualize attention patterns.
This version saves all visualizations to files instead of displaying them.
"""

import torch
import math
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files
import matplotlib.pyplot as plt
from src.attention import scaled_dot_product_attention
import os
from datetime import datetime

# Global variable for visualization directory (will be set in main)
viz_dir = None

def visualize_attention():
    """Create and visualize different attention patterns"""
    global viz_dir
    """Create and visualize different attention patterns"""
    
    # Test 1: Simple pattern to understand attention
    print("\n" + "="*60)
    print("TEST 1: SIMPLE ATTENTION PATTERN")
    print("="*60)
    
    seq_len = 4
    d_k = 2
    
    # Create queries that have clear patterns
    q = torch.tensor([[1.0, 0.0],   # Query 1: attends to first dimension
                      [0.0, 1.0],   # Query 2: attends to second dimension  
                      [1.0, 0.0],   # Query 3: like Query 1
                      [0.5, 0.5]])  # Query 4: attends to both
    
    k = q.clone()  # Same as queries
    v = torch.eye(seq_len)  # Identity matrix to see where attention goes
    
    print("\nInput Data:")
    print(f"Query shape: {q.shape}")
    print(f"Key shape: {k.shape}")
    print(f"Value shape: {v.shape}")
    
    print("\nQueries:")
    print(q)
    
    # Calculate attention
    print("\nComputing attention...")
    output, weights = scaled_dot_product_attention(q, k, v)
    
    # Check if output is None (function not implemented)
    if output is None or weights is None:
        print("\nERROR: The attention function is returning None!")
        print("Make sure you've implemented all the TODOs in src/attention.py")
        return
    
    print("Attention computed successfully!")
    
    print("\nResults:")
    print("Attention weights shape:", weights.shape)
    print("\nAttention weights:")
    print(weights)
    print("\nRow sums (should all be 1.0):")
    print(weights.sum(dim=1))
    
    # Verify row sums
    row_sums = weights.sum(dim=1)
    all_sum_to_one = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    print(f"\nAll rows sum to 1.0: {all_sum_to_one}")
    
    # Create visualization
    print("\nCreating visualization...")
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Attention weights heatmap
    plt.subplot(1, 3, 1)
    im = plt.imshow(weights.detach().numpy(), cmap='Blues', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('Attention Weights\n(who looks at whom)', fontsize=12, fontweight='bold')
    plt.xlabel('Key positions')
    plt.ylabel('Query positions')
    
    # Add text annotations
    for i in range(seq_len):
        for j in range(seq_len):
            plt.text(j, i, f'{weights[i,j]:.2f}', 
                    ha='center', va='center', 
                    color='white' if weights[i,j] > 0.5 else 'black')
    
    # Plot 2: Output visualization
    plt.subplot(1, 3, 2)
    im = plt.imshow(output.detach().numpy(), cmap='Greens', aspect='auto')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('Output\n(weighted values)', fontsize=12, fontweight='bold')
    plt.xlabel('Value dimensions')
    plt.ylabel('Sequence positions')
    
    # Plot 3: Attention pattern interpretation
    plt.subplot(1, 3, 3)
    # Create a bar chart showing attention distribution for each query
    x = range(seq_len)
    width = 0.2
    
    for i in range(seq_len):
        plt.bar([xi + i*width for xi in x], weights[i].detach().numpy(), 
                width, label=f'Query {i+1}', alpha=0.8)
    
    plt.xlabel('Key positions')
    plt.ylabel('Attention weight')
    plt.title('Attention Distribution\nby Query', fontsize=12, fontweight='bold')
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(viz_dir, 'attention_simple_pattern.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {filename}")
    plt.close()
    
    # Test 2: With masking
    print("\n" + "="*60)
    print("TEST 2: CAUSAL MASK (for decoder)")
    print("="*60)
    
    # Create a causal mask (can only attend to previous positions)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len) * -float('inf'), diagonal=1)
    print("\nCausal mask (upper triangle = -inf):")
    print(causal_mask)
    
    print("\nComputing masked attention...")
    output_masked, weights_masked = scaled_dot_product_attention(q, k, v, mask=causal_mask)
    
    print("Masked attention computed!")
    print("\nMasked attention weights:")
    print(weights_masked)
    
    # Create visualization for masked attention
    print("\nCreating masked attention visualization...")
    plt.figure(figsize=(10, 5))
    
    # Plot 1: Causal mask visualization
    plt.subplot(1, 2, 1)
    mask_viz = causal_mask.clone()
    mask_viz[mask_viz == -float('inf')] = -1  # Replace -inf with -1 for visualization
    im = plt.imshow(mask_viz.numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=0)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('Causal Mask\n(red = blocked)', fontsize=12, fontweight='bold')
    plt.xlabel('Key positions')
    plt.ylabel('Query positions')
    
    # Add grid
    for i in range(seq_len + 1):
        plt.axhline(i - 0.5, color='black', linewidth=0.5)
        plt.axvline(i - 0.5, color='black', linewidth=0.5)
    
    # Plot 2: Masked attention weights
    plt.subplot(1, 2, 2)
    im = plt.imshow(weights_masked.detach().numpy(), cmap='Blues', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('Attention Weights with Causal Mask\n(can only look backwards)', fontsize=12, fontweight='bold')
    plt.xlabel('Key positions')
    plt.ylabel('Query positions')
    
    # Add text annotations
    for i in range(seq_len):
        for j in range(seq_len):
            if weights_masked[i,j] > 0.01:  # Only show non-zero weights
                plt.text(j, i, f'{weights_masked[i,j]:.2f}', 
                        ha='center', va='center',
                        color='white' if weights_masked[i,j] > 0.5 else 'black')
    
    plt.tight_layout()
    filename = os.path.join(viz_dir, 'attention_causal_mask.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {filename}")
    plt.close()


def test_scaling_effect():
    """Demonstrate why we scale by sqrt(d_k)"""
    global viz_dir
    """Demonstrate why we scale by sqrt(d_k)"""
    print("\n" + "="*60)
    print("TEST 3: WHY SCALE BY sqrt(d_k)?")
    print("="*60)
    
    seq_len = 4
    dimensions = [1, 4, 16, 64, 256]
    
    print("\nTesting different dimensions...")
    print("-" * 50)
    print(f"{'d_k':>6} | {'Unscaled Var':>12} | {'Scaled Var':>12} | {'Ratio':>8}")
    print("-" * 50)
    
    unscaled_vars = []
    scaled_vars = []
    
    for d_k in dimensions:
        q = torch.randn(seq_len, d_k)
        k = torch.randn(seq_len, d_k)
        v = torch.randn(seq_len, d_k)
        
        # Calculate scores without scaling
        scores_unscaled = torch.matmul(q, k.transpose(-2, -1))
        scores_scaled = scores_unscaled / math.sqrt(d_k)
        
        # Calculate variance
        var_unscaled = scores_unscaled.var().item()
        var_scaled = scores_scaled.var().item()
        
        unscaled_vars.append(var_unscaled)
        scaled_vars.append(var_scaled)
        
        ratio = var_unscaled / var_scaled if var_scaled > 0 else 0
        
        print(f"{d_k:6d} | {var_unscaled:12.2f} | {var_scaled:12.2f} | {ratio:8.2f}x")
    
    print("-" * 50)
    
    print("\nKey Insights:")
    print("• Without scaling, variance grows linearly with d_k")
    print("• This causes softmax to become very peaked (one-hot)")
    print("• Peaked softmax → vanishing gradients → poor learning")
    print("• Scaling keeps variance ~1.0 regardless of dimension")
    
    # Create visualization
    print("\nCreating scaling effect visualization...")
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Variance comparison
    plt.subplot(1, 2, 1)
    x = range(len(dimensions))
    width = 0.35
    
    plt.bar([xi - width/2 for xi in x], unscaled_vars, width, 
            label='Unscaled', color='red', alpha=0.7)
    plt.bar([xi + width/2 for xi in x], scaled_vars, width, 
            label='Scaled by √d_k', color='blue', alpha=0.7)
    
    plt.xlabel('Dimension (d_k)')
    plt.ylabel('Variance of scores')
    plt.title('Variance Growth with Dimension', fontsize=12, fontweight='bold')
    plt.xticks(x, dimensions)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 2: Softmax distribution comparison
    plt.subplot(1, 2, 2)
    
    # Show softmax distribution for large d_k
    d_k_demo = 64
    scores_demo = torch.randn(1, 8) * math.sqrt(d_k_demo)  # Simulate large d_k scores
    
    softmax_unscaled = torch.softmax(scores_demo, dim=-1)
    softmax_scaled = torch.softmax(scores_demo / math.sqrt(d_k_demo), dim=-1)
    
    positions = range(8)
    plt.plot(positions, softmax_unscaled.squeeze().numpy(), 'r-o', 
             label='Unscaled (peaked)', linewidth=2, markersize=8)
    plt.plot(positions, softmax_scaled.squeeze().numpy(), 'b-o', 
             label='Scaled (smooth)', linewidth=2, markersize=8)
    
    plt.xlabel('Position')
    plt.ylabel('Attention weight')
    plt.title(f'Softmax Distribution (d_k={d_k_demo})', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    filename = os.path.join(viz_dir, 'scaling_effect.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {filename}")
    plt.close()


def create_summary_visualization():
    """Create a summary visualization explaining attention mechanism"""
    global viz_dir
    """Create a summary visualization explaining attention mechanism"""
    print("\n" + "="*60)
    print("CREATING ATTENTION MECHANISM SUMMARY")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Attention Mechanism Summary', fontsize=16, fontweight='bold')
    
    # 1. Formula visualization
    ax = axes[0, 0]
    ax.text(0.5, 0.7, r'Attention(Q,K,V) = softmax($\frac{QK^T}{\sqrt{d_k}}$)V', 
            fontsize=14, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.4, 'Where:', fontsize=12, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.3, 'Q = Queries (what am I looking for?)', fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.2, 'K = Keys (what information is available?)', fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.1, 'V = Values (actual information to retrieve)', fontsize=10, ha='center', transform=ax.transAxes)
    ax.set_title('The Attention Formula', fontsize=12)
    ax.axis('off')
    
    # 2. Step by step process
    ax = axes[0, 1]
    steps = [
        '1. Compute similarity: Q @ K.T',
        '2. Scale: divide by √d_k',
        '3. Apply mask (optional)',
        '4. Softmax: convert to probabilities',
        '5. Weight values: attention @ V'
    ]
    for i, step in enumerate(steps):
        ax.text(0.1, 0.8 - i*0.15, step, fontsize=10, transform=ax.transAxes)
    ax.set_title('Step-by-Step Process', fontsize=12)
    ax.axis('off')
    
    # 3. Attention pattern example
    ax = axes[1, 0]
    example_attention = torch.tensor([
        [0.7, 0.2, 0.1, 0.0],
        [0.3, 0.5, 0.2, 0.0],
        [0.1, 0.3, 0.4, 0.2],
        [0.0, 0.1, 0.3, 0.6]
    ])
    im = ax.imshow(example_attention, cmap='Blues', aspect='auto')
    ax.set_title('Example Attention Pattern', fontsize=12)
    ax.set_xlabel('Keys')
    ax.set_ylabel('Queries')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{example_attention[i,j]:.1f}', 
                   ha='center', va='center',
                   color='white' if example_attention[i,j] > 0.5 else 'black')
    
    # 4. Key concepts
    ax = axes[1, 1]
    concepts = [
        'Each row sums to 1.0',
        'Self-attention: Q, K, V from same sequence',
        'Parallel computation (unlike RNNs)',
        'Direct long-range dependencies',
        'Learned attention patterns'
    ]
    for i, concept in enumerate(concepts):
        ax.text(0.1, 0.8 - i*0.15, concept, fontsize=10, transform=ax.transAxes)
    ax.set_title('Key Concepts', fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    filename = os.path.join(viz_dir, 'attention_summary.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved summary to: {filename}")
    plt.close()


if __name__ == "__main__":
    print("\n" + "*" * 60)
    print("TRANSFORMER ATTENTION VISUALIZATION SCRIPT")
    print("*" * 60)
    
    # Set up directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    viz_dir = os.path.join(script_dir, 'visualizations')
    
    # Create visualizations directory
    print(f"\nCreating visualizations directory...")
    print(f"   Script directory: {script_dir}")
    print(f"   Visualization directory: {viz_dir}")
    
    try:
        os.makedirs(viz_dir, exist_ok=True)
        if os.path.exists(viz_dir):
            print("   Directory ready!")
        else:
            print("   Could not verify directory creation")
            # Use current directory as fallback
            viz_dir = os.getcwd()
            print(f"   Using fallback directory: {viz_dir}")
    except Exception as e:
        print(f"   Error creating directory: {e}")
        viz_dir = os.getcwd()
        print(f"   Using current directory: {viz_dir}")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nTimestamp: {timestamp}")
    print(f"Working directory: {os.getcwd()}")
    
    try:
        # Run all visualizations
        visualize_attention()
        test_scaling_effect()
        create_summary_visualization()
        
        print("\n" + "=" * 60)
        print("ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\nGenerated files:")
        if os.path.exists(viz_dir):
            files = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
            if files:
                for filename in files:
                    print(f"   • {os.path.join(viz_dir, filename)}")
            else:
                print("   No PNG files found in visualizations directory")
        else:
            print("   Visualizations directory not found")
        
        print("\nNext steps:")
        print("1. Check the 'visualizations' folder for the generated images")
        print("2. Run the tests: pytest tests/test_attention.py -v")
        print("3. Move on to implementing multi-head attention!")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR OCCURRED!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nCommon issues:")
        print("1. Make sure you've implemented all TODOs in src/attention.py")
        print("2. Check that all imports are correct")
        print("3. Ensure PyTorch is installed: pip install torch")
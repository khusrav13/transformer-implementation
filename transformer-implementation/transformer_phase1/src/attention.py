import torch
import torch.nn.functional as F
import math


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Calculate attention weights and apply them to values.
    
    The attention mechanism can be described as:
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    
    Args:
        query: Tensor of shape [seq_len, d_k]
        key: Tensor of shape [seq_len, d_k]  
        value: Tensor of shape [seq_len, d_v]
        mask: Optional tensor of shape [seq_len, seq_len]
              Values should be 0 or -inf (0 = attend, -inf = don't attend)
        
    Returns:
        output: Tensor of shape [seq_len, d_v]
        attention_weights: Tensor of shape [seq_len, seq_len]
    """
    
    # Get the dimension of the key vectors (d_k)
    d_k = query.size(-1)
    
    # Step 1: Calculate attention scores
    # Compute Q @ K^T (hint: use torch.matmul and torch.transpose)
    # Expected shape: [seq_len, seq_len]
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Step 2: Scale the scores
    # Divide scores by sqrt(d_k)
    # Think: Why do we scale? (Hint: it's about gradient stability)
    scores = scores / math.sqrt(d_k)
    
    # Step 3: Apply mask if provided
    if mask is not None:
        # Add mask to scores (remember: -inf becomes 0 after softmax)
        scores = scores + mask
    
    # Step 4: Apply softmax to get attention weights
    # Apply softmax along the last dimension
    # Each row should sum to 1.0
    attention_weights = F.softmax(scores, dim=-1)
    
    # Step 5: Apply attention weights to values
    # Multiply attention_weights with values
    # Expected shape: [seq_len, d_v]
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights
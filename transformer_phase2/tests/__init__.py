"""
Test package for Transformer implementation.
Provides test utilities and fixtures.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
TEST_DTYPE = __import__('torch').float32
VISUALIZATIONS_DIR = Path(__file__).parent / 'visualizations'

# Create visualizations directory if it doesn't exist
VISUALIZATIONS_DIR.mkdir(exist_ok=True)
(VISUALIZATIONS_DIR / 'attention_patterns').mkdir(exist_ok=True)
(VISUALIZATIONS_DIR / 'positional_plots').mkdir(exist_ok=True)
(VISUALIZATIONS_DIR / 'gradient_flow').mkdir(exist_ok=True)
(VISUALIZATIONS_DIR / 'layer_outputs').mkdir(exist_ok=True)

# Common test utilities
def get_test_device():
    """Get the device for testing."""
    return TEST_DEVICE

def create_test_tensor(shape, requires_grad=False):
    """Create a test tensor with the specified shape."""
    import torch
    return torch.randn(shape, device=TEST_DEVICE, dtype=TEST_DTYPE, requires_grad=requires_grad)

def assert_shape(tensor, expected_shape, name="tensor"):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, \
        f"{name} shape mismatch: got {tensor.shape}, expected {expected_shape}"

def assert_close(actual, expected, rtol=1e-5, atol=1e-8, name="values"):
    """Assert two tensors are close."""
    import torch
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=name)

# Export test utilities
__all__ = [
    'TEST_DEVICE',
    'TEST_DTYPE',
    'VISUALIZATIONS_DIR',
    'get_test_device',
    'create_test_tensor',
    'assert_shape',
    'assert_close'
]
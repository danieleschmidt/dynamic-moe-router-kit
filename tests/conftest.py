"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def sample_input():
    """Sample input tensor for testing."""
    return {"batch_size": 32, "seq_len": 128, "hidden_dim": 768}


@pytest.fixture
def router_config():
    """Standard router configuration for testing."""
    return {
        "input_dim": 768,
        "num_experts": 8,
        "min_experts": 1,
        "max_experts": 4,
    }


@pytest.fixture
def complexity_scores():
    """Sample complexity scores for testing."""
    import numpy as np
    return np.random.rand(32, 128)  # [batch, seq]


# Backend availability checks
def pytest_configure():
    """Configure pytest with backend availability markers."""
    import sys
    
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False
        
    try:
        import jax
        jax_available = True
    except ImportError:
        jax_available = False
        
    try:
        import tensorflow as tf
        tf_available = True
    except ImportError:
        tf_available = False
    
    # Store availability for test filtering
    sys.torch_available = torch_available
    sys.jax_available = jax_available
    sys.tf_available = tf_available


def pytest_collection_modifyitems(config, items):
    """Skip tests for unavailable backends."""
    import sys
    
    skip_torch = pytest.mark.skip(reason="PyTorch not available")
    skip_jax = pytest.mark.skip(reason="JAX not available")
    skip_tf = pytest.mark.skip(reason="TensorFlow not available")
    
    for item in items:
        if "torch" in item.keywords and not getattr(sys, 'torch_available', False):
            item.add_marker(skip_torch)
        elif "jax" in item.keywords and not getattr(sys, 'jax_available', False):
            item.add_marker(skip_jax)
        elif "tf" in item.keywords and not getattr(sys, 'tf_available', False):
            item.add_marker(skip_tf)
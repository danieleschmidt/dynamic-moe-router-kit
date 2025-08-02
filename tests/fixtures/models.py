"""Model fixtures for testing."""

import pytest
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Test model configuration."""
    name: str
    num_experts: int
    hidden_dim: int
    intermediate_dim: int
    min_experts: int
    max_experts: int
    complexity_estimator: str = "gradient_norm"


@pytest.fixture
def small_model_config():
    """Small model configuration for fast tests."""
    return ModelConfig(
        name="small_test_model",
        num_experts=4,
        hidden_dim=128,
        intermediate_dim=512,
        min_experts=1,
        max_experts=2
    )


@pytest.fixture
def medium_model_config():
    """Medium model configuration for integration tests."""
    return ModelConfig(
        name="medium_test_model",
        num_experts=8,
        hidden_dim=768,
        intermediate_dim=3072,
        min_experts=1,
        max_experts=4
    )


@pytest.fixture
def large_model_config():
    """Large model configuration for performance tests."""
    return ModelConfig(
        name="large_test_model",
        num_experts=32,
        hidden_dim=4096,
        intermediate_dim=16384,
        min_experts=2,
        max_experts=8
    )


@pytest.fixture
def complexity_estimator_configs():
    """Different complexity estimator configurations."""
    return {
        "gradient_norm": {"type": "gradient_norm", "normalize": True},
        "attention_entropy": {"type": "attention_entropy", "temperature": 1.0},
        "perplexity_proxy": {"type": "perplexity_proxy", "window_size": 10},
        "random": {"type": "random", "seed": 42}
    }


@pytest.fixture
def routing_strategies():
    """Different routing strategy configurations."""
    return {
        "top_k": {"strategy": "top_k", "load_balance_factor": 0.01},
        "threshold": {"strategy": "threshold", "threshold": 0.5},
        "learned": {"strategy": "learned", "temperature": 2.0}
    }


@pytest.fixture(params=["torch", "jax", "tf"])
def backend_name(request):
    """Parametrized fixture for different backends."""
    return request.param


@pytest.fixture
def test_data_shapes():
    """Common data shapes for testing."""
    return {
        "tiny": {"batch_size": 2, "seq_len": 8, "hidden_dim": 64},
        "small": {"batch_size": 4, "seq_len": 32, "hidden_dim": 128},
        "medium": {"batch_size": 8, "seq_len": 128, "hidden_dim": 768},
        "large": {"batch_size": 16, "seq_len": 512, "hidden_dim": 4096}
    }


@pytest.fixture
def expert_utilization_targets():
    """Target expert utilization patterns for testing load balancing."""
    return {
        "uniform": [0.125] * 8,  # Perfect balance for 8 experts
        "skewed": [0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],  # Imbalanced
        "bimodal": [0.3, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.3]  # Two peaks
    }


@pytest.fixture
def performance_thresholds():
    """Performance thresholds for benchmark validation."""
    return {
        "flop_reduction_min": 0.20,  # At least 20% FLOP reduction
        "quality_degradation_max": 0.02,  # At most 2% quality loss
        "latency_overhead_max": 0.15,  # At most 15% latency overhead
        "memory_overhead_max": 0.10,  # At most 10% memory overhead
        "expert_utilization_cv_max": 0.5  # Coefficient of variation for balance
    }
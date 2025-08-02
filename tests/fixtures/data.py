"""Data fixtures for testing."""

import pytest
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


@pytest.fixture
def synthetic_text_data():
    """Generate synthetic text-like data for testing."""
    def _generate(batch_size: int = 4, seq_len: int = 128, vocab_size: int = 1000):
        # Simulate token IDs with realistic patterns
        np.random.seed(42)
        
        # Create some structure in the data
        data = []
        for _ in range(batch_size):
            # Mix of common tokens (lower IDs) and rare tokens (higher IDs)
            common_tokens = np.random.choice(100, size=seq_len//2, p=None)
            rare_tokens = np.random.choice(vocab_size - 100, size=seq_len//2) + 100
            
            # Interleave for more realistic patterns
            sequence = np.empty(seq_len, dtype=np.int32)
            sequence[::2] = common_tokens[:seq_len//2]
            sequence[1::2] = rare_tokens[:seq_len//2]
            
            data.append(sequence)
        
        return np.array(data)
    
    return _generate


@pytest.fixture
def complexity_patterns():
    """Generate different complexity patterns for testing."""
    def _patterns(batch_size: int = 4, seq_len: int = 128):
        patterns = {}
        
        # Uniform complexity (easy)
        patterns["uniform"] = np.full((batch_size, seq_len), 0.3)
        
        # Increasing complexity
        patterns["increasing"] = np.tile(
            np.linspace(0.1, 0.9, seq_len), (batch_size, 1)
        )
        
        # Random spikes
        base = np.full((batch_size, seq_len), 0.2)
        spike_positions = np.random.choice(seq_len, size=seq_len//10, replace=False)
        base[:, spike_positions] = 0.8
        patterns["spiky"] = base
        
        # Sinusoidal pattern
        x = np.linspace(0, 4 * np.pi, seq_len)
        patterns["sinusoidal"] = np.tile(
            0.5 + 0.3 * np.sin(x), (batch_size, 1)
        )
        
        # Bimodal distribution
        low_complexity = np.random.beta(2, 5, (batch_size, seq_len//2)) * 0.4
        high_complexity = 0.6 + np.random.beta(2, 5, (batch_size, seq_len//2)) * 0.4
        patterns["bimodal"] = np.concatenate([low_complexity, high_complexity], axis=1)
        
        return patterns
    
    return _patterns


@pytest.fixture
def attention_weights():
    """Generate realistic attention weight patterns."""
    def _generate(batch_size: int = 4, num_heads: int = 12, seq_len: int = 128):
        np.random.seed(42)
        
        # Create attention patterns with some structure
        weights = []
        for _ in range(batch_size):
            batch_weights = []
            for head in range(num_heads):
                # Different heads focus on different patterns
                if head % 3 == 0:  # Local attention
                    w = np.eye(seq_len) + np.eye(seq_len, k=1) + np.eye(seq_len, k=-1)
                elif head % 3 == 1:  # Global attention
                    w = np.ones((seq_len, seq_len)) / seq_len
                else:  # Random attention
                    w = np.random.rand(seq_len, seq_len)
                
                # Normalize to proper attention weights
                w = w / w.sum(axis=-1, keepdims=True)
                batch_weights.append(w)
            
            weights.append(np.array(batch_weights))
        
        return np.array(weights)
    
    return _generate


@pytest.fixture
def gradient_data():
    """Generate synthetic gradient data for complexity estimation."""
    def _generate(batch_size: int = 4, seq_len: int = 128, hidden_dim: int = 768):
        np.random.seed(42)
        
        # Create gradients with varying magnitudes
        gradients = {}
        
        # Low magnitude gradients (easy examples)
        gradients["low"] = np.random.normal(0, 0.01, (batch_size, seq_len, hidden_dim))
        
        # Medium magnitude gradients
        gradients["medium"] = np.random.normal(0, 0.1, (batch_size, seq_len, hidden_dim))
        
        # High magnitude gradients (difficult examples)
        gradients["high"] = np.random.normal(0, 1.0, (batch_size, seq_len, hidden_dim))
        
        # Mixed gradients (realistic scenario)
        mixed = np.random.normal(0, 0.1, (batch_size, seq_len, hidden_dim))
        # Add some high-magnitude positions
        high_positions = np.random.choice(
            seq_len, size=seq_len//10, replace=False
        )
        mixed[:, high_positions, :] *= 10
        gradients["mixed"] = mixed
        
        return gradients
    
    return _generate


@pytest.fixture
def benchmark_datasets():
    """Standard benchmark datasets for evaluation."""
    return {
        "synthetic_easy": {
            "description": "Simple patterns, low complexity",
            "complexity_mean": 0.2,
            "complexity_std": 0.1,
            "expected_flop_reduction": 0.45
        },
        "synthetic_hard": {
            "description": "Complex patterns, high complexity",
            "complexity_mean": 0.8,
            "complexity_std": 0.15,
            "expected_flop_reduction": 0.15
        },
        "synthetic_mixed": {
            "description": "Mixed complexity patterns",
            "complexity_mean": 0.5,
            "complexity_std": 0.3,
            "expected_flop_reduction": 0.35
        }
    }


@pytest.fixture
def error_injection_scenarios():
    """Scenarios for testing error handling and robustness."""
    return {
        "nan_inputs": {
            "description": "NaN values in input tensors",
            "injection_fn": lambda x: np.where(np.random.rand(*x.shape) < 0.01, np.nan, x)
        },
        "inf_inputs": {
            "description": "Infinite values in input tensors", 
            "injection_fn": lambda x: np.where(np.random.rand(*x.shape) < 0.01, np.inf, x)
        },
        "zero_gradients": {
            "description": "All-zero gradients",
            "injection_fn": lambda x: np.zeros_like(x)
        },
        "extreme_values": {
            "description": "Extremely large values",
            "injection_fn": lambda x: x * np.where(np.random.rand(*x.shape) < 0.01, 1e6, 1.0)
        }
    }


@pytest.fixture
def load_balancing_scenarios():
    """Test scenarios for load balancing evaluation."""
    return {
        "perfect_balance": {
            "target_utilization": "uniform",
            "tolerance": 0.05,
            "description": "All experts used equally"
        },
        "moderate_imbalance": {
            "target_utilization": "slightly_skewed", 
            "tolerance": 0.2,
            "description": "Some preference but not extreme"
        },
        "extreme_imbalance": {
            "target_utilization": "highly_skewed",
            "tolerance": 0.5,
            "description": "Strong expert preferences"
        }
    }
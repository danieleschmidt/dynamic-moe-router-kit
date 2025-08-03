"""Tests for complexity estimators."""

import pytest
import numpy as np

from dynamic_moe_router.estimator import (
    GradientNormEstimator,
    AttentionEntropyEstimator,
    PerplexityProxyEstimator,
    ThresholdEstimator,
    get_estimator,
    ESTIMATOR_REGISTRY
)


class TestComplexityEstimators:
    """Test suite for complexity estimators."""
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input data."""
        batch_size, seq_len, hidden_dim = 2, 8, 64
        return np.random.randn(batch_size, seq_len, hidden_dim)
    
    def test_gradient_norm_estimator(self, sample_input):
        """Test gradient norm estimator."""
        estimator = GradientNormEstimator(normalize=True)
        scores = estimator.estimate(sample_input)
        
        # Check output shape
        assert scores.shape == sample_input.shape[:2]  # [batch, seq_len]
        
        # Check score range (should be normalized)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)
        
        # Test with different aggregation methods
        estimator_l1 = GradientNormEstimator(aggregation='l1')
        scores_l1 = estimator_l1.estimate(sample_input)
        assert scores_l1.shape == scores.shape
    
    def test_attention_entropy_estimator(self, sample_input):
        """Test attention entropy estimator."""
        estimator = AttentionEntropyEstimator(normalize=True)
        
        # Test without attention weights (fallback mode)
        scores = estimator.estimate(sample_input)
        assert scores.shape == sample_input.shape[:2]
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)
        
        # Test with provided attention weights
        batch_size, seq_len = sample_input.shape[:2]
        num_heads = 4
        attention_weights = np.random.rand(batch_size, num_heads, seq_len, seq_len)
        
        # Normalize attention weights to be valid probabilities
        attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
        
        scores_with_attn = estimator.estimate(sample_input, attention_weights=attention_weights)
        assert scores_with_attn.shape == sample_input.shape[:2]
    
    def test_perplexity_proxy_estimator(self, sample_input):
        """Test perplexity proxy estimator."""
        estimator = PerplexityProxyEstimator(temperature=1.0, normalize=True)
        
        # Test without logits (fallback mode)
        scores = estimator.estimate(sample_input)
        assert scores.shape == sample_input.shape[:2]
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)
        
        # Test with provided logits
        batch_size, seq_len = sample_input.shape[:2]
        vocab_size = 1000
        logits = np.random.randn(batch_size, seq_len, vocab_size)
        
        scores_with_logits = estimator.estimate(sample_input, logits=logits)
        assert scores_with_logits.shape == sample_input.shape[:2]
    
    def test_threshold_estimator(self, sample_input):
        """Test threshold estimator."""
        estimator = ThresholdEstimator(normalize=True)
        scores = estimator.estimate(sample_input)
        
        assert scores.shape == sample_input.shape[:2]
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)
        
        # Test with custom threshold function
        def custom_threshold(hidden_states, **kwargs):
            # Return constant complexity
            return np.ones(hidden_states.shape[:2]) * 0.5
        
        custom_estimator = ThresholdEstimator(threshold_fn=custom_threshold)
        custom_scores = custom_estimator.estimate(sample_input)
        assert np.allclose(custom_scores, 0.5)
    
    def test_estimator_calibration(self, sample_input):
        """Test estimator calibration functionality."""
        estimator = GradientNormEstimator(normalize=True)
        
        # Generate multiple samples for calibration
        samples = [sample_input + np.random.randn(*sample_input.shape) * 0.1 for _ in range(5)]
        
        # Calibrate estimator
        estimator.calibrate(samples, target_range=(0.1, 0.9))
        
        # Test estimation with calibration
        scores = estimator.estimate(sample_input)
        assert scores.shape == sample_input.shape[:2]
        
        # Check that calibration stats were set
        assert estimator._calibration_stats is not None
        assert 'mean' in estimator._calibration_stats
        assert 'std' in estimator._calibration_stats
    
    def test_estimator_registry(self):
        """Test estimator registry and factory function."""
        # Check all registered estimators
        expected_estimators = {
            'gradient_norm': GradientNormEstimator,
            'attention_entropy': AttentionEntropyEstimator,
            'perplexity_proxy': PerplexityProxyEstimator,
            'threshold': ThresholdEstimator
        }
        
        for name, cls in expected_estimators.items():
            assert name in ESTIMATOR_REGISTRY
            assert ESTIMATOR_REGISTRY[name] == cls
        
        # Test factory function
        for name in expected_estimators:
            estimator = get_estimator(name)
            assert isinstance(estimator, expected_estimators[name])
        
        # Test invalid estimator name
        with pytest.raises(ValueError, match="Unknown estimator"):
            get_estimator("invalid_estimator")
    
    def test_estimator_with_kwargs(self):
        """Test estimator creation with keyword arguments."""
        # Test with various kwargs
        estimator = get_estimator("gradient_norm", aggregation="l1", normalize=False)
        assert estimator.aggregation == "l1"
        assert estimator.normalize == False
        
        estimator = get_estimator("attention_entropy", head_aggregation="max")
        assert estimator.head_aggregation == "max"
        
        estimator = get_estimator("perplexity_proxy", temperature=2.0)
        assert estimator.temperature == 2.0
    
    def test_estimator_consistency(self, sample_input):
        """Test that estimators produce consistent results."""
        estimator = GradientNormEstimator(normalize=True)
        
        # Multiple calls should produce the same result
        scores1 = estimator.estimate(sample_input)
        scores2 = estimator.estimate(sample_input)
        
        np.testing.assert_allclose(scores1, scores2, rtol=1e-10)
    
    def test_estimator_input_validation(self):
        """Test input validation for estimators."""
        estimator = GradientNormEstimator()
        
        # Test with invalid input shapes
        invalid_input = np.random.randn(10)  # 1D instead of 3D
        
        # Should handle gracefully or raise informative error
        try:
            scores = estimator.estimate(invalid_input)
            # If it doesn't raise an error, check the output is reasonable
            assert scores.shape[0] == invalid_input.shape[0]
        except (IndexError, ValueError):
            # Expected for invalid input shapes
            pass
    
    def test_estimator_edge_cases(self):
        """Test estimators with edge cases."""
        # Very small input
        small_input = np.random.randn(1, 1, 8)
        estimator = GradientNormEstimator()
        scores = estimator.estimate(small_input)
        assert scores.shape == (1, 1)
        
        # Zero input
        zero_input = np.zeros((2, 4, 16))
        scores = estimator.estimate(zero_input)
        assert scores.shape == (2, 4)
        assert np.all(np.isfinite(scores))  # Should not produce NaN/inf
        
        # Very large input
        large_input = np.random.randn(1, 1, 16) * 1000
        scores = estimator.estimate(large_input)
        assert np.all(np.isfinite(scores))
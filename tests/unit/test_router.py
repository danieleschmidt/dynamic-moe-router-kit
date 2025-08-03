"""Tests for dynamic router."""

import pytest
import numpy as np

from dynamic_moe_router.router import DynamicRouter, AdaptiveRouter
from dynamic_moe_router.estimator import GradientNormEstimator


class TestDynamicRouter:
    """Test suite for dynamic router."""
    
    @pytest.fixture
    def router_config(self):
        """Basic router configuration."""
        return {
            'input_dim': 64,
            'num_experts': 8,
            'min_experts': 1,
            'max_experts': 4,
            'complexity_estimator': 'gradient_norm'
        }
    
    @pytest.fixture
    def sample_input(self):
        """Sample input data."""
        return np.random.randn(2, 8, 64)  # [batch, seq_len, hidden_dim]
    
    def test_router_initialization(self, router_config):
        """Test router initialization."""
        router = DynamicRouter(**router_config)
        
        assert router.input_dim == 64
        assert router.num_experts == 8
        assert router.min_experts == 1
        assert router.max_experts == 4
        assert isinstance(router.complexity_estimator, GradientNormEstimator)
    
    def test_router_validation(self):
        """Test router parameter validation."""
        # Invalid min_experts
        with pytest.raises(ValueError, match="min_experts must be"):
            DynamicRouter(input_dim=64, num_experts=8, min_experts=0)
        
        with pytest.raises(ValueError, match="min_experts must be"):
            DynamicRouter(input_dim=64, num_experts=8, min_experts=10)
        
        # Invalid max_experts
        with pytest.raises(ValueError, match="max_experts must be"):
            DynamicRouter(input_dim=64, num_experts=8, max_experts=0)
        
        with pytest.raises(ValueError, match="max_experts must be"):
            DynamicRouter(input_dim=64, num_experts=8, min_experts=3, max_experts=2)
    
    def test_router_network_initialization(self, router_config):
        """Test router network weight initialization."""
        router = DynamicRouter(**router_config)
        
        # Initially weights should be None
        assert router.router_weights is None
        assert router.router_bias is None
        
        # Initialize weights
        router.initialize_router_network()
        
        assert router.router_weights is not None
        assert router.router_bias is not None
        assert router.router_weights.shape == (64, 8)
        assert router.router_bias.shape == (8,)
    
    def test_basic_routing(self, router_config, sample_input):
        """Test basic routing functionality."""
        router = DynamicRouter(**router_config)
        result = router.route(sample_input)
        
        # Check required output keys
        required_keys = [
            'expert_indices', 'expert_weights', 'num_experts_per_token',
            'complexity_scores', 'routing_info'
        ]
        for key in required_keys:
            assert key in result
        
        # Check output shapes
        batch_size, seq_len = sample_input.shape[:2]
        max_k = router.max_experts
        
        assert result['expert_indices'].shape == (batch_size, seq_len, max_k)
        assert result['expert_weights'].shape == (batch_size, seq_len, max_k)
        assert result['num_experts_per_token'].shape == (batch_size, seq_len)
        assert result['complexity_scores'].shape == (batch_size, seq_len)
        
        # Check value constraints
        assert np.all(result['num_experts_per_token'] >= router.min_experts)
        assert np.all(result['num_experts_per_token'] <= router.max_experts)
        assert np.all(result['expert_indices'] >= -1)  # -1 for padding
        assert np.all(result['expert_indices'] < router.num_experts)
        assert np.all(result['expert_weights'] >= 0.0)
        assert np.all(result['expert_weights'] <= 1.0)
    
    def test_routing_with_return_logits(self, router_config, sample_input):
        """Test routing with router logits returned."""
        router = DynamicRouter(**router_config)
        result = router.route(sample_input, return_router_logits=True)
        
        assert 'router_logits' in result
        batch_size, seq_len = sample_input.shape[:2]
        assert result['router_logits'].shape == (batch_size, seq_len, router.num_experts)
    
    def test_expert_count_computation(self, router_config):
        """Test expert count computation from complexity scores."""
        router = DynamicRouter(**router_config)
        
        # Test with various complexity scores
        complexity_scores = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
        expert_counts = router._compute_expert_counts(complexity_scores)
        
        # Should be between min and max experts
        assert np.all(expert_counts >= router.min_experts)
        assert np.all(expert_counts <= router.max_experts)
        
        # Higher complexity should generally use more experts
        assert expert_counts[0, 0] <= expert_counts[0, -1]
    
    def test_routing_strategies(self, router_config, sample_input):
        """Test different routing strategies."""
        strategies = ['top_k', 'threshold']
        
        for strategy in strategies:
            config = router_config.copy()
            config['routing_strategy'] = strategy
            
            router = DynamicRouter(**config)
            result = router.route(sample_input)
            
            # Should produce valid outputs regardless of strategy
            assert result['expert_indices'].shape == (2, 8, 4)
            assert np.all(result['expert_weights'] >= 0.0)
    
    def test_load_balancing(self, router_config, sample_input):
        """Test load balancing functionality."""
        # Router with load balancing
        config = router_config.copy()
        config['load_balancing'] = True
        router_balanced = DynamicRouter(**config)
        
        # Router without load balancing
        config['load_balancing'] = False
        router_unbalanced = DynamicRouter(**config)
        
        # Process multiple batches
        for _ in range(3):
            router_balanced.route(sample_input)
            router_unbalanced.route(sample_input)
        
        # Check that load balancing router has usage history
        assert len(router_balanced.expert_usage_history) > 0
        
        # Get usage statistics
        stats_balanced = router_balanced.get_expert_usage_stats()
        stats_unbalanced = router_unbalanced.get_expert_usage_stats()
        
        assert 'load_balance_score' in stats_balanced
        assert 'load_balance_score' in stats_unbalanced
    
    def test_noise_factor(self, router_config, sample_input):
        """Test routing noise factor."""
        # Router with noise
        config = router_config.copy()
        config['noise_factor'] = 0.1
        router_noisy = DynamicRouter(**config)
        
        # Router without noise
        config['noise_factor'] = 0.0
        router_clean = DynamicRouter(**config)
        
        # Initialize with same seed for comparison
        router_noisy.initialize_router_network(seed=42)
        router_clean.initialize_router_network(seed=42)
        
        result_noisy = router_noisy.route(sample_input, return_router_logits=True)
        result_clean = router_clean.route(sample_input, return_router_logits=True)
        
        # Logits should be different due to noise
        assert not np.allclose(
            result_noisy['router_logits'], 
            result_clean['router_logits']
        )
    
    def test_routing_info_statistics(self, router_config, sample_input):
        """Test routing info statistics."""
        router = DynamicRouter(**router_config)
        result = router.route(sample_input)
        
        routing_info = result['routing_info']
        
        # Check required statistics
        required_stats = [
            'avg_experts_per_token', 'total_expert_calls', 'flop_reduction',
            'expert_utilization', 'complexity_stats'
        ]
        for stat in required_stats:
            assert stat in routing_info
        
        # Check value ranges
        assert 0 <= routing_info['flop_reduction'] <= 1.0
        assert routing_info['avg_experts_per_token'] >= router.min_experts
        assert routing_info['avg_experts_per_token'] <= router.max_experts
        assert len(routing_info['expert_utilization']) == router.num_experts


class TestAdaptiveRouter:
    """Test suite for adaptive router."""
    
    def test_adaptive_router_initialization(self):
        """Test adaptive router initialization."""
        router = AdaptiveRouter(
            input_dim=32,
            num_experts=4,
            adaptation_rate=0.05
        )
        
        assert router.adaptation_rate == 0.05
        assert len(router.complexity_thresholds) == router.max_experts + 1
        assert len(router.performance_history) == 0
    
    def test_threshold_adaptation(self):
        """Test threshold adaptation based on performance."""
        router = AdaptiveRouter(
            input_dim=32,
            num_experts=4,
            adaptation_rate=0.1
        )
        
        initial_thresholds = router.complexity_thresholds.copy()
        
        # Simulate performance feedback
        router.update_thresholds(0.8)  # Good performance
        router.update_thresholds(0.9)  # Better performance
        
        # Thresholds should have been adjusted
        assert not np.allclose(router.complexity_thresholds, initial_thresholds)
        
        # Simulate poor performance
        router.update_thresholds(0.5)  # Worse performance
        
        assert len(router.performance_history) == 3
    
    def test_adaptive_expert_counts(self):
        """Test adaptive expert count computation."""
        router = AdaptiveRouter(
            input_dim=32,
            num_experts=4,
            max_experts=3
        )
        
        # Test with sample complexity scores
        complexity_scores = np.array([[0.1, 0.3, 0.6, 0.9]])
        expert_counts = router._compute_expert_counts(complexity_scores)
        
        assert expert_counts.shape == (1, 4)
        assert np.all(expert_counts >= router.min_experts)
        assert np.all(expert_counts <= router.max_experts)
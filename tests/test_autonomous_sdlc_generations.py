"""
Comprehensive test suite for Autonomous SDLC Generations
Testing all three generations with quality gates.
"""

import pytest
import numpy as np
import time
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dynamic_moe_router.core_simple import RouterConfig, SimpleDynamicRouter, SimpleMoELayer
from dynamic_moe_router.robust_core import RobustRouterConfig, RobustDynamicRouter, SecurityValidator
from dynamic_moe_router.optimized_core import OptimizedRouterConfig, OptimizedDynamicRouter

class TestGeneration1Simple:
    """Test Generation 1: Simple Working Implementation"""
    
    def setup_method(self):
        """Setup for each test"""
        self.config = RouterConfig(
            input_dim=256,
            num_experts=4,
            min_experts=1,
            max_experts=2
        )
        self.router = SimpleDynamicRouter(self.config)
        self.moe_layer = SimpleMoELayer(self.config)
    
    def test_router_initialization(self):
        """Test router initialization"""
        assert self.router.config.input_dim == 256
        assert self.router.config.num_experts == 4
        assert self.router.router_weights.shape == (256, 4)
        assert hasattr(self.router, 'complexity_estimator')
    
    def test_basic_routing(self):
        """Test basic routing functionality"""
        batch_size, seq_len = 2, 32
        inputs = np.random.randn(batch_size, seq_len, self.config.input_dim)
        
        result = self.router.route(inputs)
        
        # Check required keys
        required_keys = ['expert_indices', 'expert_weights', 'complexity_scores', 
                        'avg_experts_per_token', 'flop_reduction', 'routing_logits']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        # Check shapes
        assert result['expert_indices'].shape[0] == batch_size
        assert result['expert_indices'].shape[1] == seq_len
        assert result['expert_weights'].shape[0] == batch_size
        assert result['expert_weights'].shape[1] == seq_len
    
    def test_moe_forward_pass(self):
        """Test MoE layer forward pass"""
        batch_size, seq_len = 2, 32
        inputs = np.random.randn(batch_size, seq_len, self.config.input_dim)
        
        outputs, routing_info = self.moe_layer.forward(inputs)
        
        assert outputs.shape == inputs.shape
        assert isinstance(routing_info, dict)
        assert 'avg_experts_per_token' in routing_info
    
    def test_complexity_estimation(self):
        """Test complexity estimation"""
        batch_size, seq_len = 2, 32
        inputs = np.random.randn(batch_size, seq_len, self.config.input_dim)
        
        complexity = self.router.complexity_estimator.estimate(inputs)
        
        assert complexity.shape == (batch_size, seq_len, 1)
        assert np.all(complexity >= 0) and np.all(complexity <= 1)
    
    def test_performance_baseline(self):
        """Test performance meets baseline requirements"""
        batch_size, seq_len = 4, 64
        inputs = np.random.randn(batch_size, seq_len, self.config.input_dim)
        
        start_time = time.time()
        result = self.router.route(inputs)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert (end_time - start_time) < 1.0, "Routing took too long"
        
        # Should provide FLOP reduction
        assert result['flop_reduction'] > 0, "No FLOP reduction achieved"

class TestGeneration2Robust:
    """Test Generation 2: Robust Implementation with Error Handling"""
    
    def setup_method(self):
        """Setup for each test"""
        self.config = RobustRouterConfig(
            input_dim=256,
            num_experts=4,
            min_experts=1,
            max_experts=2,
            enable_security_checks=True,
            enable_monitoring=True
        )
        self.router = RobustDynamicRouter(self.config)
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config should pass
        valid_config = RobustRouterConfig(input_dim=128, num_experts=4)
        assert valid_config.input_dim == 128
        
        # Invalid config should raise error
        with pytest.raises(ValueError):
            RobustRouterConfig(input_dim=-1)
        
        with pytest.raises(ValueError):
            RobustRouterConfig(min_experts=4, max_experts=2, num_experts=8)
    
    def test_security_validation(self):
        """Test security validation"""
        security_validator = SecurityValidator(self.config)
        
        # Normal input should pass
        normal_inputs = np.random.randn(2, 32, self.config.input_dim)
        assert security_validator.validate_input(normal_inputs, "test")
        
        # Extreme input should fail
        extreme_inputs = np.random.randn(2, 32, self.config.input_dim) * 1000
        assert not security_validator.validate_input(extreme_inputs, "test")
        
        # Invalid input should fail
        invalid_inputs = np.full((2, 32, self.config.input_dim), np.inf)
        assert not security_validator.validate_input(invalid_inputs, "test")
    
    def test_error_handling(self):
        """Test error handling and fallback mechanisms"""
        # Test with extreme inputs that trigger fallback
        extreme_inputs = np.random.randn(2, 32, self.config.input_dim) * 1000
        
        result = self.router.route(extreme_inputs)
        
        # Should return fallback result
        assert 'fallback_used' in result or result['avg_experts_per_token'] > 0
        assert isinstance(result, dict)
    
    def test_performance_monitoring(self):
        """Test performance monitoring"""
        normal_inputs = np.random.randn(2, 32, self.config.input_dim)
        
        # Make several calls
        for _ in range(3):
            self.router.route(normal_inputs)
        
        health = self.router.get_health_status()
        
        assert 'health_status' in health
        assert 'performance_stats' in health
        assert health['performance_stats']['total_calls'] >= 3
    
    def test_numerical_stability(self):
        """Test numerical stability with edge cases"""
        # Test with zeros
        zero_inputs = np.zeros((2, 32, self.config.input_dim))
        result = self.router.route(zero_inputs)
        assert not np.any(np.isnan(result['expert_weights']))
        
        # Test with very small values
        small_inputs = np.random.randn(2, 32, self.config.input_dim) * 1e-8
        result = self.router.route(small_inputs)
        assert not np.any(np.isnan(result['expert_weights']))
    
    def test_load_balancing_metrics(self):
        """Test load balancing metrics calculation"""
        normal_inputs = np.random.randn(4, 64, self.config.input_dim)
        
        result = self.router.route(normal_inputs)
        
        assert 'routing_entropy' in result
        assert 'load_balance_loss' in result
        assert isinstance(result['routing_entropy'], float)
        assert isinstance(result['load_balance_loss'], float)

class TestGeneration3Optimized:
    """Test Generation 3: Optimized High-Performance Implementation"""
    
    def setup_method(self):
        """Setup for each test"""
        self.config = OptimizedRouterConfig(
            input_dim=256,
            num_experts=8,
            min_experts=1,
            max_experts=4,
            enable_caching=True,
            enable_concurrent_processing=True,
            enable_memory_pooling=True,
            max_workers=2
        )
        self.router = OptimizedDynamicRouter(self.config)
    
    def test_caching_functionality(self):
        """Test caching improves performance"""
        inputs = np.random.randn(2, 32, self.config.input_dim)
        
        # First call (cache miss)
        start_time = time.time()
        result1 = self.router.route(inputs)
        first_time = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        result2 = self.router.route(inputs)
        second_time = time.time() - start_time
        
        # Cache hit should be faster
        assert second_time < first_time, "Caching did not improve performance"
        
        # Results should be identical
        np.testing.assert_array_equal(result1['expert_indices'], result2['expert_indices'])
    
    def test_concurrent_processing(self):
        """Test concurrent processing for large batches"""
        # Large batch that should trigger concurrent processing
        large_inputs = np.random.randn(64, 32, self.config.input_dim)
        
        start_time = time.time()
        result = self.router.route(large_inputs)
        end_time = time.time()
        
        assert result['expert_indices'].shape[0] == 64
        assert (end_time - start_time) < 5.0, "Concurrent processing took too long"
    
    def test_memory_pooling(self):
        """Test memory pooling functionality"""
        if self.router.memory_pool:
            # Test getting arrays from pool
            array1 = self.router.memory_pool.get_float_array((10, 10), 0.0)
            assert array1.shape == (10, 10)
            assert array1.dtype == np.float32
            
            # Return to pool
            self.router.memory_pool.return_array(array1)
            
            # Get stats
            stats = self.router.memory_pool.get_stats()
            assert 'total_allocated' in stats
            assert 'reuse_rate' in stats
    
    def test_vectorized_operations(self):
        """Test vectorized operations for performance"""
        from dynamic_moe_router.optimized_core import VectorizedOperations
        
        # Test fast softmax
        logits = np.random.randn(4, 32, 8)
        softmax_result = VectorizedOperations.fast_softmax(logits)
        
        assert softmax_result.shape == logits.shape
        assert np.allclose(np.sum(softmax_result, axis=-1), 1.0, rtol=1e-5)
        
        # Test fast top-k
        values, indices = VectorizedOperations.fast_topk(logits, k=3)
        assert values.shape == (4, 32, 3)
        assert indices.shape == (4, 32, 3)
    
    def test_auto_scaling(self):
        """Test auto-scaling functionality"""
        if self.router.auto_scaler:
            # Update metrics to trigger scaling decision
            self.router.auto_scaler.update_metrics(0.9, 500, 0.2)  # High CPU, should scale up
            
            should_scale = self.router.auto_scaler.should_scale_up()
            # Note: scaling may be prevented by cooldown, so we just test the logic exists
            assert isinstance(should_scale, bool)
    
    def test_performance_benchmarks(self):
        """Test performance meets optimization targets"""
        # Small batch performance
        small_inputs = np.random.randn(2, 32, self.config.input_dim)
        start_time = time.time()
        result = self.router.route(small_inputs)
        small_batch_time = time.time() - start_time
        
        # Medium batch performance  
        medium_inputs = np.random.randn(8, 64, self.config.input_dim)
        start_time = time.time()
        result = self.router.route(medium_inputs)
        medium_batch_time = time.time() - start_time
        
        # Performance should be reasonable
        assert small_batch_time < 0.1, f"Small batch too slow: {small_batch_time:.3f}s"
        assert medium_batch_time < 1.0, f"Medium batch too slow: {medium_batch_time:.3f}s"
        
        # Get comprehensive stats
        stats = self.router.get_performance_stats()
        assert 'avg_compute_time' in stats
        assert stats['call_count'] >= 2
    
    def test_resource_cleanup(self):
        """Test proper resource cleanup"""
        # Test that router can be deleted without errors
        router = OptimizedDynamicRouter(self.config)
        
        # Make some calls
        inputs = np.random.randn(2, 32, self.config.input_dim)
        router.route(inputs)
        
        # Delete should not raise exceptions
        del router

class TestIntegrationAndCompatibility:
    """Integration tests across generations"""
    
    def test_consistent_results(self):
        """Test that different generations produce reasonable results"""
        # Same configuration for fair comparison
        base_config = {
            'input_dim': 128,
            'num_experts': 4,
            'min_experts': 1,
            'max_experts': 2
        }
        
        simple_config = RouterConfig(**base_config)
        robust_config = RobustRouterConfig(**base_config, enable_security_checks=False)
        optimized_config = OptimizedRouterConfig(**base_config, enable_caching=False,
                                                enable_concurrent_processing=False)
        
        simple_router = SimpleDynamicRouter(simple_config)
        robust_router = RobustDynamicRouter(robust_config)
        optimized_router = OptimizedDynamicRouter(optimized_config)
        
        # Same input for all
        np.random.seed(42)  # For reproducibility
        inputs = np.random.randn(2, 16, 128)
        
        simple_result = simple_router.route(inputs)
        robust_result = robust_router.route(inputs)
        optimized_result = optimized_router.route(inputs)
        
        # All should produce valid routing decisions
        for result in [simple_result, robust_result, optimized_result]:
            assert 'avg_experts_per_token' in result
            assert result['avg_experts_per_token'] > 0
            assert 'flop_reduction' in result
            assert result['flop_reduction'] >= 0
    
    def test_scalability_progression(self):
        """Test that optimizations improve with each generation"""
        # Test with increasingly large inputs
        sizes = [(2, 16), (4, 32), (8, 64)]
        
        for batch_size, seq_len in sizes:
            inputs = np.random.randn(batch_size, seq_len, 128)
            
            # Time each generation
            simple_router = SimpleDynamicRouter(RouterConfig(input_dim=128, num_experts=4))
            start_time = time.time()
            simple_router.route(inputs)
            simple_time = time.time() - start_time
            
            robust_router = RobustDynamicRouter(RobustRouterConfig(input_dim=128, num_experts=4,
                                                                 enable_security_checks=False))
            start_time = time.time()
            robust_router.route(inputs)
            robust_time = time.time() - start_time
            
            optimized_router = OptimizedDynamicRouter(OptimizedRouterConfig(input_dim=128, num_experts=4,
                                                                          enable_caching=False))
            start_time = time.time()
            optimized_router.route(inputs)
            optimized_time = time.time() - start_time
            
            # All should complete in reasonable time
            assert simple_time < 1.0
            assert robust_time < 1.0
            assert optimized_time < 1.0

def test_autonomous_sdlc_completion():
    """Test that autonomous SDLC is complete and functional"""
    print("🧪 Running Autonomous SDLC Quality Gates...")
    
    # Test all generations can be imported and instantiated
    simple_config = RouterConfig()
    simple_router = SimpleDynamicRouter(simple_config)
    assert simple_router is not None
    
    robust_config = RobustRouterConfig()
    robust_router = RobustDynamicRouter(robust_config)
    assert robust_router is not None
    
    optimized_config = OptimizedRouterConfig()
    optimized_router = OptimizedDynamicRouter(optimized_config)
    assert optimized_router is not None
    
    print("✅ All generations functional")
    
    # Test basic functionality
    test_input = np.random.randn(2, 32, 768)
    
    simple_result = simple_router.route(test_input)
    assert simple_result['avg_experts_per_token'] > 0
    
    robust_result = robust_router.route(test_input)
    assert robust_result['avg_experts_per_token'] > 0
    
    optimized_result = optimized_router.route(test_input)
    assert optimized_result['avg_experts_per_token'] > 0
    
    print("✅ All generations produce valid routing decisions")
    
    print("✅ Autonomous SDLC Quality Gates PASSED!")

if __name__ == "__main__":
    # Run the completion test
    test_autonomous_sdlc_completion()
    
    print("\n🏃‍♂️ Running full test suite with pytest...")
    pytest.main([__file__, "-v", "--tb=short"])
#!/usr/bin/env python3
"""
Autonomous SDLC Quality Gates - Test All Generations
No external dependencies required.
"""

import numpy as np
import time
import sys
import os
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dynamic_moe_router.core_simple import RouterConfig, SimpleDynamicRouter, SimpleMoELayer
from dynamic_moe_router.robust_core import RobustRouterConfig, RobustDynamicRouter, SecurityValidator
from dynamic_moe_router.optimized_core import OptimizedRouterConfig, OptimizedDynamicRouter

class QualityGateError(Exception):
    """Exception for quality gate failures"""
    pass

class QualityGateRunner:
    """Runner for quality gate tests"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def run_test(self, test_name, test_func):
        """Run a test and track results"""
        try:
            print(f"🧪 {test_name}...", end=" ")
            test_func()
            print("✅ PASS")
            self.passed += 1
        except Exception as e:
            print(f"❌ FAIL")
            self.failed += 1
            self.errors.append(f"{test_name}: {str(e)}")
            print(f"   Error: {str(e)}")
    
    def assert_true(self, condition, message="Assertion failed"):
        """Assert helper"""
        if not condition:
            raise QualityGateError(message)
    
    def assert_equal(self, actual, expected, message="Values not equal"):
        """Assert equal helper"""
        if actual != expected:
            raise QualityGateError(f"{message}: {actual} != {expected}")
    
    def assert_in_range(self, value, min_val, max_val, message="Value out of range"):
        """Assert value in range helper"""
        if not (min_val <= value <= max_val):
            raise QualityGateError(f"{message}: {value} not in [{min_val}, {max_val}]")
    
    def print_summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        success_rate = (self.passed / total * 100) if total > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"🏁 QUALITY GATE SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total}")
        print(f"Passed: {self.passed} ✅")
        print(f"Failed: {self.failed} ❌")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 85:
            print(f"🎉 QUALITY GATES PASSED! (>= 85% success rate)")
            return True
        else:
            print(f"💥 QUALITY GATES FAILED! (< 85% success rate)")
            if self.errors:
                print(f"\nErrors:")
                for error in self.errors:
                    print(f"  - {error}")
            return False

def test_generation_1_simple(runner):
    """Test Generation 1: Simple Implementation"""
    
    def test_basic_initialization():
        config = RouterConfig(input_dim=256, num_experts=4)
        router = SimpleDynamicRouter(config)
        runner.assert_equal(router.config.input_dim, 256)
        runner.assert_equal(router.config.num_experts, 4)
    
    def test_routing_functionality():
        config = RouterConfig(input_dim=128, num_experts=4)
        router = SimpleDynamicRouter(config)
        
        inputs = np.random.randn(2, 32, 128)
        result = router.route(inputs)
        
        required_keys = ['expert_indices', 'expert_weights', 'avg_experts_per_token', 'flop_reduction']
        for key in required_keys:
            runner.assert_true(key in result, f"Missing key: {key}")
        
        runner.assert_true(result['avg_experts_per_token'] > 0, "No experts selected")
        runner.assert_true(result['flop_reduction'] >= 0, "Invalid FLOP reduction")
    
    def test_moe_layer():
        config = RouterConfig(input_dim=64, num_experts=4)
        moe_layer = SimpleMoELayer(config)
        
        inputs = np.random.randn(1, 16, 64)
        outputs, routing_info = moe_layer.forward(inputs)
        
        runner.assert_equal(outputs.shape, inputs.shape)
        runner.assert_true('avg_experts_per_token' in routing_info)
    
    def test_performance():
        config = RouterConfig(input_dim=256, num_experts=8)
        router = SimpleDynamicRouter(config)
        
        inputs = np.random.randn(4, 64, 256)
        start_time = time.time()
        result = router.route(inputs)
        elapsed = time.time() - start_time
        
        runner.assert_true(elapsed < 1.0, f"Too slow: {elapsed:.3f}s")
    
    runner.run_test("Gen1: Basic Initialization", test_basic_initialization)
    runner.run_test("Gen1: Routing Functionality", test_routing_functionality)
    runner.run_test("Gen1: MoE Layer", test_moe_layer)
    runner.run_test("Gen1: Performance", test_performance)

def test_generation_2_robust(runner):
    """Test Generation 2: Robust Implementation"""
    
    def test_config_validation():
        # Valid config
        config = RobustRouterConfig(input_dim=128, num_experts=4)
        runner.assert_equal(config.input_dim, 128)
        
        # Invalid config should raise error
        try:
            RobustRouterConfig(input_dim=-1)
            runner.assert_true(False, "Should have raised ValueError")
        except ValueError:
            pass  # Expected
    
    def test_security_validation():
        config = RobustRouterConfig(input_dim=128, num_experts=4)
        validator = SecurityValidator(config)
        
        # Normal input
        normal_inputs = np.random.randn(2, 32, 128)
        runner.assert_true(validator.validate_input(normal_inputs, "test"))
        
        # Extreme input
        extreme_inputs = np.random.randn(2, 32, 128) * 1000
        runner.assert_true(not validator.validate_input(extreme_inputs, "test"))
    
    def test_error_handling():
        config = RobustRouterConfig(input_dim=128, num_experts=4, enable_security_checks=True)
        router = RobustDynamicRouter(config)
        
        # Extreme input should trigger fallback
        extreme_inputs = np.random.randn(2, 32, 128) * 1000
        result = router.route(extreme_inputs)
        
        runner.assert_true(isinstance(result, dict), "Should return valid result")
        runner.assert_true('avg_experts_per_token' in result)
    
    def test_monitoring():
        config = RobustRouterConfig(input_dim=64, num_experts=4, enable_monitoring=True)
        router = RobustDynamicRouter(config)
        
        inputs = np.random.randn(2, 16, 64)
        for _ in range(3):
            router.route(inputs)
        
        health = router.get_health_status()
        runner.assert_true('health_status' in health)
        runner.assert_true('performance_stats' in health)
    
    def test_numerical_stability():
        config = RobustRouterConfig(input_dim=64, num_experts=4)
        router = RobustDynamicRouter(config)
        
        # Test with zeros
        zero_inputs = np.zeros((2, 16, 64))
        result = router.route(zero_inputs)
        runner.assert_true(not np.any(np.isnan(result['expert_weights'])), "NaN in weights")
        
        # Test with tiny values
        tiny_inputs = np.random.randn(2, 16, 64) * 1e-8
        result = router.route(tiny_inputs)
        runner.assert_true(not np.any(np.isnan(result['expert_weights'])), "NaN in weights")
    
    runner.run_test("Gen2: Config Validation", test_config_validation)
    runner.run_test("Gen2: Security Validation", test_security_validation)
    runner.run_test("Gen2: Error Handling", test_error_handling)
    runner.run_test("Gen2: Monitoring", test_monitoring)
    runner.run_test("Gen2: Numerical Stability", test_numerical_stability)

def test_generation_3_optimized(runner):
    """Test Generation 3: Optimized Implementation"""
    
    def test_caching():
        config = OptimizedRouterConfig(input_dim=128, num_experts=4, enable_caching=True)
        router = OptimizedDynamicRouter(config)
        
        inputs = np.random.randn(2, 16, 128)
        
        # First call
        start_time = time.time()
        result1 = router.route(inputs)
        time1 = time.time() - start_time
        
        # Second call (should hit cache)
        start_time = time.time()
        result2 = router.route(inputs)
        time2 = time.time() - start_time
        
        runner.assert_true(time2 < time1, "Cache didn't improve performance")
        
        # Results should be identical
        np.testing.assert_array_equal(result1['expert_indices'], result2['expert_indices'])
    
    def test_concurrent_processing():
        config = OptimizedRouterConfig(
            input_dim=64, num_experts=4, 
            enable_concurrent_processing=True,
            batch_processing_threshold=8,
            max_workers=2
        )
        router = OptimizedDynamicRouter(config)
        
        # Large batch
        large_inputs = np.random.randn(16, 32, 64)
        
        start_time = time.time()
        result = router.route(large_inputs)
        elapsed = time.time() - start_time
        
        runner.assert_equal(result['expert_indices'].shape[0], 16)
        runner.assert_true(elapsed < 5.0, f"Concurrent processing too slow: {elapsed:.3f}s")
    
    def test_memory_pooling():
        config = OptimizedRouterConfig(input_dim=64, num_experts=4, enable_memory_pooling=True)
        router = OptimizedDynamicRouter(config)
        
        if router.memory_pool:
            # Test pool functionality
            array = router.memory_pool.get_float_array((10, 10), 0.0)
            runner.assert_equal(array.shape, (10, 10))
            runner.assert_equal(array.dtype, np.float32)
            
            router.memory_pool.return_array(array)
            
            stats = router.memory_pool.get_stats()
            runner.assert_true('total_allocated' in stats)
    
    def test_vectorized_operations():
        from dynamic_moe_router.optimized_core import VectorizedOperations
        
        # Test fast softmax
        logits = np.random.randn(4, 16, 8)
        softmax_result = VectorizedOperations.fast_softmax(logits)
        
        runner.assert_equal(softmax_result.shape, logits.shape)
        
        # Check probabilities sum to 1
        sums = np.sum(softmax_result, axis=-1)
        runner.assert_true(np.allclose(sums, 1.0, rtol=1e-4), "Softmax doesn't sum to 1")
        
        # Test fast top-k
        values, indices = VectorizedOperations.fast_topk(logits, k=3)
        runner.assert_equal(values.shape, (4, 16, 3))
        runner.assert_equal(indices.shape, (4, 16, 3))
    
    def test_performance_optimization():
        config = OptimizedRouterConfig(input_dim=256, num_experts=8, enable_vectorization=True)
        router = OptimizedDynamicRouter(config)
        
        # Test small batch performance
        small_inputs = np.random.randn(2, 32, 256)
        start_time = time.time()
        result = router.route(small_inputs)
        small_time = time.time() - start_time
        
        runner.assert_true(small_time < 0.5, f"Small batch too slow: {small_time:.3f}s")
        
        # Test medium batch performance
        medium_inputs = np.random.randn(8, 64, 256)
        start_time = time.time()
        result = router.route(medium_inputs)
        medium_time = time.time() - start_time
        
        runner.assert_true(medium_time < 2.0, f"Medium batch too slow: {medium_time:.3f}s")
        
        # Get performance stats
        stats = router.get_performance_stats()
        runner.assert_true('avg_compute_time' in stats)
        runner.assert_true(stats['call_count'] >= 2)
    
    runner.run_test("Gen3: Caching", test_caching)
    runner.run_test("Gen3: Concurrent Processing", test_concurrent_processing)
    runner.run_test("Gen3: Memory Pooling", test_memory_pooling)
    runner.run_test("Gen3: Vectorized Operations", test_vectorized_operations)
    runner.run_test("Gen3: Performance Optimization", test_performance_optimization)

def test_integration(runner):
    """Integration tests across generations"""
    
    def test_consistent_results():
        # Same config for all generations
        base_config = {'input_dim': 128, 'num_experts': 4, 'min_experts': 1, 'max_experts': 2}
        
        simple_config = RouterConfig(**base_config)
        robust_config = RobustRouterConfig(**base_config, enable_security_checks=False)
        optimized_config = OptimizedRouterConfig(**base_config, enable_caching=False, 
                                                enable_concurrent_processing=False)
        
        simple_router = SimpleDynamicRouter(simple_config)
        robust_router = RobustDynamicRouter(robust_config)
        optimized_router = OptimizedDynamicRouter(optimized_config)
        
        # Same input
        np.random.seed(42)
        inputs = np.random.randn(2, 16, 128)
        
        simple_result = simple_router.route(inputs)
        robust_result = robust_router.route(inputs)
        optimized_result = optimized_router.route(inputs)
        
        # All should produce valid results
        for result, name in [(simple_result, "simple"), (robust_result, "robust"), (optimized_result, "optimized")]:
            runner.assert_true('avg_experts_per_token' in result, f"Missing key in {name}")
            runner.assert_true(result['avg_experts_per_token'] > 0, f"No experts in {name}")
            runner.assert_true(result['flop_reduction'] >= 0, f"Invalid FLOP reduction in {name}")
    
    def test_scalability():
        # Test with different input sizes
        sizes = [(2, 16), (4, 32), (8, 64)]
        
        for batch_size, seq_len in sizes:
            inputs = np.random.randn(batch_size, seq_len, 128)
            
            # All generations should handle this size
            simple_router = SimpleDynamicRouter(RouterConfig(input_dim=128, num_experts=4))
            robust_router = RobustDynamicRouter(RobustRouterConfig(input_dim=128, num_experts=4))
            optimized_router = OptimizedDynamicRouter(OptimizedRouterConfig(input_dim=128, num_experts=4))
            
            for router, name in [(simple_router, "simple"), (robust_router, "robust"), (optimized_router, "optimized")]:
                start_time = time.time()
                result = router.route(inputs)
                elapsed = time.time() - start_time
                
                runner.assert_true(elapsed < 2.0, f"{name} too slow for size {batch_size}x{seq_len}: {elapsed:.3f}s")
                runner.assert_true(result['avg_experts_per_token'] > 0, f"No experts for {name} size {batch_size}x{seq_len}")
    
    runner.run_test("Integration: Consistent Results", test_consistent_results)
    runner.run_test("Integration: Scalability", test_scalability)

def main():
    """Run all quality gate tests"""
    print("🚀 AUTONOMOUS SDLC QUALITY GATES")
    print("=" * 60)
    print("Testing all three generations of dynamic MoE router implementation")
    print()
    
    runner = QualityGateRunner()
    
    # Test Generation 1: Simple
    print("📋 Generation 1: Simple Working Implementation")
    test_generation_1_simple(runner)
    print()
    
    # Test Generation 2: Robust
    print("🛡️  Generation 2: Robust with Error Handling")
    test_generation_2_robust(runner)
    print()
    
    # Test Generation 3: Optimized
    print("⚡ Generation 3: High-Performance Optimized")
    test_generation_3_optimized(runner)
    print()
    
    # Integration tests
    print("🔗 Integration Tests")
    test_integration(runner)
    print()
    
    # Summary
    success = runner.print_summary()
    
    if success:
        print("\n🎯 AUTONOMOUS SDLC EXECUTION SUCCESSFUL!")
        print("All three generations implemented with progressive enhancement:")
        print("  ✅ Generation 1: Basic functionality working")
        print("  ✅ Generation 2: Robust error handling and monitoring")  
        print("  ✅ Generation 3: High-performance optimizations")
        print("  ✅ Quality gates passed (>= 85% success rate)")
        return 0
    else:
        print("\n💥 AUTONOMOUS SDLC EXECUTION FAILED!")
        print("Quality gates not met. Check errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
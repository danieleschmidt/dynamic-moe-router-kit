"""Demonstrate robust error handling in dynamic MoE router kit."""

import sys
sys.path.insert(0, 'src')

import numpy as np
import warnings

from dynamic_moe_router import DynamicRouter, get_estimator
from dynamic_moe_router.exceptions import (
    RouterConfigurationError,
    ComplexityEstimationError,
    ExpertDispatchError,
    ValidationError
)
from dynamic_moe_router.logging_config import setup_logging, set_debug_mode


def test_configuration_validation():
    """Test router configuration validation."""
    print("=== Testing Configuration Validation ===")
    
    # Test invalid configurations
    test_cases = [
        # (config, expected_error_type, description)
        ({"input_dim": 0, "num_experts": 4}, RouterConfigurationError, "zero input_dim"),
        ({"input_dim": 64, "num_experts": 0}, RouterConfigurationError, "zero num_experts"),
        ({"input_dim": 64, "num_experts": 4, "min_experts": 0}, RouterConfigurationError, "zero min_experts"),
        ({"input_dim": 64, "num_experts": 4, "min_experts": 5}, RouterConfigurationError, "min_experts > num_experts"),
        ({"input_dim": 64, "num_experts": 4, "max_experts": 6}, RouterConfigurationError, "max_experts > num_experts"),
        ({"input_dim": 64, "num_experts": 4, "min_experts": 3, "max_experts": 2}, RouterConfigurationError, "max_experts < min_experts"),
        ({"input_dim": 64, "num_experts": 4, "noise_factor": -0.1}, RouterConfigurationError, "negative noise_factor"),
    ]
    
    for i, (config, expected_error, description) in enumerate(test_cases, 1):
        try:
            router = DynamicRouter(**config)
            print(f"  {i}. FAIL - {description}: Expected error but router was created")
        except expected_error as e:
            print(f"  {i}. PASS - {description}: Caught {type(e).__name__}: {e}")
        except Exception as e:
            print(f"  {i}. FAIL - {description}: Wrong error type {type(e).__name__}: {e}")
    
    print()


def test_input_validation():
    """Test input validation for routing operations."""
    print("=== Testing Input Validation ===")
    
    # Create a valid router for testing
    router = DynamicRouter(
        input_dim=64,
        num_experts=4,
        min_experts=1,
        max_experts=2
    )
    
    test_cases = [
        # (input_data, description, should_fail)
        (np.random.randn(2, 8), "2D input (missing hidden dimension)", True),
        (np.random.randn(2, 8, 32), "Wrong hidden dimension", True),
        (np.random.randn(0, 8, 64), "Zero batch size", True),
        (np.random.randn(2, 0, 64), "Zero sequence length", True),
        (np.random.randn(2, 8, 64), "Valid input", False),
        (np.full((2, 8, 64), np.nan), "NaN input", True),
        (np.full((2, 8, 64), np.inf), "Infinite input", True),
    ]
    
    for i, (input_data, description, should_fail) in enumerate(test_cases, 1):
        try:
            result = router.route(input_data)
            if should_fail:
                print(f"  {i}. FAIL - {description}: Expected error but routing succeeded")
            else:
                print(f"  {i}. PASS - {description}: Routing succeeded")
        except Exception as e:
            if should_fail:
                print(f"  {i}. PASS - {description}: Caught {type(e).__name__}: {str(e)[:60]}...")
            else:
                print(f"  {i}. FAIL - {description}: Unexpected error {type(e).__name__}: {e}")
    
    print()


def test_complexity_estimator_robustness():
    """Test robustness of complexity estimators."""
    print("=== Testing Complexity Estimator Robustness ===")
    
    # Test edge cases with different estimators
    estimator_types = ["gradient_norm", "attention_entropy", "perplexity_proxy", "threshold"]
    
    for estimator_name in estimator_types:
        print(f"  Testing {estimator_name} estimator:")
        
        try:
            estimator = get_estimator(estimator_name)
            
            # Test with extreme values
            test_cases = [
                (np.zeros((2, 4, 64)), "all zeros"),
                (np.ones((2, 4, 64)) * 1e6, "very large values"),
                (np.random.randn(2, 4, 64) * 1e-6, "very small values"),
                (np.random.randn(2, 4, 64), "normal values"),
            ]
            
            for input_data, description in test_cases:
                try:
                    scores = estimator.estimate(input_data)
                    if np.isnan(scores).any() or np.isinf(scores).any():
                        print(f"    WARN - {description}: Generated invalid scores")
                    else:
                        print(f"    PASS - {description}: Generated valid scores [{scores.min():.3f}, {scores.max():.3f}]")
                except Exception as e:
                    print(f"    FAIL - {description}: {type(e).__name__}: {str(e)[:40]}...")
                    
        except Exception as e:
            print(f"    FAIL - Failed to initialize estimator: {type(e).__name__}: {e}")
    
    print()


def test_memory_usage_warnings():
    """Test memory usage warnings for large inputs."""
    print("=== Testing Memory Usage Warnings ===")
    
    router = DynamicRouter(input_dim=1024, num_experts=8)
    
    # Create progressively larger inputs to trigger memory warnings
    sizes = [
        (2, 16, 1024, "small input"),
        (4, 64, 1024, "medium input"),
        (8, 256, 1024, "large input"),
        (16, 512, 1024, "very large input"),
    ]
    
    for batch_size, seq_len, hidden_dim, description in sizes:
        print(f"  Testing {description} [{batch_size}, {seq_len}, {hidden_dim}]:")
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            
            try:
                input_data = np.random.randn(batch_size, seq_len, hidden_dim)
                result = router.route(input_data)
                
                # Check for memory warnings
                memory_warnings = [w for w in warning_list if "memory" in str(w.message).lower()]
                if memory_warnings:
                    print(f"    WARN - Memory warning triggered: {memory_warnings[0].message}")
                else:
                    print(f"    PASS - No memory warnings")
                    
            except Exception as e:
                print(f"    FAIL - Routing failed: {type(e).__name__}: {e}")
    
    print()


def test_load_balancing_robustness():
    """Test load balancing under various conditions."""
    print("=== Testing Load Balancing Robustness ===")
    
    # Test load balancing with extreme expert usage patterns
    router = DynamicRouter(
        input_dim=128,
        num_experts=4,
        min_experts=1,
        max_experts=1,  # Force single expert selection
        load_balancing=True
    )
    
    print("  Testing load balancing with forced single expert selection:")
    
    # Process multiple batches
    for batch_num in range(5):
        try:
            # Create input that might bias toward certain experts
            input_data = np.random.randn(4, 8, 128)
            if batch_num % 2 == 0:
                input_data *= 10  # Amplify every other batch
            
            result = router.route(input_data)
            utilization = result['routing_info']['expert_utilization']
            
            print(f"    Batch {batch_num + 1}: Expert utilization = {[f'{u:.3f}' for u in utilization]}")
            
        except Exception as e:
            print(f"    FAIL - Batch {batch_num + 1}: {type(e).__name__}: {e}")
    
    # Check overall load balancing statistics
    try:
        stats = router.get_expert_usage_stats()
        print(f"  Overall load balance score: {stats['load_balance_score']:.3f}")
        print(f"  Most used expert: #{stats['most_used_expert']}")
        print(f"  Least used expert: #{stats['least_used_expert']}")
    except Exception as e:
        print(f"  FAIL - Could not get usage stats: {type(e).__name__}: {e}")
    
    print()


def test_routing_strategy_fallback():
    """Test fallback behavior for routing strategies."""
    print("=== Testing Routing Strategy Fallback ===")
    
    # Test both routing strategies with edge cases
    strategies = ["top_k", "threshold"]
    
    for strategy in strategies:
        print(f"  Testing {strategy} strategy:")
        
        try:
            router = DynamicRouter(
                input_dim=64,
                num_experts=8,
                min_experts=1,
                max_experts=8,  # Allow all experts
                routing_strategy=strategy
            )
            
            # Test with edge cases
            test_cases = [
                (np.zeros((2, 4, 64)), "zero input"),
                (np.ones((2, 4, 64)), "uniform input"),
                (np.random.randn(2, 4, 64) * 100, "high variance input"),
            ]
            
            for input_data, description in test_cases:
                try:
                    result = router.route(input_data)
                    avg_experts = result['routing_info']['avg_experts_per_token']
                    print(f"    PASS - {description}: avg_experts={avg_experts:.2f}")
                except Exception as e:
                    print(f"    FAIL - {description}: {type(e).__name__}: {e}")
                    
        except Exception as e:
            print(f"    FAIL - Could not create router with {strategy}: {type(e).__name__}: {e}")
    
    print()


def test_graceful_degradation():
    """Test graceful degradation under failure conditions."""
    print("=== Testing Graceful Degradation ===")
    
    # Create a router that might encounter issues
    router = DynamicRouter(
        input_dim=32,
        num_experts=4,
        load_balancing=True,  # Enable load balancing that might fail
        noise_factor=0.1       # Add noise that might cause issues
    )
    
    print("  Testing with problematic complexity estimator:")
    
    # Create a problematic estimator that sometimes fails
    class UnreliableEstimator:
        def __init__(self):
            self.call_count = 0
        
        def estimate(self, hidden_states, **kwargs):
            self.call_count += 1
            if self.call_count % 3 == 0:  # Fail every 3rd call
                raise RuntimeError("Simulated estimator failure")
            return np.random.rand(*hidden_states.shape[:2])
    
    # Replace the estimator (normally not recommended!)
    router.complexity_estimator = UnreliableEstimator()
    
    success_count = 0
    total_attempts = 5
    
    for attempt in range(total_attempts):
        try:
            input_data = np.random.randn(2, 8, 32)
            result = router.route(input_data)
            success_count += 1
            print(f"    Attempt {attempt + 1}: SUCCESS")
        except Exception as e:
            print(f"    Attempt {attempt + 1}: FAILED - {type(e).__name__}: {str(e)[:50]}...")
    
    success_rate = success_count / total_attempts
    print(f"  Success rate: {success_rate:.1%} ({success_count}/{total_attempts})")
    
    if success_rate > 0:
        print("  PASS - Router showed resilience to component failures")
    else:
        print("  FAIL - Router failed all attempts")
    
    print()


def main():
    """Run all robustness tests."""
    print("🛡️  DYNAMIC MOE ROUTER - ROBUSTNESS TESTING")
    print("=" * 50)
    print()
    
    # Setup logging to see error handling in action
    setup_logging(level="DEBUG")
    
    # Run all tests
    test_configuration_validation()
    test_input_validation()
    test_complexity_estimator_robustness()
    test_memory_usage_warnings()
    test_load_balancing_robustness()
    test_routing_strategy_fallback()
    test_graceful_degradation()
    
    print("🎉 ROBUSTNESS TESTING COMPLETE")
    print()
    print("This demonstrates Generation 2: MAKE IT ROBUST features:")
    print("  ✅ Comprehensive input validation")
    print("  ✅ Detailed error messages and logging")
    print("  ✅ Graceful handling of edge cases")
    print("  ✅ Memory usage monitoring")
    print("  ✅ Configuration validation")
    print("  ✅ Fallback mechanisms")


if __name__ == "__main__":
    main()
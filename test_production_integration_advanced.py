"""Advanced integration test for production MoE router with all optimizations."""

import numpy as np
import time
import logging
from typing import Dict, Any

# Import our modules
from src.dynamic_moe_router.production_fixed import ProductionRouter, ProductionConfig, RouterFactory
from src.dynamic_moe_router.enhanced_resilience import ResilientRouter, CircuitConfig, RetryPolicy
from src.dynamic_moe_router.high_performance_scaling import PerformanceOptimizer, PerformanceConfig
from src.dynamic_moe_router.global_deployment import GlobalDeploymentManager, GlobalConfig, Region

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_production_router_basic():
    """Test basic production router functionality."""
    logger.info("Testing basic production router...")
    
    config = ProductionConfig(
        input_dim=512,
        num_experts=8,
        min_experts=1,
        max_experts=4
    )
    
    router = ProductionRouter(config)
    
    # Test routing
    hidden_states = np.random.randn(2, 10, 512).astype(np.float32)
    result = router.route(hidden_states)
    
    assert 'expert_indices' in result
    assert 'expert_weights' in result
    assert 'production_info' in result
    assert 'request_id' in result['production_info']
    
    logger.info("✓ Basic production router test passed")
    return True


def test_resilient_router():
    """Test resilient router with circuit breaker and retry."""
    logger.info("Testing resilient router...")
    
    # Create base router
    config = ProductionConfig(input_dim=256, num_experts=4)
    base_router = ProductionRouter(config)
    
    # Create resilient wrapper
    circuit_config = CircuitConfig(failure_threshold=3, recovery_timeout=1.0)
    retry_policy = RetryPolicy(max_retries=2, base_delay=0.01)
    
    resilient_router = ResilientRouter(
        base_router=base_router,
        circuit_config=circuit_config,
        retry_policy=retry_policy,
        fallback_strategy="load_balance"
    )
    
    # Test normal operation
    hidden_states = np.random.randn(1, 5, 256).astype(np.float32)
    result = resilient_router.route(hidden_states)
    
    assert 'expert_indices' in result
    stats = resilient_router.get_resilience_stats()
    assert 'circuit_breaker' in stats
    
    logger.info("✓ Resilient router test passed")
    return True


def test_performance_optimizer():
    """Test performance optimization features."""
    logger.info("Testing performance optimizer...")
    
    config = PerformanceConfig(
        cache_size=100,
        enable_result_caching=True,
        enable_dynamic_batching=True,
        enable_vectorization=True
    )
    
    optimizer = PerformanceOptimizer(config)
    
    # Create mock router function
    def mock_router(hidden_states):
        return {
            'expert_indices': np.random.randint(0, 4, size=(hidden_states.shape[0], hidden_states.shape[1], 2)),
            'expert_weights': np.random.random((hidden_states.shape[0], hidden_states.shape[1], 2)),
            'routing_info': {'avg_experts_per_token': 2.0}
        }
    
    # Test caching
    cached_router = optimizer.optimized_routing_cache(mock_router)
    
    test_input = np.random.randn(1, 3, 128).astype(np.float32)
    result1 = cached_router(test_input)
    result2 = cached_router(test_input)  # Should hit cache
    
    # Test batch processing
    batch_requests = [np.random.randn(1, 3, 128) for _ in range(5)]
    batch_results = optimizer.batch_route(batch_requests, mock_router)
    
    assert len(batch_results) == 5
    
    stats = optimizer.get_performance_stats()
    assert 'total_requests' in stats
    assert 'cache_stats' in stats
    
    optimizer.shutdown()
    logger.info("✓ Performance optimizer test passed")
    return True


def test_global_deployment():
    """Test global deployment features."""
    logger.info("Testing global deployment...")
    
    global_config = GlobalConfig(
        primary_region=Region.US_EAST,
        secondary_regions=[Region.EU_WEST, Region.ASIA_PACIFIC],
        enable_i18n=True,
        enable_gdpr_compliance=True
    )
    
    global_manager = GlobalDeploymentManager(global_config)
    
    # Test request processing with compliance
    request_data = {
        'user_id': 'test123',
        'data': np.random.randn(2, 5, 256).astype(np.float32)
    }
    
    processed_request = global_manager.process_global_request(
        request_data,
        user_region='eu-west-1',
        user_language='de'
    )
    
    assert '_compliance_info' in processed_request
    assert '_global_info' in processed_request
    
    # Test localization
    message = global_manager.get_localized_message(
        'router_initialized',
        language='es'
    )
    assert isinstance(message, str)
    assert len(message) > 0
    
    # Test health check
    health = global_manager.health_check_global()
    assert 'global_deployment_status' in health
    assert 'load_balancing_stats' in health
    
    logger.info("✓ Global deployment test passed")
    return True


def test_end_to_end_integration():
    """Test complete end-to-end integration."""
    logger.info("Testing end-to-end integration...")
    
    # Create global deployment
    global_config = GlobalConfig(
        primary_region=Region.US_EAST,
        enable_gdpr_compliance=True,
        supported_languages=['en', 'es', 'de']
    )
    global_manager = GlobalDeploymentManager(global_config)
    
    # Create performance-optimized production router
    perf_config = PerformanceConfig(
        cache_size=50,
        enable_result_caching=True,
        enable_dynamic_batching=True
    )
    optimizer = PerformanceOptimizer(perf_config)
    
    # Create production router
    router_config = ProductionConfig(
        input_dim=384,
        num_experts=6,
        min_experts=1,
        max_experts=3
    )
    router = ProductionRouter(router_config)
    
    # Wrap with resilience
    resilient_router = ResilientRouter(
        base_router=router,
        fallback_strategy="load_balance"
    )
    
    # Apply performance optimization
    optimized_route = optimizer.optimized_routing_cache(resilient_router.route)
    
    # Test complete pipeline
    for i in range(5):
        # Process global request
        request_data = {
            'batch_id': f'batch_{i}',
            'hidden_states': np.random.randn(2, 8, 384).astype(np.float32)
        }
        
        processed_request = global_manager.process_global_request(
            request_data,
            user_region='eu-west-1' if i % 2 == 0 else 'us-east-1',
            user_language='de' if i % 2 == 0 else 'en'
        )
        
        # Route with optimization
        result = optimized_route(processed_request['hidden_states'])
        
        # Verify result structure
        assert 'expert_indices' in result
        assert 'expert_weights' in result
        assert 'routing_info' in result
        
        logger.info(f"Processed batch {i}: {result['routing_info']['avg_experts_per_token']:.2f} experts/token")
    
    # Get comprehensive stats
    perf_stats = optimizer.get_performance_stats()
    resilience_stats = resilient_router.get_resilience_stats()
    global_health = global_manager.health_check_global()
    
    logger.info(f"Cache hit rate: {perf_stats['cache_hit_rate']:.2%}")
    logger.info(f"Circuit breaker state: {resilience_stats['circuit_breaker']['state']}")
    logger.info(f"Global deployment status: {global_health['global_deployment_status']}")
    
    optimizer.shutdown()
    logger.info("✓ End-to-end integration test passed")
    return True


def test_router_factory():
    """Test router factory patterns."""
    logger.info("Testing router factory...")
    
    # Test optimized routers
    inference_router = RouterFactory.create_optimized_for_inference()
    training_router = RouterFactory.create_optimized_for_training()
    
    # Test both routers
    test_input = np.random.randn(1, 5, 768).astype(np.float32)
    
    inference_result = inference_router.route(test_input)
    assert 'expert_indices' in inference_result
    
    training_input = np.random.randn(1, 5, 1024).astype(np.float32)
    training_result = training_router.route(training_input)
    assert 'expert_indices' in training_result
    
    logger.info("✓ Router factory test passed")
    return True


def run_performance_benchmark():
    """Run performance benchmark."""
    logger.info("Running performance benchmark...")
    
    # Create optimized router
    config = ProductionConfig(input_dim=512, num_experts=8)
    router = ProductionRouter(config)
    
    # Benchmark parameters
    batch_sizes = [1, 4, 16, 32]
    sequence_lengths = [10, 50, 100]
    num_iterations = 10
    
    results = {}
    
    for batch_size in batch_sizes:
        for seq_len in sequence_lengths:
            times = []
            
            for _ in range(num_iterations):
                hidden_states = np.random.randn(batch_size, seq_len, 512).astype(np.float32)
                
                start_time = time.time()
                result = router.route(hidden_states)
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = (batch_size * seq_len) / avg_time
            
            results[f"batch_{batch_size}_seq_{seq_len}"] = {
                'avg_time_ms': avg_time * 1000,
                'std_time_ms': std_time * 1000,
                'throughput_tokens_per_sec': throughput
            }
            
            logger.info(f"Batch {batch_size}, Seq {seq_len}: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms, "
                       f"{throughput:.1f} tokens/sec")
    
    logger.info("✓ Performance benchmark completed")
    return results


def main():
    """Run all tests and benchmarks."""
    logger.info("Starting comprehensive production integration tests...")
    
    tests = [
        test_production_router_basic,
        test_resilient_router,
        test_performance_optimizer,
        test_global_deployment,
        test_router_factory,
        test_end_to_end_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test_func.__name__} failed: {e}")
    
    logger.info(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All production integration tests passed!")
        
        # Run performance benchmark
        logger.info("\nRunning performance benchmark...")
        benchmark_results = run_performance_benchmark()
        
        logger.info("\n🚀 Production MoE Router is ready for deployment!")
        return True
    else:
        logger.error("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
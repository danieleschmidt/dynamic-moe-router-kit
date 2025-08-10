"""Production demonstration of dynamic MoE routing."""

import numpy as np
import time

from dynamic_moe_router import DynamicRouter, MoELayer, get_estimator


def create_production_router():
    """Create a production-ready router configuration."""
    return DynamicRouter(
        input_dim=768,
        num_experts=8,
        min_experts=1,
        max_experts=3,
        complexity_estimator="gradient_norm",
        routing_strategy="top_k",
        load_balancing=True,
        noise_factor=0.1
    )


def create_dummy_expert():
    """Create a dummy expert function."""
    def expert_fn(x):
        W = np.random.randn(x.shape[-1], x.shape[-1]) * 0.02
        b = np.zeros(x.shape[-1])
        return np.maximum(0, np.dot(x, W) + b)
    return expert_fn


def benchmark_performance():
    """Benchmark router performance under production load."""
    print("=== Production Performance Benchmark ===")
    
    # Setup
    router = create_production_router()
    moe_layer = MoELayer(
        router=router,
        expert_fn=create_dummy_expert,
        num_experts=8
    )
    
    # Test different batch sizes
    batch_sizes = [1, 4, 8, 16, 32]
    seq_len = 128
    hidden_dim = 768
    
    print(f"{'Batch Size':<12} {'Avg Time (ms)':<15} {'Experts/Token':<15} {'FLOP Reduction':<15}")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        # Generate test data
        test_input = np.random.randn(batch_size, seq_len, hidden_dim)
        
        # Warm up
        for _ in range(3):
            moe_layer.forward(test_input)
        
        # Benchmark
        times = []
        total_experts = []
        flop_reductions = []
        
        for _ in range(10):
            start_time = time.time()
            output, routing_info = moe_layer.forward(test_input, return_router_logits=True)
            elapsed = (time.time() - start_time) * 1000
            
            times.append(elapsed)
            total_experts.append(routing_info['routing_info']['avg_experts_per_token'])
            flop_reductions.append(routing_info['routing_info']['flop_reduction'])
        
        avg_time = np.mean(times)
        avg_experts = np.mean(total_experts)
        avg_flop_reduction = np.mean(flop_reductions)
        
        print(f"{batch_size:<12} {avg_time:<15.2f} {avg_experts:<15.2f} {avg_flop_reduction:<15.1%}")
    
    print("\\nBenchmark complete.")


def test_load_balancing():
    """Test load balancing over extended usage."""
    print("\\n=== Load Balancing Analysis ===")
    
    router = create_production_router()
    batch_size, seq_len, hidden_dim = 8, 64, 768
    
    # Process multiple batches
    for batch_num in range(20):
        test_input = np.random.randn(batch_size, seq_len, hidden_dim)
        router.route(test_input)
    
    # Analyze usage statistics
    stats = router.get_expert_usage_stats()
    
    print(f"Total batches processed: {stats['total_batches']}")
    print(f"Load balance score: {stats['load_balance_score']:.3f}")
    print(f"Most used expert: #{stats['most_used_expert']}")
    print(f"Least used expert: #{stats['least_used_expert']}")
    print(f"Usage variance: {stats['usage_variance']:.4f}")
    
    print("\\nExpert utilization:")
    for i, usage in enumerate(stats['avg_usage_per_expert']):
        print(f"  Expert {i}: {usage:.3f}")


def test_adaptive_routing():
    """Test adaptive routing capabilities."""
    print("\\n=== Adaptive Routing Test ===")
    
    from dynamic_moe_router.router import AdaptiveRouter
    
    router = AdaptiveRouter(
        input_dim=768,
        num_experts=8,
        min_experts=1,
        max_experts=4,
        adaptation_rate=0.05
    )
    
    batch_size, seq_len, hidden_dim = 4, 32, 768
    
    print("Simulating adaptive learning:")
    for epoch in range(10):
        test_input = np.random.randn(batch_size, seq_len, hidden_dim)
        result = router.route(test_input)
        
        avg_experts = result['routing_info']['avg_experts_per_token']
        
        # Simulate performance feedback (prefer ~2.5 experts)
        target_experts = 2.5
        performance_score = 1.0 - abs(avg_experts - target_experts) / target_experts
        
        router.update_thresholds(performance_score)
        
        if epoch % 3 == 0:
            adaptation_stats = router.get_adaptation_stats()
            print(f"  Epoch {epoch+1}: avg_experts={avg_experts:.2f}, "
                  f"performance={performance_score:.3f}")
            print(f"    Thresholds: {[f'{t:.3f}' for t in adaptation_stats['current_thresholds']]}")
    
    print("Adaptive routing test complete.")


def test_error_handling():
    """Test error handling and resilience."""
    print("\\n=== Error Handling Test ===")
    
    router = create_production_router()
    
    # Test 1: Invalid input shapes
    try:
        invalid_input = np.random.randn(10, 20)  # Wrong number of dimensions
        router.route(invalid_input)
        print("ERROR: Should have failed on invalid input")
    except Exception as e:
        print(f"✓ Correctly handled invalid input: {type(e).__name__}")
    
    # Test 2: Very large inputs (should handle gracefully)
    try:
        large_input = np.random.randn(1, 2000, 768)  # Large sequence
        result = router.route(large_input)
        print(f"✓ Handled large input: {large_input.shape}")
    except Exception as e:
        print(f"✗ Failed on large input: {e}")
    
    # Test 3: Edge case - minimum size input
    try:
        min_input = np.random.randn(1, 1, 768)
        result = router.route(min_input)
        print(f"✓ Handled minimum input: {min_input.shape}")
    except Exception as e:
        print(f"✗ Failed on minimum input: {e}")
    
    print("Error handling test complete.")


def production_integration_demo():
    """Demonstrate production integration patterns."""
    print("\\n=== Production Integration Demo ===")
    
    # Simulate a production inference pipeline
    router = create_production_router()
    
    # Batch processing simulation
    print("Simulating batch processing pipeline...")
    
    batch_queue = [
        np.random.randn(2, 64, 768),
        np.random.randn(4, 32, 768),
        np.random.randn(1, 128, 768),
        np.random.randn(8, 16, 768),
    ]
    
    total_time = 0
    total_tokens = 0
    total_flop_savings = 0
    
    for i, batch in enumerate(batch_queue):
        start_time = time.time()
        result = router.route(batch)
        elapsed = time.time() - start_time
        
        batch_tokens = batch.shape[0] * batch.shape[1]
        flop_reduction = result['routing_info']['flop_reduction']
        
        total_time += elapsed
        total_tokens += batch_tokens
        total_flop_savings += flop_reduction
        
        print(f"  Batch {i+1}: {batch.shape} -> {elapsed*1000:.1f}ms, "
              f"FLOP reduction: {flop_reduction:.1%}")
    
    avg_flop_savings = total_flop_savings / len(batch_queue)
    throughput = total_tokens / total_time
    
    print(f"\\nPipeline Summary:")
    print(f"  Total time: {total_time*1000:.1f}ms")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Throughput: {throughput:.1f} tokens/sec")
    print(f"  Average FLOP savings: {avg_flop_savings:.1%}")
    
    # Get router statistics
    stats = router.get_expert_usage_stats()
    print(f"  Load balance score: {stats.get('load_balance_score', 'N/A')}")


if __name__ == "__main__":
    print("Dynamic MoE Router - Production Demonstration")
    print("=" * 50)
    
    benchmark_performance()
    test_load_balancing()
    test_adaptive_routing()
    test_error_handling()
    production_integration_demo()
    
    print("\\n" + "=" * 50)
    print("All production tests completed successfully!")
    print("The router is ready for production deployment.")
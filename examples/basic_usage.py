"""Basic usage examples for dynamic MoE router kit."""

import numpy as np
from dynamic_moe_router import (
    DynamicRouter,
    MoELayer, 
    FLOPProfiler,
    get_estimator
)


def create_dummy_expert():
    """Create a dummy expert function for demonstration."""
    def expert_fn(x):
        # Simple linear transformation + ReLU
        W = np.random.randn(x.shape[-1], x.shape[-1]) * 0.02
        b = np.zeros(x.shape[-1])
        return np.maximum(0, np.dot(x, W) + b)
    return expert_fn


def basic_dynamic_routing_example():
    """Demonstrate basic dynamic routing functionality."""
    print("=== Basic Dynamic Routing Example ===")
    
    # Configuration
    batch_size, seq_len, hidden_dim = 4, 32, 768
    num_experts = 8
    min_experts, max_experts = 1, 4
    
    # Create router
    router = DynamicRouter(
        input_dim=hidden_dim,
        num_experts=num_experts,
        min_experts=min_experts,
        max_experts=max_experts,
        complexity_estimator="gradient_norm",
        routing_strategy="top_k"
    )
    
    # Create MoE layer
    moe_layer = MoELayer(
        router=router,
        expert_fn=create_dummy_expert,
        num_experts=num_experts
    )
    
    # Generate dummy input
    hidden_states = np.random.randn(batch_size, seq_len, hidden_dim)
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Number of experts: {num_experts}")
    print(f"Expert range: {min_experts}-{max_experts} per token")
    
    # Forward pass with profiling
    with FLOPProfiler() as profiler:
        output, routing_info = moe_layer.forward(
            hidden_states, return_router_logits=True
        )
        
        # Profile the routing decision
        profiler.profile_routing_decision(
            hidden_states.shape, num_experts, routing_info
        )
    
    print(f"Output shape: {output.shape}")
    print(f"Average experts per token: {routing_info['routing_info']['avg_experts_per_token']:.2f}")
    print(f"FLOP reduction: {routing_info['routing_info']['flop_reduction']:.1%}")
    print(f"Expert utilization: {routing_info['routing_info']['expert_utilization'][:4]}...")
    
    # Print profiler summary
    print("\\nProfiler Summary:")
    print(profiler.summary())


def compare_complexity_estimators():
    """Compare different complexity estimation strategies."""
    print("\\n=== Complexity Estimator Comparison ===")
    
    # Test data
    batch_size, seq_len, hidden_dim = 2, 16, 512
    hidden_states = np.random.randn(batch_size, seq_len, hidden_dim)
    
    estimators = {
        "gradient_norm": get_estimator("gradient_norm"),
        "attention_entropy": get_estimator("attention_entropy"), 
        "perplexity_proxy": get_estimator("perplexity_proxy"),
        "threshold": get_estimator("threshold")
    }
    
    print(f"Input shape: {hidden_states.shape}")
    print("\\nComplexity scores by estimator:")
    
    for name, estimator in estimators.items():
        scores = estimator.estimate(hidden_states)
        print(f"{name:18}: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}, range=[{np.min(scores):.3f}, {np.max(scores):.3f}]")


def routing_strategy_comparison():
    """Compare top-k vs threshold routing strategies."""
    print("\\n=== Routing Strategy Comparison ===")
    
    # Configuration
    batch_size, seq_len, hidden_dim = 2, 8, 256
    num_experts = 6
    hidden_states = np.random.randn(batch_size, seq_len, hidden_dim)
    
    strategies = ["top_k", "threshold"]
    
    for strategy in strategies:
        print(f"\\n{strategy.upper()} Strategy:")
        
        router = DynamicRouter(
            input_dim=hidden_dim,
            num_experts=num_experts,
            min_experts=1,
            max_experts=3,
            complexity_estimator="gradient_norm",
            routing_strategy=strategy
        )
        
        result = router.route(hidden_states, return_router_logits=True)
        
        print(f"  Average experts per token: {result['routing_info']['avg_experts_per_token']:.2f}")
        print(f"  FLOP reduction: {result['routing_info']['flop_reduction']:.1%}")
        print(f"  Expert utilization variance: {np.var(result['routing_info']['expert_utilization']):.4f}")


def adaptive_routing_example():
    """Demonstrate adaptive threshold learning."""
    print("\\n=== Adaptive Routing Example ===")
    
    from dynamic_moe_router.router import AdaptiveRouter
    
    # Create adaptive router
    router = AdaptiveRouter(
        input_dim=256,
        num_experts=4,
        min_experts=1,
        max_experts=3,
        adaptation_rate=0.05
    )
    
    # Simulate training with performance feedback
    print("Simulating adaptive threshold learning:")
    
    batch_size, seq_len, hidden_dim = 2, 8, 256
    
    for epoch in range(5):
        # Generate dummy data
        hidden_states = np.random.randn(batch_size, seq_len, hidden_dim)
        
        # Route and get results
        result = router.route(hidden_states)
        avg_experts = result['routing_info']['avg_experts_per_token']
        
        # Simulate performance score (prefer moderate expert usage)
        target_experts = 2.0
        performance_score = 1.0 - abs(avg_experts - target_experts) / target_experts
        
        # Update thresholds based on performance
        router.update_thresholds(performance_score)
        
        print(f"  Epoch {epoch+1}: avg_experts={avg_experts:.2f}, performance={performance_score:.3f}")
        print(f"    Thresholds: {router.complexity_thresholds.tolist()}")


def load_balancing_demo():
    """Demonstrate load balancing functionality."""
    print("\\n=== Load Balancing Demo ===")
    
    router = DynamicRouter(
        input_dim=128,
        num_experts=4,
        min_experts=1,
        max_experts=2,
        load_balancing=True
    )
    
    # Process multiple batches to see load balancing in action
    batch_size, seq_len, hidden_dim = 3, 6, 128
    
    print("Processing batches with load balancing enabled:")
    
    for batch_num in range(3):
        hidden_states = np.random.randn(batch_size, seq_len, hidden_dim)
        result = router.route(hidden_states)
        
        expert_util = result['routing_info']['expert_utilization']
        print(f"  Batch {batch_num+1}: Expert utilization = {[f'{u:.2f}' for u in expert_util]}")
    
    # Get overall usage statistics
    stats = router.get_expert_usage_stats()
    print(f"\\nOverall statistics:")
    print(f"  Most used expert: #{stats['most_used_expert']}")
    print(f"  Least used expert: #{stats['least_used_expert']}")
    print(f"  Load balance score: {stats['load_balance_score']:.3f}")


if __name__ == "__main__":
    # Run all examples
    basic_dynamic_routing_example()
    compare_complexity_estimators()
    routing_strategy_comparison()
    adaptive_routing_example()
    load_balancing_demo()
    
    print("\\n=== All Examples Complete ===")
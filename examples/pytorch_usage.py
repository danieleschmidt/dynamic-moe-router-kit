"""PyTorch usage examples for dynamic MoE router kit."""

import torch
import torch.nn as nn
import numpy as np

# Import dynamic MoE components
from dynamic_moe_router import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from dynamic_moe_router.torch import (
        TorchDynamicRouter,
        TorchMoELayer,
        TorchGradientNormEstimator,
        LinearExpert,
        GLUExpert,
        patch_model_with_dynamic_routing
    )
    from dynamic_moe_router import FLOPProfiler


def basic_pytorch_example():
    """Demonstrate basic PyTorch dynamic routing."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Please install PyTorch to run this example.")
        return
    
    print("=== Basic PyTorch Dynamic Routing Example ===")
    
    # Configuration
    batch_size, seq_len, hidden_dim = 4, 32, 768
    num_experts = 8
    min_experts, max_experts = 1, 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    
    # Create complexity estimator
    estimator = TorchGradientNormEstimator()
    
    # Create router
    router = TorchDynamicRouter(
        input_dim=hidden_dim,
        num_experts=num_experts,
        min_experts=min_experts,
        max_experts=max_experts,
        complexity_estimator=estimator,
        routing_strategy="top_k"
    ).to(device)
    
    # Expert factory function
    def expert_fn():
        return LinearExpert(hidden_dim, hidden_dim * 4).to(device)
    
    # Create MoE layer
    moe_layer = TorchMoELayer(
        router=router,
        expert_fn=expert_fn,
        num_experts=num_experts
    ).to(device)
    
    # Generate dummy input
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Number of experts: {num_experts}")
    print(f"Expert range: {min_experts}-{max_experts} per token")
    
    # Forward pass with profiling
    with FLOPProfiler() as profiler:
        with torch.no_grad():  # Inference mode
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
    print("\nProfiler Summary:")
    print(profiler.summary())
    
    return moe_layer, output


def gradient_based_training_example():
    """Demonstrate training with gradient-based complexity estimation."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping gradient-based training example.")
        return
    
    print("\n=== Gradient-Based Training Example ===")
    
    # Configuration
    batch_size, seq_len, hidden_dim = 2, 16, 256
    num_experts = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create components
    estimator = TorchGradientNormEstimator()
    router = TorchDynamicRouter(
        input_dim=hidden_dim,
        num_experts=num_experts,
        min_experts=1,
        max_experts=2,
        complexity_estimator=estimator
    ).to(device)
    
    def expert_fn():
        return GLUExpert(hidden_dim).to(device)
    
    moe_layer = TorchMoELayer(
        router=router,
        expert_fn=expert_fn,
        num_experts=num_experts
    ).to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(moe_layer.parameters(), lr=0.001)
    
    # Training loop
    print("Training with dynamic expert routing:")
    
    for epoch in range(5):
        # Generate synthetic training data
        inputs = torch.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=True)
        targets = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs, routing_info = moe_layer.forward(inputs, return_router_logits=True)
        
        # Simple MSE loss
        loss = torch.nn.functional.mse_loss(outputs, targets)
        
        # Add load balancing loss (optional)
        expert_utilization = torch.tensor(routing_info['routing_info']['expert_utilization'], device=device)
        balance_loss = torch.var(expert_utilization) * 0.01  # Small weight for load balancing
        
        total_loss = loss + balance_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        avg_experts = routing_info['routing_info']['avg_experts_per_token']
        flop_reduction = routing_info['routing_info']['flop_reduction']
        
        print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, Balance={balance_loss.item():.4f}, "
              f"Experts={avg_experts:.2f}, FLOP↓={flop_reduction:.1%}")


def attention_entropy_example():
    """Demonstrate attention entropy-based complexity estimation."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping attention entropy example.")
        return
    
    print("\n=== Attention Entropy Complexity Estimation ===")
    
    from dynamic_moe_router.torch import TorchAttentionEntropyEstimator
    
    # Configuration
    batch_size, seq_len, hidden_dim = 2, 12, 512
    num_heads = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create attention entropy estimator
    estimator = TorchAttentionEntropyEstimator(head_aggregation='mean')
    
    # Create router with attention entropy
    router = TorchDynamicRouter(
        input_dim=hidden_dim,
        num_experts=6,
        min_experts=1,
        max_experts=3,
        complexity_estimator=estimator
    ).to(device)
    
    # Generate dummy attention weights (simulating transformer attention)
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Simulate attention weights [batch, num_heads, seq_len, seq_len]
    attention_weights = torch.softmax(
        torch.randn(batch_size, num_heads, seq_len, seq_len, device=device), dim=-1
    )
    
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Route using attention entropy
    with torch.no_grad():
        routing_result = router.route(
            hidden_states, 
            return_router_logits=True,
            attention_weights=attention_weights
        )
    
    complexity_scores = routing_result['complexity_scores']
    print(f"Complexity scores shape: {complexity_scores.shape}")
    print(f"Complexity range: [{complexity_scores.min():.3f}, {complexity_scores.max():.3f}]")
    print(f"Average experts per token: {routing_result['routing_info']['avg_experts_per_token']:.2f}")
    
    # Compare with pseudo-attention (when attention weights not available)
    print("\nComparing with pseudo-attention:")
    routing_result_pseudo = router.route(hidden_states, return_router_logits=True)
    complexity_pseudo = routing_result_pseudo['complexity_scores']
    print(f"Pseudo-attention complexity range: [{complexity_pseudo.min():.3f}, {complexity_pseudo.max():.3f}]")


def model_patching_example():
    """Demonstrate patching existing models with dynamic routing."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping model patching example.")
        return
    
    print("\n=== Model Patching Example ===")
    
    # Create a simple transformer-like model
    class SimpleTransformer(nn.Module):
        def __init__(self, hidden_dim=256, num_layers=2):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    'attention': nn.MultiheadAttention(hidden_dim, 4, batch_first=True),
                    'ffn': nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 4),
                        nn.ReLU(),
                        nn.Linear(hidden_dim * 4, hidden_dim)
                    ),
                    'norm1': nn.LayerNorm(hidden_dim),
                    'norm2': nn.LayerNorm(hidden_dim)
                }) for _ in range(num_layers)
            ])
        
        def forward(self, x):
            for layer in self.layers:
                # Self-attention
                attn_out, _ = layer['attention'](x, x, x)
                x = layer['norm1'](x + attn_out)
                
                # FFN
                ffn_out = layer['ffn'](x)
                x = layer['norm2'](x + ffn_out)
            return x
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleTransformer(hidden_dim=256).to(device)
    
    print("Original model:")
    for name, _ in model.named_modules():
        if 'ffn' in name and not name.endswith(('weight', 'bias', '0', '1', '2')):
            print(f"  {name}")
    
    # Patch FFN layers with dynamic MoE
    try:
        patched_model = patch_model_with_dynamic_routing(
            model,
            target_layers=['layers.0.ffn', 'layers.1.ffn'],  # Replace FFN layers
            min_experts_ratio=0.25,
            max_experts_ratio=0.75,
            num_experts=4,
            complexity_metric="gradient_norm"
        )
        
        print("\nModel successfully patched with dynamic MoE routing!")
        
        # Test the patched model
        batch_size, seq_len, hidden_dim = 2, 16, 256
        test_input = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        
        with torch.no_grad():
            output = patched_model(test_input)
            print(f"Patched model output shape: {output.shape}")
            
    except Exception as e:
        print(f"Model patching failed: {e}")
        print("This is expected for the demonstration - patching requires specific model structures.")


def performance_comparison():
    """Compare dynamic vs static MoE performance."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping performance comparison.")
        return
    
    print("\n=== Performance Comparison ===")
    
    from dynamic_moe_router import ComparisonProfiler
    
    # Configuration
    batch_size, seq_len, hidden_dim = 4, 64, 512
    num_experts = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dynamic MoE layer
    estimator = TorchGradientNormEstimator()
    router = TorchDynamicRouter(
        input_dim=hidden_dim,
        num_experts=num_experts,
        min_experts=1,
        max_experts=3,  # Use 1-3 experts instead of all 8
        complexity_estimator=estimator
    ).to(device)
    
    def expert_fn():
        return LinearExpert(hidden_dim, hidden_dim * 4).to(device)
    
    moe_layer = TorchMoELayer(
        router=router,
        expert_fn=expert_fn,
        num_experts=num_experts
    ).to(device)
    
    # Test input
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Create comparison profiler
    comparison_profiler = ComparisonProfiler()
    
    # Profile dynamic MoE
    print("Profiling dynamic MoE...")
    with comparison_profiler.dynamic_profiler:
        with torch.no_grad():
            output, routing_info = moe_layer.forward(
                hidden_states, return_router_logits=True
            )
            
            comparison_profiler.dynamic_profiler.profile_routing_decision(
                hidden_states.shape, num_experts, routing_info
            )
    
    # Set static baseline (simulated)
    import time
    static_time_start = time.time()
    # Simulate static MoE computation (all experts for all tokens)
    time.sleep(0.01)  # Simulate computation time
    static_time = time.time() - static_time_start
    
    comparison_profiler.set_static_baseline(
        batch_size, seq_len, hidden_dim, num_experts, static_time
    )
    
    # Get comparison results
    results = comparison_profiler.compare_performance()
    
    print("\nPerformance Comparison Results:")
    print(f"  FLOP speedup: {results['flop_speedup']:.2f}x")
    print(f"  FLOP reduction: {results['flop_reduction_percent']:.1f}%")
    print(f"  Dynamic FLOPs: {results['dynamic_flops']:,}")
    print(f"  Static FLOPs: {results['static_flops']:,}")
    
    # Get efficiency metrics
    efficiency = comparison_profiler.dynamic_profiler.compute_efficiency_metrics(
        batch_size, seq_len, hidden_dim, num_experts
    )
    
    print(f"\nEfficiency Metrics:")
    print(f"  Utilization efficiency: {efficiency['utilization_efficiency']:.1%}")
    print(f"  Load balance score: {efficiency['load_balance_score']:.3f}")
    print(f"  Expert calls: {efficiency['total_expert_calls']} / {efficiency['max_possible_calls']}")


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch is not available. Please install PyTorch and try again.")
        print("Installation: pip install torch")
        exit(1)
    
    # Run all PyTorch examples
    basic_pytorch_example()
    gradient_based_training_example()
    attention_entropy_example()
    model_patching_example()
    performance_comparison()
    
    print("\n=== All PyTorch Examples Complete ===")
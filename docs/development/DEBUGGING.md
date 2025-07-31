# Debugging Guide for Dynamic MoE Router

## Overview
This guide provides comprehensive debugging strategies for dynamic MoE routing issues.

## Common Issues and Solutions

### 1. Expert Selection Problems

#### Issue: Uneven Expert Utilization
**Symptoms**: Some experts are heavily utilized while others remain idle

**Debugging Steps**:
```python
# Monitor expert selection distribution
import torch
from collections import Counter

def debug_expert_selection(router_logits, k_values):
    """Debug expert selection patterns."""
    expert_counts = Counter()
    
    for batch_idx in range(router_logits.size(0)):
        for seq_idx in range(router_logits.size(1)):
            k = k_values[batch_idx, seq_idx]
            top_experts = router_logits[batch_idx, seq_idx].topk(k).indices
            
            for expert_id in top_experts:
                expert_counts[expert_id.item()] += 1
    
    # Print utilization distribution
    total_selections = sum(expert_counts.values())
    for expert_id in range(router_logits.size(-1)):
        utilization = expert_counts[expert_id] / total_selections * 100
        print(f"Expert {expert_id}: {utilization:.2f}% utilization")
    
    return expert_counts
```

**Solutions**:
- Adjust load balancing loss weight
- Review complexity estimation logic
- Check expert capacity constraints

#### Issue: Poor Routing Decisions
**Symptoms**: High computational cost with minimal performance gain

**Debugging**:
```python
def analyze_routing_efficiency(complexity_scores, k_values, performance_metrics):
    """Analyze routing efficiency."""
    import matplotlib.pyplot as plt
    
    # Plot complexity vs k selection
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.scatter(complexity_scores.flatten(), k_values.flatten(), alpha=0.5)
    plt.xlabel('Complexity Score')
    plt.ylabel('Experts Selected (k)')
    plt.title('Complexity vs Expert Selection')
    
    plt.subplot(132)
    plt.hist(complexity_scores.flatten(), bins=50, alpha=0.7)
    plt.xlabel('Complexity Score')
    plt.ylabel('Frequency')
    plt.title('Complexity Distribution')
    
    plt.subplot(133)
    plt.hist(k_values.flatten(), bins=range(1, 9), alpha=0.7)
    plt.xlabel('Experts Selected')
    plt.ylabel('Frequency')
    plt.title('Expert Selection Distribution')
    
    plt.tight_layout()
    plt.savefig('routing_analysis.png')
    plt.show()
```

### 2. Performance Issues

#### Issue: High Inference Latency
**Symptoms**: Slower inference compared to static MoE

**Profiling Code**:
```python
import time
import torch.profiler

def profile_inference_latency(model, inputs):
    """Profile inference latency with detailed breakdown."""
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # Warm up
        for _ in range(10):
            _ = model(inputs)
        
        torch.cuda.synchronize()  # If using CUDA
        
        # Actual profiling
        start_time = time.time()
        for _ in range(100):
            outputs = model(inputs)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_latency = (end_time - start_time) / 100
        
    # Export trace for analysis
    prof.export_chrome_trace("inference_trace.json")
    
    # Print top operations
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    return avg_latency, prof
```

#### Issue: Memory Leaks
**Symptoms**: Gradually increasing memory usage over time

**Memory Debugging**:
```python
import gc
import tracemalloc
from collections import defaultdict

def debug_memory_usage():
    """Debug memory usage patterns."""
    tracemalloc.start()
    
    memory_snapshots = []
    
    def take_snapshot(label):
        gc.collect()  # Force garbage collection
        snapshot = tracemalloc.take_snapshot()
        memory_snapshots.append((label, snapshot))
        
        # Print current memory usage
        current, peak = tracemalloc.get_traced_memory()
        print(f"{label}: Current={current/1024/1024:.1f}MB, Peak={peak/1024/1024:.1f}MB")
    
    take_snapshot("Initial")
    
    # Your model operations here
    # model = load_model()
    # for batch in data_loader:
    #     outputs = model(batch)
    #     take_snapshot(f"Batch {i}")
    
    # Analyze memory growth
    if len(memory_snapshots) > 1:
        first_snapshot = memory_snapshots[0][1]
        last_snapshot = memory_snapshots[-1][1]
        
        top_stats = last_snapshot.compare_to(first_snapshot, 'lineno')
        print("\nTop 10 memory differences:")
        for stat in top_stats[:10]:
            print(stat)
```

### 3. Numerical Stability Issues

#### Issue: NaN or Inf Values in Routing
**Symptoms**: Model outputs become NaN during training

**Debugging**:
```python
def check_numerical_stability(tensor, name="tensor"):
    """Check for numerical issues in tensors."""
    if torch.isnan(tensor).any():
        print(f"WARNING: NaN detected in {name}")
        print(f"NaN positions: {torch.isnan(tensor).nonzero()}")
        
    if torch.isinf(tensor).any():
        print(f"WARNING: Inf detected in {name}")
        print(f"Inf positions: {torch.isinf(tensor).nonzero()}")
        
    # Check for extremely large values
    max_val = tensor.abs().max().item()
    if max_val > 1e6:
        print(f"WARNING: Large values in {name}: max={max_val}")
        
    return not (torch.isnan(tensor).any() or torch.isinf(tensor).any())

# Use in router code
def debug_router_forward(self, inputs):
    """Router forward pass with numerical stability checks."""
    check_numerical_stability(inputs, "inputs")
    
    # Complexity estimation
    complexity = self.complexity_estimator(inputs)
    check_numerical_stability(complexity, "complexity")
    
    # Router logits
    router_logits = self.router_network(inputs)
    check_numerical_stability(router_logits, "router_logits")
    
    # Expert selection
    k = self.compute_k(complexity)
    expert_indices = router_logits.topk(k, dim=-1).indices
    
    return expert_indices, complexity
```

### 4. Training Issues

#### Issue: Training Instability
**Symptoms**: Loss spikes, gradient explosions, poor convergence

**Gradient Debugging**:
```python
def debug_gradients(model, loss):
    """Debug gradient flow and magnitudes."""
    loss.backward()
    
    total_norm = 0
    param_count = 0
    
    gradient_info = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += param.numel()
            
            gradient_info[name] = {
                'norm': param_norm.item(),
                'mean': param.grad.data.mean().item(),
                'std': param.grad.data.std().item(),
                'max': param.grad.data.max().item(),
                'min': param.grad.data.min().item()
            }
    
    total_norm = total_norm ** (1. / 2)
    
    print(f"Total gradient norm: {total_norm:.4f}")
    
    # Print top gradients by norm
    sorted_grads = sorted(gradient_info.items(), 
                         key=lambda x: x[1]['norm'], reverse=True)
    
    print("\nTop 5 gradients by norm:")
    for name, info in sorted_grads[:5]:
        print(f"  {name}: norm={info['norm']:.4f}, mean={info['mean']:.6f}")
    
    return gradient_info, total_norm
```

## Debugging Tools and Utilities

### 1. Router State Inspector
```python
class RouterInspector:
    """Inspector for router internal state."""
    
    def __init__(self, router):
        self.router = router
        self.routing_history = []
        
    def inspect_routing_decision(self, inputs, complexity, k_values, expert_indices):
        """Inspect a single routing decision."""
        routing_info = {
            'timestamp': time.time(),
            'input_shape': inputs.shape,
            'complexity_stats': {
                'mean': complexity.mean().item(),
                'std': complexity.std().item(),
                'min': complexity.min().item(),
                'max': complexity.max().item()
            },
            'k_stats': {
                'mean': k_values.float().mean().item(),
                'std': k_values.float().std().item(),
                'min': k_values.min().item(),
                'max': k_values.max().item()
            },
            'expert_selection': expert_indices.cpu().numpy().tolist()
        }
        
        self.routing_history.append(routing_info)
        return routing_info
        
    def generate_report(self, output_file="routing_report.json"):
        """Generate comprehensive routing report."""
        import json
        
        report = {
            'total_decisions': len(self.routing_history),
            'time_range': {
                'start': min(h['timestamp'] for h in self.routing_history),
                'end': max(h['timestamp'] for h in self.routing_history)
            },
            'complexity_trends': self._analyze_complexity_trends(),
            'expert_utilization': self._analyze_expert_utilization(),
            'routing_history': self.routing_history
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        return report
```

### 2. Performance Regression Detector
```python
class PerformanceMonitor:
    """Monitor for performance regressions."""
    
    def __init__(self, baseline_metrics=None):
        self.baseline_metrics = baseline_metrics or {}
        self.current_metrics = {}
        
    def record_metric(self, name, value):
        """Record a performance metric."""
        if name not in self.current_metrics:
            self.current_metrics[name] = []
        self.current_metrics[name].append(value)
        
    def check_regression(self, threshold=0.1):
        """Check for performance regressions."""
        regressions = {}
        
        for metric, values in self.current_metrics.items():
            if metric in self.baseline_metrics:
                current_avg = sum(values) / len(values)
                baseline_avg = self.baseline_metrics[metric]
                
                regression = (current_avg - baseline_avg) / baseline_avg
                
                if regression > threshold:
                    regressions[metric] = {
                        'regression_percent': regression * 100,
                        'baseline': baseline_avg,
                        'current': current_avg
                    }
        
        return regressions
```

## Integration with Development Tools

### VS Code Debugging Configuration
The `.vscode/launch.json` file includes debugging configurations for:
- Single test debugging
- Full test suite debugging  
- PyTorch model debugging
- JAX model debugging
- Performance benchmarking

### Jupyter Notebook Debugging
```python
# In Jupyter notebooks
%load_ext autoreload
%autoreload 2

# For detailed profiling
%load_ext line_profiler
%lprun -f router_function model(inputs)

# For memory profiling
%load_ext memory_profiler
%memit model(inputs)
```

## Automated Debugging

### Pre-commit Hook for Debugging Checks
```yaml
- repo: local
  hooks:
    - id: check-debug-statements
      name: Check for debug statements
      entry: python scripts/check_debug.py
      language: system
      files: \.py$
```

### Continuous Performance Testing
Set up automated performance regression tests in CI/CD:
```python
def test_routing_performance_regression():
    """Test for routing performance regressions."""
    baseline_latency = 0.1  # seconds
    
    start_time = time.time()
    outputs = model(test_inputs)
    actual_latency = time.time() - start_time
    
    assert actual_latency < baseline_latency * 1.2, \
        f"Performance regression detected: {actual_latency:.3f}s > {baseline_latency*1.2:.3f}s"
```

This debugging guide provides comprehensive tools and strategies for identifying and resolving issues in dynamic MoE routing systems.
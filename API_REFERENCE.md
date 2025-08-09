# 📚 Dynamic MoE Router Kit - API Reference

## Overview

Complete API reference for the Dynamic MoE Router Kit, covering all classes, methods, and configuration options for production deployment.

---

## Core Components

### 🧠 DynamicRouter

The main routing component that performs intelligent expert selection.

```python
class DynamicRouter:
    """Core dynamic routing algorithm for MoE models."""
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        min_experts: int = 1,
        max_experts: Optional[int] = None,
        complexity_estimator: Union[str, ComplexityEstimator] = "gradient_norm",
        routing_strategy: str = "top_k",
        load_balancing: bool = True,
        noise_factor: float = 0.0,
        **estimator_kwargs
    ):
        """Initialize dynamic router.
        
        Args:
            input_dim: Input feature dimension
            num_experts: Total number of experts available
            min_experts: Minimum experts to route per token
            max_experts: Maximum experts to route per token
            complexity_estimator: Complexity estimation method or instance
            routing_strategy: "top_k" or "threshold"
            load_balancing: Enable load balancing across experts
            noise_factor: Noise factor for regularization (0.0-1.0)
            **estimator_kwargs: Additional arguments for complexity estimator
        """
```

#### Methods

##### `route(hidden_states, return_router_logits=False, **kwargs)`

Main routing method that selects experts for each token.

**Parameters:**
- `hidden_states` (Array): Input tensor [batch, seq_len, hidden_dim]
- `return_router_logits` (bool): Whether to return raw router scores
- `**kwargs`: Additional arguments for complexity estimation

**Returns:**
```python
{
    'expert_indices': Array,        # Selected expert indices [batch, seq_len, k]
    'expert_weights': Array,        # Expert combination weights [batch, seq_len, k]
    'num_experts_per_token': Array, # Number of experts used [batch, seq_len]
    'complexity_scores': Array,     # Token complexity scores [batch, seq_len]
    'routing_info': Dict[str, Any], # Additional routing statistics
    'router_logits': Array,         # Raw router scores (if requested)
}
```

**Raises:**
- `RouterConfigurationError`: Invalid configuration
- `ComplexityEstimationError`: Complexity estimation failed
- `ExpertDispatchError`: Expert selection failed
- `ValidationError`: Input validation failed

##### `get_expert_usage_stats()`

Get expert usage statistics for monitoring.

**Returns:**
```python
{
    'total_batches': int,           # Number of processed batches
    'avg_usage_per_expert': List[float], # Average usage per expert
    'usage_variance': float,        # Variance in expert usage
    'most_used_expert': int,        # Index of most used expert
    'least_used_expert': int,       # Index of least used expert
    'load_balance_score': float,    # Load balance quality (0-1)
}
```

---

### 🔄 AdaptiveRouter

Enhanced router with adaptive complexity thresholds.

```python
class AdaptiveRouter(DynamicRouter):
    """Enhanced router with adaptive complexity thresholds."""
    
    def __init__(self, adaptation_rate: float = 0.01, **kwargs):
        """Initialize adaptive router.
        
        Args:
            adaptation_rate: Learning rate for threshold adaptation
            **kwargs: Arguments passed to DynamicRouter
        """
```

#### Methods

##### `update_thresholds(performance_score)`

Update complexity thresholds based on performance feedback.

**Parameters:**
- `performance_score` (float): Performance metric (higher is better)

---

### 📊 ComplexityEstimator

Base class for complexity estimation algorithms.

```python
class ComplexityEstimator(ABC):
    """Base class for input complexity estimation algorithms."""
    
    def __init__(self, normalize: bool = True, epsilon: float = 1e-8):
        """Initialize complexity estimator.
        
        Args:
            normalize: Whether to normalize complexity scores to [0, 1]
            epsilon: Small constant for numerical stability
        """
    
    @abstractmethod
    def estimate(self, hidden_states: Any, **kwargs) -> Any:
        """Estimate complexity scores for input tokens.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            **kwargs: Additional context (attention weights, gradients, etc.)
            
        Returns:
            Complexity scores [batch, seq_len] in range [0, 1]
        """
```

#### Built-in Estimators

##### `GradientNormEstimator`
```python
class GradientNormEstimator(ComplexityEstimator):
    """Estimate complexity based on gradient norms."""
    
    def __init__(self, 
                 percentile: float = 90.0, 
                 smoothing_factor: float = 0.1,
                 **kwargs):
```

##### `AttentionEntropyEstimator`
```python
class AttentionEntropyEstimator(ComplexityEstimator):
    """Estimate complexity based on attention entropy."""
    
    def __init__(self,
                 head_aggregation: str = "mean",
                 temperature: float = 1.0,
                 **kwargs):
```

##### `PerplexityProxyEstimator`
```python
class PerplexityProxyEstimator(ComplexityEstimator):
    """Estimate complexity based on model confidence."""
    
    def __init__(self,
                 temperature: float = 1.0,
                 **kwargs):
```

##### `ThresholdEstimator`
```python
class ThresholdEstimator(ComplexityEstimator):
    """Simple threshold-based estimator for testing."""
    
    def __init__(self,
                 threshold_fn: Optional[Callable] = None,
                 **kwargs):
```

---

### 🏗️ MoELayer

Dynamic Mixture-of-Experts layer implementation.

```python
class MoELayer:
    """Dynamic Mixture-of-Experts layer with adaptive routing."""
    
    def __init__(
        self,
        router: DynamicRouter,
        expert_fn: Callable[[], Any],
        num_experts: int,
        expert_capacity_factor: float = 1.25,
        dropout_rate: float = 0.0,
        use_bias: bool = True
    ):
        """Initialize MoE layer.
        
        Args:
            router: DynamicRouter instance for expert selection
            expert_fn: Function that creates expert networks
            num_experts: Number of expert networks
            expert_capacity_factor: Capacity factor for expert load
            dropout_rate: Dropout rate for regularization
            use_bias: Whether to use bias in expert networks
        """
```

#### Methods

##### `forward(hidden_states, return_router_logits=False, **kwargs)`

Forward pass through dynamic MoE layer.

**Parameters:**
- `hidden_states` (Array): Input tensor [batch, seq_len, hidden_dim]
- `return_router_logits` (bool): Whether to return routing information
- `**kwargs`: Additional arguments for router

**Returns:**
- If `return_router_logits=False`: Output tensor [batch, seq_len, hidden_dim]
- If `return_router_logits=True`: (output, routing_info)

##### `get_performance_summary()`

Get performance summary for the MoE layer.

**Returns:**
```python
{
    'total_forward_calls': int,     # Number of forward passes
    'total_expert_calls': int,      # Total expert invocations
    'avg_experts_per_forward': float, # Average experts per forward pass
    'computational_efficiency': float, # Efficiency vs static MoE
    'router_stats': Dict[str, Any], # Router statistics
}
```

#### Specialized MoE Layers

##### `SparseMoELayer`
```python
class SparseMoELayer(MoELayer):
    """Sparse MoE layer with capacity constraints."""
    
    def __init__(self, capacity_factor: float = 1.25, **kwargs):
```

##### `LayerNormMoE`
```python
class LayerNormMoE(MoELayer):
    """MoE layer with layer normalization and residual connections."""
    
    def __init__(self, layer_norm_eps: float = 1e-5, **kwargs):
```

---

## Optimization Components

### ⚡ create_optimized_router

Create an optimized, scalable router wrapper.

```python
def create_optimized_router(
    router,
    enable_autoscaling: bool = True,
    enable_parallel: bool = True,
    max_workers: int = 4,
    **scaling_kwargs
):
    """Create an optimized, scalable router wrapper.
    
    Args:
        router: Base router to optimize
        enable_autoscaling: Enable automatic scaling
        enable_parallel: Enable parallel processing
        max_workers: Maximum worker threads
        **scaling_kwargs: Additional scaling configuration
        
    Returns:
        OptimizedRouter instance
    """
```

#### OptimizedRouter Methods

##### `get_optimization_stats()`

Get comprehensive optimization statistics.

**Returns:**
```python
{
    'performance': Dict[str, Any],  # Performance metrics
    'autoscaler': Dict[str, Any],   # Autoscaling status
    'memory_pool': Dict[str, Any],  # Memory pool statistics
}
```

---

### 🗄️ create_cached_router

Create a cached wrapper around a router.

```python
def create_cached_router(
    router,
    cache_size: int = 1000,
    adaptive: bool = True
):
    """Create a cached wrapper around a router.
    
    Args:
        router: Base router to cache
        cache_size: Maximum cache size
        adaptive: Use adaptive cache sizing
        
    Returns:
        CachedRouter instance
    """
```

#### CachedRouter Methods

##### `get_cache_stats()`

Get cache performance statistics.

**Returns:**
```python
{
    'size': int,                    # Current cache size
    'max_size': int,                # Maximum cache size
    'hit_count': int,               # Cache hits
    'miss_count': int,              # Cache misses
    'hit_rate': float,              # Hit rate (0-1)
    'ttl_seconds': float,           # Time-to-live
}
```

##### `clear_cache()`

Clear the routing cache.

---

### 📊 create_monitoring_wrapper

Create a monitoring wrapper around a router.

```python
def create_monitoring_wrapper(
    router,
    enable_circuit_breaker: bool = True,
    **monitor_kwargs
):
    """Create a monitoring wrapper around a router.
    
    Args:
        router: Base router to monitor
        enable_circuit_breaker: Enable circuit breaker pattern
        **monitor_kwargs: Additional monitoring configuration
        
    Returns:
        MonitoredRouter instance
    """
```

#### MonitoredRouter Methods

##### `get_monitoring_summary()`

Get comprehensive monitoring summary.

**Returns:**
```python
{
    'performance': Dict[str, Any],  # Performance metrics
    'health': Dict[str, Any],       # Health check results
    'circuit_breaker_state': str,  # Circuit breaker status
}
```

---

## Benchmarking Components

### 🏃‍♂️ DynamicMoEBenchmark

Comprehensive benchmarking suite.

```python
class DynamicMoEBenchmark:
    """Comprehensive benchmarking suite for dynamic MoE routing."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark suite.
        
        Args:
            output_dir: Directory to save benchmark results
        """
```

#### Methods

##### `run_latency_benchmark(config)`

Benchmark routing latency across different configurations.

**Parameters:**
- `config` (BenchmarkConfig): Benchmark configuration

**Returns:**
- List[BenchmarkResult]: Latency benchmark results

##### `run_throughput_benchmark(config, concurrent_requests=None)`

Benchmark throughput under concurrent load.

**Parameters:**
- `config` (BenchmarkConfig): Benchmark configuration
- `concurrent_requests` (List[int]): Concurrent request levels

**Returns:**
- List[BenchmarkResult]: Throughput benchmark results

##### `run_scaling_benchmark(config)`

Benchmark scaling performance with optimizations.

**Parameters:**
- `config` (BenchmarkConfig): Benchmark configuration

**Returns:**
- List[BenchmarkResult]: Scaling benchmark results

##### `save_results(filename=None)`

Save benchmark results to JSON file.

**Parameters:**
- `filename` (str, optional): Output filename

##### `generate_report()`

Generate a summary report of benchmark results.

**Returns:**
- `str`: Markdown formatted report

### 📋 BenchmarkConfig

Configuration for benchmark runs.

```python
@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    
    name: str
    num_runs: int = 100
    warmup_runs: int = 10
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    hidden_dims: List[int] = None
    num_experts_list: List[int] = None
    min_experts_list: List[int] = None
    max_experts_list: List[int] = None
```

### 📈 BenchmarkResult

Results from a benchmark run.

```python
@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    
    config_name: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
```

---

## Profiling Components

### 🔍 FLOPProfiler

FLOP profiling for routing operations.

```python
class FLOPProfiler:
    """Profile FLOP usage in dynamic routing."""
    
    def __init__(self):
        """Initialize FLOP profiler."""
    
    def profile_routing(self, router, hidden_states):
        """Profile routing operation FLOPs."""
    
    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
```

### 📊 ComparisonProfiler

Compare static vs dynamic routing performance.

```python
class ComparisonProfiler:
    """Compare static vs dynamic MoE performance."""
    
    def __init__(self):
        """Initialize comparison profiler."""
    
    def compare_routing_strategies(self, configs, test_data):
        """Compare different routing strategies."""
    
    def generate_comparison_report(self) -> str:
        """Generate comparison report."""
```

---

## Configuration Reference

### Environment Variables

```bash
# Core settings
DYNAMIC_MOE_LOG_LEVEL=INFO          # Logging level
DYNAMIC_MOE_CACHE_SIZE=1000         # Default cache size
DYNAMIC_MOE_MAX_WORKERS=4           # Default worker count

# Performance settings
DYNAMIC_MOE_ENABLE_CACHING=true     # Enable caching by default
DYNAMIC_MOE_ENABLE_MONITORING=true  # Enable monitoring by default
DYNAMIC_MOE_AUTOSCALE=true         # Enable autoscaling

# Monitoring settings
DYNAMIC_MOE_METRICS_PORT=9090       # Metrics server port
DYNAMIC_MOE_HEALTH_PORT=8080        # Health check port
DYNAMIC_MOE_ALERT_WEBHOOK=""        # Alert webhook URL

# Security settings
DYNAMIC_MOE_API_KEY_REQUIRED=false  # Require API key
DYNAMIC_MOE_SSL_ENABLED=false       # Enable SSL
DYNAMIC_MOE_RATE_LIMIT=1000         # Requests per hour
```

### Configuration Files

#### `config.yaml`
```yaml
# Dynamic MoE Router Configuration
router:
  input_dim: 768
  num_experts: 8
  min_experts: 1
  max_experts: 4
  complexity_estimator: "gradient_norm"
  routing_strategy: "top_k"
  load_balancing: true
  noise_factor: 0.1

optimization:
  enable_autoscaling: true
  enable_parallel: true
  enable_caching: true
  max_workers: 4
  cache_size: 1000

monitoring:
  enable_circuit_breaker: true
  alert_thresholds:
    avg_latency_ms: 50.0
    error_rate: 0.05
    load_balance_variance: 0.1
    memory_usage_mb: 1000.0

benchmarking:
  num_runs: 100
  warmup_runs: 10
  batch_sizes: [1, 8, 16, 32]
  sequence_lengths: [128, 256, 512]
```

---

## Error Handling

### Exception Hierarchy

```python
DynamicMoEError                     # Base exception
├── RouterConfigurationError        # Configuration errors
├── ComplexityEstimationError      # Complexity estimation failures
├── ExpertDispatchError            # Expert selection failures
├── LoadBalancingError             # Load balancing issues
├── ValidationError                # Input validation errors
├── BackendError                   # Backend-specific errors
├── MemoryError                    # Memory-related errors
├── ConvergenceError               # Algorithm convergence issues
├── ModelPatchingError             # Model patching failures
└── ProfilingError                 # Profiling operation failures
```

### Error Handling Best Practices

```python
from dynamic_moe_router import DynamicRouter
from dynamic_moe_router.exceptions import (
    RouterConfigurationError,
    ComplexityEstimationError,
    ExpertDispatchError
)

try:
    router = DynamicRouter(
        input_dim=768,
        num_experts=8,
        min_experts=1,
        max_experts=4
    )
    
    result = router.route(hidden_states)
    
except RouterConfigurationError as e:
    logger.error(f"Router configuration invalid: {e}")
    # Handle configuration error
    
except ComplexityEstimationError as e:
    logger.error(f"Complexity estimation failed: {e}")
    # Handle complexity estimation error
    
except ExpertDispatchError as e:
    logger.error(f"Expert dispatch failed: {e}")
    # Handle expert dispatch error
    
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Handle unexpected errors
```

---

## Performance Guidelines

### Recommended Configuration Ranges

| Parameter | Small Models | Medium Models | Large Models |
|-----------|--------------|---------------|--------------|
| `input_dim` | 256-512 | 768-1024 | 1024-4096 |
| `num_experts` | 4-8 | 8-16 | 16-64 |
| `min_experts` | 1-2 | 1-2 | 2-4 |
| `max_experts` | 2-4 | 4-8 | 8-16 |
| `cache_size` | 500-1000 | 1000-5000 | 5000-10000 |
| `max_workers` | 2-4 | 4-8 | 8-16 |

### Performance Optimization Checklist

- ✅ Use appropriate cache size for your workload
- ✅ Enable autoscaling for variable load
- ✅ Use parallel processing for high throughput
- ✅ Monitor expert load balancing
- ✅ Configure circuit breaker for resilience
- ✅ Set appropriate complexity estimator
- ✅ Tune noise factor for regularization
- ✅ Use profiling to identify bottlenecks

---

This API reference provides comprehensive documentation for all components of the Dynamic MoE Router Kit. For additional examples and tutorials, see the `examples/` directory and README.md.
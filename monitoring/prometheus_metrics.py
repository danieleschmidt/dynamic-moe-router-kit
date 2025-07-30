"""Prometheus metrics collection for dynamic MoE router monitoring."""

from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import time
import functools
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Core routing metrics
ROUTING_REQUESTS_TOTAL = Counter(
    'dynamic_moe_routing_requests_total',
    'Total number of routing requests',
    ['model_name', 'backend', 'status']
)

ROUTING_DURATION_SECONDS = Histogram(
    'dynamic_moe_routing_duration_seconds', 
    'Time spent on routing decisions',
    ['model_name', 'backend'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

EXPERTS_USED_PER_TOKEN = Histogram(
    'dynamic_moe_experts_used_per_token',
    'Number of experts used per token',
    ['model_name'],
    buckets=[1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32]
)

# Performance metrics
FLOPS_COMPUTED = Counter(
    'dynamic_moe_flops_computed_total',
    'Total FLOPs computed',
    ['model_name', 'layer_type']
)

FLOPS_SAVED = Counter(
    'dynamic_moe_flops_saved_total', 
    'Total FLOPs saved through dynamic routing',
    ['model_name']
)

MEMORY_USAGE_BYTES = Gauge(
    'dynamic_moe_memory_usage_bytes',
    'Current memory usage in bytes',
    ['model_name', 'component']
)

# Model health metrics
MODEL_LOAD_BALANCE = Gauge(
    'dynamic_moe_load_balance_score',
    'Load balance score across experts (0-1)',
    ['model_name', 'layer_id']
)

COMPLEXITY_SCORE_DISTRIBUTION = Histogram(
    'dynamic_moe_complexity_score',
    'Distribution of token complexity scores',
    ['model_name', 'estimator_type'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Error metrics
ROUTING_ERRORS_TOTAL = Counter(
    'dynamic_moe_routing_errors_total',
    'Total routing errors',
    ['model_name', 'error_type']
)

EXPERT_FAILURES_TOTAL = Counter(
    'dynamic_moe_expert_failures_total',
    'Total expert computation failures', 
    ['model_name', 'expert_id', 'failure_type']
)


class MetricsCollector:
    """Centralized metrics collection for dynamic MoE routing."""
    
    def __init__(self, model_name: str, enable_detailed_metrics: bool = True):
        self.model_name = model_name
        self.enable_detailed = enable_detailed_metrics
        self._start_time = time.time()
        
    def record_routing_request(self, backend: str, status: str = 'success'):
        """Record a routing request."""
        ROUTING_REQUESTS_TOTAL.labels(
            model_name=self.model_name,
            backend=backend, 
            status=status
        ).inc()
    
    def record_routing_duration(self, duration: float, backend: str):
        """Record routing decision duration."""
        ROUTING_DURATION_SECONDS.labels(
            model_name=self.model_name,
            backend=backend
        ).observe(duration)
    
    def record_experts_used(self, num_experts: float):
        """Record number of experts used per token."""
        EXPERTS_USED_PER_TOKEN.labels(
            model_name=self.model_name
        ).observe(num_experts)
    
    def record_flops(self, flops_computed: int, flops_saved: int, layer_type: str = 'moe'):
        """Record FLOP computation and savings."""
        FLOPS_COMPUTED.labels(
            model_name=self.model_name,
            layer_type=layer_type
        ).inc(flops_computed)
        
        FLOPS_SAVED.labels(
            model_name=self.model_name
        ).inc(flops_saved)
    
    def update_memory_usage(self, memory_bytes: int, component: str):
        """Update current memory usage."""
        MEMORY_USAGE_BYTES.labels(
            model_name=self.model_name,
            component=component
        ).set(memory_bytes)
    
    def record_load_balance(self, score: float, layer_id: int):
        """Record load balance score for a layer."""
        MODEL_LOAD_BALANCE.labels(
            model_name=self.model_name,
            layer_id=str(layer_id)
        ).set(score)
    
    def record_complexity_score(self, score: float, estimator_type: str):
        """Record complexity score distribution."""
        COMPLEXITY_SCORE_DISTRIBUTION.labels(
            model_name=self.model_name,
            estimator_type=estimator_type
        ).observe(score)
    
    def record_error(self, error_type: str):
        """Record routing error."""
        ROUTING_ERRORS_TOTAL.labels(
            model_name=self.model_name,
            error_type=error_type
        ).inc()
    
    def record_expert_failure(self, expert_id: int, failure_type: str):
        """Record expert computation failure."""
        EXPERT_FAILURES_TOTAL.labels(
            model_name=self.model_name,
            expert_id=str(expert_id),
            failure_type=failure_type
        ).inc()


def monitor_routing_performance(metrics_collector: MetricsCollector, backend: str):
    """Decorator to monitor routing performance."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                metrics_collector.record_routing_request(backend, 'success')
                return result
            except Exception as e:
                metrics_collector.record_routing_request(backend, 'error')
                metrics_collector.record_error(type(e).__name__)
                raise
            finally:
                duration = time.time() - start_time
                metrics_collector.record_routing_duration(duration, backend)
        
        return wrapper
    return decorator


def track_expert_usage(metrics_collector: MetricsCollector):
    """Decorator to track expert usage patterns."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Extract routing information from result
            if isinstance(result, tuple) and len(result) > 1:
                _, routing_info = result
                if isinstance(routing_info, dict):
                    if 'avg_experts_per_token' in routing_info:
                        metrics_collector.record_experts_used(
                            routing_info['avg_experts_per_token']
                        )
                    
                    if 'load_balance_score' in routing_info:
                        layer_id = routing_info.get('layer_id', 0)
                        metrics_collector.record_load_balance(
                            routing_info['load_balance_score'], layer_id
                        )
            
            return result
        return wrapper
    return decorator


class PerformanceProfiler:
    """Performance profiler with metrics integration."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._profiles = {}
    
    def start_profile(self, profile_name: str):
        """Start a performance profile."""
        self._profiles[profile_name] = {
            'start_time': time.time(),
            'start_memory': self._get_memory_usage()
        }
    
    def end_profile(self, profile_name: str) -> Dict[str, Any]:
        """End a performance profile and record metrics."""
        if profile_name not in self._profiles:
            logger.warning(f"Profile {profile_name} not found")
            return {}
        
        profile = self._profiles.pop(profile_name)
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        duration = end_time - profile['start_time']
        memory_delta = end_memory - profile['start_memory']
        
        # Record metrics
        self.metrics.update_memory_usage(end_memory, profile_name)
        
        return {
            'duration': duration,
            'memory_delta': memory_delta,
            'peak_memory': end_memory
        }
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            logger.warning("psutil not available, memory tracking disabled")
            return 0


def start_metrics_server(port: int = 8000, registry=None):
    """Start Prometheus metrics server."""
    try:
        start_http_server(port, registry=registry)
        logger.info(f"Metrics server started on port {port}")
        logger.info(f"Metrics available at http://localhost:{port}/metrics")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
        raise


# Example usage and testing
if __name__ == "__main__":
    # Initialize metrics collection
    collector = MetricsCollector("test-model")
    profiler = PerformanceProfiler(collector)
    
    # Start metrics server
    start_metrics_server(8000)
    
    # Simulate some metrics
    import random
    
    for i in range(100):
        # Simulate routing
        collector.record_routing_request("torch", "success")
        collector.record_routing_duration(random.uniform(0.001, 0.1), "torch")
        collector.record_experts_used(random.uniform(1, 8))
        
        # Simulate complexity scores
        collector.record_complexity_score(
            random.uniform(0, 1), "gradient_norm"
        )
        
        # Simulate performance profiling
        profiler.start_profile("inference")
        time.sleep(0.01)  # Simulate work
        profile_result = profiler.end_profile("inference")
        
        time.sleep(0.1)
    
    print("Metrics collection demo completed")
    print("View metrics at http://localhost:8000/metrics")
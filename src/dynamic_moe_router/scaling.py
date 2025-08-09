"""Auto-scaling and optimization features for dynamic MoE routing."""

import concurrent.futures
import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, Optional

import numpy as np

from .exceptions import DynamicMoEError
from .monitoring import PerformanceMonitor


class AutoScaler:
    """Automatic scaling for MoE routing based on load and performance metrics."""

    def __init__(
        self,
        min_experts: int = 2,
        max_experts: int = 16,
        target_latency_ms: float = 50.0,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        stabilization_window: int = 10,
        min_stable_period: float = 30.0
    ):
        self.min_experts = min_experts
        self.max_experts = max_experts
        self.target_latency_ms = target_latency_ms
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.stabilization_window = stabilization_window
        self.min_stable_period = min_stable_period

        # Scaling state
        self.current_experts = min_experts
        self.last_scale_time = 0
        self.decision_history = deque(maxlen=stabilization_window)

        # Metrics tracking
        self.performance_monitor = PerformanceMonitor()

    def should_scale(self, current_metrics: Dict[str, Any]) -> Optional[str]:
        """Determine if scaling is needed based on current metrics.
        
        Returns:
            'up', 'down', or None
        """
        # Check if enough time has passed since last scaling
        if time.time() - self.last_scale_time < self.min_stable_period:
            return None

        latency_ms = current_metrics.get('latency_p95_ms', 0)
        utilization = current_metrics.get('expert_utilization_avg', 0)
        error_rate = current_metrics.get('error_rate', 0)

        # Don't scale if system is unstable (high error rate)
        if error_rate > 0.1:
            return None

        # Scale up conditions
        if (latency_ms > self.target_latency_ms * 1.5 or
            utilization > self.scale_up_threshold) and \
           self.current_experts < self.max_experts:
            decision = 'up'
        # Scale down conditions
        elif (latency_ms < self.target_latency_ms * 0.7 and
              utilization < self.scale_down_threshold) and \
             self.current_experts > self.min_experts:
            decision = 'down'
        else:
            decision = None

        # Add to decision history for stabilization
        self.decision_history.append(decision)

        # Only scale if decision is consistent
        if len(self.decision_history) >= self.stabilization_window:
            recent_decisions = list(self.decision_history)[-5:]  # Last 5 decisions
            if all(d == decision for d in recent_decisions) and decision:
                self.last_scale_time = time.time()
                return decision

        return None

    def execute_scaling(self, router, direction: str) -> bool:
        """Execute scaling operation on router.
        
        Returns:
            True if scaling was successful
        """
        try:
            if direction == 'up' and self.current_experts < self.max_experts:
                new_experts = min(self.max_experts, int(self.current_experts * 1.5))
                self._scale_router(router, new_experts)
                self.current_experts = new_experts
                return True
            elif direction == 'down' and self.current_experts > self.min_experts:
                new_experts = max(self.min_experts, int(self.current_experts * 0.7))
                self._scale_router(router, new_experts)
                self.current_experts = new_experts
                return True
        except Exception as e:
            raise DynamicMoEError(f"Scaling failed: {e}")

        return False

    def _scale_router(self, router, new_expert_count: int):
        """Modify router configuration for new expert count."""
        # Update router parameters
        old_max = router.max_experts
        router.max_experts = min(new_expert_count, router.num_experts)

        # Reinitialize router network if needed
        if hasattr(router, 'initialize_router_network'):
            router.initialize_router_network()

        print(f"Scaled from {old_max} to {router.max_experts} max experts")


class LoadBalancer:
    """Advanced load balancing with multiple strategies."""

    def __init__(self, strategy: str = "adaptive"):
        self.strategy = strategy
        self.expert_loads = defaultdict(float)
        self.expert_capacities = defaultdict(float)
        self.response_times = defaultdict(lambda: deque(maxlen=100))

    def balance_load(self, expert_indices: np.ndarray, expert_weights: np.ndarray,
                    routing_info: Dict[str, Any]) -> np.ndarray:
        """Apply advanced load balancing to expert weights."""

        if self.strategy == "adaptive":
            return self._adaptive_balancing(expert_indices, expert_weights, routing_info)
        elif self.strategy == "capacity_aware":
            return self._capacity_aware_balancing(expert_indices, expert_weights)
        elif self.strategy == "latency_aware":
            return self._latency_aware_balancing(expert_indices, expert_weights)
        else:
            return expert_weights  # No balancing

    def _adaptive_balancing(self, expert_indices: np.ndarray, expert_weights: np.ndarray,
                           routing_info: Dict[str, Any]) -> np.ndarray:
        """Adaptive load balancing based on current system state."""
        balanced_weights = expert_weights.copy()

        # Get expert utilization from routing info
        expert_utilization = routing_info.get('expert_utilization', [])
        if not expert_utilization:
            return balanced_weights

        # Calculate balance penalty for overused experts
        mean_util = np.mean(expert_utilization)
        std_util = np.std(expert_utilization)

        if std_util > 0.1:  # Apply balancing if there's significant imbalance
            for i in range(expert_indices.shape[0]):  # batch
                for j in range(expert_indices.shape[1]):  # sequence
                    for k in range(expert_indices.shape[2]):  # experts
                        expert_idx = expert_indices[i, j, k]
                        if expert_idx >= 0 and expert_idx < len(expert_utilization):
                            util = expert_utilization[expert_idx]
                            if util > mean_util + std_util:
                                # Penalize overused experts
                                penalty = min(0.5, (util - mean_util) / (2 * std_util))
                                balanced_weights[i, j, k] *= (1 - penalty)

        # Renormalize weights
        for i in range(balanced_weights.shape[0]):
            for j in range(balanced_weights.shape[1]):
                row_sum = np.sum(balanced_weights[i, j])
                if row_sum > 0:
                    balanced_weights[i, j] /= row_sum

        return balanced_weights

    def _capacity_aware_balancing(self, expert_indices: np.ndarray,
                                 expert_weights: np.ndarray) -> np.ndarray:
        """Balance load based on expert capacities."""
        # Placeholder for capacity-aware balancing
        return expert_weights

    def _latency_aware_balancing(self, expert_indices: np.ndarray,
                               expert_weights: np.ndarray) -> np.ndarray:
        """Balance load based on expert response times."""
        # Placeholder for latency-aware balancing
        return expert_weights

    def update_expert_metrics(self, expert_id: int, load: float,
                             response_time: float, capacity: float = 1.0):
        """Update metrics for an expert."""
        self.expert_loads[expert_id] = load
        self.expert_capacities[expert_id] = capacity
        self.response_times[expert_id].append(response_time)


class ParallelRouter:
    """Router with parallel expert computation."""

    def __init__(self, base_router, max_workers: int = 4):
        self.router = base_router
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        # Delegate attribute access
        for attr in dir(base_router):
            if not attr.startswith('_') and attr not in ['route']:
                setattr(self, attr, getattr(base_router, attr))

    def route(self, hidden_states, return_router_logits=False, **kwargs):
        """Parallel routing with concurrent expert computation."""
        # Get routing decisions from base router
        routing_result = self.router.route(hidden_states, return_router_logits, **kwargs)

        # For demonstration, just return base result
        # In a full implementation, this would parallelize expert computations
        return routing_result

    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


class BatchOptimizer:
    """Optimize batch processing for better throughput."""

    def __init__(self, target_batch_size: int = 32, max_delay_ms: float = 10.0):
        self.target_batch_size = target_batch_size
        self.max_delay_ms = max_delay_ms
        self.pending_requests = []
        self.batch_lock = threading.Lock()
        self.last_batch_time = time.time()

    def add_request(self, request_data: Any) -> concurrent.futures.Future:
        """Add a request to the batch queue."""
        future = concurrent.futures.Future()

        with self.batch_lock:
            self.pending_requests.append((request_data, future))

            # Process batch if we hit target size or max delay
            current_time = time.time()
            should_process = (
                len(self.pending_requests) >= self.target_batch_size or
                (current_time - self.last_batch_time) * 1000 > self.max_delay_ms
            )

            if should_process:
                self._process_batch()

        return future

    def _process_batch(self):
        """Process the current batch of requests."""
        if not self.pending_requests:
            return

        batch_requests = self.pending_requests.copy()
        self.pending_requests.clear()
        self.last_batch_time = time.time()

        # Process batch (placeholder - would integrate with actual router)
        for request_data, future in batch_requests:
            # Simulate processing
            try:
                result = self._process_single_request(request_data)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

    def _process_single_request(self, request_data: Any) -> Any:
        """Process a single request."""
        # Placeholder implementation
        return {"processed": True, "data": request_data}


class MemoryPool:
    """Memory pool for efficient tensor allocation and reuse."""

    def __init__(self, max_pool_size: int = 100):
        self.max_pool_size = max_pool_size
        self.tensor_pools = defaultdict(lambda: deque())
        self.pool_lock = threading.Lock()

    def get_tensor(self, shape: tuple, dtype: type = np.float32) -> np.ndarray:
        """Get a tensor from the pool or create a new one."""
        pool_key = (shape, dtype)

        with self.pool_lock:
            if self.tensor_pools[pool_key]:
                return self.tensor_pools[pool_key].popleft()

        # Create new tensor if pool is empty
        return np.zeros(shape, dtype=dtype)

    def return_tensor(self, tensor: np.ndarray):
        """Return a tensor to the pool for reuse."""
        pool_key = (tensor.shape, tensor.dtype)

        with self.pool_lock:
            if len(self.tensor_pools[pool_key]) < self.max_pool_size:
                # Clear tensor data and return to pool
                tensor.fill(0)
                self.tensor_pools[pool_key].append(tensor)

    def clear_pools(self):
        """Clear all memory pools."""
        with self.pool_lock:
            self.tensor_pools.clear()


def create_optimized_router(router, enable_autoscaling: bool = True,
                          enable_parallel: bool = True, max_workers: int = 4,
                          **scaling_kwargs):
    """Create an optimized, scalable router wrapper."""

    class OptimizedRouter:
        def __init__(self, base_router):
            self.router = base_router

            # Add optimization components
            if enable_autoscaling:
                self.autoscaler = AutoScaler(**scaling_kwargs)
            else:
                self.autoscaler = None

            self.load_balancer = LoadBalancer(strategy="adaptive")

            if enable_parallel:
                self.parallel_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            else:
                self.parallel_executor = None

            self.memory_pool = MemoryPool()
            self.performance_monitor = PerformanceMonitor()

            # Delegate attribute access
            for attr in dir(base_router):
                if not attr.startswith('_') and attr not in ['route']:
                    setattr(self, attr, getattr(base_router, attr))

        def route(self, hidden_states, return_router_logits=False, **kwargs):
            """Optimized routing with scaling and load balancing."""
            start_time = time.time()

            try:
                # Get base routing result
                result = self.router.route(hidden_states, return_router_logits, **kwargs)

                # Apply load balancing
                if 'expert_weights' in result:
                    result['expert_weights'] = self.load_balancer.balance_load(
                        result['expert_indices'],
                        result['expert_weights'],
                        result.get('routing_info', {})
                    )

                # Check if autoscaling is needed
                if self.autoscaler:
                    current_metrics = self._get_current_metrics()
                    scale_decision = self.autoscaler.should_scale(current_metrics)
                    if scale_decision:
                        self.autoscaler.execute_scaling(self.router, scale_decision)

                # Record performance metrics
                duration_ms = (time.time() - start_time) * 1000
                self.performance_monitor.record_call(duration_ms, success=True)

                return result

            except Exception as e:
                # Record failure
                duration_ms = (time.time() - start_time) * 1000
                self.performance_monitor.record_call(duration_ms, success=False)
                raise e

        def _get_current_metrics(self) -> Dict[str, Any]:
            """Get current performance metrics for autoscaling."""
            perf_summary = self.performance_monitor.get_metrics_summary()

            return {
                'latency_p95_ms': perf_summary.get('latency_stats', {}).get('p95_ms', 0),
                'error_rate': perf_summary.get('error_rate', 0),
                'expert_utilization_avg': 0.5  # Placeholder
            }

        def get_optimization_stats(self) -> Dict[str, Any]:
            """Get comprehensive optimization statistics."""
            return {
                'performance': self.performance_monitor.get_metrics_summary(),
                'autoscaler': {
                    'current_experts': self.autoscaler.current_experts if self.autoscaler else 'disabled',
                    'last_scale_time': self.autoscaler.last_scale_time if self.autoscaler else 0
                },
                'memory_pool': {
                    'pool_sizes': {k: len(v) for k, v in self.memory_pool.tensor_pools.items()}
                }
            }

        def __del__(self):
            """Cleanup resources."""
            if hasattr(self, 'parallel_executor') and self.parallel_executor:
                self.parallel_executor.shutdown(wait=False)

    return OptimizedRouter(router)


# Utility functions for performance optimization
def optimize_tensor_operations(func: Callable) -> Callable:
    """Decorator to optimize tensor operations."""
    def wrapper(*args, **kwargs):
        # Could add tensor operation optimizations here
        return func(*args, **kwargs)
    return wrapper


class ResourceManager:
    """Manage computational resources across multiple routers."""

    def __init__(self, max_memory_mb: float = 1000.0, max_cpu_percent: float = 80.0):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.active_routers = {}

    def register_router(self, router_id: str, router: Any):
        """Register a router for resource management."""
        self.active_routers[router_id] = {
            'router': router,
            'last_active': time.time(),
            'memory_usage': 0.0,
            'cpu_usage': 0.0
        }

    def check_resource_usage(self) -> Dict[str, Any]:
        """Check current resource usage across all routers."""
        total_memory = sum(info['memory_usage'] for info in self.active_routers.values())
        avg_cpu = np.mean([info['cpu_usage'] for info in self.active_routers.values()]) if self.active_routers else 0

        return {
            'total_memory_mb': total_memory,
            'avg_cpu_percent': avg_cpu,
            'memory_limit_exceeded': total_memory > self.max_memory_mb,
            'cpu_limit_exceeded': avg_cpu > self.max_cpu_percent,
            'active_routers': len(self.active_routers)
        }

"""Performance optimization utilities for dynamic MoE routing."""

import threading
import time
import warnings
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .logging_config import PerformanceLogger


class MemoryPool:
    """Memory pool for reusing tensor allocations."""

    def __init__(self, max_size_mb: float = 100):
        self.max_size_mb = max_size_mb
        self.pools: Dict[Tuple, list] = defaultdict(list)
        self._lock = threading.Lock()
        self._total_size_bytes = 0

    def get_tensor(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """Get a tensor from the pool or allocate new one."""
        pool_key = (shape, dtype)

        with self._lock:
            if self.pools[pool_key]:
                tensor = self.pools[pool_key].pop()
                tensor.fill(0)  # Zero out reused tensor
                return tensor

        # Allocate new tensor
        return np.zeros(shape, dtype=dtype)

    def return_tensor(self, tensor: np.ndarray) -> None:
        """Return a tensor to the pool."""
        shape = tensor.shape
        dtype = tensor.dtype
        pool_key = (shape, dtype)

        # Check if pool is not full
        tensor_size_bytes = tensor.nbytes

        with self._lock:
            if (self._total_size_bytes + tensor_size_bytes) / (1024 * 1024) < self.max_size_mb:
                self.pools[pool_key].append(tensor)
                self._total_size_bytes += tensor_size_bytes
            # Otherwise, let tensor be garbage collected

    def clear(self) -> None:
        """Clear all pools."""
        with self._lock:
            self.pools.clear()
            self._total_size_bytes = 0


class RouterCache:
    """LRU cache for router computations."""

    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[float] = None):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self.cache:
                return None

            value, timestamp = self.cache[key]

            # Check TTL
            if self.ttl_seconds and time.time() - timestamp > self.ttl_seconds:
                del self.cache[key]
                del self._access_times[key]
                return None

            # Update access time
            self._access_times[key] = time.time()
            return value

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        current_time = time.time()

        with self._lock:
            # Evict old entries if cache is full
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()

            self.cache[key] = (value, current_time)
            self._access_times[key] = current_time

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_times:
            return

        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        del self.cache[lru_key]
        del self._access_times[lru_key]

    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self.cache.clear()
            self._access_times.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'utilization': len(self.cache) / self.max_size,
                'ttl_seconds': self.ttl_seconds
            }


class BatchProcessor:
    """Efficient batch processing for routing operations."""

    def __init__(self, optimal_batch_size: int = 32):
        self.optimal_batch_size = optimal_batch_size
        self.performance_logger = PerformanceLogger("batch_processor")

    def auto_batch_route(
        self,
        router_fn: Callable,
        inputs: list,
        **kwargs
    ) -> list:
        """Automatically batch routing operations for efficiency."""
        if len(inputs) <= self.optimal_batch_size:
            # Process all at once
            return self._process_batch(router_fn, inputs, **kwargs)

        # Process in optimal-sized batches
        results = []
        for i in range(0, len(inputs), self.optimal_batch_size):
            batch = inputs[i:i + self.optimal_batch_size]
            batch_results = self._process_batch(router_fn, batch, **kwargs)
            results.extend(batch_results)

        return results

    def _process_batch(self, router_fn: Callable, batch: list, **kwargs) -> list:
        """Process a single batch."""
        self.performance_logger.start_timer(f"batch_size_{len(batch)}")

        try:
            # Stack inputs if they're numpy arrays
            if batch and hasattr(batch[0], 'shape'):
                stacked_input = np.stack(batch, axis=0)
                stacked_result = router_fn(stacked_input, **kwargs)

                # Unstack results
                results = []
                for i in range(len(batch)):
                    result = {}
                    for key, value in stacked_result.items():
                        if hasattr(value, '__getitem__') and len(value) == len(batch):
                            result[key] = value[i]
                        else:
                            result[key] = value
                    results.append(result)

                return results
            else:
                # Process individually
                return [router_fn(inp, **kwargs) for inp in batch]

        finally:
            self.performance_logger.end_timer(f"batch_size_{len(batch)}")


class VectorizedOperations:
    """Vectorized implementations of common operations."""

    @staticmethod
    def fast_softmax(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
        """Optimized softmax computation with numerical stability."""
        if temperature != 1.0:
            x = x / temperature

        # Numerical stability: subtract max
        x_max = np.max(x, axis=axis, keepdims=True)
        x_shifted = x - x_max

        # Compute exponentials
        exp_x = np.exp(x_shifted)

        # Normalize
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    @staticmethod
    def fast_top_k(x: np.ndarray, k: int, axis: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized top-k selection."""
        if k >= x.shape[axis]:
            # Return all elements sorted in descending order
            indices = np.argsort(x, axis=axis)
            if axis == -1:
                indices = indices[..., ::-1]  # Descending order
            else:
                indices = np.flip(indices, axis=axis)
            values = np.take_along_axis(x, indices, axis=axis)
            return values, indices

        # Simple but reliable implementation using argsort
        # For small k, argpartition would be more efficient, but argsort is more stable
        sorted_indices = np.argsort(x, axis=axis)

        # Get top-k in descending order
        if axis == -1:
            top_k_indices = sorted_indices[..., -k:]  # Last k elements
            top_k_indices = top_k_indices[..., ::-1]  # Reverse for descending
        else:
            # Handle arbitrary axis
            slices = [slice(None)] * x.ndim
            slices[axis] = slice(-k, None)
            top_k_indices = sorted_indices[tuple(slices)]
            top_k_indices = np.flip(top_k_indices, axis=axis)

        # Get corresponding values
        top_k_values = np.take_along_axis(x, top_k_indices, axis=axis)

        return top_k_values, top_k_indices

    @staticmethod
    def fast_expert_dispatch(
        hidden_states: np.ndarray,
        expert_indices: np.ndarray,
        expert_weights: np.ndarray
    ) -> np.ndarray:
        """Vectorized expert dispatching."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        _, _, max_k = expert_indices.shape

        # Initialize output
        output = np.zeros_like(hidden_states)

        # Create masks for each expert
        for expert_id in range(expert_indices.max() + 1):
            if expert_id == -1:  # Skip padding
                continue

            # Find all positions assigned to this expert
            expert_mask = (expert_indices == expert_id)

            if not expert_mask.any():
                continue

            # Get positions and weights
            batch_idx, seq_idx, k_idx = np.where(expert_mask)
            weights = expert_weights[batch_idx, seq_idx, k_idx]

            # Gather inputs for this expert
            expert_inputs = hidden_states[batch_idx, seq_idx]  # [num_tokens, hidden_dim]

            # Simulate expert computation (would be replaced with actual expert)
            expert_outputs = expert_inputs * 0.9  # Placeholder computation

            # Weight and scatter back
            weighted_outputs = expert_outputs * weights[:, np.newaxis]

            # Add to output
            np.add.at(output, (batch_idx, seq_idx), weighted_outputs)

        return output


def cache_complexity_scores(cache_size: int = 500):
    """Decorator to cache complexity estimation results."""
    cache = RouterCache(max_size=cache_size, ttl_seconds=60.0)  # 1 minute TTL

    def decorator(func):
        @wraps(func)
        def wrapper(self, hidden_states, **kwargs):
            # Create cache key from input hash and kwargs
            input_hash = hash(hidden_states.data.tobytes()[:1000])  # Hash first 1000 bytes
            kwargs_key = str(sorted(kwargs.items()))
            cache_key = f"{input_hash}_{kwargs_key}"

            # Try cache first
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Compute and cache
            result = func(self, hidden_states, **kwargs)
            cache.set(cache_key, result)

            return result

        # Add cache management methods
        wrapper.cache_stats = cache.stats
        wrapper.clear_cache = cache.clear

        return wrapper

    return decorator


def profile_performance(operation_name: str):
    """Decorator to profile function performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = PerformanceLogger(operation_name)
            logger.start_timer(func.__name__)

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                logger.end_timer(func.__name__)

        return wrapper

    return decorator


class AdaptiveOptimizer:
    """Adaptive optimization based on runtime performance."""

    def __init__(self):
        self.performance_history: Dict[str, list] = defaultdict(list)
        self.optimal_configs: Dict[str, Any] = {}

    def record_performance(
        self,
        operation: str,
        config: Dict[str, Any],
        duration: float,
        memory_mb: float = 0
    ) -> None:
        """Record performance for an operation configuration."""
        perf_score = 1.0 / (duration + 0.001)  # Higher is better
        if memory_mb > 0:
            perf_score /= (1 + memory_mb / 1000)  # Penalize memory usage

        self.performance_history[operation].append({
            'config': config,
            'duration': duration,
            'memory_mb': memory_mb,
            'score': perf_score
        })

        # Keep only recent history
        if len(self.performance_history[operation]) > 100:
            self.performance_history[operation] = self.performance_history[operation][-50:]

        # Update optimal config
        self._update_optimal_config(operation)

    def _update_optimal_config(self, operation: str) -> None:
        """Update optimal configuration for an operation."""
        history = self.performance_history[operation]
        if len(history) < 5:  # Need some data
            return

        # Find best performing config
        best_entry = max(history, key=lambda x: x['score'])
        self.optimal_configs[operation] = best_entry['config']

    def get_optimal_config(self, operation: str) -> Optional[Dict[str, Any]]:
        """Get optimal configuration for an operation."""
        return self.optimal_configs.get(operation)

    def suggest_batch_size(self, operation: str, default: int = 32) -> int:
        """Suggest optimal batch size based on performance history."""
        optimal_config = self.get_optimal_config(operation)
        if optimal_config and 'batch_size' in optimal_config:
            return optimal_config['batch_size']
        return default


# Global instances
_memory_pool = MemoryPool()
_adaptive_optimizer = AdaptiveOptimizer()


def get_memory_pool() -> MemoryPool:
    """Get global memory pool instance."""
    return _memory_pool


def get_adaptive_optimizer() -> AdaptiveOptimizer:
    """Get global adaptive optimizer instance."""
    return _adaptive_optimizer


def optimize_for_throughput(func):
    """Decorator to optimize function for maximum throughput."""
    optimizer = get_adaptive_optimizer()

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract configuration from kwargs
        config = {k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool))}

        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            optimizer.record_performance(func.__name__, config, duration)

    return wrapper


def enable_fast_math():
    """Enable fast math optimizations where available."""
    # Set NumPy to use optimized BLAS
    try:
        import os
        os.environ['OMP_NUM_THREADS'] = str(min(8, os.cpu_count() or 1))
        os.environ['MKL_NUM_THREADS'] = str(min(8, os.cpu_count() or 1))
    except Exception:
        pass

    # Set NumPy error handling for performance
    np.seterr(all='ignore')  # Ignore numerical warnings for speed

    warnings.filterwarnings('ignore', category=RuntimeWarning)


def benchmark_operation(
    func: Callable,
    args: Tuple,
    kwargs: Dict[str, Any],
    num_trials: int = 10,
    warmup_trials: int = 2
) -> Dict[str, float]:
    """Benchmark an operation and return performance statistics."""
    # Warmup
    for _ in range(warmup_trials):
        func(*args, **kwargs)

    # Benchmark
    times = []
    for _ in range(num_trials):
        start_time = time.time()
        func(*args, **kwargs)
        times.append(time.time() - start_time)

    times = np.array(times)

    return {
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'median_time': float(np.median(times)),
        'p95_time': float(np.percentile(times, 95)),
        'throughput_ops_per_sec': 1.0 / np.mean(times)
    }

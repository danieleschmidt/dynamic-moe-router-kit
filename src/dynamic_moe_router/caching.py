"""Intelligent caching system for dynamic MoE routing."""

import hashlib
import pickle
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np


class RoutingCache:
    """LRU cache for routing decisions with intelligent invalidation."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._timestamps = {}
        self._hit_count = 0
        self._miss_count = 0
        self._lock = threading.RLock()

    def get(self, input_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached routing result if valid."""
        with self._lock:
            if input_hash not in self._cache:
                self._miss_count += 1
                return None

            # Check TTL
            timestamp = self._timestamps[input_hash]
            if time.time() - timestamp > self.ttl_seconds:
                self._remove(input_hash)
                self._miss_count += 1
                return None

            # Move to end (most recently used)
            result = self._cache.pop(input_hash)
            self._cache[input_hash] = result
            self._hit_count += 1

            return result

    def put(self, input_hash: str, routing_result: Dict[str, Any]):
        """Cache routing result with timestamp."""
        with self._lock:
            # Remove oldest if at capacity
            if len(self._cache) >= self.max_size and input_hash not in self._cache:
                oldest_key = next(iter(self._cache))
                self._remove(oldest_key)

            self._cache[input_hash] = routing_result.copy()
            self._timestamps[input_hash] = time.time()

            # Move to end
            if input_hash in self._cache:
                self._cache.move_to_end(input_hash)

    def _remove(self, key: str):
        """Remove entry from cache."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_requests if total_requests > 0 else 0.0

        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hit_count': self._hit_count,
            'miss_count': self._miss_count,
            'hit_rate': hit_rate,
            'ttl_seconds': self.ttl_seconds
        }

    def cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._timestamps.items()
            if current_time - timestamp > self.ttl_seconds
        ]

        with self._lock:
            for key in expired_keys:
                self._remove(key)


class AdaptiveCache:
    """Cache that adapts size and TTL based on usage patterns."""

    def __init__(self, initial_size: int = 500, min_size: int = 100, max_size: int = 5000):
        self.min_size = min_size
        self.max_size = max_size
        self.current_size = initial_size

        self.cache = RoutingCache(max_size=self.current_size, ttl_seconds=300.0)

        # Adaptation parameters
        self._adaptation_interval = 100  # Adapt every N requests
        self._request_count = 0
        self._recent_hit_rates = []
        self._window_size = 10

    def get(self, input_hash: str) -> Optional[Dict[str, Any]]:
        """Get with adaptive behavior tracking."""
        result = self.cache.get(input_hash)
        self._request_count += 1

        # Periodic adaptation
        if self._request_count % self._adaptation_interval == 0:
            self._adapt_parameters()

        return result

    def put(self, input_hash: str, routing_result: Dict[str, Any]):
        """Put with size management."""
        self.cache.put(input_hash, routing_result)

    def _adapt_parameters(self):
        """Adapt cache parameters based on recent performance."""
        stats = self.cache.get_stats()
        current_hit_rate = stats['hit_rate']

        self._recent_hit_rates.append(current_hit_rate)
        if len(self._recent_hit_rates) > self._window_size:
            self._recent_hit_rates.pop(0)

        if len(self._recent_hit_rates) < 3:
            return  # Not enough data

        avg_hit_rate = np.mean(self._recent_hit_rates)

        # Adapt cache size based on hit rate
        if avg_hit_rate < 0.3 and self.current_size > self.min_size:
            # Poor hit rate - reduce cache size
            new_size = max(self.min_size, int(self.current_size * 0.8))
            self._resize_cache(new_size)
        elif avg_hit_rate > 0.7 and self.current_size < self.max_size:
            # Good hit rate - increase cache size
            new_size = min(self.max_size, int(self.current_size * 1.2))
            self._resize_cache(new_size)

    def _resize_cache(self, new_size: int):
        """Resize the underlying cache."""
        if new_size != self.current_size:
            # Create new cache with new size, preserving most recent entries
            old_cache = self.cache._cache
            old_timestamps = self.cache._timestamps

            # Keep most recent entries up to new size
            items_to_keep = list(old_cache.items())[-min(len(old_cache), new_size):]

            # Create new cache
            self.cache = RoutingCache(max_size=new_size, ttl_seconds=self.cache.ttl_seconds)

            # Restore kept entries
            for key, value in items_to_keep:
                if key in old_timestamps:
                    self.cache._cache[key] = value
                    self.cache._timestamps[key] = old_timestamps[key]

            self.current_size = new_size


class InputHasher:
    """Intelligent hashing for input tensors and routing parameters."""

    @staticmethod
    def hash_routing_input(
        hidden_states: Any,
        router_config: Dict[str, Any],
        precision: int = 4
    ) -> str:
        """Create hash for routing input that balances precision with cache effectiveness."""

        # Hash tensor shape and dtype info
        shape_hash = hashlib.md5(str(hidden_states.shape).encode()).hexdigest()[:8]

        # Hash a sample of the tensor values (for performance)
        if hasattr(hidden_states, 'flatten'):
            flat_tensor = hidden_states.flatten()
            # Sample subset for large tensors
            sample_size = min(1000, len(flat_tensor))
            indices = np.linspace(0, len(flat_tensor) - 1, sample_size, dtype=int)
            sample_values = flat_tensor[indices]

            # Round to reduce precision for better caching
            rounded_values = np.round(sample_values, precision)
            tensor_hash = hashlib.md5(rounded_values.tobytes()).hexdigest()[:8]
        else:
            tensor_hash = hashlib.md5(str(hidden_states).encode()).hexdigest()[:8]

        # Hash router configuration
        config_str = str(sorted(router_config.items()))
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        # Combine all hashes
        combined_hash = f"{shape_hash}_{tensor_hash}_{config_hash}"

        return combined_hash

    @staticmethod
    def should_cache(hidden_states: Any, routing_result: Dict[str, Any]) -> bool:
        """Determine if routing result is worth caching."""

        # Don't cache very small inputs (overhead not worth it)
        if hasattr(hidden_states, 'size'):
            if hidden_states.size < 100:
                return False

        # Don't cache if routing was very fast (< 1ms equivalent in complexity)
        routing_info = routing_result.get('routing_info', {})
        avg_experts = routing_info.get('avg_experts_per_token', 1)

        # Cache if using multiple experts (more expensive computation)
        if avg_experts > 1.5:
            return True

        # Cache based on complexity scores variance
        complexity_stats = routing_info.get('complexity_stats', {})
        if complexity_stats.get('std', 0) > 0.1:  # High variance = worth caching
            return True

        return False


def create_cached_router(router, cache_size: int = 1000, adaptive: bool = True):
    """Create a cached wrapper around a router."""

    class CachedRouter:
        def __init__(self, base_router):
            self.router = base_router

            if adaptive:
                self.cache = AdaptiveCache(initial_size=cache_size)
            else:
                self.cache = RoutingCache(max_size=cache_size)

            self.hasher = InputHasher()

            # Delegate attribute access
            for attr in dir(base_router):
                if not attr.startswith('_') and attr not in ['route']:
                    setattr(self, attr, getattr(base_router, attr))

        def route(self, hidden_states, return_router_logits=False, **kwargs):
            """Cached routing with intelligent cache management."""

            # Create router config for hashing
            router_config = {
                'num_experts': self.router.num_experts,
                'min_experts': self.router.min_experts,
                'max_experts': self.router.max_experts,
                'routing_strategy': self.router.routing_strategy,
                'return_router_logits': return_router_logits,
                'kwargs_hash': hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()[:8]
            }

            # Generate cache key
            cache_key = self.hasher.hash_routing_input(hidden_states, router_config)

            # Try cache first
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Cache miss - compute routing
            result = self.router.route(hidden_states, return_router_logits, **kwargs)

            # Cache if worthwhile
            if self.hasher.should_cache(hidden_states, result):
                self.cache.put(cache_key, result)

            return result

        def get_cache_stats(self) -> Dict[str, Any]:
            """Get cache performance statistics."""
            if hasattr(self.cache, 'cache'):
                return self.cache.cache.get_stats()
            else:
                return self.cache.get_stats()

        def clear_cache(self):
            """Clear the routing cache."""
            if hasattr(self.cache, 'cache'):
                self.cache.cache.clear()
            else:
                self.cache.clear()

    return CachedRouter(router)


class ResultCompressor:
    """Compress routing results to save memory in cache."""

    @staticmethod
    def compress_result(routing_result: Dict[str, Any]) -> bytes:
        """Compress routing result for storage."""
        # Remove large arrays that can be recomputed if needed
        compressed_result = routing_result.copy()

        # Keep only essential information
        essential_keys = [
            'expert_indices', 'expert_weights', 'num_experts_per_token',
            'routing_info'
        ]

        compressed_result = {k: v for k, v in compressed_result.items()
                           if k in essential_keys}

        return pickle.dumps(compressed_result)

    @staticmethod
    def decompress_result(compressed_data: bytes) -> Dict[str, Any]:
        """Decompress routing result."""
        return pickle.loads(compressed_data)


# Background cache maintenance
class CacheMaintenanceWorker:
    """Background worker for cache maintenance tasks."""

    def __init__(self, caches: List[RoutingCache]):
        self.caches = caches
        self._stop_event = threading.Event()
        self._worker_thread = None

    def start(self):
        """Start background maintenance worker."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(target=self._maintenance_loop)
            self._worker_thread.daemon = True
            self._worker_thread.start()

    def stop(self):
        """Stop background maintenance worker."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)

    def _maintenance_loop(self):
        """Main maintenance loop."""
        while not self._stop_event.is_set():
            for cache in self.caches:
                try:
                    cache.cleanup_expired()
                except Exception:
                    # Log error but continue with other caches
                    pass

            # Wait before next maintenance cycle
            self._stop_event.wait(timeout=60.0)  # Run every minute

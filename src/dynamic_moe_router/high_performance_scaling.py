"""High-performance scaling and optimization for dynamic MoE routing."""

import logging
import time
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional, Callable, Tuple
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""
    # Caching
    cache_size: int = 10000
    enable_result_caching: bool = True
    cache_ttl_seconds: float = 300.0  # 5 minutes
    
    # Batching
    max_batch_size: int = 128
    batch_timeout_ms: float = 10.0
    enable_dynamic_batching: bool = True
    
    # Threading and concurrency
    thread_pool_size: int = 8
    max_concurrent_requests: int = 100
    enable_async_processing: bool = True
    
    # Memory optimization
    enable_memory_pooling: bool = True
    memory_pool_size: int = 1000
    enable_garbage_collection: bool = True
    gc_interval_seconds: float = 60.0
    
    # Computational optimization
    enable_vectorization: bool = True
    enable_jit_compilation: bool = True
    enable_gradient_checkpointing: bool = True


class AdaptiveCache:
    """Adaptive LRU cache with TTL and performance monitoring."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: float = 300.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._access_times = {}
        self._insert_times = {}
        self._lock = threading.RLock()
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with TTL check."""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            # Check TTL
            if time.time() - self._insert_times[key] > self.ttl_seconds:
                self._evict(key)
                self.misses += 1
                return None
            
            # Update access time for LRU
            self._access_times[key] = time.time()
            self.hits += 1
            return self._cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache with automatic eviction."""
        with self._lock:
            current_time = time.time()
            
            # Remove old entry if exists
            if key in self._cache:
                del self._cache[key]
                del self._access_times[key]
                del self._insert_times[key]
            
            # Evict if at capacity
            if len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Add new entry
            self._cache[key] = value
            self._access_times[key] = current_time
            self._insert_times[key] = current_time
    
    def _evict(self, key: str) -> None:
        """Evict specific key."""
        if key in self._cache:
            del self._cache[key]
            del self._access_times[key]
            del self._insert_times[key]
            self.evictions += 1
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        self._evict(lru_key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._insert_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(total_requests, 1)
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate': hit_rate,
                'utilization': len(self._cache) / self.max_size
            }


class BatchProcessor:
    """Dynamic batching processor for efficient inference."""
    
    def __init__(self, max_batch_size: int = 128, timeout_ms: float = 10.0):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self._pending_requests = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._shutdown = False
        
        # Start background batch processor
        self._processor_thread = threading.Thread(target=self._process_batches)
        self._processor_thread.daemon = True
        self._processor_thread.start()
    
    def submit(self, request_data: Any, callback: Callable) -> None:
        """Submit request for batch processing."""
        with self._condition:
            if self._shutdown:
                raise RuntimeError("BatchProcessor is shut down")
            
            self._pending_requests.append({
                'data': request_data,
                'callback': callback,
                'timestamp': time.time()
            })
            
            # Notify processor if batch is full
            if len(self._pending_requests) >= self.max_batch_size:
                self._condition.notify()
    
    def _process_batches(self) -> None:
        """Background thread to process batches."""
        while not self._shutdown:
            with self._condition:
                # Wait for requests or timeout
                self._condition.wait(timeout=self.timeout_ms / 1000.0)
                
                if not self._pending_requests or self._shutdown:
                    continue
                
                # Extract batch
                batch_size = min(len(self._pending_requests), self.max_batch_size)
                batch_requests = self._pending_requests[:batch_size]
                self._pending_requests = self._pending_requests[batch_size:]
            
            # Process batch in executor
            if batch_requests:
                self._executor.submit(self._execute_batch, batch_requests)
    
    def _execute_batch(self, batch_requests: List[Dict[str, Any]]) -> None:
        """Execute a batch of requests."""
        try:
            # Extract data and callbacks
            batch_data = [req['data'] for req in batch_requests]
            callbacks = [req['callback'] for req in batch_requests]
            
            # Process batch (placeholder - actual implementation would call router)
            results = self._mock_batch_process(batch_data)
            
            # Execute callbacks with results
            for callback, result in zip(callbacks, results):
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Callback execution failed: {e}")
                    
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Call callbacks with error
            for req in batch_requests:
                try:
                    req['callback'](None, error=e)
                except Exception:
                    pass
    
    def _mock_batch_process(self, batch_data: List[Any]) -> List[Any]:
        """Mock batch processing (replace with actual router calls)."""
        return [{'processed': True, 'data': data} for data in batch_data]
    
    def shutdown(self) -> None:
        """Shutdown batch processor."""
        self._shutdown = True
        with self._condition:
            self._condition.notify_all()
        
        self._processor_thread.join(timeout=5.0)
        self._executor.shutdown(wait=True)


class MemoryPool:
    """Memory pool for reusing arrays and reducing allocations."""
    
    def __init__(self, pool_size: int = 1000):
        self.pool_size = pool_size
        self._arrays = {}  # dtype -> shape -> list of arrays
        self._lock = threading.Lock()
        self.allocations = 0
        self.reuses = 0
    
    def get_array(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """Get array from pool or allocate new one."""
        cache_key = (tuple(shape), dtype)
        
        with self._lock:
            if cache_key in self._arrays and self._arrays[cache_key]:
                array = self._arrays[cache_key].pop()
                self.reuses += 1
                # Clear the array
                array.fill(0)
                return array
        
        # Allocate new array
        self.allocations += 1
        return np.zeros(shape, dtype=dtype)
    
    def return_array(self, array: np.ndarray) -> None:
        """Return array to pool for reuse."""
        cache_key = (tuple(array.shape), array.dtype)
        
        with self._lock:
            if cache_key not in self._arrays:
                self._arrays[cache_key] = []
            
            # Only keep up to pool_size arrays per type
            if len(self._arrays[cache_key]) < self.pool_size:
                self._arrays[cache_key].append(array)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            total_arrays = sum(len(arrays) for arrays in self._arrays.values())
            reuse_rate = self.reuses / max(self.allocations + self.reuses, 1)
            
            return {
                'total_pooled_arrays': total_arrays,
                'array_types': len(self._arrays),
                'allocations': self.allocations,
                'reuses': self.reuses,
                'reuse_rate': reuse_rate
            }


class PerformanceOptimizer:
    """Comprehensive performance optimization manager."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        
        # Initialize components
        self.cache = AdaptiveCache(config.cache_size, config.cache_ttl_seconds)
        self.batch_processor = BatchProcessor(config.max_batch_size, config.batch_timeout_ms)
        self.memory_pool = MemoryPool(config.memory_pool_size)
        
        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=config.thread_pool_size)
        
        # Performance tracking
        self.total_requests = 0
        self.cache_enabled_requests = 0
        self.batched_requests = 0
        
        # Garbage collection
        if config.enable_garbage_collection:
            self._start_gc_thread()
    
    def optimized_routing_cache(self, router_func: Callable) -> Callable:
        """Decorator for caching router results."""
        @wraps(router_func)
        def wrapper(*args, **kwargs):
            self.total_requests += 1
            
            if not self.config.enable_result_caching:
                return router_func(*args, **kwargs)
            
            # Create cache key from input hash
            cache_key = self._create_cache_key(args, kwargs)
            
            # Try cache first
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.cache_enabled_requests += 1
                return cached_result
            
            # Compute result and cache
            result = router_func(*args, **kwargs)
            self.cache.put(cache_key, result)
            
            return result
        
        return wrapper
    
    def async_route(self, router_func: Callable, *args, **kwargs) -> asyncio.Future:
        """Asynchronous routing with optimization."""
        if not self.config.enable_async_processing:
            return asyncio.get_event_loop().run_in_executor(
                None, router_func, *args, **kwargs
            )
        
        # Submit to thread pool
        future = self.executor.submit(router_func, *args, **kwargs)
        
        # Convert to asyncio future
        loop = asyncio.get_event_loop()
        async_future = loop.create_future()
        
        def done_callback(thread_future):
            try:
                result = thread_future.result()
                loop.call_soon_threadsafe(async_future.set_result, result)
            except Exception as e:
                loop.call_soon_threadsafe(async_future.set_exception, e)
        
        future.add_done_callback(done_callback)
        return async_future
    
    def batch_route(self, requests: List[Any], router_func: Callable) -> List[Any]:
        """Batch multiple routing requests for efficiency."""
        if not self.config.enable_dynamic_batching or len(requests) == 1:
            return [router_func(req) for req in requests]
        
        self.batched_requests += len(requests)
        
        # Process in optimal batch sizes
        results = []
        for i in range(0, len(requests), self.config.max_batch_size):
            batch = requests[i:i + self.config.max_batch_size]
            
            # Combine inputs for vectorized processing
            if self.config.enable_vectorization:
                batch_result = self._vectorized_route(batch, router_func)
            else:
                batch_result = [router_func(req) for req in batch]
            
            results.extend(batch_result)
        
        return results
    
    def _vectorized_route(self, batch: List[Any], router_func: Callable) -> List[Any]:
        """Vectorized routing for better performance."""
        try:
            # Stack inputs if they're numpy arrays
            if all(isinstance(item, np.ndarray) for item in batch):
                stacked_input = np.stack(batch, axis=0)
                batch_result = router_func(stacked_input)
                
                # Split results back
                if isinstance(batch_result, dict):
                    return self._split_batch_result(batch_result, len(batch))
                else:
                    return [batch_result[i] for i in range(len(batch))]
            else:
                # Fallback to individual processing
                return [router_func(req) for req in batch]
                
        except Exception as e:
            logger.warning(f"Vectorized routing failed, falling back: {e}")
            return [router_func(req) for req in batch]
    
    def _split_batch_result(self, batch_result: Dict[str, Any], batch_size: int) -> List[Dict[str, Any]]:
        """Split batched result back into individual results."""
        individual_results = []
        
        for i in range(batch_size):
            result = {}
            for key, value in batch_result.items():
                if isinstance(value, np.ndarray) and len(value.shape) > 0:
                    result[key] = value[i]
                else:
                    result[key] = value
            individual_results.append(result)
        
        return individual_results
    
    def _create_cache_key(self, args: Tuple, kwargs: Dict) -> str:
        """Create cache key from function arguments."""
        try:
            # Simple hash-based key (could be more sophisticated)
            key_parts = []
            
            for arg in args:
                if isinstance(arg, np.ndarray):
                    key_parts.append(f"arr_{arg.shape}_{hash(arg.data.tobytes())}")
                else:
                    key_parts.append(str(hash(str(arg))))
            
            for k, v in sorted(kwargs.items()):
                if isinstance(v, np.ndarray):
                    key_parts.append(f"{k}_arr_{v.shape}_{hash(v.data.tobytes())}")
                else:
                    key_parts.append(f"{k}_{hash(str(v))}")
            
            return "_".join(key_parts)
            
        except Exception:
            # Fallback to simple string representation
            return f"{hash(str(args))}_{hash(str(sorted(kwargs.items())))}"
    
    def _start_gc_thread(self) -> None:
        """Start garbage collection thread."""
        def gc_worker():
            import gc
            while True:
                time.sleep(self.config.gc_interval_seconds)
                collected = gc.collect()
                if collected > 0:
                    logger.debug(f"Garbage collector freed {collected} objects")
        
        gc_thread = threading.Thread(target=gc_worker, daemon=True)
        gc_thread.start()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        cache_hit_rate = self.cache_enabled_requests / max(self.total_requests, 1)
        batch_rate = self.batched_requests / max(self.total_requests, 1)
        
        return {
            'total_requests': self.total_requests,
            'cache_hit_rate': cache_hit_rate,
            'batch_rate': batch_rate,
            'cache_stats': self.cache.get_stats(),
            'memory_pool_stats': self.memory_pool.get_stats(),
            'config': {
                'cache_enabled': self.config.enable_result_caching,
                'batching_enabled': self.config.enable_dynamic_batching,
                'async_enabled': self.config.enable_async_processing,
                'vectorization_enabled': self.config.enable_vectorization
            }
        }
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.total_requests = 0
        self.cache_enabled_requests = 0
        self.batched_requests = 0
        self.cache.hits = 0
        self.cache.misses = 0
        self.cache.evictions = 0
        self.memory_pool.allocations = 0
        self.memory_pool.reuses = 0
    
    def shutdown(self) -> None:
        """Shutdown performance optimizer."""
        self.batch_processor.shutdown()
        self.executor.shutdown(wait=True)
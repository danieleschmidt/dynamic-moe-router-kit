"""High-performance optimizations for dynamic MoE routing."""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Advanced performance optimization for MoE routing."""
    
    def __init__(
        self,
        enable_caching: bool = True,
        cache_size: int = 1000,
        enable_vectorization: bool = True,
        enable_batching: bool = True,
        batch_timeout_ms: int = 10,
        max_batch_size: int = 128,
        enable_async: bool = True,
        thread_pool_size: int = 4
    ):
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.enable_vectorization = enable_vectorization
        self.enable_batching = enable_batching
        self.batch_timeout_ms = batch_timeout_ms
        self.max_batch_size = max_batch_size
        self.enable_async = enable_async
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.batch_operations = 0
        self.total_operations = 0
        
        # Thread pool for async operations
        if enable_async:
            self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        else:
            self.thread_pool = None
        
        # Batching queue
        if enable_batching:
            self.batch_queue = []
            self.batch_lock = threading.Lock()
            self.batch_condition = threading.Condition(self.batch_lock)
            self.batch_worker_running = False
            self._start_batch_worker()
        
        logger.info(f"Performance optimizer initialized with optimizations: "
                   f"caching={enable_caching}, vectorization={enable_vectorization}, "
                   f"batching={enable_batching}, async={enable_async}")
    
    def optimized_routing_cache(self, func):
        """Decorator for intelligent routing result caching."""
        
        if not self.enable_caching:
            return func
        
        @lru_cache(maxsize=self.cache_size)
        def cached_computation(input_hash: str, config_hash: str):
            """Cached computation with hash-based keys."""
            return func.__wrapped_call__(input_hash, config_hash)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enable_caching:
                return func(*args, **kwargs)
            
            # Create hash keys for inputs and config
            input_hash = self._hash_inputs(args, kwargs)
            config_hash = self._hash_config(kwargs)
            
            try:
                self.cache_hits += 1
                return cached_computation(input_hash, config_hash)
            except TypeError:
                # Fallback for non-hashable inputs
                self.cache_misses += 1
                return func(*args, **kwargs)
        
        # Store original function for cached_computation
        func.__wrapped_call__ = func
        return wrapper
    
    def vectorized_expert_selection(
        self,
        router_logits: np.ndarray,
        k_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized vectorized expert selection."""
        
        if not self.enable_vectorization:
            return self._standard_expert_selection(router_logits, k_values)
        
        batch_size, seq_len, num_experts = router_logits.shape
        max_k = int(np.max(k_values))
        
        # Vectorized top-k selection
        indices = np.argpartition(router_logits, -max_k, axis=-1)[..., -max_k:]
        
        # Sort the top-k indices by logit values
        batch_indices = np.arange(batch_size)[:, None, None]
        seq_indices = np.arange(seq_len)[None, :, None]
        
        # Get logits for top-k indices
        top_k_logits = router_logits[batch_indices, seq_indices, indices]
        
        # Sort indices by logit values (descending)
        sort_indices = np.argsort(-top_k_logits, axis=-1)
        sorted_indices = np.take_along_axis(indices, sort_indices, axis=-1)
        sorted_logits = np.take_along_axis(top_k_logits, sort_indices, axis=-1)
        
        # Prepare outputs
        expert_indices = np.zeros((batch_size, seq_len, max_k), dtype=np.int32)
        expert_weights = np.zeros((batch_size, seq_len, max_k), dtype=np.float32)
        
        # Vectorized softmax computation
        for i in range(batch_size):
            for j in range(seq_len):
                k_ij = k_values[i, j]
                if k_ij > 0:
                    # Select top k_ij experts
                    selected_indices = sorted_indices[i, j, :k_ij]
                    selected_logits = sorted_logits[i, j, :k_ij]
                    
                    # Compute softmax weights
                    exp_logits = np.exp(selected_logits - np.max(selected_logits))
                    weights = exp_logits / np.sum(exp_logits)
                    
                    expert_indices[i, j, :k_ij] = selected_indices
                    expert_weights[i, j, :k_ij] = weights
        
        return expert_indices, expert_weights
    
    def _standard_expert_selection(
        self,
        router_logits: np.ndarray,
        k_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Standard (non-vectorized) expert selection for fallback."""
        
        batch_size, seq_len, num_experts = router_logits.shape
        max_k = int(np.max(k_values))
        
        expert_indices = np.zeros((batch_size, seq_len, max_k), dtype=np.int32)
        expert_weights = np.zeros((batch_size, seq_len, max_k), dtype=np.float32)
        
        for i in range(batch_size):
            for j in range(seq_len):
                k_ij = k_values[i, j]
                if k_ij > 0:
                    # Get top-k indices
                    top_k_indices = np.argpartition(router_logits[i, j], -k_ij)[-k_ij:]
                    top_k_logits = router_logits[i, j, top_k_indices]
                    
                    # Sort by logit values
                    sort_order = np.argsort(-top_k_logits)
                    sorted_indices = top_k_indices[sort_order]
                    sorted_logits = top_k_logits[sort_order]
                    
                    # Compute softmax weights
                    exp_logits = np.exp(sorted_logits - np.max(sorted_logits))
                    weights = exp_logits / np.sum(exp_logits)
                    
                    expert_indices[i, j, :k_ij] = sorted_indices
                    expert_weights[i, j, :k_ij] = weights
        
        return expert_indices, expert_weights
    
    def _hash_inputs(self, args: Tuple, kwargs: Dict) -> str:
        """Create hash for caching inputs."""
        try:
            # Hash tensor shapes and key properties, not full values
            hash_components = []
            
            for arg in args[1:]:  # Skip self
                if hasattr(arg, 'shape'):
                    hash_components.append(f"shape_{arg.shape}")
                    if hasattr(arg, 'dtype'):
                        hash_components.append(f"dtype_{arg.dtype}")
                else:
                    hash_components.append(str(arg))
            
            # Include relevant kwargs
            for key in ['return_router_logits', 'return_load_balancing_loss']:
                if key in kwargs:
                    hash_components.append(f"{key}_{kwargs[key]}")
            
            return "_".join(hash_components)
        except Exception:
            return f"fallback_{time.time()}"
    
    def _hash_config(self, kwargs: Dict) -> str:
        """Create hash for router configuration."""
        try:
            config_keys = ['routing_strategy', 'temperature', 'noise_factor']
            hash_components = [f"{k}_{kwargs.get(k, 'default')}" for k in config_keys]
            return "_".join(hash_components)
        except Exception:
            return "default_config"
    
    def _start_batch_worker(self):
        """Start background worker for batch processing."""
        if not self.enable_batching or self.batch_worker_running:
            return
        
        def batch_worker():
            self.batch_worker_running = True
            while self.batch_worker_running:
                with self.batch_condition:
                    # Wait for batches or timeout
                    self.batch_condition.wait(timeout=self.batch_timeout_ms / 1000.0)
                    
                    if len(self.batch_queue) > 0:
                        batch = self.batch_queue[:]
                        self.batch_queue.clear()
                        
                        # Process batch outside lock
                        self._process_batch(batch)
        
        batch_thread = threading.Thread(target=batch_worker, daemon=True)
        batch_thread.start()
    
    def _process_batch(self, batch: List[Dict]):
        """Process a batch of routing requests."""
        if not batch:
            return
        
        self.batch_operations += 1
        logger.debug(f"Processing batch of {len(batch)} requests")
        
        # Group similar requests for more efficient processing
        grouped_batches = self._group_similar_requests(batch)
        
        for group in grouped_batches:
            try:
                self._execute_batch_group(group)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Fallback to individual processing
                for request in group:
                    try:
                        request['callback'](request['func'](*request['args'], **request['kwargs']))
                    except Exception as fallback_error:
                        request['error_callback'](fallback_error)
    
    def _group_similar_requests(self, batch: List[Dict]) -> List[List[Dict]]:
        """Group similar requests for more efficient batch processing."""
        # Simple grouping by input shape for now
        groups = {}
        
        for request in batch:
            # Extract shape information
            shape_key = "unknown"
            if request['args'] and hasattr(request['args'][1], 'shape'):
                shape_key = str(request['args'][1].shape)
            
            if shape_key not in groups:
                groups[shape_key] = []
            groups[shape_key].append(request)
        
        return list(groups.values())
    
    def _execute_batch_group(self, group: List[Dict]):
        """Execute a group of similar requests efficiently."""
        # For now, execute individually
        # TODO: Implement true batch execution for compatible requests
        for request in group:
            try:
                result = request['func'](*request['args'], **request['kwargs'])
                request['callback'](result)
            except Exception as e:
                request['error_callback'](e)
    
    async def async_route(self, router_func, *args, **kwargs):
        """Asynchronous routing wrapper."""
        if not self.enable_async or not self.thread_pool:
            return router_func(*args, **kwargs)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool, router_func, *args, **kwargs
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance optimization statistics."""
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / max(cache_total, 1)
        
        return {
            "caching": {
                "enabled": self.enable_caching,
                "hit_rate": cache_hit_rate,
                "total_hits": self.cache_hits,
                "total_misses": self.cache_misses
            },
            "vectorization": {
                "enabled": self.enable_vectorization
            },
            "batching": {
                "enabled": self.enable_batching,
                "total_batch_operations": self.batch_operations,
                "queue_size": len(self.batch_queue) if self.enable_batching else 0
            },
            "async": {
                "enabled": self.enable_async,
                "thread_pool_size": self.thread_pool._max_workers if self.thread_pool else 0
            },
            "total_operations": self.total_operations
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.cache_hits = 0
        self.cache_misses = 0
        self.batch_operations = 0
        self.total_operations = 0
    
    def shutdown(self):
        """Cleanup resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if self.enable_batching:
            self.batch_worker_running = False
            with self.batch_condition:
                self.batch_condition.notify_all()


class ConcurrentRouter:
    """Thread-safe concurrent router wrapper."""
    
    def __init__(
        self,
        base_router,
        max_concurrent_requests: int = 100,
        request_timeout_seconds: float = 30.0,
        enable_request_pooling: bool = True
    ):
        self.base_router = base_router
        self.max_concurrent_requests = max_concurrent_requests
        self.request_timeout_seconds = request_timeout_seconds
        self.enable_request_pooling = enable_request_pooling
        
        # Concurrency control
        self.semaphore = threading.Semaphore(max_concurrent_requests)
        self.request_count = 0
        self.request_lock = threading.Lock()
        
        # Request pooling
        if enable_request_pooling:
            self.request_pool = ThreadPoolExecutor(max_workers=max_concurrent_requests)
        else:
            self.request_pool = None
        
        # Performance tracking
        self.active_requests = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.avg_response_time = 0.0
        self.start_time = time.time()
        
        logger.info(f"Concurrent router initialized with {max_concurrent_requests} max concurrent requests")
    
    def route(self, *args, **kwargs):
        """Thread-safe routing with concurrency control."""
        
        start_time = time.time()
        
        # Check concurrency limits
        if not self.semaphore.acquire(timeout=self.request_timeout_seconds):
            raise RuntimeError("Request timeout: too many concurrent requests")
        
        try:
            with self.request_lock:
                self.total_requests += 1
                self.active_requests += 1
            
            # Execute routing
            result = self.base_router.route(*args, **kwargs)
            
            # Update performance metrics
            response_time = time.time() - start_time
            self._update_performance_metrics(response_time, success=True)
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_performance_metrics(response_time, success=False)
            raise
        
        finally:
            with self.request_lock:
                self.active_requests -= 1
            self.semaphore.release()
    
    async def async_route(self, *args, **kwargs):
        """Asynchronous routing interface."""
        if not self.request_pool:
            # Fallback to synchronous
            return self.route(*args, **kwargs)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.request_pool, self.route, *args, **kwargs)
    
    def _update_performance_metrics(self, response_time: float, success: bool):
        """Update performance tracking metrics."""
        with self.request_lock:
            if not success:
                self.failed_requests += 1
            
            # Update average response time with exponential moving average
            alpha = 0.1  # Smoothing factor
            self.avg_response_time = (
                alpha * response_time + (1 - alpha) * self.avg_response_time
            )
    
    def get_concurrency_stats(self) -> Dict[str, Any]:
        """Get concurrency performance statistics."""
        with self.request_lock:
            uptime = time.time() - self.start_time
            error_rate = self.failed_requests / max(self.total_requests, 1)
            throughput = self.total_requests / max(uptime, 1)
            
            return {
                "active_requests": self.active_requests,
                "total_requests": self.total_requests,
                "failed_requests": self.failed_requests,
                "error_rate": error_rate,
                "avg_response_time_ms": self.avg_response_time * 1000,
                "throughput_req_per_sec": throughput,
                "uptime_seconds": uptime,
                "max_concurrent": self.max_concurrent_requests,
                "available_slots": self.semaphore._value
            }
    
    def reset_stats(self):
        """Reset concurrency statistics."""
        with self.request_lock:
            self.total_requests = 0
            self.failed_requests = 0
            self.avg_response_time = 0.0
            self.start_time = time.time()
    
    def shutdown(self):
        """Cleanup concurrent resources."""
        if self.request_pool:
            self.request_pool.shutdown(wait=True)


# Global performance optimizer
default_performance_optimizer = PerformanceOptimizer()
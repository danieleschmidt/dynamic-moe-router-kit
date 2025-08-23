"""
Generation 3: Optimized Core - Dynamic MoE Router
Making it scale with performance optimization, caching, and concurrent processing.
"""

import numpy as np
import logging
import time
import threading
import multiprocessing as mp
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps, lru_cache
import hashlib
import pickle
from collections import defaultdict, deque
import concurrent.futures
import sys
import gc

# Import from previous generations  
try:
    from .robust_core import RobustRouterConfig, SecurityValidator, PerformanceMonitor, retry_on_failure
except ImportError:
    from robust_core import RobustRouterConfig, SecurityValidator, PerformanceMonitor, retry_on_failure

logger = logging.getLogger('dynamic_moe_router.optimized')

@dataclass
class OptimizedRouterConfig(RobustRouterConfig):
    """Optimized configuration with performance tuning parameters"""
    # Caching parameters
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl_seconds: int = 300  # 5 minutes
    
    # Concurrency parameters
    enable_concurrent_processing: bool = True
    max_workers: int = 4
    batch_processing_threshold: int = 32
    
    # Memory optimization
    enable_memory_pooling: bool = True
    memory_pool_size: int = 100
    enable_garbage_collection: bool = True
    gc_frequency: int = 100  # calls between GC
    
    # Performance optimization
    enable_vectorization: bool = True
    use_fast_math: bool = True
    enable_compiler_optimizations: bool = True
    
    # Auto-scaling parameters
    enable_auto_scaling: bool = True
    cpu_threshold_scale_up: float = 0.8
    cpu_threshold_scale_down: float = 0.3
    memory_threshold_mb: int = 1000
    response_time_threshold_ms: int = 100
    
    # Load balancing
    enable_load_balancing: bool = True
    load_balance_strategy: str = "round_robin"  # round_robin, least_loaded, weighted
    
    def __post_init__(self):
        super().__post_init__()
        
        # Optimize worker count based on system
        if self.max_workers <= 0:
            self.max_workers = min(8, mp.cpu_count())

class AdaptiveCache:
    """High-performance adaptive cache with TTL and LRU eviction"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
    
    def _generate_key(self, inputs: np.ndarray, config_hash: str) -> str:
        """Generate cache key from inputs and config"""
        # Use input shape and hash of small sample for key
        shape_key = f"{inputs.shape}"
        sample_hash = hashlib.md5(inputs.flatten()[:100].tobytes()).hexdigest()[:8]
        return f"{shape_key}_{sample_hash}_{config_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with TTL check"""
        with self.lock:
            current_time = time.time()
            
            if key not in self.cache:
                self.miss_count += 1
                return None
            
            # Check TTL
            if current_time - self.creation_times[key] > self.ttl_seconds:
                self._evict_key(key)
                self.miss_count += 1
                return None
            
            # Update access time for LRU
            self.access_times[key] = current_time
            self.hit_count += 1
            
            return pickle.loads(self.cache[key])
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache with LRU eviction"""
        with self.lock:
            current_time = time.time()
            
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            # Store serialized value for deep copy behavior
            self.cache[key] = pickle.dumps(value)
            self.access_times[key] = current_time
            self.creation_times[key] = current_time
    
    def _evict_key(self, key: str) -> None:
        """Evict specific key"""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key] 
            del self.creation_times[key]
    
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._evict_key(lru_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / max(total_requests, 1)
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

class MemoryPool:
    """Memory pool for reducing allocation overhead"""
    
    def __init__(self, pool_size: int = 100):
        self.pool_size = pool_size
        self.float_pools = defaultdict(deque)
        self.int_pools = defaultdict(deque)
        self.lock = threading.Lock()
        self.allocated_count = 0
        self.reused_count = 0
    
    def get_float_array(self, shape: Tuple[int, ...], fill_value: float = 0.0) -> np.ndarray:
        """Get float array from pool or create new"""
        with self.lock:
            pool = self.float_pools[shape]
            
            if pool:
                array = pool.popleft()
                array.fill(fill_value)
                self.reused_count += 1
                return array
            else:
                self.allocated_count += 1
                return np.full(shape, fill_value, dtype=np.float32)
    
    def get_int_array(self, shape: Tuple[int, ...], fill_value: int = 0) -> np.ndarray:
        """Get int array from pool or create new"""
        with self.lock:
            pool = self.int_pools[shape]
            
            if pool:
                array = pool.popleft()
                array.fill(fill_value)
                self.reused_count += 1
                return array
            else:
                self.allocated_count += 1
                return np.full(shape, fill_value, dtype=np.int32)
    
    def return_array(self, array: np.ndarray) -> None:
        """Return array to pool"""
        with self.lock:
            if array.dtype == np.float32:
                pool = self.float_pools[array.shape]
            elif array.dtype == np.int32:
                pool = self.int_pools[array.shape]
            else:
                return  # Don't pool unsupported dtypes
            
            if len(pool) < self.pool_size:
                pool.append(array)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics"""
        total_pooled = sum(len(pool) for pools in [self.float_pools, self.int_pools] 
                          for pool in pools.values())
        reuse_rate = self.reused_count / max(self.allocated_count + self.reused_count, 1)
        
        return {
            'total_allocated': self.allocated_count,
            'total_reused': self.reused_count,
            'reuse_rate': reuse_rate,
            'arrays_pooled': total_pooled,
            'pool_types': len(self.float_pools) + len(self.int_pools)
        }

class AutoScaler:
    """Automatic scaling based on system metrics"""
    
    def __init__(self, config: OptimizedRouterConfig):
        self.config = config
        self.current_workers = config.max_workers
        self.cpu_usage_history = deque(maxlen=10)
        self.memory_usage_history = deque(maxlen=10)
        self.response_time_history = deque(maxlen=10)
        self.last_scale_time = 0
        self.scale_cooldown = 30  # seconds
        
    def should_scale_up(self) -> bool:
        """Check if we should scale up resources"""
        if not self.config.enable_auto_scaling:
            return False
        
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
        
        if len(self.cpu_usage_history) < 3:
            return False
        
        avg_cpu = np.mean(list(self.cpu_usage_history)[-3:])
        avg_response_time = np.mean(list(self.response_time_history)[-3:])
        
        return (avg_cpu > self.config.cpu_threshold_scale_up or 
                avg_response_time > self.config.response_time_threshold_ms / 1000.0)
    
    def should_scale_down(self) -> bool:
        """Check if we should scale down resources"""
        if not self.config.enable_auto_scaling:
            return False
        
        if self.current_workers <= 1:
            return False
        
        if time.time() - self.last_scale_time < self.scale_cooldown * 2:  # Longer cooldown for scale down
            return False
        
        if len(self.cpu_usage_history) < 5:
            return False
        
        avg_cpu = np.mean(list(self.cpu_usage_history)[-5:])
        return avg_cpu < self.config.cpu_threshold_scale_down
    
    def update_metrics(self, cpu_usage: float, memory_mb: float, response_time_s: float):
        """Update system metrics"""
        self.cpu_usage_history.append(cpu_usage)
        self.memory_usage_history.append(memory_mb)
        self.response_time_history.append(response_time_s)
    
    def scale_up(self) -> int:
        """Scale up resources"""
        new_workers = min(self.current_workers + 2, self.config.max_workers * 2)
        self.current_workers = new_workers
        self.last_scale_time = time.time()
        logger.info(f"🔺 Scaled up to {new_workers} workers")
        return new_workers
    
    def scale_down(self) -> int:
        """Scale down resources"""
        new_workers = max(self.current_workers - 1, 1)
        self.current_workers = new_workers
        self.last_scale_time = time.time()
        logger.info(f"🔻 Scaled down to {new_workers} workers")
        return new_workers

class VectorizedOperations:
    """Optimized vectorized operations for better performance"""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def precompute_softmax_table(max_logit: float = 10.0, steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Precompute softmax lookup table for common values"""
        logits = np.linspace(-max_logit, max_logit, steps)
        exp_logits = np.exp(logits)
        return logits, exp_logits
    
    @staticmethod
    def fast_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Optimized softmax with temperature scaling"""
        if temperature != 1.0:
            logits = logits / temperature
        
        # Subtract max for stability (vectorized)
        max_logits = np.max(logits, axis=-1, keepdims=True)
        shifted = logits - max_logits
        
        # Fast exp using vectorized operations
        exp_shifted = np.exp(np.clip(shifted, -10, 10))  # Prevent overflow
        sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)
        
        return exp_shifted / (sum_exp + 1e-8)
    
    @staticmethod
    def fast_topk(array: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fast top-k selection using argpartition"""
        if k >= array.shape[-1]:
            indices = np.argsort(array, axis=-1)
            values = np.take_along_axis(array, indices, axis=-1)
            return values, indices
        
        # Use argpartition for better performance than full sort
        partition_indices = np.argpartition(array, -k, axis=-1)[..., -k:]
        
        # Sort only the top-k elements
        topk_indices = np.take_along_axis(
            np.argsort(array, axis=-1), 
            partition_indices, axis=-1
        )
        topk_values = np.take_along_axis(array, topk_indices, axis=-1)
        
        return topk_values, topk_indices
    
    @staticmethod
    def fast_complexity_estimate(inputs: np.ndarray, method: str = "variance") -> np.ndarray:
        """Fast complexity estimation using different methods"""
        if method == "variance":
            # Use ddof=1 for unbiased variance
            variance = np.var(inputs, axis=-1, keepdims=True, ddof=1)
            return np.tanh(variance / (np.std(variance) + 1e-8))
        
        elif method == "l2_norm":
            norms = np.linalg.norm(inputs, axis=-1, keepdims=True)
            return norms / (np.max(norms) + 1e-8)
        
        elif method == "entropy":
            # Approximate entropy using histogram
            hist, _ = np.histogram(inputs.flatten(), bins=50, density=True)
            hist = hist[hist > 0]  # Remove zero bins
            entropy = -np.sum(hist * np.log(hist))
            return np.full((inputs.shape[0], inputs.shape[1], 1), 
                          np.tanh(entropy))
        
        else:
            return np.full((inputs.shape[0], inputs.shape[1], 1), 0.5)

class OptimizedDynamicRouter:
    """Generation 3: High-performance optimized dynamic router"""
    
    def __init__(self, config: OptimizedRouterConfig):
        self.config = config
        self.security_validator = SecurityValidator(config)
        self.performance_monitor = PerformanceMonitor()
        
        # Performance optimization components
        self.cache = AdaptiveCache(config.cache_size, config.cache_ttl_seconds) if config.enable_caching else None
        self.memory_pool = MemoryPool(config.memory_pool_size) if config.enable_memory_pooling else None
        self.auto_scaler = AutoScaler(config) if config.enable_auto_scaling else None
        
        # Initialize optimized routing network
        self._initialize_optimized_network()
        
        # Concurrent processing setup
        self.executor = None
        if config.enable_concurrent_processing:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Performance tracking
        self.call_count = 0
        self.total_compute_time = 0
        self.cache_hits = 0
        
        logger.info(f"🚀 Initialized OptimizedDynamicRouter with advanced optimizations")
    
    def _initialize_optimized_network(self):
        """Initialize optimized routing network"""
        np.random.seed(42)
        
        # Use float32 for better performance
        std = np.sqrt(2.0 / (self.config.input_dim + self.config.num_experts))
        self.router_weights = np.random.normal(
            0, std, (self.config.input_dim, self.config.num_experts)
        ).astype(np.float32)
        
        self.router_bias = np.zeros(self.config.num_experts, dtype=np.float32)
        
        # Precomputed constants for optimization
        self.expert_range = self.config.max_experts - self.config.min_experts
        self.inv_num_experts = 1.0 / self.config.num_experts
        
        logger.debug("Optimized routing network initialized")
    
    def route(self, inputs: np.ndarray) -> Dict[str, Any]:
        """High-performance routing with all optimizations"""
        
        start_time = time.time()
        self.call_count += 1
        
        try:
            # Check cache first
            if self.cache:
                cache_key = self.cache._generate_key(inputs, self._get_config_hash())
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    self.cache_hits += 1
                    logger.debug("✅ Cache hit - returning cached result")
                    return cached_result
            
            # Security validation (if enabled)
            if self.config.enable_security_checks:
                if not self.security_validator.validate_input(inputs, "routing"):
                    return self._emergency_fallback_routing(inputs)
            
            # Determine processing strategy
            batch_size = inputs.shape[0]
            if (self.config.enable_concurrent_processing and 
                batch_size >= self.config.batch_processing_threshold and 
                self.executor):
                result = self._concurrent_route(inputs)
            else:
                result = self._optimized_route(inputs)
            
            # Cache result
            if self.cache:
                self.cache.put(cache_key, result)
            
            # Update auto-scaler metrics
            if self.auto_scaler:
                compute_time = time.time() - start_time
                self.auto_scaler.update_metrics(0.5, 100, compute_time)  # Mock CPU/memory
                
                if self.auto_scaler.should_scale_up():
                    new_workers = self.auto_scaler.scale_up()
                    self._resize_thread_pool(new_workers)
                elif self.auto_scaler.should_scale_down():
                    new_workers = self.auto_scaler.scale_down()
                    self._resize_thread_pool(new_workers)
            
            # Periodic garbage collection
            if (self.config.enable_garbage_collection and 
                self.call_count % self.config.gc_frequency == 0):
                gc.collect()
            
            self.total_compute_time += time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Optimized routing failed: {e}")
            return self._emergency_fallback_routing(inputs)
    
    def _optimized_route(self, inputs: np.ndarray) -> Dict[str, Any]:
        """Core optimized routing logic"""
        batch_size, seq_len, dim = inputs.shape
        
        # Fast complexity estimation
        if self.config.enable_vectorization:
            complexity = VectorizedOperations.fast_complexity_estimate(inputs, "variance")
        else:
            variance = np.var(inputs, axis=-1, keepdims=True)
            complexity = np.tanh(variance)
        
        # Vectorized expert count calculation
        k_experts = (self.config.min_experts + 
                    self.expert_range * complexity)
        k_experts = np.round(k_experts).astype(np.int32)
        
        # Optimized routing logits computation
        if inputs.dtype != np.float32:
            inputs = inputs.astype(np.float32)
        
        routing_logits = np.dot(inputs, self.router_weights) + self.router_bias
        
        if self.config.routing_temperature != 1.0:
            routing_logits /= self.config.routing_temperature
        
        # Fast top-k selection
        if self.config.enable_vectorization:
            _, expert_indices = VectorizedOperations.fast_topk(
                routing_logits, self.config.max_experts
            )
        else:
            expert_indices = np.argsort(routing_logits, axis=-1)[..., -self.config.max_experts:]
        
        # Fast softmax for weights
        selected_logits = np.take_along_axis(routing_logits, expert_indices, axis=-1)
        
        if self.config.enable_vectorization:
            expert_weights = VectorizedOperations.fast_softmax(
                selected_logits, self.config.routing_temperature
            )
        else:
            expert_weights = self._stable_softmax(selected_logits)
        
        # Efficient metrics calculation
        avg_experts = np.mean(k_experts)
        flop_reduction = 1.0 - (avg_experts * self.inv_num_experts)
        
        # Use memory pool for result arrays if available
        if self.memory_pool:
            routing_entropy_array = self.memory_pool.get_float_array((1,), 0.0)
            load_balance_array = self.memory_pool.get_float_array((1,), 0.0)
        else:
            routing_entropy_array = np.array([0.0])
            load_balance_array = np.array([0.0])
        
        routing_entropy = self._fast_routing_entropy(expert_weights)
        load_balance_loss = self._fast_load_balance_loss(expert_weights)
        
        result = {
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'complexity_scores': complexity,
            'avg_experts_per_token': float(avg_experts),
            'flop_reduction': float(flop_reduction),
            'routing_logits': routing_logits,
            'routing_entropy': float(routing_entropy),
            'load_balance_loss': float(load_balance_loss)
        }
        
        # Return arrays to pool
        if self.memory_pool:
            self.memory_pool.return_array(routing_entropy_array)
            self.memory_pool.return_array(load_balance_array)
        
        return result
    
    def _concurrent_route(self, inputs: np.ndarray) -> Dict[str, Any]:
        """Concurrent processing for large batches"""
        batch_size = inputs.shape[0]
        chunk_size = max(1, batch_size // self.auto_scaler.current_workers)
        
        # Split batch into chunks
        chunks = []
        for i in range(0, batch_size, chunk_size):
            chunk = inputs[i:i + chunk_size]
            chunks.append(chunk)
        
        # Process chunks concurrently
        futures = []
        for chunk in chunks:
            future = self.executor.submit(self._optimized_route, chunk)
            futures.append(future)
        
        # Collect results
        chunk_results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result(timeout=5.0)  # 5s timeout
                chunk_results.append(result)
            except Exception as e:
                logger.error(f"Chunk processing failed: {e}")
                # Create fallback result for this chunk
                chunk_results.append(self._emergency_fallback_routing(chunks[0]))
        
        # Merge results
        return self._merge_chunk_results(chunk_results)
    
    def _merge_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge results from concurrent chunk processing"""
        if not chunk_results:
            return {}
        
        merged = {}
        
        # Concatenate arrays
        array_keys = ['expert_indices', 'expert_weights', 'complexity_scores', 'routing_logits']
        for key in array_keys:
            if key in chunk_results[0]:
                merged[key] = np.concatenate([result[key] for result in chunk_results], axis=0)
        
        # Average scalar metrics
        scalar_keys = ['avg_experts_per_token', 'flop_reduction', 'routing_entropy', 'load_balance_loss']
        for key in scalar_keys:
            if key in chunk_results[0]:
                values = [result[key] for result in chunk_results if key in result]
                merged[key] = np.mean(values) if values else 0.0
        
        return merged
    
    def _fast_routing_entropy(self, weights: np.ndarray) -> float:
        """Fast routing entropy calculation"""
        # Vectorized entropy calculation
        safe_weights = np.maximum(weights, 1e-8)
        entropy = -np.sum(weights * np.log(safe_weights), axis=-1)
        return float(np.mean(entropy))
    
    def _fast_load_balance_loss(self, weights: np.ndarray) -> float:
        """Fast load balance loss calculation"""
        expert_usage = np.mean(weights, axis=(0, 1))
        ideal_usage = 1.0 / len(expert_usage)
        balance_loss = np.sum((expert_usage - ideal_usage) ** 2)
        return float(balance_loss)
    
    def _stable_softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax (fallback)"""
        shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(np.clip(shifted_logits, -10, 10))
        return exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-8)
    
    def _resize_thread_pool(self, new_size: int):
        """Resize thread pool for auto-scaling"""
        if self.executor and new_size != self.executor._max_workers:
            # Shutdown current executor
            self.executor.shutdown(wait=False)
            
            # Create new executor with updated size
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=new_size)
            logger.debug(f"Thread pool resized to {new_size} workers")
    
    def _emergency_fallback_routing(self, inputs: np.ndarray) -> Dict[str, Any]:
        """Emergency fallback routing"""
        batch_size, seq_len, dim = inputs.shape
        
        # Use static top-2 experts as fallback
        expert_indices = np.tile([0, 1], (batch_size, seq_len, 1)).astype(np.int32)
        expert_weights = np.full((batch_size, seq_len, 2), 0.5, dtype=np.float32)
        
        return {
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'complexity_scores': np.full((batch_size, seq_len, 1), 0.5, dtype=np.float32),
            'avg_experts_per_token': 2.0,
            'flop_reduction': 0.75,
            'routing_logits': np.zeros((batch_size, seq_len, self.config.num_experts), dtype=np.float32),
            'routing_entropy': 0.693,  # ln(2)
            'load_balance_loss': 0.0,
            'fallback_used': True
        }
    
    def _get_config_hash(self) -> str:
        """Get configuration hash"""
        config_str = str(sorted(self.config.__dict__.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            'call_count': self.call_count,
            'total_compute_time': self.total_compute_time,
            'avg_compute_time': self.total_compute_time / max(self.call_count, 1),
            'cache_hit_rate': self.cache_hits / max(self.call_count, 1) if self.cache else 0,
        }
        
        if self.cache:
            stats['cache_stats'] = self.cache.get_stats()
        
        if self.memory_pool:
            stats['memory_pool_stats'] = self.memory_pool.get_stats()
        
        if self.auto_scaler:
            stats['auto_scaler_workers'] = self.auto_scaler.current_workers
        
        return stats
    
    def __del__(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=False)

def demonstrate_optimized_moe():
    """Demonstrate the optimized MoE implementation"""
    print("⚡ Generation 3: Optimized Dynamic MoE Router Demo")
    print("=" * 55)
    
    # Create optimized configuration
    config = OptimizedRouterConfig(
        input_dim=768,
        num_experts=8,
        min_experts=1,
        max_experts=4,
        enable_caching=True,
        enable_concurrent_processing=True,
        enable_memory_pooling=True,
        enable_auto_scaling=True,
        enable_vectorization=True,
        max_workers=4
    )
    
    # Create optimized router
    router = OptimizedDynamicRouter(config)
    
    # Test with different batch sizes
    print("\n🔬 Testing small batch (cache miss)...")
    small_inputs = np.random.randn(2, 128, config.input_dim).astype(np.float32)
    
    start_time = time.time()
    routing_info = router.route(small_inputs)
    small_time = time.time() - start_time
    
    print(f"✅ Small batch: avg_experts={routing_info['avg_experts_per_token']:.2f}, "
          f"time={small_time*1000:.2f}ms")
    
    # Test cache hit
    print("\n💾 Testing cache hit...")
    start_time = time.time()
    routing_info2 = router.route(small_inputs)
    cache_time = time.time() - start_time
    
    print(f"✅ Cache hit: time={cache_time*1000:.2f}ms "
          f"(speedup: {small_time/cache_time:.1f}x)")
    
    # Test large batch (concurrent processing)
    print("\n⚡ Testing large batch (concurrent processing)...")
    large_inputs = np.random.randn(64, 256, config.input_dim).astype(np.float32)
    
    start_time = time.time()
    routing_info = router.route(large_inputs)
    large_time = time.time() - start_time
    
    print(f"✅ Large batch: avg_experts={routing_info['avg_experts_per_token']:.2f}, "
          f"time={large_time*1000:.2f}ms")
    
    # Show performance statistics
    perf_stats = router.get_performance_stats()
    print(f"\n📊 Performance Statistics:")
    print(f"   Total calls: {perf_stats['call_count']}")
    print(f"   Cache hit rate: {perf_stats['cache_hit_rate']:.1%}")
    print(f"   Avg compute time: {perf_stats['avg_compute_time']*1000:.2f}ms")
    
    if 'cache_stats' in perf_stats:
        cache_stats = perf_stats['cache_stats']
        print(f"   Cache efficiency: {cache_stats['hit_rate']:.1%}")
    
    if 'memory_pool_stats' in perf_stats:
        pool_stats = perf_stats['memory_pool_stats']
        print(f"   Memory reuse rate: {pool_stats['reuse_rate']:.1%}")
    
    print(f"   Current workers: {perf_stats.get('auto_scaler_workers', 'N/A')}")
    
    print("\n✅ Generation 3 Complete: High-performance optimizations active!")
    
    return router, perf_stats

if __name__ == "__main__":
    demonstrate_optimized_moe()
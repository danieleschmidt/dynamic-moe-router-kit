"""Advanced caching and optimization for dynamic MoE routing."""

import logging
import time
import threading
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

import numpy as np

from .exceptions import ProfilingError
from .security import get_security_monitor

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float
    access_count: int
    computation_time: float
    hit_probability: float = 0.0
    
    def __post_init__(self):
        self.last_access = self.timestamp


class AdaptiveCache:
    """Adaptive LRU cache with predictive prefetching."""
    
    def __init__(self,
                 max_size: int = 1000,
                 ttl: float = 3600.0,  # 1 hour
                 enable_prefetch: bool = True,
                 prefetch_threshold: float = 0.7):
        self.max_size = max_size
        self.ttl = ttl
        self.enable_prefetch = enable_prefetch
        self.prefetch_threshold = prefetch_threshold
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._access_patterns: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cache-prefetch")
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._prefetch_hits = 0
        self._evictions = 0
        
        # Security integration
        self._security_monitor = get_security_monitor()
    
    def get(self, key: str) -> Tuple[Any, bool]:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check TTL
                if time.time() - entry.timestamp > self.ttl:
                    del self._cache[key]
                    self._misses += 1
                    return None, False
                
                # Update access metadata
                entry.access_count += 1
                entry.last_access = time.time()
                self._access_patterns[key].append(entry.last_access)
                
                # Move to end (LRU)
                self._cache.move_to_end(key)
                
                self._hits += 1
                return entry.value, True
            else:
                self._misses += 1
                return None, False
    
    def put(self, key: str, value: Any, computation_time: float = 0.0) -> None:
        """Put value in cache with eviction if needed."""
        with self._lock:
            # Security check
            if self._is_suspicious_key(key):
                self._security_monitor.log_event(
                    'suspicious_cache_key',
                    'medium',
                    f'Blocked suspicious cache key: {key[:16]}...'
                )
                return
            
            current_time = time.time()
            
            if key in self._cache:
                # Update existing entry
                self._cache[key].value = value
                self._cache[key].timestamp = current_time
                self._cache[key].computation_time = computation_time
                self._cache.move_to_end(key)
            else:
                # Add new entry
                entry = CacheEntry(
                    value=value,
                    timestamp=current_time,
                    access_count=1,
                    computation_time=computation_time
                )
                self._cache[key] = entry
                
                # Evict if necessary
                while len(self._cache) > self.max_size:
                    self._evict_lru()
                
                # Schedule prefetch prediction
                if self.enable_prefetch:
                    self._executor.submit(self._update_prefetch_predictions, key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            evicted_key, _ = self._cache.popitem(last=False)
            self._evictions += 1
            logger.debug(f"Evicted cache entry: {evicted_key}")
    
    def _is_suspicious_key(self, key: str) -> bool:
        """Check if cache key looks suspicious."""
        # Check for unusual patterns
        if len(key) > 256:  # Abnormally long keys
            return True
        if key.count('../') > 2:  # Path traversal attempts
            return True
        if any(char in key for char in ['<', '>', '&', '"', "'"]):  # Injection attempts
            return True
        return False
    
    def _update_prefetch_predictions(self, key: str) -> None:
        """Update prefetch predictions based on access patterns."""
        try:
            with self._lock:
                if key not in self._access_patterns:
                    return
                
                accesses = self._access_patterns[key]
                if len(accesses) < 3:  # Need minimum history
                    return
                
                # Calculate access frequency and predict next access
                recent_accesses = accesses[-10:]  # Last 10 accesses
                intervals = [recent_accesses[i+1] - recent_accesses[i] 
                           for i in range(len(recent_accesses)-1)]
                
                if intervals:
                    avg_interval = np.mean(intervals)
                    last_access = recent_accesses[-1]
                    predicted_next = last_access + avg_interval
                    
                    # Update hit probability
                    if key in self._cache:
                        time_until_predicted = predicted_next - time.time()
                        if time_until_predicted > 0:
                            self._cache[key].hit_probability = 1.0 / (1.0 + time_until_predicted / 3600)
        
        except Exception as e:
            logger.warning(f"Prefetch prediction failed for {key}: {e}")
    
    def prefetch_candidates(self) -> List[str]:
        """Get keys that are good candidates for prefetching."""
        with self._lock:
            candidates = []
            for key, entry in self._cache.items():
                if (entry.hit_probability > self.prefetch_threshold and
                    entry.access_count > 3):
                    candidates.append(key)
            return sorted(candidates, key=lambda k: self._cache[k].hit_probability, reverse=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'prefetch_hits': self._prefetch_hits,
                'avg_computation_time': np.mean([e.computation_time for e in self._cache.values()]) if self._cache else 0
            }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_patterns.clear()
            logger.info("Cache cleared")


class RouterCache:
    """Specialized cache for routing decisions."""
    
    def __init__(self, max_size: int = 5000):
        self.cache = AdaptiveCache(max_size=max_size, ttl=1800.0)  # 30 minutes
        self.complexity_cache = AdaptiveCache(max_size=2000, ttl=900.0)  # 15 minutes
    
    def get_routing_decision(self, input_hash: str) -> Tuple[Optional[Dict[str, Any]], bool]:
        """Get cached routing decision."""
        return self.cache.get(f"route_{input_hash}")
    
    def cache_routing_decision(self, input_hash: str, decision: Dict[str, Any], computation_time: float = 0.0) -> None:
        """Cache routing decision."""
        self.cache.put(f"route_{input_hash}", decision, computation_time)
    
    def get_complexity_scores(self, input_hash: str) -> Tuple[Optional[Any], bool]:
        """Get cached complexity scores."""
        return self.complexity_cache.get(f"complexity_{input_hash}")
    
    def cache_complexity_scores(self, input_hash: str, scores: Any, computation_time: float = 0.0) -> None:
        """Cache complexity scores."""
        self.complexity_cache.put(f"complexity_{input_hash}", scores, computation_time)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive caching statistics."""
        return {
            'routing_cache': self.cache.get_stats(),
            'complexity_cache': self.complexity_cache.get_stats()
        }


class BatchProcessor:
    """Optimized batch processing for routing operations."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="batch-processor"
        )
    
    def process_batch_parallel(self,
                             batch_items: List[Any],
                             process_func: Callable,
                             chunk_size: Optional[int] = None) -> List[Any]:
        """Process batch items in parallel chunks."""
        if chunk_size is None:
            chunk_size = max(1, len(batch_items) // self.max_workers)
        
        # Create chunks
        chunks = [batch_items[i:i + chunk_size] 
                 for i in range(0, len(batch_items), chunk_size)]
        
        # Process chunks in parallel
        future_to_chunk = {
            self.executor.submit(self._process_chunk, chunk, process_func): i
            for i, chunk in enumerate(chunks)
        }
        
        # Collect results
        results = [None] * len(chunks)
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                results[chunk_idx] = future.result()
            except Exception as e:
                logger.error(f"Chunk {chunk_idx} processing failed: {e}")
                results[chunk_idx] = []
        
        # Flatten results
        flattened = []
        for chunk_result in results:
            if chunk_result:
                flattened.extend(chunk_result)
        
        return flattened
    
    def _process_chunk(self, chunk: List[Any], process_func: Callable) -> List[Any]:
        """Process a single chunk of items."""
        return [process_func(item) for item in chunk]


class ComputeOptimizer:
    """Optimizes computation patterns for MoE routing."""
    
    def __init__(self):
        self._computation_profiles: Dict[str, List[float]] = defaultdict(list)
        self._optimal_strategies: Dict[str, str] = {}
        
    def profile_computation(self, operation: str, duration: float) -> None:
        """Profile computation time for an operation."""
        self._computation_profiles[operation].append(duration)
        
        # Analyze and update optimal strategy
        if len(self._computation_profiles[operation]) >= 10:
            self._analyze_optimal_strategy(operation)
    
    def _analyze_optimal_strategy(self, operation: str) -> None:
        """Analyze computation profile to determine optimal strategy."""
        durations = self._computation_profiles[operation][-50:]  # Recent 50 samples
        
        avg_duration = np.mean(durations)
        duration_variance = np.var(durations)
        
        # Simple strategy selection based on characteristics
        if avg_duration < 0.001:  # Very fast operations
            strategy = "inline"
        elif avg_duration > 0.1 and duration_variance < 0.001:  # Slow but consistent
            strategy = "batch"
        elif duration_variance > 0.01:  # High variance
            strategy = "adaptive"
        else:
            strategy = "parallel"
        
        self._optimal_strategies[operation] = strategy
        logger.debug(f"Optimal strategy for {operation}: {strategy} (avg: {avg_duration:.4f}s)")
    
    def get_optimal_strategy(self, operation: str) -> str:
        """Get optimal strategy for an operation."""
        return self._optimal_strategies.get(operation, "parallel")
    
    def get_profiles(self) -> Dict[str, Dict[str, float]]:
        """Get computation profiles summary."""
        profiles = {}
        for op, durations in self._computation_profiles.items():
            if durations:
                profiles[op] = {
                    'avg_duration': np.mean(durations[-50:]),
                    'std_duration': np.std(durations[-50:]),
                    'sample_count': len(durations),
                    'optimal_strategy': self.get_optimal_strategy(op)
                }
        return profiles


# Global instances
_global_router_cache = RouterCache()
_global_batch_processor = BatchProcessor()
_global_compute_optimizer = ComputeOptimizer()


def get_router_cache() -> RouterCache:
    """Get global router cache."""
    return _global_router_cache


def get_batch_processor() -> BatchProcessor:
    """Get global batch processor."""
    return _global_batch_processor


def get_compute_optimizer() -> ComputeOptimizer:
    """Get global compute optimizer."""
    return _global_compute_optimizer


def compute_input_hash(tensor: Any, **kwargs) -> str:
    """Compute stable hash for input tensor and parameters."""
    try:
        # Create hash from tensor properties and kwargs
        hash_components = []
        
        if hasattr(tensor, 'shape'):
            hash_components.append(str(tensor.shape))
        if hasattr(tensor, 'dtype'):
            hash_components.append(str(tensor.dtype))
        if hasattr(tensor, 'tobytes'):
            # Sample a few elements for efficiency
            if tensor.size > 1000:
                sampled = tensor.flat[::tensor.size//100]  # Sample ~100 elements
                hash_components.append(sampled.tobytes())
            else:
                hash_components.append(tensor.tobytes())
        
        # Add kwargs
        sorted_kwargs = sorted(kwargs.items())
        hash_components.append(str(sorted_kwargs))
        
        # Compute hash
        combined = ''.join(hash_components).encode('utf-8')
        return hashlib.sha256(combined).hexdigest()
        
    except Exception as e:
        logger.warning(f"Failed to compute input hash: {e}")
        return hashlib.sha256(str(time.time()).encode()).hexdigest()
"""Ultra-optimized router with advanced scaling features."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

from .router import DynamicRouter
from .caching_advanced import (
    get_router_cache, 
    get_batch_processor, 
    get_compute_optimizer,
    compute_input_hash
)
from .monitoring import EnhancedPerformanceMonitor
from .security import create_secure_sanitizer
from .exceptions import ExpertDispatchError, ProfilingError

logger = logging.getLogger(__name__)


class UltraOptimizedRouter(DynamicRouter):
    """Ultra-high performance router with advanced optimization features."""
    
    def __init__(self,
                 *args,
                 enable_caching: bool = True,
                 enable_parallel_processing: bool = True,
                 enable_compute_optimization: bool = True,
                 cache_hit_ratio_threshold: float = 0.3,
                 parallel_threshold_size: int = 1000,  # Min tokens to use parallel processing
                 max_parallel_workers: int = None,
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        
        # Optimization features
        self.enable_caching = enable_caching
        self.enable_parallel_processing = enable_parallel_processing
        self.enable_compute_optimization = enable_compute_optimization
        self.cache_hit_ratio_threshold = cache_hit_ratio_threshold
        self.parallel_threshold_size = parallel_threshold_size
        
        # Parallel processing setup
        if max_parallel_workers is None:
            max_parallel_workers = min(8, mp.cpu_count())
        self.max_parallel_workers = max_parallel_workers
        
        # Cache integration
        if self.enable_caching:
            self.router_cache = get_router_cache()
            
        # Batch processor
        if self.enable_parallel_processing:
            self.batch_processor = get_batch_processor()
            
        # Compute optimizer
        if self.enable_compute_optimization:
            self.compute_optimizer = get_compute_optimizer()
            
        # Performance tracking
        self.optimization_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_executions': 0,
            'sequential_executions': 0,
            'total_compute_time_saved': 0.0
        }
        
        logger.info(f"Initialized UltraOptimizedRouter with caching={enable_caching}, "
                   f"parallel={enable_parallel_processing}, optimization={enable_compute_optimization}")
    
    def route(self, hidden_states: Any, return_router_logits: bool = False, **complexity_kwargs) -> Dict[str, Any]:
        """Ultra-optimized routing with caching, parallelization, and adaptive optimization."""
        start_time = time.time()
        
        # Compute input hash for caching
        input_hash = None
        if self.enable_caching:
            input_hash = compute_input_hash(hidden_states, **complexity_kwargs)
            
            # Try cache first
            cached_result, cache_hit = self.router_cache.get_routing_decision(input_hash)
            if cache_hit:
                self.optimization_stats['cache_hits'] += 1
                if self.enable_monitoring:
                    duration_ms = (time.time() - start_time) * 1000
                    self.performance_monitor.record_call(duration_ms, True, cache_hit=True)
                
                logger.debug("Cache hit for routing decision")
                return cached_result
            else:
                self.optimization_stats['cache_misses'] += 1
        
        # Determine processing strategy
        batch_size, seq_len, hidden_dim = hidden_states.shape
        total_tokens = batch_size * seq_len
        
        use_parallel = (self.enable_parallel_processing and 
                       total_tokens >= self.parallel_threshold_size)
        
        # Execute routing with optimal strategy
        if use_parallel:
            result = self._route_parallel(hidden_states, return_router_logits, **complexity_kwargs)
            self.optimization_stats['parallel_executions'] += 1
        else:
            result = self._route_sequential(hidden_states, return_router_logits, **complexity_kwargs)
            self.optimization_stats['sequential_executions'] += 1
        
        # Cache result if caching enabled
        if self.enable_caching and input_hash:
            computation_time = time.time() - start_time
            self.router_cache.cache_routing_decision(input_hash, result, computation_time)
        
        # Profile computation for optimization
        if self.enable_compute_optimization:
            total_time = time.time() - start_time
            strategy = "parallel" if use_parallel else "sequential"
            self.compute_optimizer.profile_computation(f"route_{strategy}", total_time)
        
        return result
    
    def _route_parallel(self, hidden_states: Any, return_router_logits: bool = False, **complexity_kwargs) -> Dict[str, Any]:
        """Parallel processing version of routing."""
        start_time = time.time()
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        try:
            # Step 1: Parallel complexity estimation
            complexity_start = time.time()
            complexity_scores = self._estimate_complexity_parallel(hidden_states, **complexity_kwargs)
            complexity_time = time.time() - complexity_start
            
            # Step 2: Parallel expert count computation
            num_experts_per_token = self._compute_expert_counts_parallel(complexity_scores)
            
            # Step 3: Parallel router logits computation
            router_logits = self._compute_router_logits_parallel(hidden_states)
            
            # Step 4: Add noise if needed (parallel)
            if self.noise_factor > 0:
                router_logits = self._add_noise_parallel(router_logits)
            
            # Step 5: Parallel expert selection
            expert_indices, expert_weights = self._select_experts_parallel(
                router_logits, num_experts_per_token
            )
            
            # Step 6: Load balancing (if enabled)
            if self.load_balancing:
                expert_weights = self._apply_load_balancing(expert_indices, expert_weights, router_logits)
            
            # Compile results
            routing_stats = self._compute_routing_stats(expert_indices, num_experts_per_token, complexity_scores)
            
            result = {
                'expert_indices': expert_indices,
                'expert_weights': expert_weights,
                'num_experts_per_token': num_experts_per_token,
                'complexity_scores': complexity_scores,
                'routing_info': routing_stats
            }
            
            if return_router_logits:
                result['router_logits'] = router_logits
            
            # Add optimization metadata
            result['optimization_info'] = {
                'strategy': 'parallel',
                'complexity_estimation_time': complexity_time,
                'total_parallel_time': time.time() - start_time
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Parallel routing failed: {e}, falling back to sequential")
            return self._route_sequential(hidden_states, return_router_logits, **complexity_kwargs)
    
    def _route_sequential(self, hidden_states: Any, return_router_logits: bool = False, **complexity_kwargs) -> Dict[str, Any]:
        """Sequential (standard) routing with optimizations."""
        # Use the parent class implementation with monitoring
        return super()._route_internal(hidden_states, return_router_logits, **complexity_kwargs)
    
    def _estimate_complexity_parallel(self, hidden_states: Any, **kwargs) -> Any:
        """Parallel complexity estimation."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if seq_len < 32:  # Too small for parallel processing
            return self.complexity_estimator.estimate(hidden_states, **kwargs)
        
        # Split sequence into chunks for parallel processing
        chunk_size = max(4, seq_len // self.max_parallel_workers)
        chunks = []
        
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            chunk = hidden_states[:, i:end_idx, :]
            chunks.append((chunk, kwargs))
        
        # Process chunks in parallel
        def process_chunk(chunk_data):
            chunk, chunk_kwargs = chunk_data
            return self.complexity_estimator.estimate(chunk, **chunk_kwargs)
        
        try:
            chunk_results = self.batch_processor.process_batch_parallel(chunks, process_chunk)
            
            # Concatenate results
            if chunk_results:
                return np.concatenate(chunk_results, axis=1)  # Concatenate along sequence dimension
            else:
                return self.complexity_estimator.estimate(hidden_states, **kwargs)
                
        except Exception as e:
            logger.warning(f"Parallel complexity estimation failed: {e}")
            return self.complexity_estimator.estimate(hidden_states, **kwargs)
    
    def _compute_expert_counts_parallel(self, complexity_scores: Any) -> Any:
        """Parallel expert count computation."""
        # This is typically a simple operation, but we can vectorize it efficiently
        return super()._compute_expert_counts(complexity_scores)
    
    def _compute_router_logits_parallel(self, hidden_states: Any) -> Any:
        """Parallel router logits computation using optimized matrix operations."""
        if self.router_weights is None:
            self.initialize_router_network()
        
        # Use numpy's optimized matrix multiplication (already parallel via BLAS)
        return np.dot(hidden_states, self.router_weights) + self.router_bias
    
    def _add_noise_parallel(self, router_logits: Any) -> Any:
        """Parallel noise addition."""
        # Vectorized noise addition (already optimal)
        noise = np.random.randn(*router_logits.shape) * self.noise_factor
        return router_logits + noise
    
    def _select_experts_parallel(self, router_logits: Any, num_experts_per_token: Any) -> Tuple[Any, Any]:
        """Parallel expert selection with optimized algorithms."""
        batch_size, seq_len, num_experts = router_logits.shape
        max_k = self.max_experts
        
        # Pre-allocate output arrays
        expert_indices = np.full((batch_size, seq_len, max_k), -1, dtype=int)
        expert_weights = np.zeros((batch_size, seq_len, max_k))
        
        if self.routing_strategy == "top_k":
            return self._top_k_selection_vectorized(
                router_logits, num_experts_per_token, expert_indices, expert_weights
            )
        else:
            return super()._select_experts(router_logits, num_experts_per_token)
    
    def _top_k_selection_vectorized(self, router_logits: Any, num_experts_per_token: Any,
                                   expert_indices: Any, expert_weights: Any) -> Tuple[Any, Any]:
        """Vectorized top-k selection for better performance."""
        batch_size, seq_len, num_experts = router_logits.shape
        
        # Vectorized approach for common case where all tokens use same k
        unique_ks = np.unique(num_experts_per_token)
        
        if len(unique_ks) == 1:
            # All tokens use the same k - highly optimized path
            k = unique_ks[0]
            return self._uniform_k_selection(router_logits, k, expert_indices, expert_weights)
        else:
            # Mixed k values - use original approach
            return super()._top_k_selection(
                router_logits, num_experts_per_token, expert_indices, expert_weights
            )
    
    def _uniform_k_selection(self, router_logits: Any, k: int, 
                           expert_indices: Any, expert_weights: Any) -> Tuple[Any, Any]:
        """Highly optimized selection when all tokens use the same k."""
        batch_size, seq_len, num_experts = router_logits.shape
        
        # Reshape for batch processing
        flattened_logits = router_logits.reshape(-1, num_experts)  # [batch*seq, num_experts]
        
        # Vectorized top-k selection
        top_k_indices = np.argpartition(flattened_logits, -k, axis=1)[:, -k:]
        
        # Sort the top-k indices by their values
        for i in range(flattened_logits.shape[0]):
            token_logits = flattened_logits[i]
            sorted_idx = np.argsort(token_logits[top_k_indices[i]])[::-1]
            top_k_indices[i] = top_k_indices[i][sorted_idx]
        
        # Reshape back
        top_k_indices = top_k_indices.reshape(batch_size, seq_len, k)
        
        # Compute weights using vectorized operations
        for b in range(batch_size):
            for s in range(seq_len):
                selected_logits = router_logits[b, s, top_k_indices[b, s]]
                weights = self._softmax(selected_logits)
                
                expert_indices[b, s, :k] = top_k_indices[b, s]
                expert_weights[b, s, :k] = weights
        
        return expert_indices, expert_weights
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        base_stats = super().get_expert_usage_stats()
        
        # Add optimization-specific stats
        base_stats['optimization'] = self.optimization_stats.copy()
        
        if self.enable_caching:
            base_stats['caching'] = self.router_cache.get_comprehensive_stats()
        
        if self.enable_compute_optimization:
            base_stats['compute_profiles'] = self.compute_optimizer.get_profiles()
        
        # Calculate optimization metrics
        total_cache_requests = self.optimization_stats['cache_hits'] + self.optimization_stats['cache_misses']
        if total_cache_requests > 0:
            cache_hit_ratio = self.optimization_stats['cache_hits'] / total_cache_requests
            base_stats['cache_hit_ratio'] = cache_hit_ratio
        
        total_executions = self.optimization_stats['parallel_executions'] + self.optimization_stats['sequential_executions']
        if total_executions > 0:
            parallel_ratio = self.optimization_stats['parallel_executions'] / total_executions
            base_stats['parallel_execution_ratio'] = parallel_ratio
        
        return base_stats
    
    def optimize_for_workload(self, sample_inputs: List[Any], sample_size: int = 100) -> Dict[str, Any]:
        """Analyze workload and optimize router configuration."""
        logger.info(f"Analyzing workload with {len(sample_inputs)} samples...")
        
        # Profile sample inputs
        timing_stats = []
        cache_performance = []
        
        for i, input_tensor in enumerate(sample_inputs[:sample_size]):
            if i % 10 == 0:
                logger.debug(f"Profiling sample {i+1}/{min(sample_size, len(sample_inputs))}")
            
            # Time sequential execution
            start_time = time.time()
            result_seq = self._route_sequential(input_tensor)
            seq_time = time.time() - start_time
            
            # Time parallel execution (if enabled)
            par_time = None
            if self.enable_parallel_processing:
                start_time = time.time()
                result_par = self._route_parallel(input_tensor)
                par_time = time.time() - start_time
            
            timing_stats.append({
                'input_shape': input_tensor.shape,
                'sequential_time': seq_time,
                'parallel_time': par_time,
                'tokens': input_tensor.shape[0] * input_tensor.shape[1]
            })
        
        # Analyze results and provide recommendations
        recommendations = self._analyze_workload_profile(timing_stats)
        
        logger.info(f"Workload analysis complete. Recommendations: {recommendations}")
        return recommendations
    
    def _analyze_workload_profile(self, timing_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze timing statistics and provide optimization recommendations."""
        if not timing_stats:
            return {'error': 'No timing statistics available'}
        
        # Analyze sequential vs parallel performance
        parallel_wins = 0
        total_comparisons = 0
        
        avg_seq_time = np.mean([s['sequential_time'] for s in timing_stats])
        
        parallel_times = [s['parallel_time'] for s in timing_stats if s['parallel_time'] is not None]
        if parallel_times:
            avg_par_time = np.mean(parallel_times)
            parallel_wins = sum(1 for s in timing_stats 
                              if s['parallel_time'] and s['parallel_time'] < s['sequential_time'])
            total_comparisons = len([s for s in timing_stats if s['parallel_time']])
        else:
            avg_par_time = None
        
        # Token size analysis
        token_counts = [s['tokens'] for s in timing_stats]
        avg_tokens = np.mean(token_counts)
        
        recommendations = {
            'avg_sequential_time': avg_seq_time,
            'avg_parallel_time': avg_par_time,
            'parallel_win_rate': parallel_wins / max(total_comparisons, 1),
            'avg_tokens_per_batch': avg_tokens,
            'sample_count': len(timing_stats)
        }
        
        # Provide specific recommendations
        if avg_par_time and avg_par_time < avg_seq_time * 0.8:
            recommendations['recommendation'] = 'Enable parallel processing by default'
            recommendations['suggested_parallel_threshold'] = int(np.percentile(token_counts, 25))
        elif parallel_wins / max(total_comparisons, 1) > 0.7:
            recommendations['recommendation'] = 'Enable parallel processing for large batches'
            recommendations['suggested_parallel_threshold'] = int(np.percentile(token_counts, 50))
        else:
            recommendations['recommendation'] = 'Sequential processing is more efficient'
            recommendations['suggested_parallel_threshold'] = int(np.percentile(token_counts, 90))
        
        return recommendations
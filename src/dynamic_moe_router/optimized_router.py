"""High-performance optimized router implementation."""

import logging
from typing import Any, Dict, Tuple

import numpy as np

from .logging_config import PerformanceLogger
from .performance import (
    VectorizedOperations,
    get_adaptive_optimizer,
    get_memory_pool,
    optimize_for_throughput,
)
from .router import DynamicRouter

logger = logging.getLogger(__name__)


class OptimizedDynamicRouter(DynamicRouter):
    """High-performance optimized dynamic router with caching and vectorization."""

    def __init__(self, enable_caching: bool = True, enable_profiling: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.enable_caching = enable_caching
        self.enable_profiling = enable_profiling
        self.memory_pool = get_memory_pool()
        self.adaptive_optimizer = get_adaptive_optimizer()
        self.performance_logger = PerformanceLogger(f"router_{id(self)}")

        # Performance counters
        self.total_routing_time = 0.0
        self.total_routes = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Vectorized operations
        self.vectorized_ops = VectorizedOperations()

        logger.info(f"Initialized OptimizedDynamicRouter with caching={enable_caching}")

    @optimize_for_throughput
    def route(
        self,
        hidden_states: Any,
        return_router_logits: bool = False,
        **complexity_kwargs
    ) -> Dict[str, Any]:
        """Optimized routing with caching and vectorization."""
        if self.enable_profiling:
            self.performance_logger.start_timer("total_route")

        try:
            batch_size, seq_len, hidden_dim = hidden_states.shape

            # Record memory usage
            if self.enable_profiling:
                memory_mb = hidden_states.nbytes / (1024 * 1024)
                self.performance_logger.log_memory_usage("input", memory_mb)

            # Step 1: Optimized complexity estimation
            if self.enable_profiling:
                self.performance_logger.start_timer("complexity_estimation")

            complexity_scores = self._estimate_complexity_optimized(
                hidden_states, **complexity_kwargs
            )

            if self.enable_profiling:
                self.performance_logger.end_timer("complexity_estimation")

            # Step 2: Vectorized expert count computation
            num_experts_per_token = self._compute_expert_counts_vectorized(complexity_scores)

            # Step 3: Optimized router logits computation
            if self.enable_profiling:
                self.performance_logger.start_timer("router_logits")

            router_logits = self._compute_router_logits_optimized(hidden_states)

            if self.noise_factor > 0:
                router_logits = self._add_routing_noise_optimized(router_logits)

            if self.enable_profiling:
                self.performance_logger.end_timer("router_logits")

            # Step 4: High-performance expert selection
            if self.enable_profiling:
                self.performance_logger.start_timer("expert_selection")

            expert_indices, expert_weights = self._select_experts_vectorized(
                router_logits, num_experts_per_token
            )

            if self.enable_profiling:
                self.performance_logger.end_timer("expert_selection")

            # Step 5: Optimized load balancing
            if self.load_balancing:
                if self.enable_profiling:
                    self.performance_logger.start_timer("load_balancing")

                expert_weights = self._apply_load_balancing_optimized(
                    expert_indices, expert_weights, router_logits
                )

                if self.enable_profiling:
                    self.performance_logger.end_timer("load_balancing")

            # Compile results
            result = {
                'expert_indices': expert_indices,
                'expert_weights': expert_weights,
                'num_experts_per_token': num_experts_per_token,
                'complexity_scores': complexity_scores,
                'routing_info': self._compute_routing_stats_optimized(
                    expert_indices, num_experts_per_token, complexity_scores
                )
            }

            if return_router_logits:
                result['router_logits'] = router_logits

            # Update performance stats
            self.total_routes += 1

            return result

        finally:
            if self.enable_profiling:
                total_time = self.performance_logger.end_timer("total_route")
                self.total_routing_time += total_time

    def _estimate_complexity_optimized(self, hidden_states: Any, **kwargs) -> Any:
        """Optimized complexity estimation with caching."""
        if self.enable_caching:
            # Try cache first
            cache_key = self._create_cache_key(hidden_states, kwargs)
            cached_result = getattr(self.complexity_estimator, '_cache', {}).get(cache_key)

            if cached_result is not None:
                self.cache_hits += 1
                return cached_result
            else:
                self.cache_misses += 1

        # Compute complexity scores
        complexity_scores = self.complexity_estimator.estimate(hidden_states, **kwargs)

        # Cache result if enabled
        if self.enable_caching:
            if not hasattr(self.complexity_estimator, '_cache'):
                self.complexity_estimator._cache = {}

            # Simple LRU-like cache with size limit
            cache = self.complexity_estimator._cache
            if len(cache) > 100:  # Simple cache eviction
                # Remove oldest entries (simplified)
                keys_to_remove = list(cache.keys())[:20]
                for key in keys_to_remove:
                    del cache[key]

            cache[cache_key] = complexity_scores

        return complexity_scores

    def _create_cache_key(self, hidden_states: Any, kwargs: Dict[str, Any]) -> str:
        """Create cache key from inputs."""
        # Use shape and data hash for cache key
        shape_key = str(hidden_states.shape)

        # Sample a few values for quick hash (don't use full tensor for performance)
        sample_size = min(100, hidden_states.size)
        flat_data = hidden_states.flatten()
        sample_indices = np.linspace(0, flat_data.size - 1, sample_size, dtype=int)
        sample_data = flat_data[sample_indices]
        data_hash = hash(sample_data.tobytes())

        kwargs_key = str(sorted(kwargs.items())) if kwargs else ""

        return f"{shape_key}_{data_hash}_{kwargs_key}"

    def _compute_expert_counts_vectorized(self, complexity_scores: Any) -> Any:
        """Vectorized expert count computation."""
        # More efficient vectorized computation
        expert_range = self.max_experts - self.min_experts
        raw_counts = self.min_experts + expert_range * complexity_scores

        # Use memory pool for intermediate arrays if beneficial
        if complexity_scores.size > 1000:
            temp_array = self.memory_pool.get_tensor(complexity_scores.shape)
            np.round(raw_counts, out=temp_array)
            expert_counts = np.clip(temp_array, self.min_experts, self.max_experts).astype(int)
            self.memory_pool.return_tensor(temp_array)
        else:
            expert_counts = np.clip(np.round(raw_counts), self.min_experts, self.max_experts).astype(int)

        return expert_counts

    def _compute_router_logits_optimized(self, hidden_states: Any) -> Any:
        """Optimized router logits computation."""
        if self.router_weights is None:
            self.initialize_router_network()

        # Use optimized BLAS operations
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Reshape for efficient matrix multiplication
        reshaped_states = hidden_states.reshape(-1, hidden_dim)  # [batch*seq, hidden_dim]

        # Efficient matrix multiplication
        logits_flat = np.dot(reshaped_states, self.router_weights) + self.router_bias

        # Reshape back
        logits = logits_flat.reshape(batch_size, seq_len, self.num_experts)

        return logits

    def _add_routing_noise_optimized(self, router_logits: Any) -> Any:
        """Optimized noise addition."""
        # Reuse noise array from memory pool for large tensors
        if router_logits.size > 1000:
            noise = self.memory_pool.get_tensor(router_logits.shape)
            np.random.randn(*router_logits.shape, out=noise)
            noise *= self.noise_factor
            result = router_logits + noise
            self.memory_pool.return_tensor(noise)
            return result
        else:
            noise = np.random.randn(*router_logits.shape) * self.noise_factor
            return router_logits + noise

    def _select_experts_vectorized(
        self,
        router_logits: Any,
        num_experts_per_token: Any
    ) -> Tuple[Any, Any]:
        """Highly optimized expert selection using vectorized operations."""
        batch_size, seq_len, num_experts = router_logits.shape
        max_k = self.max_experts

        # Initialize output arrays from memory pool
        expert_indices = self.memory_pool.get_tensor((batch_size, seq_len, max_k), dtype=np.int32)
        expert_weights = self.memory_pool.get_tensor((batch_size, seq_len, max_k))

        expert_indices.fill(-1)  # Initialize with padding value
        expert_weights.fill(0.0)

        try:
            if self.routing_strategy == "top_k":
                return self._top_k_selection_vectorized(
                    router_logits, num_experts_per_token, expert_indices, expert_weights
                )
            else:
                return self._threshold_selection_optimized(
                    router_logits, num_experts_per_token, expert_indices, expert_weights
                )
        except Exception:
            # Return arrays to pool on error
            self.memory_pool.return_tensor(expert_indices)
            self.memory_pool.return_tensor(expert_weights)
            raise

    def _top_k_selection_vectorized(
        self,
        router_logits: Any,
        num_experts_per_token: Any,
        expert_indices: Any,
        expert_weights: Any
    ) -> Tuple[Any, Any]:
        """Fully vectorized top-k selection."""
        batch_size, seq_len, num_experts = router_logits.shape

        # Check if all tokens use the same k (common case)
        unique_ks = np.unique(num_experts_per_token)

        if len(unique_ks) == 1:
            # Highly optimized path for uniform k
            k = int(unique_ks[0])
            if k > 0:
                # Vectorized top-k for all positions at once
                top_values, top_indices = self.vectorized_ops.fast_top_k(
                    router_logits, k, axis=-1
                )

                # Vectorized softmax
                weights = self.vectorized_ops.fast_softmax(top_values, axis=-1)

                # Store results
                expert_indices[:, :, :k] = top_indices
                expert_weights[:, :, :k] = weights
        else:
            # Handle variable k efficiently
            self._handle_variable_k_optimized(
                router_logits, num_experts_per_token, expert_indices, expert_weights
            )

        return expert_indices, expert_weights

    def _handle_variable_k_optimized(
        self,
        router_logits: Any,
        num_experts_per_token: Any,
        expert_indices: Any,
        expert_weights: Any
    ) -> None:
        """Handle variable k values efficiently."""
        batch_size, seq_len = num_experts_per_token.shape

        # Group positions by k value for batch processing
        k_groups = {}
        for k in np.unique(num_experts_per_token):
            k = int(k)
            if k > 0:
                mask = (num_experts_per_token == k)
                positions = np.where(mask)
                if len(positions[0]) > 0:
                    k_groups[k] = positions

        # Process each k group
        for k, (batch_indices, seq_indices) in k_groups.items():
            if len(batch_indices) == 0:
                continue

            # Extract logits for this k group
            group_logits = router_logits[batch_indices, seq_indices]  # [group_size, num_experts]

            # Vectorized top-k for this group
            top_values, top_indices = self.vectorized_ops.fast_top_k(
                group_logits, k, axis=-1
            )

            # Vectorized softmax
            weights = self.vectorized_ops.fast_softmax(top_values, axis=-1)

            # Store results back
            expert_indices[batch_indices, seq_indices, :k] = top_indices
            expert_weights[batch_indices, seq_indices, :k] = weights

    def _threshold_selection_optimized(
        self,
        router_logits: Any,
        num_experts_per_token: Any,
        expert_indices: Any,
        expert_weights: Any
    ) -> Tuple[Any, Any]:
        """Optimized threshold-based selection."""
        # This is inherently harder to vectorize, but we can optimize the loop
        batch_size, seq_len, num_experts = router_logits.shape

        # Pre-sort logits for all positions
        sorted_indices = np.argsort(router_logits, axis=-1)[..., ::-1]  # Descending
        sorted_logits = np.take_along_axis(router_logits, sorted_indices, axis=-1)

        for b in range(batch_size):
            for s in range(seq_len):
                k = int(num_experts_per_token[b, s])

                if k > 0:
                    # Use pre-sorted arrays
                    token_sorted_logits = sorted_logits[b, s]
                    token_sorted_indices = sorted_indices[b, s]

                    if k >= num_experts:
                        # Use all experts
                        selected_indices = token_sorted_indices
                        selected_logits = token_sorted_logits
                    else:
                        # Use threshold approach
                        threshold = token_sorted_logits[k-1]
                        mask = token_sorted_logits >= threshold

                        selected_indices = token_sorted_indices[mask]
                        selected_logits = token_sorted_logits[mask]

                        # Limit to max_k
                        if len(selected_indices) > self.max_experts:
                            selected_indices = selected_indices[:self.max_experts]
                            selected_logits = selected_logits[:self.max_experts]

                    if len(selected_indices) > 0:
                        weights = self.vectorized_ops.fast_softmax(selected_logits)

                        num_selected = len(selected_indices)
                        expert_indices[b, s, :num_selected] = selected_indices
                        expert_weights[b, s, :num_selected] = weights

        return expert_indices, expert_weights

    def _apply_load_balancing_optimized(
        self,
        expert_indices: Any,
        expert_weights: Any,
        router_logits: Any
    ) -> Any:
        """Optimized load balancing."""
        # Use parent implementation but with vectorized expert usage tracking
        expert_usage = np.zeros(self.num_experts)

        # Vectorized usage counting
        valid_indices = expert_indices[expert_indices >= 0]
        valid_weights_flat = expert_weights[expert_indices >= 0]

        # Accumulate usage efficiently
        np.add.at(expert_usage, valid_indices, valid_weights_flat)

        # Add to history
        self.expert_usage_history.append(expert_usage)
        if len(self.expert_usage_history) > self.max_history_length:
            self.expert_usage_history.pop(0)

        # Apply load balancing penalty if needed
        if len(self.expert_usage_history) > 1:
            avg_usage = np.mean(self.expert_usage_history, axis=0)
            usage_variance = np.var(avg_usage)

            if usage_variance > 0.1:
                penalty_factor = 0.9
                threshold = np.mean(avg_usage) + np.std(avg_usage)
                overused_experts = avg_usage > threshold

                # Vectorized penalty application
                penalty_mask = np.isin(expert_indices, np.where(overused_experts)[0])
                expert_weights[penalty_mask] *= penalty_factor

                # Vectorized renormalization
                for b in range(expert_indices.shape[0]):
                    for s in range(expert_indices.shape[1]):
                        token_weights = expert_weights[b, s]
                        valid_mask = token_weights > 0
                        if valid_mask.any():
                            token_weights[valid_mask] /= np.sum(token_weights[valid_mask])

        return expert_weights

    def _compute_routing_stats_optimized(
        self,
        expert_indices: Any,
        num_experts_per_token: Any,
        complexity_scores: Any
    ) -> Dict[str, Any]:
        """Optimized routing statistics computation."""
        # Vectorized statistics computation
        batch_size, seq_len = complexity_scores.shape
        total_tokens = batch_size * seq_len

        # Vectorized expert utilization
        valid_indices = expert_indices[expert_indices >= 0]
        expert_counts = np.bincount(valid_indices, minlength=self.num_experts)
        total_expert_calls = np.sum(expert_counts)

        expert_utilization = (
            expert_counts / total_expert_calls if total_expert_calls > 0
            else np.zeros(self.num_experts)
        )

        # Vectorized statistics
        avg_experts = float(np.mean(num_experts_per_token))
        static_flops = total_tokens * self.num_experts
        dynamic_flops = np.sum(num_experts_per_token)
        flop_reduction = 1.0 - (dynamic_flops / static_flops) if static_flops > 0 else 0.0

        return {
            'avg_experts_per_token': avg_experts,
            'total_expert_calls': int(dynamic_flops),
            'flop_reduction': flop_reduction,
            'expert_utilization': expert_utilization.tolist(),
            'complexity_stats': {
                'mean': float(np.mean(complexity_scores)),
                'std': float(np.std(complexity_scores)),
                'min': float(np.min(complexity_scores)),
                'max': float(np.max(complexity_scores))
            }
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        avg_time = self.total_routing_time / max(1, self.total_routes)
        cache_hit_rate = self.cache_hits / max(1, self.cache_hits + self.cache_misses)

        return {
            'total_routes': self.total_routes,
            'total_routing_time': self.total_routing_time,
            'avg_routing_time': avg_time,
            'routes_per_second': 1.0 / avg_time if avg_time > 0 else 0,
            'cache_hit_rate': cache_hit_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'memory_pool_stats': {
                'total_size_mb': self.memory_pool._total_size_bytes / (1024 * 1024),
                'pool_count': len(self.memory_pool.pools)
            }
        }

    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self.total_routing_time = 0.0
        self.total_routes = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.memory_pool.clear()

        # Clear estimator cache if it exists
        if hasattr(self.complexity_estimator, '_cache'):
            self.complexity_estimator._cache.clear()

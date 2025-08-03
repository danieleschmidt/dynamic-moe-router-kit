"""Dynamic expert routing for Mixture-of-Experts models."""

import math
from typing import Dict, Any, Optional, Tuple, Union, List

import numpy as np

from .estimator import ComplexityEstimator, get_estimator


class DynamicRouter:
    """Core dynamic routing algorithm for MoE models.
    
    The router analyzes input complexity and adaptively selects
    the optimal number of experts for each token/sample.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        min_experts: int = 1,
        max_experts: Optional[int] = None,
        complexity_estimator: Union[str, ComplexityEstimator] = "gradient_norm",
        routing_strategy: str = "top_k",
        load_balancing: bool = True,
        noise_factor: float = 0.0,
        **estimator_kwargs
    ):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.min_experts = min_experts
        self.max_experts = max_experts or num_experts
        self.routing_strategy = routing_strategy
        self.load_balancing = load_balancing
        self.noise_factor = noise_factor
        
        # Validate parameters
        if self.min_experts < 1 or self.min_experts > self.num_experts:
            raise ValueError(f"min_experts must be in [1, {self.num_experts}]")
        if self.max_experts < self.min_experts or self.max_experts > self.num_experts:
            raise ValueError(f"max_experts must be in [{self.min_experts}, {self.num_experts}]")
        
        # Initialize complexity estimator
        if isinstance(complexity_estimator, str):
            self.complexity_estimator = get_estimator(complexity_estimator, **estimator_kwargs)
        else:
            self.complexity_estimator = complexity_estimator
        
        # Router network parameters (to be initialized in subclasses)
        self.router_weights = None
        self.router_bias = None
        
        # Expert utilization tracking for load balancing
        self.expert_usage_history: List[np.ndarray] = []
        self.max_history_length = 1000
    
    def initialize_router_network(self, seed: int = 42):
        """Initialize router network parameters."""
        np.random.seed(seed)
        
        # Simple linear projection to expert logits
        self.router_weights = np.random.randn(self.input_dim, self.num_experts) * 0.02
        self.router_bias = np.zeros(self.num_experts)
    
    def route(
        self, 
        hidden_states: Any, 
        return_router_logits: bool = False,
        **complexity_kwargs
    ) -> Dict[str, Any]:
        """Perform dynamic expert routing.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            return_router_logits: Whether to return raw router scores
            **complexity_kwargs: Additional args for complexity estimation
            
        Returns:
            Dictionary containing:
            - expert_indices: Selected expert indices [batch, seq_len, k]
            - expert_weights: Expert combination weights [batch, seq_len, k]
            - num_experts_per_token: Number of experts used [batch, seq_len]
            - complexity_scores: Token complexity scores [batch, seq_len]
            - router_logits: Raw router scores (if requested)
            - routing_info: Additional routing statistics
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Step 1: Estimate input complexity
        complexity_scores = self.complexity_estimator.estimate(
            hidden_states, **complexity_kwargs
        )
        
        # Step 2: Determine number of experts per token
        num_experts_per_token = self._compute_expert_counts(complexity_scores)
        
        # Step 3: Compute router logits
        router_logits = self._compute_router_logits(hidden_states)
        
        # Step 4: Apply noise for regularization
        if self.noise_factor > 0:
            router_logits = self._add_routing_noise(router_logits)
        
        # Step 5: Select experts based on strategy
        expert_indices, expert_weights = self._select_experts(
            router_logits, num_experts_per_token
        )
        
        # Step 6: Apply load balancing
        if self.load_balancing:
            expert_weights = self._apply_load_balancing(
                expert_indices, expert_weights, router_logits
            )
        
        # Compile results
        result = {
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'num_experts_per_token': num_experts_per_token,
            'complexity_scores': complexity_scores,
            'routing_info': self._compute_routing_stats(
                expert_indices, num_experts_per_token, complexity_scores
            )
        }
        
        if return_router_logits:
            result['router_logits'] = router_logits
        
        return result
    
    def _compute_expert_counts(self, complexity_scores: Any) -> Any:
        """Convert complexity scores to expert counts.
        
        Args:
            complexity_scores: Normalized complexity in [0, 1]
            
        Returns:
            Number of experts per token
        """
        # Linear interpolation between min and max experts
        expert_range = self.max_experts - self.min_experts
        raw_counts = self.min_experts + expert_range * complexity_scores
        
        # Round to integers and clamp
        expert_counts = np.round(raw_counts).astype(int)
        expert_counts = np.clip(expert_counts, self.min_experts, self.max_experts)
        
        return expert_counts
    
    def _compute_router_logits(self, hidden_states: Any) -> Any:
        """Compute expert selection logits.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            
        Returns:
            Router logits [batch, seq_len, num_experts]
        """
        if self.router_weights is None:
            self.initialize_router_network()
        
        # Linear projection: [batch, seq_len, hidden_dim] @ [hidden_dim, num_experts]
        logits = np.dot(hidden_states, self.router_weights) + self.router_bias
        
        return logits
    
    def _add_routing_noise(self, router_logits: Any) -> Any:
        """Add noise to router logits for regularization."""
        noise_shape = router_logits.shape
        noise = np.random.randn(*noise_shape) * self.noise_factor
        return router_logits + noise
    
    def _select_experts(
        self, 
        router_logits: Any, 
        num_experts_per_token: Any
    ) -> Tuple[Any, Any]:
        """Select experts based on routing strategy.
        
        Args:
            router_logits: Expert scores [batch, seq_len, num_experts]
            num_experts_per_token: Number of experts to select [batch, seq_len]
            
        Returns:
            Tuple of (expert_indices, expert_weights)
        """
        batch_size, seq_len, num_experts = router_logits.shape
        max_k = self.max_experts
        
        # Initialize output arrays
        expert_indices = np.full((batch_size, seq_len, max_k), -1, dtype=int)
        expert_weights = np.zeros((batch_size, seq_len, max_k))
        
        if self.routing_strategy == "top_k":
            return self._top_k_selection(
                router_logits, num_experts_per_token, expert_indices, expert_weights
            )
        elif self.routing_strategy == "threshold":
            return self._threshold_selection(
                router_logits, num_experts_per_token, expert_indices, expert_weights
            )
        else:
            raise ValueError(f"Unknown routing strategy: {self.routing_strategy}")
    
    def _top_k_selection(
        self,
        router_logits: Any,
        num_experts_per_token: Any,
        expert_indices: Any,
        expert_weights: Any
    ) -> Tuple[Any, Any]:
        """Top-k expert selection strategy."""
        batch_size, seq_len, num_experts = router_logits.shape
        
        for b in range(batch_size):
            for s in range(seq_len):
                k = num_experts_per_token[b, s]
                
                # Get top-k expert indices
                token_logits = router_logits[b, s]
                top_indices = np.argpartition(token_logits, -k)[-k:]
                sorted_indices = top_indices[np.argsort(token_logits[top_indices])[::-1]]
                
                # Compute softmax weights over selected experts
                selected_logits = token_logits[sorted_indices]
                weights = self._softmax(selected_logits)
                
                # Store results
                expert_indices[b, s, :k] = sorted_indices
                expert_weights[b, s, :k] = weights
        
        return expert_indices, expert_weights
    
    def _threshold_selection(
        self,
        router_logits: Any,
        num_experts_per_token: Any,
        expert_indices: Any,
        expert_weights: Any
    ) -> Tuple[Any, Any]:
        """Threshold-based expert selection strategy."""
        # Compute dynamic threshold based on desired expert count
        batch_size, seq_len, num_experts = router_logits.shape
        
        for b in range(batch_size):
            for s in range(seq_len):
                k = num_experts_per_token[b, s]
                token_logits = router_logits[b, s]
                
                # Set threshold to select approximately k experts
                if k >= num_experts:
                    threshold = float('-inf')
                else:
                    sorted_logits = np.sort(token_logits)[::-1]
                    threshold = sorted_logits[k-1] if k > 0 else sorted_logits[0]
                
                # Select experts above threshold
                selected_mask = token_logits >= threshold
                selected_indices = np.where(selected_mask)[0]
                
                # Limit to max_k experts
                if len(selected_indices) > self.max_experts:
                    selected_indices = selected_indices[:self.max_experts]
                
                # Compute weights
                if len(selected_indices) > 0:
                    selected_logits = token_logits[selected_indices]
                    weights = self._softmax(selected_logits)
                    
                    # Store results
                    num_selected = len(selected_indices)
                    expert_indices[b, s, :num_selected] = selected_indices
                    expert_weights[b, s, :num_selected] = weights
        
        return expert_indices, expert_weights
    
    def _apply_load_balancing(
        self,
        expert_indices: Any,
        expert_weights: Any,
        router_logits: Any
    ) -> Any:
        """Apply load balancing to expert weights."""
        # Track expert usage
        batch_size, seq_len, max_k = expert_indices.shape
        expert_usage = np.zeros(self.num_experts)
        
        # Count expert usage in current batch
        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(max_k):
                    expert_idx = expert_indices[b, s, k]
                    if expert_idx >= 0:
                        expert_usage[expert_idx] += expert_weights[b, s, k]
        
        # Add to history
        self.expert_usage_history.append(expert_usage)
        if len(self.expert_usage_history) > self.max_history_length:
            self.expert_usage_history.pop(0)
        
        # Compute load balancing penalty
        if len(self.expert_usage_history) > 1:
            avg_usage = np.mean(self.expert_usage_history, axis=0)
            usage_variance = np.var(avg_usage)
            
            # Penalize overused experts (simple approach)
            if usage_variance > 0.1:  # Threshold for applying penalty
                penalty_factor = 0.9  # Reduce weights by 10%
                overused_experts = avg_usage > np.mean(avg_usage) + np.std(avg_usage)
                
                for b in range(batch_size):
                    for s in range(seq_len):
                        for k in range(max_k):
                            expert_idx = expert_indices[b, s, k]
                            if expert_idx >= 0 and overused_experts[expert_idx]:
                                expert_weights[b, s, k] *= penalty_factor
                
                # Renormalize weights
                for b in range(batch_size):
                    for s in range(seq_len):
                        token_weights = expert_weights[b, s]
                        valid_weights = token_weights[token_weights > 0]
                        if len(valid_weights) > 0:
                            token_weights[token_weights > 0] /= np.sum(valid_weights)
        
        return expert_weights
    
    def _compute_routing_stats(self, expert_indices: Any, num_experts_per_token: Any, complexity_scores: Any) -> Dict[str, Any]:
        """Compute routing statistics for monitoring."""
        batch_size, seq_len = complexity_scores.shape
        total_tokens = batch_size * seq_len
        
        # Average experts per token
        avg_experts = float(np.mean(num_experts_per_token))
        
        # Expert utilization distribution
        expert_counts = np.bincount(
            expert_indices[expert_indices >= 0].flatten(),
            minlength=self.num_experts
        )
        expert_utilization = expert_counts / np.sum(expert_counts) if np.sum(expert_counts) > 0 else expert_counts
        
        # FLOP reduction estimate
        static_flops = total_tokens * self.num_experts  # If all experts were used
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
    
    def _softmax(self, x: Any, axis: int = -1) -> Any:
        """Compute softmax activation."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def get_expert_usage_stats(self) -> Dict[str, Any]:
        """Get expert usage statistics."""
        if not self.expert_usage_history:
            return {'message': 'No usage history available'}
        
        history = np.array(self.expert_usage_history)
        avg_usage = np.mean(history, axis=0)
        usage_variance = np.var(avg_usage)
        
        return {
            'total_batches': len(self.expert_usage_history),
            'avg_usage_per_expert': avg_usage.tolist(),
            'usage_variance': float(usage_variance),
            'most_used_expert': int(np.argmax(avg_usage)),
            'least_used_expert': int(np.argmin(avg_usage)),
            'load_balance_score': 1.0 / (1.0 + usage_variance)  # Higher is better
        }


class AdaptiveRouter(DynamicRouter):
    """Enhanced router with adaptive complexity thresholds.
    
    This router learns optimal complexity thresholds based on
    observed performance and adjusts routing decisions accordingly.
    """
    
    def __init__(self, adaptation_rate: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.adaptation_rate = adaptation_rate
        self.complexity_thresholds = np.linspace(0.0, 1.0, self.max_experts + 1)
        self.performance_history: List[float] = []
    
    def update_thresholds(self, performance_score: float):
        """Update complexity thresholds based on performance feedback.
        
        Args:
            performance_score: Performance metric (higher is better)
        """
        self.performance_history.append(performance_score)
        
        if len(self.performance_history) > 1:
            # Simple gradient-based update
            perf_gradient = performance_score - self.performance_history[-2]
            
            # Adjust thresholds to encourage better performance
            if perf_gradient > 0:
                # Performance improved - slight increase in complexity sensitivity
                self.complexity_thresholds[1:-1] *= (1 + self.adaptation_rate)
            else:
                # Performance degraded - reduce complexity sensitivity
                self.complexity_thresholds[1:-1] *= (1 - self.adaptation_rate)
            
            # Keep thresholds sorted and bounded
            self.complexity_thresholds = np.clip(self.complexity_thresholds, 0.0, 1.0)
            self.complexity_thresholds = np.sort(self.complexity_thresholds)
    
    def _compute_expert_counts(self, complexity_scores: Any) -> Any:
        """Compute expert counts using adaptive thresholds."""
        expert_counts = np.ones_like(complexity_scores, dtype=int) * self.min_experts
        
        for i in range(self.min_experts, self.max_experts):
            threshold = self.complexity_thresholds[i]
            mask = complexity_scores >= threshold
            expert_counts[mask] = i + 1
        
        return expert_counts
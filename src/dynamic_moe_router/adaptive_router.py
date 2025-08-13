"""Enhanced adaptive router with load balancing and dynamic expert selection."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .router import DynamicRouter
from .estimator import ComplexityEstimator, get_estimator
from .exceptions import (
    ComplexityEstimationError,
    ExpertDispatchError,
    RouterConfigurationError,
)
from .validation import (
    validate_tensor_shape,
    validate_router_config,
    check_memory_usage
)

logger = logging.getLogger(__name__)


class AdaptiveLoadBalancer:
    """Adaptive load balancing for expert selection."""
    
    def __init__(
        self,
        num_experts: int,
        alpha: float = 0.01,  # Load balancing strength
        smoothing_factor: float = 0.9,  # Historical averaging
        capacity_factor: float = 1.25  # Expert capacity multiplier
    ):
        self.num_experts = num_experts
        self.alpha = alpha
        self.smoothing_factor = smoothing_factor
        self.capacity_factor = capacity_factor
        
        # Track expert utilization
        self.expert_counts = np.zeros(num_experts)
        self.total_tokens = 0
        
    def update_counts(self, expert_indices: np.ndarray) -> None:
        """Update expert utilization counts."""
        unique_experts, counts = np.unique(expert_indices, return_counts=True)
        
        # Update with smoothing
        new_counts = np.zeros(self.num_experts)
        new_counts[unique_experts] = counts
        
        self.expert_counts = (
            self.smoothing_factor * self.expert_counts +
            (1 - self.smoothing_factor) * new_counts
        )
        self.total_tokens += expert_indices.size
        
    def compute_load_balancing_loss(self, router_logits: np.ndarray) -> float:
        """Compute auxiliary load balancing loss."""
        if self.total_tokens == 0:
            return 0.0
            
        # Compute expert probabilities
        expert_probs = np.exp(router_logits - np.max(router_logits, axis=-1, keepdims=True))
        expert_probs = expert_probs / np.sum(expert_probs, axis=-1, keepdims=True)
        
        # Average probability per expert
        avg_probs = np.mean(expert_probs, axis=(0, 1))
        
        # Fraction of tokens assigned to each expert
        expert_fractions = self.expert_counts / max(self.total_tokens, 1)
        
        # Load balancing loss (encourage uniform distribution)
        lb_loss = self.alpha * np.sum(avg_probs * expert_fractions)
        
        return float(lb_loss)
    
    def apply_load_balancing(self, router_logits: np.ndarray) -> np.ndarray:
        """Apply load balancing to router logits."""
        if self.total_tokens == 0:
            return router_logits
            
        # Compute load balancing penalty
        expert_fractions = self.expert_counts / max(self.total_tokens, 1)
        
        # Penalize overused experts
        penalty = self.alpha * expert_fractions
        adjusted_logits = router_logits - penalty[np.newaxis, np.newaxis, :]
        
        return adjusted_logits


class EnhancedDynamicRouter(DynamicRouter):
    """Enhanced dynamic router with adaptive load balancing and improved expert selection."""
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        min_experts: int = 1,
        max_experts: Optional[int] = None,
        complexity_estimator: Union[str, ComplexityEstimator] = "gradient_norm",
        routing_strategy: str = "adaptive_top_k",
        load_balancing: bool = True,
        noise_factor: float = 0.1,
        temperature: float = 1.0,
        expert_capacity_factor: float = 1.25,
        adaptive_threshold: bool = True,
        **estimator_kwargs
    ):
        super().__init__(
            input_dim=input_dim,
            num_experts=num_experts,
            min_experts=min_experts,
            max_experts=max_experts,
            complexity_estimator=complexity_estimator,
            routing_strategy=routing_strategy,
            load_balancing=load_balancing,
            noise_factor=noise_factor,
            **estimator_kwargs
        )
        
        # Enhanced parameters
        self.temperature = temperature
        self.expert_capacity_factor = expert_capacity_factor
        self.adaptive_threshold = adaptive_threshold
        
        # Load balancer
        if load_balancing:
            self.load_balancer = AdaptiveLoadBalancer(
                num_experts=num_experts,
                capacity_factor=expert_capacity_factor
            )
        else:
            self.load_balancer = None
            
        # Adaptive thresholding
        self.complexity_history = []
        self.max_history = 1000
        
        logger.info(f"Enhanced router initialized with {routing_strategy} strategy")
    
    def _compute_adaptive_k(self, complexity_scores: np.ndarray) -> np.ndarray:
        """Compute adaptive number of experts based on complexity and history."""
        batch_size, seq_len = complexity_scores.shape
        
        if self.adaptive_threshold and len(self.complexity_history) > 10:
            # Use historical complexity distribution for adaptive thresholding
            hist_mean = np.mean(self.complexity_history)
            hist_std = np.std(self.complexity_history)
            
            # Normalize complexity relative to history
            normalized_complexity = (complexity_scores - hist_mean) / max(hist_std, 1e-6)
            normalized_complexity = np.clip(normalized_complexity, -2, 2)  # Clip outliers
            complexity_scores = (normalized_complexity + 2) / 4  # Map to [0, 1]
        
        # Update history
        self.complexity_history.extend(complexity_scores.flatten())
        if len(self.complexity_history) > self.max_history:
            self.complexity_history = self.complexity_history[-self.max_history:]
        
        # Compute k using enhanced strategy
        if self.routing_strategy == "adaptive_top_k":
            # Sigmoid-based adaptive k with steeper transition
            sigmoid_factor = 4.0  # Controls transition steepness
            k_ratio = 1 / (1 + np.exp(-sigmoid_factor * (complexity_scores - 0.5)))
            k_continuous = self.min_experts + (self.max_experts - self.min_experts) * k_ratio
        else:
            # Linear interpolation (original strategy)
            k_continuous = self.min_experts + (self.max_experts - self.min_experts) * complexity_scores
        
        # Round to integers with stochastic rounding for better distribution
        k_floor = np.floor(k_continuous).astype(int)
        k_remainder = k_continuous - k_floor
        
        # Stochastic rounding
        random_values = np.random.random((batch_size, seq_len))
        k = k_floor + (random_values < k_remainder).astype(int)
        
        # Ensure bounds
        k = np.clip(k, self.min_experts, self.max_experts)
        
        return k
    
    def _select_experts_with_noise(self, router_logits: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Select experts with optional noise and load balancing."""
        batch_size, seq_len, num_experts = router_logits.shape
        
        # Apply load balancing if enabled
        if self.load_balancer is not None:
            router_logits = self.load_balancer.apply_load_balancing(router_logits)
        
        # Add noise for exploration
        if self.noise_factor > 0:
            noise = np.random.normal(0, self.noise_factor, router_logits.shape)
            router_logits = router_logits + noise
        
        # Apply temperature scaling
        router_logits = router_logits / self.temperature
        
        # Select top-k experts for each token
        expert_indices = np.zeros((batch_size, seq_len, self.max_experts), dtype=int)
        expert_weights = np.zeros((batch_size, seq_len, self.max_experts))
        
        for i in range(batch_size):
            for j in range(seq_len):
                k_ij = k[i, j]
                
                # Get top-k indices
                top_k_indices = np.argpartition(router_logits[i, j], -k_ij)[-k_ij:]
                top_k_logits = router_logits[i, j, top_k_indices]
                
                # Compute softmax weights
                exp_logits = np.exp(top_k_logits - np.max(top_k_logits))
                weights = exp_logits / np.sum(exp_logits)
                
                # Store results
                expert_indices[i, j, :k_ij] = top_k_indices
                expert_weights[i, j, :k_ij] = weights
        
        return expert_indices, expert_weights
    
    def route(
        self,
        hidden_states: Any,
        return_router_logits: bool = False,
        return_load_balancing_loss: bool = False,
        **complexity_kwargs
    ) -> Dict[str, Any]:
        """Enhanced routing with adaptive load balancing."""
        
        try:
            # Validate inputs
            validate_tensor_shape(
                hidden_states,
                expected_dims=3,
                min_shape=(1, 1, 1),
                name="hidden_states"
            )
            
            batch_size, seq_len, hidden_dim = hidden_states.shape
            
            if hidden_dim != self.input_dim:
                raise RouterConfigurationError(
                    f"Input hidden_dim ({hidden_dim}) doesn't match router input_dim ({self.input_dim})"
                )
            
            # Estimate complexity
            try:
                complexity_scores = self.complexity_estimator.estimate(hidden_states, **complexity_kwargs)
                
                if isinstance(complexity_scores, (int, float)):
                    complexity_scores = np.full((batch_size, seq_len), complexity_scores)
                elif len(complexity_scores.shape) == 1:
                    complexity_scores = np.tile(complexity_scores, (batch_size, 1))
                    
            except Exception as e:
                raise ComplexityEstimationError(f"Complexity estimation failed: {e}")
            
            # Compute adaptive k
            k = self._compute_adaptive_k(complexity_scores)
            
            # Compute router logits
            if hasattr(hidden_states, 'numpy'):
                hidden_np = hidden_states.numpy()
            else:
                hidden_np = np.array(hidden_states)
            
            # Reshape for matrix multiplication
            hidden_reshaped = hidden_np.reshape(-1, self.input_dim)
            logits_reshaped = np.dot(hidden_reshaped, self.router_weights) + self.router_bias
            router_logits = logits_reshaped.reshape(batch_size, seq_len, self.num_experts)
            
            # Select experts with enhanced strategy
            expert_indices, expert_weights = self._select_experts_with_noise(router_logits, k)
            
            # Update load balancer
            if self.load_balancer is not None:
                valid_indices = expert_indices[expert_indices < self.num_experts]
                if len(valid_indices) > 0:
                    self.load_balancer.update_counts(valid_indices)
            
            # Prepare routing info
            routing_info = {
                "avg_experts_per_token": float(np.mean(k)),
                "max_experts_per_token": int(np.max(k)),
                "min_experts_per_token": int(np.min(k)),
                "complexity_mean": float(np.mean(complexity_scores)),
                "complexity_std": float(np.std(complexity_scores)),
                "routing_efficiency": float(np.mean(k)) / self.num_experts,
                "flop_reduction": 1.0 - (float(np.mean(k)) / self.num_experts)
            }
            
            # Add load balancing loss if requested
            if return_load_balancing_loss and self.load_balancer is not None:
                routing_info["load_balancing_loss"] = self.load_balancer.compute_load_balancing_loss(router_logits)
            
            # Build result dictionary
            result = {
                "expert_indices": expert_indices,
                "expert_weights": expert_weights,
                "num_experts_per_token": k,
                "complexity_scores": complexity_scores,
                "routing_info": routing_info
            }
            
            if return_router_logits:
                result["router_logits"] = router_logits
            
            logger.debug(f"Routing completed: avg_k={routing_info['avg_experts_per_token']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced routing failed: {e}")
            raise ExpertDispatchError(f"Enhanced routing failed: {e}")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        stats = {
            "num_experts": self.num_experts,
            "min_experts": self.min_experts,
            "max_experts": self.max_experts,
            "routing_strategy": self.routing_strategy,
            "temperature": self.temperature,
            "noise_factor": self.noise_factor,
            "load_balancing_enabled": self.load_balancer is not None
        }
        
        if self.load_balancer is not None:
            stats.update({
                "total_tokens_processed": self.load_balancer.total_tokens,
                "expert_utilization": self.load_balancer.expert_counts.tolist(),
                "load_balancing_alpha": self.load_balancer.alpha
            })
        
        if self.complexity_history:
            stats.update({
                "complexity_history_size": len(self.complexity_history),
                "complexity_mean": float(np.mean(self.complexity_history)),
                "complexity_std": float(np.std(self.complexity_history))
            })
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset routing statistics."""
        if self.load_balancer is not None:
            self.load_balancer.expert_counts = np.zeros(self.num_experts)
            self.load_balancer.total_tokens = 0
        
        self.complexity_history = []
        logger.info("Routing statistics reset")
"""
Quadratic Attention-Gated Dynamic Routing for MoE Models

This module implements a novel 2024 research algorithm that establishes a
connection between MoE frameworks and attention mechanisms using quadratic
gating for more expressive and efficient expert selection.

Based on recent research showing quadratic gating can serve as a more
expressive alternative to traditional MoE routing, with improved sample
efficiency and theoretical guarantees.

Key Innovations:
- Quadratic gating mechanism mimicking self-attention patterns
- Dynamic expert allocation based on attention importance scores
- Hierarchical expert selection with multi-head routing
- Theoretical connection between attention and expert routing

Author: Terry (Terragon Labs)
Research Period: 2024 Advanced MoE Routing Algorithms
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .estimator import ComplexityEstimator, get_estimator
from .exceptions import (
    ComplexityEstimationError,
    ExpertDispatchError,
    RouterConfigurationError,
)
from .validation import (
    validate_complexity_scores,
    validate_expert_indices,
    validate_expert_weights,
    validate_router_config,
    validate_tensor_shape,
)

logger = logging.getLogger(__name__)


class QuadraticAttentionGate:
    """
    Quadratic gating mechanism that establishes a connection between
    self-attention and expert routing.
    
    The gate computes attention-like scores using quadratic transformations:
    Gate(x) = softmax(x^T W_q W_k^T x + b)
    
    This design allows for more expressive routing patterns compared to
    traditional linear gating functions.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_bias: bool = True
    ):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.num_heads = num_heads
        self.head_dim = head_dim or input_dim // num_heads
        self.dropout = dropout
        self.temperature = temperature
        self.use_bias = use_bias
        
        # Validate dimensions
        if self.head_dim * self.num_heads != self.input_dim:
            raise RouterConfigurationError(
                f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
            )
        
        # Initialize quadratic gating parameters
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize quadratic gating parameters with Xavier/Glorot initialization."""
        # Query and key transformation matrices for each head
        scale = math.sqrt(2.0 / (self.input_dim + self.head_dim))
        
        self.W_query = np.random.normal(
            0, scale, (self.num_heads, self.input_dim, self.head_dim)
        ).astype(np.float32)
        
        self.W_key = np.random.normal(
            0, scale, (self.num_heads, self.input_dim, self.head_dim)
        ).astype(np.float32)
        
        # Expert projection matrix
        expert_scale = math.sqrt(2.0 / (self.num_heads + self.num_experts))
        self.W_expert = np.random.normal(
            0, expert_scale, (self.num_heads, self.num_experts)
        ).astype(np.float32)
        
        # Bias terms
        if self.use_bias:
            self.bias = np.zeros((self.num_experts,), dtype=np.float32)
        else:
            self.bias = None
            
    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute quadratic attention-gated routing scores.
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            routing_scores: Expert routing probabilities (batch_size, seq_len, num_experts)
            routing_info: Dictionary with attention patterns and statistics
        """
        batch_size, seq_len, _ = inputs.shape
        
        # Compute multi-head attention-like scores
        attention_scores = []
        attention_patterns = []
        
        for head in range(self.num_heads):
            # Transform inputs to queries and keys
            queries = np.dot(inputs, self.W_query[head])  # (B, L, head_dim)
            keys = np.dot(inputs, self.W_key[head])       # (B, L, head_dim)
            
            # Compute quadratic attention scores
            # Score(i,j) = q_i^T k_j / sqrt(head_dim)
            head_scores = np.matmul(queries, keys.transpose(0, 2, 1)) / math.sqrt(self.head_dim)
            
            # Apply temperature scaling
            head_scores = head_scores / self.temperature
            
            # Softmax over sequence dimension for attention pattern
            attention_pattern = self._softmax(head_scores, axis=-1)
            attention_patterns.append(attention_pattern)
            
            # Aggregate attention scores across sequence for expert routing
            # Use max pooling to capture most important attention patterns
            aggregated_scores = np.max(head_scores, axis=-1)  # (B, L)
            attention_scores.append(aggregated_scores)
            
        # Stack multi-head scores
        multi_head_scores = np.stack(attention_scores, axis=-1)  # (B, L, num_heads)
        
        # Project to expert space using learned transformation
        expert_logits = np.dot(multi_head_scores, self.W_expert.T)  # (B, L, num_experts)
        
        # Add bias if enabled
        if self.bias is not None:
            expert_logits = expert_logits + self.bias
            
        # Apply final softmax for expert probabilities
        routing_scores = self._softmax(expert_logits, axis=-1)
        
        # Compute routing statistics
        routing_info = {
            'attention_patterns': np.stack(attention_patterns, axis=1),  # (B, num_heads, L, L)
            'expert_logits': expert_logits,
            'multi_head_scores': multi_head_scores,
            'routing_entropy': self._compute_entropy(routing_scores),
            'attention_entropy': self._compute_attention_entropy(attention_patterns),
            'expert_utilization': np.mean(routing_scores, axis=(0, 1)),
            'temperature': self.temperature
        }
        
        return routing_scores, routing_info
        
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax implementation."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        
    def _compute_entropy(self, probs: np.ndarray) -> np.ndarray:
        """Compute entropy of probability distributions."""
        # Add small epsilon to prevent log(0)
        eps = 1e-10
        return -np.sum(probs * np.log(probs + eps), axis=-1)
        
    def _compute_attention_entropy(self, attention_patterns: List[np.ndarray]) -> np.ndarray:
        """Compute average entropy across attention heads."""
        entropies = []
        for pattern in attention_patterns:
            entropy = self._compute_entropy(pattern)
            entropies.append(np.mean(entropy, axis=-1))  # Average over sequence
        return np.stack(entropies, axis=-1)  # (B, L, num_heads)


class DynamicAttentionAllocationRouter:
    """
    Dynamic expert allocation based on attention importance scores.
    
    Implements the DA-MoE algorithm (2024) that dynamically determines
    the number of experts (K) based on token importance derived from
    attention mechanisms.
    
    Key features:
    - Token importance scoring using attention mechanisms
    - Dynamic K determination per token
    - Load balancing with importance-aware routing
    - Efficient expert allocation strategies
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        min_experts: int = 1,
        max_experts: Optional[int] = None,
        importance_threshold: float = 0.5,
        allocation_strategy: str = "sigmoid",  # "sigmoid", "linear", "exponential"
        load_balancing_factor: float = 0.01,
        attention_aggregation: str = "max",  # "max", "mean", "weighted"
        **kwargs
    ):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.min_experts = min_experts
        self.max_experts = max_experts or num_experts
        self.importance_threshold = importance_threshold
        self.allocation_strategy = allocation_strategy
        self.load_balancing_factor = load_balancing_factor
        self.attention_aggregation = attention_aggregation
        
        # Validate configuration
        validate_router_config({
            'input_dim': input_dim,
            'num_experts': num_experts,
            'min_experts': min_experts,
            'max_experts': self.max_experts
        })
        
        # Initialize attention gate for importance scoring
        self.attention_gate = QuadraticAttentionGate(
            input_dim=input_dim,
            num_experts=num_experts,
            **kwargs
        )
        
        # Initialize expert routing network
        self._initialize_routing_network()
        
    def _initialize_routing_network(self):
        """Initialize the expert routing network."""
        # Simple linear transformation for expert selection
        scale = math.sqrt(2.0 / self.input_dim)
        self.W_routing = np.random.normal(
            0, scale, (self.input_dim, self.num_experts)
        ).astype(np.float32)
        self.b_routing = np.zeros(self.num_experts, dtype=np.float32)
        
        # Load balancing parameters
        self.expert_load_history = np.ones(self.num_experts, dtype=np.float32)
        
    def route(
        self,
        inputs: np.ndarray,
        attention_weights: Optional[np.ndarray] = None,
        return_routing_info: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
        """
        Perform dynamic expert routing based on attention importance.
        
        Args:
            inputs: Input tensor (batch_size, seq_len, input_dim)
            attention_weights: Optional external attention weights
            return_routing_info: Whether to return detailed routing information
            
        Returns:
            expert_indices: Selected expert indices for each token
            expert_weights: Routing weights for selected experts
            routing_info: Optional detailed routing information
        """
        batch_size, seq_len, _ = inputs.shape
        
        # Compute attention-based importance scores
        if attention_weights is not None:
            importance_scores = self._compute_external_importance(attention_weights)
        else:
            importance_scores = self._compute_internal_importance(inputs)
            
        # Dynamically determine number of experts per token
        dynamic_k = self._compute_dynamic_k(importance_scores)
        
        # Compute expert routing scores
        routing_logits = np.dot(inputs, self.W_routing) + self.b_routing
        
        # Apply load balancing
        routing_logits = self._apply_load_balancing(routing_logits)
        
        # Select top-K experts dynamically
        expert_indices, expert_weights = self._select_dynamic_experts(
            routing_logits, dynamic_k
        )
        
        # Update load history
        self._update_load_history(expert_indices)
        
        if return_routing_info:
            routing_info = {
                'importance_scores': importance_scores,
                'dynamic_k': dynamic_k,
                'routing_logits': routing_logits,
                'expert_utilization': self.expert_load_history,
                'average_experts_per_token': np.mean(dynamic_k),
                'token_complexity_distribution': self._analyze_complexity_distribution(importance_scores)
            }
            return expert_indices, expert_weights, routing_info
        
        return expert_indices, expert_weights
        
    def _compute_internal_importance(self, inputs: np.ndarray) -> np.ndarray:
        """Compute token importance using internal attention mechanism."""
        # Use quadratic attention gate to compute importance
        _, routing_info = self.attention_gate.forward(inputs)
        
        # Extract attention patterns and compute importance
        attention_patterns = routing_info['attention_patterns']  # (B, num_heads, L, L)
        
        if self.attention_aggregation == "max":
            # Maximum attention received by each token across heads
            importance = np.max(np.max(attention_patterns, axis=-1), axis=1)  # (B, L)
        elif self.attention_aggregation == "mean":
            # Average attention received across heads
            importance = np.mean(np.mean(attention_patterns, axis=-1), axis=1)  # (B, L)
        elif self.attention_aggregation == "weighted":
            # Weighted combination based on routing entropy
            routing_entropy = routing_info['routing_entropy']
            weights = 1.0 / (routing_entropy + 1e-8)  # Higher weight for lower entropy
            weights = weights / np.sum(weights, axis=1, keepdims=True)
            
            attention_scores = np.mean(attention_patterns, axis=-1)  # (B, num_heads, L)
            importance = np.sum(attention_scores * weights[..., None], axis=1)  # (B, L)
        else:
            raise ValueError(f"Unknown attention aggregation: {self.attention_aggregation}")
            
        return importance
        
    def _compute_external_importance(self, attention_weights: np.ndarray) -> np.ndarray:
        """Compute importance from external attention weights."""
        # Attention weights expected shape: (batch_size, num_heads, seq_len, seq_len)
        if self.attention_aggregation == "max":
            importance = np.max(np.max(attention_weights, axis=-1), axis=1)
        elif self.attention_aggregation == "mean":
            importance = np.mean(np.mean(attention_weights, axis=-1), axis=1)
        else:
            # Default to max aggregation
            importance = np.max(np.max(attention_weights, axis=-1), axis=1)
            
        return importance
        
    def _compute_dynamic_k(self, importance_scores: np.ndarray) -> np.ndarray:
        """Compute dynamic number of experts based on importance scores."""
        # Normalize importance scores to [0, 1]
        normalized_importance = (importance_scores - np.min(importance_scores, axis=1, keepdims=True))
        max_vals = np.max(normalized_importance, axis=1, keepdims=True)
        normalized_importance = normalized_importance / (max_vals + 1e-8)
        
        # Apply allocation strategy
        if self.allocation_strategy == "sigmoid":
            # Sigmoid-based allocation
            scaled_importance = (normalized_importance - self.importance_threshold) * 6  # Scale for sigmoid
            allocation_factor = 1.0 / (1.0 + np.exp(-scaled_importance))
        elif self.allocation_strategy == "linear":
            # Linear allocation
            allocation_factor = np.clip(
                normalized_importance / self.importance_threshold, 0.0, 1.0
            )
        elif self.allocation_strategy == "exponential":
            # Exponential allocation
            allocation_factor = np.clip(
                np.exp(normalized_importance - self.importance_threshold), 0.0, 1.0
            )
        else:
            raise ValueError(f"Unknown allocation strategy: {self.allocation_strategy}")
            
        # Map to expert count range
        expert_range = self.max_experts - self.min_experts
        dynamic_k = self.min_experts + (allocation_factor * expert_range)
        
        # Round to integers and ensure bounds
        dynamic_k = np.clip(np.round(dynamic_k).astype(int), self.min_experts, self.max_experts)
        
        return dynamic_k
        
    def _apply_load_balancing(self, routing_logits: np.ndarray) -> np.ndarray:
        """Apply load balancing to routing logits."""
        # Penalize overused experts
        load_penalty = self.load_balancing_factor * self.expert_load_history
        return routing_logits - load_penalty
        
    def _select_dynamic_experts(
        self, routing_logits: np.ndarray, dynamic_k: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select top-K experts dynamically per token."""
        batch_size, seq_len, num_experts = routing_logits.shape
        
        # Initialize output arrays
        max_k = np.max(dynamic_k)
        expert_indices = np.full((batch_size, seq_len, max_k), -1, dtype=int)
        expert_weights = np.zeros((batch_size, seq_len, max_k), dtype=np.float32)
        
        # Process each token individually
        for b in range(batch_size):
            for s in range(seq_len):
                k = dynamic_k[b, s]
                
                # Get top-k experts for this token
                logits = routing_logits[b, s]
                top_k_indices = np.argpartition(logits, -k)[-k:]
                top_k_indices = top_k_indices[np.argsort(logits[top_k_indices])[::-1]]
                
                # Compute softmax weights over selected experts
                top_k_logits = logits[top_k_indices]
                top_k_weights = np.exp(top_k_logits - np.max(top_k_logits))
                top_k_weights = top_k_weights / np.sum(top_k_weights)
                
                # Store results
                expert_indices[b, s, :k] = top_k_indices
                expert_weights[b, s, :k] = top_k_weights
                
        return expert_indices, expert_weights
        
    def _update_load_history(self, expert_indices: np.ndarray):
        """Update expert load history for load balancing."""
        # Count expert usage
        expert_counts = np.zeros(self.num_experts)
        valid_mask = expert_indices >= 0
        
        if np.any(valid_mask):
            valid_indices = expert_indices[valid_mask]
            for idx in valid_indices:
                expert_counts[idx] += 1
                
        # Update exponential moving average
        decay = 0.99
        self.expert_load_history = decay * self.expert_load_history + (1 - decay) * expert_counts
        
    def _analyze_complexity_distribution(self, importance_scores: np.ndarray) -> Dict[str, float]:
        """Analyze the distribution of token complexity."""
        flat_scores = importance_scores.flatten()
        return {
            'mean_importance': float(np.mean(flat_scores)),
            'std_importance': float(np.std(flat_scores)),
            'min_importance': float(np.min(flat_scores)),
            'max_importance': float(np.max(flat_scores)),
            'median_importance': float(np.median(flat_scores)),
            'complexity_variance': float(np.var(flat_scores))
        }


class QuadraticAttentionDynamicRouter:
    """
    Complete quadratic attention-gated dynamic routing system.
    
    Combines quadratic attention gating with dynamic expert allocation
    for state-of-the-art MoE routing performance.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        min_experts: int = 1,
        max_experts: Optional[int] = None,
        num_attention_heads: int = 8,
        complexity_estimator: Union[str, ComplexityEstimator] = "gradient_norm",
        enable_quadratic_gating: bool = True,
        enable_dynamic_allocation: bool = True,
        **kwargs
    ):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.min_experts = min_experts
        self.max_experts = max_experts or num_experts
        self.num_attention_heads = num_attention_heads
        self.enable_quadratic_gating = enable_quadratic_gating
        self.enable_dynamic_allocation = enable_dynamic_allocation
        
        # Initialize complexity estimator
        if isinstance(complexity_estimator, str):
            self.complexity_estimator = get_estimator(complexity_estimator)
        else:
            self.complexity_estimator = complexity_estimator
            
        # Initialize routing components
        if enable_quadratic_gating:
            self.quadratic_gate = QuadraticAttentionGate(
                input_dim=input_dim,
                num_experts=num_experts,
                num_heads=num_attention_heads,
                **kwargs
            )
            
        if enable_dynamic_allocation:
            self.dynamic_allocator = DynamicAttentionAllocationRouter(
                input_dim=input_dim,
                num_experts=num_experts,
                min_experts=min_experts,
                max_experts=max_experts,
                **kwargs
            )
        else:
            # Fall back to standard routing
            self._initialize_standard_routing()
            
    def _initialize_standard_routing(self):
        """Initialize standard linear routing as fallback."""
        scale = math.sqrt(2.0 / self.input_dim)
        self.W_standard = np.random.normal(
            0, scale, (self.input_dim, self.num_experts)
        ).astype(np.float32)
        self.b_standard = np.zeros(self.num_experts, dtype=np.float32)
        
    def route(
        self,
        inputs: np.ndarray,
        attention_weights: Optional[np.ndarray] = None,
        return_routing_info: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
        """
        Perform quadratic attention-gated dynamic routing.
        
        Args:
            inputs: Input tensor (batch_size, seq_len, input_dim)
            attention_weights: Optional external attention weights
            return_routing_info: Whether to return detailed routing information
            
        Returns:
            expert_indices: Selected expert indices
            expert_weights: Expert routing weights
            routing_info: Optional detailed routing information
        """
        routing_info = {}
        
        # Estimate input complexity
        try:
            complexity_scores = self.complexity_estimator.estimate(inputs)
            routing_info['complexity_scores'] = complexity_scores
        except Exception as e:
            logger.warning(f"Complexity estimation failed: {e}")
            complexity_scores = np.ones((inputs.shape[0], inputs.shape[1]))
            
        # Route using enabled components
        if self.enable_dynamic_allocation:
            # Use dynamic allocation router
            if return_routing_info:
                expert_indices, expert_weights, dynamic_info = self.dynamic_allocator.route(
                    inputs, attention_weights, return_routing_info=True
                )
                routing_info.update(dynamic_info)
            else:
                expert_indices, expert_weights = self.dynamic_allocator.route(
                    inputs, attention_weights, return_routing_info=False
                )
                
        elif self.enable_quadratic_gating:
            # Use quadratic gating only
            routing_scores, gate_info = self.quadratic_gate.forward(inputs)
            routing_info.update(gate_info)
            
            # Select top-k experts (fixed k for this mode)
            k = min(self.max_experts, max(2, self.num_experts // 4))
            expert_indices, expert_weights = self._select_top_k_experts(routing_scores, k)
            
        else:
            # Standard linear routing fallback
            routing_logits = np.dot(inputs, self.W_standard) + self.b_standard
            routing_scores = self._softmax(routing_logits, axis=-1)
            
            k = min(self.max_experts, max(2, self.num_experts // 4))
            expert_indices, expert_weights = self._select_top_k_experts(routing_scores, k)
            
            routing_info['routing_scores'] = routing_scores
            routing_info['routing_logits'] = routing_logits
            
        # Add general routing statistics
        routing_info.update({
            'algorithm': 'quadratic_attention_dynamic',
            'quadratic_gating_enabled': self.enable_quadratic_gating,
            'dynamic_allocation_enabled': self.enable_dynamic_allocation,
            'num_experts': self.num_experts,
            'input_shape': inputs.shape
        })
        
        if return_routing_info:
            return expert_indices, expert_weights, routing_info
        else:
            return expert_indices, expert_weights
            
    def _select_top_k_experts(
        self, routing_scores: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select top-k experts from routing scores."""
        batch_size, seq_len, num_experts = routing_scores.shape
        
        # Get top-k indices
        top_k_indices = np.argpartition(routing_scores, -k, axis=-1)[..., -k:]
        
        # Sort within top-k
        batch_indices = np.arange(batch_size)[:, None, None]
        seq_indices = np.arange(seq_len)[None, :, None]
        
        top_k_scores = routing_scores[batch_indices, seq_indices, top_k_indices]
        sort_indices = np.argsort(top_k_scores, axis=-1)[..., ::-1]
        
        expert_indices = np.take_along_axis(top_k_indices, sort_indices, axis=-1)
        expert_weights = np.take_along_axis(top_k_scores, sort_indices, axis=-1)
        
        # Renormalize weights
        expert_weights = expert_weights / np.sum(expert_weights, axis=-1, keepdims=True)
        
        return expert_indices, expert_weights
        
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# Export main classes
__all__ = [
    'QuadraticAttentionGate',
    'DynamicAttentionAllocationRouter', 
    'QuadraticAttentionDynamicRouter'
]
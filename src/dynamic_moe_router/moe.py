"""Dynamic Mixture-of-Experts layer implementation."""

from typing import Any, Callable, Dict, List, Union

import numpy as np

from .router import DynamicRouter


class MoELayer:
    """Dynamic Mixture-of-Experts layer with adaptive routing.
    
    This layer combines multiple expert networks with dynamic routing
    based on input complexity estimation.
    """

    def __init__(
        self,
        router: DynamicRouter,
        expert_fn: Callable[[], Any],
        num_experts: int,
        expert_capacity_factor: float = 1.25,
        dropout_rate: float = 0.0,
        use_bias: bool = True
    ):
        self.router = router
        self.num_experts = num_experts
        self.expert_capacity_factor = expert_capacity_factor
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias

        # Initialize experts
        self.experts = [expert_fn() for _ in range(num_experts)]

        # Performance tracking
        self.forward_calls = 0
        self.total_expert_calls = 0

    def forward(
        self,
        hidden_states: Any,
        return_router_logits: bool = False,
        **router_kwargs
    ) -> Union[Any, tuple]:
        """Forward pass through dynamic MoE layer.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            return_router_logits: Whether to return routing information
            **router_kwargs: Additional arguments for router
            
        Returns:
            If return_router_logits=False: output tensor [batch, seq_len, hidden_dim]
            If return_router_logits=True: (output, routing_info)
        """
        self.forward_calls += 1
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Step 1: Get routing decisions
        routing_result = self.router.route(
            hidden_states,
            return_router_logits=True,
            **router_kwargs
        )

        expert_indices = routing_result['expert_indices']
        expert_weights = routing_result['expert_weights']
        num_experts_per_token = routing_result['num_experts_per_token']

        # Step 2: Dispatch tokens to experts
        expert_outputs = self._dispatch_to_experts(
            hidden_states, expert_indices, expert_weights
        )

        # Step 3: Combine expert outputs
        output = self._combine_expert_outputs(
            expert_outputs, expert_indices, expert_weights,
            (batch_size, seq_len, hidden_dim)
        )

        # Step 4: Apply dropout if specified
        if self.dropout_rate > 0:
            output = self._apply_dropout(output)

        # Update statistics
        self.total_expert_calls += np.sum(num_experts_per_token)

        if return_router_logits:
            # Add MoE-specific statistics
            routing_result['moe_stats'] = self._compute_moe_stats(
                expert_indices, expert_outputs
            )
            return output, routing_result
        else:
            return output

    def _dispatch_to_experts(
        self,
        hidden_states: Any,
        expert_indices: Any,
        expert_weights: Any
    ) -> List[Any]:
        """Dispatch tokens to appropriate experts.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            expert_indices: Expert indices [batch, seq_len, max_k]
            expert_weights: Expert weights [batch, seq_len, max_k]
            
        Returns:
            List of expert outputs for each position
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        _, _, max_k = expert_indices.shape

        # Collect all expert computations
        expert_outputs = []

        for b in range(batch_size):
            batch_outputs = []
            for s in range(seq_len):
                token_input = hidden_states[b, s]  # [hidden_dim]
                token_outputs = []

                for k in range(max_k):
                    expert_idx = expert_indices[b, s, k]
                    if expert_idx >= 0:
                        # Compute expert output
                        expert_output = self._compute_expert_output(expert_idx, token_input)
                        token_outputs.append(expert_output)
                    else:
                        # Padding - use zero output
                        token_outputs.append(np.zeros_like(token_input))

                batch_outputs.append(token_outputs)
            expert_outputs.append(batch_outputs)

        return expert_outputs

    def _compute_expert_output(self, expert_idx: int, token_input: Any) -> Any:
        """Compute output for a specific expert.
        
        This is a framework-agnostic placeholder that will be overridden
        in framework-specific implementations.
        """
        expert = self.experts[expert_idx]

        # For base implementation, assume expert is a simple linear transformation
        if hasattr(expert, 'forward'):
            return expert.forward(token_input)
        elif hasattr(expert, '__call__'):
            return expert(token_input)
        else:
            # Fallback: simple linear transformation
            return self._simple_expert_forward(expert_idx, token_input)

    def _simple_expert_forward(self, expert_idx: int, token_input: Any) -> Any:
        """Simple expert forward pass for base implementation."""
        # Initialize expert weights if needed
        if not hasattr(self, 'expert_weights'):
            self._initialize_expert_weights(token_input.shape[-1])

        weights = self.expert_weights[expert_idx]
        bias = self.expert_bias[expert_idx] if self.use_bias else 0

        # Simple linear transformation
        output = np.dot(token_input, weights) + bias

        # Apply activation (ReLU)
        return np.maximum(0, output)

    def _initialize_expert_weights(self, hidden_dim: int):
        """Initialize expert network weights."""
        np.random.seed(42)  # For reproducibility

        # Each expert is a simple 2-layer MLP
        intermediate_dim = hidden_dim * 4  # Standard MoE expansion

        self.expert_weights = []
        self.expert_bias = []

        for i in range(self.num_experts):
            # First layer: expand
            w1 = np.random.randn(hidden_dim, intermediate_dim) * 0.02
            b1 = np.zeros(intermediate_dim) if self.use_bias else None

            # Second layer: project back
            w2 = np.random.randn(intermediate_dim, hidden_dim) * 0.02
            b2 = np.zeros(hidden_dim) if self.use_bias else None

            # Store as composite transformation
            expert_w = np.dot(w1, w2.T)  # Simplified for base implementation
            expert_b = b1 if self.use_bias else np.zeros(hidden_dim)

            self.expert_weights.append(expert_w)
            self.expert_bias.append(expert_b)

    def _combine_expert_outputs(
        self,
        expert_outputs: List[Any],
        expert_indices: Any,
        expert_weights: Any,
        output_shape: tuple
    ) -> Any:
        """Combine weighted expert outputs.
        
        Args:
            expert_outputs: List of expert outputs per position
            expert_indices: Expert indices [batch, seq_len, max_k]
            expert_weights: Expert weights [batch, seq_len, max_k]
            output_shape: Target output shape (batch, seq_len, hidden_dim)
            
        Returns:
            Combined output tensor
        """
        batch_size, seq_len, hidden_dim = output_shape
        output = np.zeros(output_shape)

        for b in range(batch_size):
            for s in range(seq_len):
                token_output = np.zeros(hidden_dim)

                # Combine outputs from selected experts
                token_expert_outputs = expert_outputs[b][s]
                for k, expert_output in enumerate(token_expert_outputs):
                    weight = expert_weights[b, s, k]
                    if weight > 0:
                        token_output += weight * expert_output

                output[b, s] = token_output

        return output

    def _apply_dropout(self, output: Any) -> Any:
        """Apply dropout to output (framework-agnostic placeholder)."""
        if self.dropout_rate > 0:
            # Simple dropout implementation
            mask = np.random.random(output.shape) > self.dropout_rate
            output = output * mask / (1 - self.dropout_rate)
        return output

    def _compute_moe_stats(self, expert_indices: Any, expert_outputs: List[Any]) -> Dict[str, Any]:
        """Compute MoE-specific statistics."""
        # Expert utilization
        used_experts = expert_indices[expert_indices >= 0]
        expert_counts = np.bincount(used_experts, minlength=self.num_experts)

        # Compute expert load balance
        total_calls = np.sum(expert_counts)
        ideal_load = total_calls / self.num_experts
        load_variance = np.var(expert_counts) if total_calls > 0 else 0.0

        # Efficiency metrics
        avg_experts_per_call = total_calls / self.forward_calls if self.forward_calls > 0 else 0

        return {
            'expert_call_counts': expert_counts.tolist(),
            'total_expert_calls': int(total_calls),
            'load_balance_variance': float(load_variance),
            'ideal_load_per_expert': float(ideal_load),
            'avg_experts_per_forward': float(avg_experts_per_call),
            'forward_calls': self.forward_calls
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the MoE layer."""
        if self.forward_calls == 0:
            return {'message': 'No forward passes performed yet'}

        # Calculate efficiency metrics
        max_possible_calls = self.forward_calls * self.num_experts
        efficiency = 1.0 - (self.total_expert_calls / max_possible_calls)

        # Get router statistics
        router_stats = self.router.get_expert_usage_stats()

        return {
            'total_forward_calls': self.forward_calls,
            'total_expert_calls': self.total_expert_calls,
            'avg_experts_per_forward': self.total_expert_calls / self.forward_calls,
            'computational_efficiency': efficiency,
            'router_stats': router_stats
        }

    def reset_stats(self):
        """Reset performance statistics."""
        self.forward_calls = 0
        self.total_expert_calls = 0
        self.router.expert_usage_history.clear()


class SparseMoELayer(MoELayer):
    """Sparse MoE layer with capacity constraints and load balancing.
    
    This implementation adds capacity constraints to prevent
    expert overload and improve training stability.
    """

    def __init__(self, capacity_factor: float = 1.25, **kwargs):
        super().__init__(**kwargs)
        self.capacity_factor = capacity_factor

    def _dispatch_to_experts(self, hidden_states: Any, expert_indices: Any, expert_weights: Any) -> List[Any]:
        """Dispatch with capacity constraints."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        total_tokens = batch_size * seq_len

        # Calculate expert capacities
        expert_capacity = int(self.capacity_factor * total_tokens / self.num_experts)
        expert_token_counts = np.zeros(self.num_experts, dtype=int)

        # Filter assignments based on capacity
        filtered_indices = np.copy(expert_indices)
        filtered_weights = np.copy(expert_weights)

        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(expert_indices.shape[2]):
                    expert_idx = expert_indices[b, s, k]
                    if expert_idx >= 0:
                        if expert_token_counts[expert_idx] < expert_capacity:
                            expert_token_counts[expert_idx] += 1
                        else:
                            # Expert at capacity - drop this assignment
                            filtered_indices[b, s, k] = -1
                            filtered_weights[b, s, k] = 0.0

        # Renormalize weights after capacity filtering
        for b in range(batch_size):
            for s in range(seq_len):
                token_weights = filtered_weights[b, s]
                valid_weights = token_weights[token_weights > 0]
                if len(valid_weights) > 0:
                    token_weights[token_weights > 0] /= np.sum(valid_weights)

        return super()._dispatch_to_experts(hidden_states, filtered_indices, filtered_weights)


class LayerNormMoE(MoELayer):
    """MoE layer with layer normalization and residual connections."""

    def __init__(self, layer_norm_eps: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm_eps = layer_norm_eps

    def forward(self, hidden_states: Any, return_router_logits: bool = False, **router_kwargs):
        """Forward with layer norm and residual connection."""
        # Store input for residual connection
        residual = hidden_states

        # Apply layer normalization to input
        normalized_input = self._layer_norm(hidden_states)

        # Standard MoE forward pass
        if return_router_logits:
            moe_output, routing_info = super().forward(
                normalized_input, return_router_logits=True, **router_kwargs
            )
        else:
            moe_output = super().forward(normalized_input, **router_kwargs)

        # Add residual connection
        output = residual + moe_output

        if return_router_logits:
            return output, routing_info
        else:
            return output

    def _layer_norm(self, x: Any) -> Any:
        """Apply layer normalization."""
        # Compute mean and variance along last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)

        # Normalize
        normalized = (x - mean) / np.sqrt(variance + self.layer_norm_eps)

        return normalized

"""PyTorch MoE layer implementations."""

from typing import Any, Callable, Dict, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..moe import MoELayer
from .router import TorchDynamicRouter


class TorchMoELayer(MoELayer, nn.Module):
    """PyTorch implementation of dynamic MoE layer."""

    def __init__(
        self,
        router: TorchDynamicRouter,
        expert_fn: Callable[[], nn.Module],
        num_experts: int,
        expert_capacity_factor: float = 1.25,
        dropout_rate: float = 0.0,
        use_bias: bool = True
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TorchMoELayer")

        # Initialize both parent classes
        MoELayer.__init__(self, router, expert_fn, num_experts, expert_capacity_factor, dropout_rate, use_bias)
        nn.Module.__init__(self)

        # Create expert networks as PyTorch modules
        self.experts = nn.ModuleList([expert_fn() for _ in range(num_experts)])

        # Dropout layer
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_router_logits: bool = False,
        **router_kwargs
    ) -> Union[torch.Tensor, tuple]:
        """Forward pass through PyTorch MoE layer.
        
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
        device = hidden_states.device

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
        output = self._dispatch_to_experts_torch(
            hidden_states, expert_indices, expert_weights
        )

        # Step 3: Apply dropout if specified
        if self.dropout is not None:
            output = self.dropout(output)

        # Update statistics
        self.total_expert_calls += torch.sum(num_experts_per_token).item()

        if return_router_logits:
            # Add MoE-specific statistics
            routing_result['moe_stats'] = self._compute_moe_stats_torch(
                expert_indices, expert_weights
            )
            return output, routing_result
        else:
            return output

    def _dispatch_to_experts_torch(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """Efficient PyTorch expert dispatching using gather/scatter operations."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        _, _, max_k = expert_indices.shape
        device = hidden_states.device

        # Initialize output
        output = torch.zeros_like(hidden_states)

        # Process each expert to maximize parallelization
        for expert_idx in range(self.num_experts):
            # Find all tokens assigned to this expert
            expert_mask = (expert_indices == expert_idx)  # [batch, seq, max_k]

            if expert_mask.any():
                # Get positions and weights for this expert
                batch_idx, seq_idx, k_idx = torch.where(expert_mask)

                if len(batch_idx) > 0:
                    # Gather input tokens for this expert
                    expert_inputs = hidden_states[batch_idx, seq_idx]  # [num_tokens, hidden_dim]

                    # Get corresponding weights
                    expert_token_weights = expert_weights[batch_idx, seq_idx, k_idx]  # [num_tokens]

                    # Process through expert
                    expert_outputs = self.experts[expert_idx](expert_inputs)  # [num_tokens, hidden_dim]

                    # Weight the outputs
                    weighted_outputs = expert_outputs * expert_token_weights.unsqueeze(-1)

                    # Scatter back to output tensor
                    output[batch_idx, seq_idx] += weighted_outputs

        return output

    def _compute_moe_stats_torch(self, expert_indices: torch.Tensor, expert_weights: torch.Tensor) -> Dict[str, Any]:
        """Compute MoE-specific statistics for PyTorch tensors."""
        # Expert utilization
        valid_indices = expert_indices[expert_indices >= 0]
        expert_counts = torch.bincount(valid_indices, minlength=self.num_experts)

        # Compute expert load balance
        total_calls = torch.sum(expert_counts).item()
        ideal_load = total_calls / self.num_experts if total_calls > 0 else 0
        load_variance = torch.var(expert_counts.float()).item() if total_calls > 0 else 0.0

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


class TorchSparseMoELayer(TorchMoELayer):
    """PyTorch sparse MoE layer with capacity constraints."""

    def __init__(self, capacity_factor: float = 1.25, **kwargs):
        super().__init__(**kwargs)
        self.capacity_factor = capacity_factor

    def _dispatch_to_experts_torch(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """Dispatch with capacity constraints."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        total_tokens = batch_size * seq_len

        # Calculate expert capacities
        expert_capacity = int(self.capacity_factor * total_tokens / self.num_experts)
        expert_token_counts = torch.zeros(self.num_experts, dtype=torch.int32, device=hidden_states.device)

        # Filter assignments based on capacity
        filtered_indices = expert_indices.clone()
        filtered_weights = expert_weights.clone()

        # Apply capacity constraints
        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(expert_indices.shape[2]):
                    expert_idx = expert_indices[b, s, k].item()
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
                valid_mask = token_weights > 0
                if valid_mask.any():
                    token_weights[valid_mask] /= token_weights[valid_mask].sum()

        return super()._dispatch_to_experts_torch(hidden_states, filtered_indices, filtered_weights)


class TorchLayerNormMoE(TorchMoELayer):
    """PyTorch MoE layer with layer normalization and residual connections."""

    def __init__(self, hidden_dim: int, layer_norm_eps: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, return_router_logits: bool = False, **router_kwargs):
        """Forward with layer norm and residual connection."""
        # Store input for residual connection
        residual = hidden_states

        # Apply layer normalization to input
        normalized_input = self.layer_norm(hidden_states)

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


# Simple expert implementations for testing and examples
class LinearExpert(nn.Module):
    """Simple linear expert for testing."""

    def __init__(self, input_dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim * 4  # Standard MoE expansion

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear2(x)


class GLUExpert(nn.Module):
    """Gated Linear Unit expert (similar to what's used in modern transformers)."""

    def __init__(self, input_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim * 4

        self.gate_linear = nn.Linear(input_dim, hidden_dim, bias=False)
        self.up_linear = nn.Linear(input_dim, hidden_dim, bias=False)
        self.down_linear = nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_linear(x))  # SiLU activation
        up = self.up_linear(x)
        return self.down_linear(gate * up)

"""PyTorch implementation of dynamic router."""

from typing import Any, Dict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..router import DynamicRouter

if not TORCH_AVAILABLE:
    # Create dummy base classes when PyTorch is not available
    class _DummyModule:
        pass
    nn = type('nn', (), {'Module': _DummyModule})()


class TorchDynamicRouter(DynamicRouter, nn.Module if TORCH_AVAILABLE else object):
    """PyTorch implementation of dynamic MoE router."""

    def __init__(self, **kwargs):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TorchDynamicRouter")

        # Initialize both parent classes
        DynamicRouter.__init__(self, **kwargs)
        nn.Module.__init__(self)

        # Create PyTorch router network
        self.router_network = nn.Linear(self.input_dim, self.num_experts)

        # Initialize with same logic as base class
        self._initialize_torch_weights()

    def _initialize_torch_weights(self):
        """Initialize PyTorch router weights."""
        torch.manual_seed(42)  # For reproducibility
        nn.init.normal_(self.router_network.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.router_network.bias)

    def route(
        self,
        hidden_states: torch.Tensor,
        return_router_logits: bool = False,
        **complexity_kwargs
    ) -> Dict[str, Any]:
        """Perform dynamic expert routing with PyTorch tensors.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            return_router_logits: Whether to return raw router scores
            **complexity_kwargs: Additional args for complexity estimation
            
        Returns:
            Dictionary with routing results (all PyTorch tensors)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        # Step 1: Estimate input complexity
        complexity_scores = self._estimate_complexity_torch(hidden_states, **complexity_kwargs)

        # Step 2: Determine number of experts per token
        num_experts_per_token = self._compute_expert_counts_torch(complexity_scores)

        # Step 3: Compute router logits
        router_logits = self.router_network(hidden_states)

        # Step 4: Apply noise for regularization
        if self.noise_factor > 0:
            noise = torch.randn_like(router_logits) * self.noise_factor
            router_logits = router_logits + noise

        # Step 5: Select experts based on strategy
        expert_indices, expert_weights = self._select_experts_torch(
            router_logits, num_experts_per_token
        )

        # Step 6: Apply load balancing
        if self.load_balancing:
            expert_weights = self._apply_load_balancing_torch(
                expert_indices, expert_weights, router_logits
            )

        # Compile results
        result = {
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'num_experts_per_token': num_experts_per_token,
            'complexity_scores': complexity_scores,
            'routing_info': self._compute_routing_stats_torch(
                expert_indices, num_experts_per_token, complexity_scores
            )
        }

        if return_router_logits:
            result['router_logits'] = router_logits

        return result

    def _estimate_complexity_torch(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Estimate complexity using PyTorch-compatible estimator."""
        if hasattr(self.complexity_estimator, 'estimate_torch'):
            return self.complexity_estimator.estimate_torch(hidden_states, **kwargs)
        else:
            # Fallback to numpy implementation and convert
            hidden_np = hidden_states.detach().cpu().numpy()
            complexity_np = self.complexity_estimator.estimate(hidden_np, **kwargs)
            return torch.from_numpy(complexity_np).to(hidden_states.device)

    def _compute_expert_counts_torch(self, complexity_scores: torch.Tensor) -> torch.Tensor:
        """Convert complexity scores to expert counts (PyTorch)."""
        expert_range = self.max_experts - self.min_experts
        raw_counts = self.min_experts + expert_range * complexity_scores

        # Round to integers and clamp
        expert_counts = torch.round(raw_counts).long()
        expert_counts = torch.clamp(expert_counts, self.min_experts, self.max_experts)

        return expert_counts

    def _select_experts_torch(
        self,
        router_logits: torch.Tensor,
        num_experts_per_token: torch.Tensor
    ) -> tuple:
        """Select experts using PyTorch operations."""
        batch_size, seq_len, num_experts = router_logits.shape
        device = router_logits.device
        max_k = self.max_experts

        # Initialize output tensors
        expert_indices = torch.full(
            (batch_size, seq_len, max_k), -1,
            dtype=torch.long, device=device
        )
        expert_weights = torch.zeros(
            (batch_size, seq_len, max_k),
            dtype=router_logits.dtype, device=device
        )

        if self.routing_strategy == "top_k":
            return self._top_k_selection_torch(
                router_logits, num_experts_per_token, expert_indices, expert_weights
            )
        elif self.routing_strategy == "threshold":
            return self._threshold_selection_torch(
                router_logits, num_experts_per_token, expert_indices, expert_weights
            )
        else:
            raise ValueError(f"Unknown routing strategy: {self.routing_strategy}")

    def _top_k_selection_torch(
        self,
        router_logits: torch.Tensor,
        num_experts_per_token: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> tuple:
        """Vectorized top-k selection for PyTorch."""
        batch_size, seq_len, num_experts = router_logits.shape
        max_k = self.max_experts

        # For efficiency, we'll handle the most common case (same k for all tokens)
        # and fall back to loop for variable k
        unique_ks = torch.unique(num_experts_per_token)

        if len(unique_ks) == 1:
            # All tokens use same number of experts - can vectorize
            k = int(unique_ks.item())
            if k > 0:
                # Get top-k indices and values
                top_logits, top_indices = torch.topk(router_logits, k, dim=-1)

                # Compute softmax weights
                weights = F.softmax(top_logits, dim=-1)

                # Store results
                expert_indices[:, :, :k] = top_indices
                expert_weights[:, :, :k] = weights
        else:
            # Variable k - need to handle token by token
            for b in range(batch_size):
                for s in range(seq_len):
                    k = int(num_experts_per_token[b, s].item())
                    if k > 0:
                        token_logits = router_logits[b, s]
                        top_logits, top_indices = torch.topk(token_logits, k)
                        weights = F.softmax(top_logits, dim=0)

                        expert_indices[b, s, :k] = top_indices
                        expert_weights[b, s, :k] = weights

        return expert_indices, expert_weights

    def _threshold_selection_torch(
        self,
        router_logits: torch.Tensor,
        num_experts_per_token: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> tuple:
        """Threshold-based selection for PyTorch."""
        batch_size, seq_len, num_experts = router_logits.shape

        # This is more complex to vectorize, so we use a loop
        for b in range(batch_size):
            for s in range(seq_len):
                k = int(num_experts_per_token[b, s].item())
                token_logits = router_logits[b, s]

                if k > 0:
                    # Set threshold to select approximately k experts
                    if k >= num_experts:
                        threshold = float('-inf')
                    else:
                        sorted_logits, _ = torch.sort(token_logits, descending=True)
                        threshold = sorted_logits[k-1].item()

                    # Select experts above threshold
                    selected_mask = token_logits >= threshold
                    selected_indices = torch.where(selected_mask)[0]

                    # Limit to max_k experts
                    if len(selected_indices) > self.max_experts:
                        # Take the highest scoring ones
                        selected_logits = token_logits[selected_indices]
                        _, top_idx = torch.topk(selected_logits, self.max_experts)
                        selected_indices = selected_indices[top_idx]

                    # Compute weights
                    if len(selected_indices) > 0:
                        selected_logits = token_logits[selected_indices]
                        weights = F.softmax(selected_logits, dim=0)

                        # Store results
                        num_selected = len(selected_indices)
                        expert_indices[b, s, :num_selected] = selected_indices
                        expert_weights[b, s, :num_selected] = weights

        return expert_indices, expert_weights

    def _apply_load_balancing_torch(
        self,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        router_logits: torch.Tensor
    ) -> torch.Tensor:
        """Apply load balancing using PyTorch operations."""
        # Convert to numpy for compatibility with base class logic
        expert_indices_np = expert_indices.detach().cpu().numpy()
        expert_weights_np = expert_weights.detach().cpu().numpy()
        router_logits_np = router_logits.detach().cpu().numpy()

        # Apply base class load balancing
        balanced_weights_np = super()._apply_load_balancing(
            expert_indices_np, expert_weights_np, router_logits_np
        )

        # Convert back to PyTorch
        return torch.from_numpy(balanced_weights_np).to(expert_weights.device)

    def _compute_routing_stats_torch(
        self,
        expert_indices: torch.Tensor,
        num_experts_per_token: torch.Tensor,
        complexity_scores: torch.Tensor
    ) -> Dict[str, Any]:
        """Compute routing statistics for PyTorch tensors."""
        batch_size, seq_len = complexity_scores.shape
        total_tokens = batch_size * seq_len

        # Average experts per token
        avg_experts = float(torch.mean(num_experts_per_token.float()).item())

        # Expert utilization distribution
        valid_indices = expert_indices[expert_indices >= 0]
        expert_counts = torch.bincount(valid_indices, minlength=self.num_experts)
        total_expert_calls = torch.sum(expert_counts).item()
        expert_utilization = (expert_counts.float() / total_expert_calls).tolist() if total_expert_calls > 0 else [0.0] * self.num_experts

        # FLOP reduction estimate
        static_flops = total_tokens * self.num_experts
        dynamic_flops = torch.sum(num_experts_per_token).item()
        flop_reduction = 1.0 - (dynamic_flops / static_flops) if static_flops > 0 else 0.0

        return {
            'avg_experts_per_token': avg_experts,
            'total_expert_calls': int(dynamic_flops),
            'flop_reduction': flop_reduction,
            'expert_utilization': expert_utilization,
            'complexity_stats': {
                'mean': float(torch.mean(complexity_scores).item()),
                'std': float(torch.std(complexity_scores).item()),
                'min': float(torch.min(complexity_scores).item()),
                'max': float(torch.max(complexity_scores).item())
            }
        }


class TorchAdaptiveRouter(TorchDynamicRouter):
    """PyTorch implementation of adaptive router."""

    def __init__(self, adaptation_rate: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.adaptation_rate = adaptation_rate

        # Store thresholds as learnable parameters
        initial_thresholds = torch.linspace(0.0, 1.0, self.max_experts + 1)
        self.register_parameter(
            'complexity_thresholds',
            nn.Parameter(initial_thresholds)
        )
        self.performance_history = []

    def update_thresholds(self, performance_score: float):
        """Update complexity thresholds based on performance feedback."""
        self.performance_history.append(performance_score)

        if len(self.performance_history) > 1:
            # Simple gradient-based update
            perf_gradient = performance_score - self.performance_history[-2]

            with torch.no_grad():
                # Adjust thresholds to encourage better performance
                if perf_gradient > 0:
                    # Performance improved
                    self.complexity_thresholds[1:-1] *= (1 + self.adaptation_rate)
                else:
                    # Performance degraded
                    self.complexity_thresholds[1:-1] *= (1 - self.adaptation_rate)

                # Keep thresholds sorted and bounded
                self.complexity_thresholds.clamp_(0.0, 1.0)
                self.complexity_thresholds.data = torch.sort(self.complexity_thresholds)[0]

    def _compute_expert_counts_torch(self, complexity_scores: torch.Tensor) -> torch.Tensor:
        """Compute expert counts using adaptive thresholds."""
        expert_counts = torch.ones_like(complexity_scores, dtype=torch.long) * self.min_experts

        for i in range(self.min_experts, self.max_experts):
            threshold = self.complexity_thresholds[i]
            mask = complexity_scores >= threshold
            expert_counts[mask] = i + 1

        return expert_counts

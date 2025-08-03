"""PyTorch implementation of dynamic router."""

from typing import Dict, Any, Optional, Union
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..router import DynamicRouter
from ..estimator import ComplexityEstimator


class TorchDynamicRouter(DynamicRouter, nn.Module):
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
        
        if self.routing_strategy == \"top_k\":\n            return self._top_k_selection_torch(\n                router_logits, num_experts_per_token, expert_indices, expert_weights\n            )\n        elif self.routing_strategy == \"threshold\":\n            return self._threshold_selection_torch(\n                router_logits, num_experts_per_token, expert_indices, expert_weights\n            )\n        else:\n            raise ValueError(f\"Unknown routing strategy: {self.routing_strategy}\")\n    \n    def _top_k_selection_torch(\n        self,\n        router_logits: torch.Tensor,\n        num_experts_per_token: torch.Tensor,\n        expert_indices: torch.Tensor,\n        expert_weights: torch.Tensor\n    ) -> tuple:\n        \"\"\"Vectorized top-k selection for PyTorch.\"\"\"\n        batch_size, seq_len, num_experts = router_logits.shape\n        max_k = self.max_experts\n        \n        # For efficiency, we'll handle the most common case (same k for all tokens)\n        # and fall back to loop for variable k\n        unique_ks = torch.unique(num_experts_per_token)\n        \n        if len(unique_ks) == 1:\n            # All tokens use same number of experts - can vectorize\n            k = int(unique_ks.item())\n            if k > 0:\n                # Get top-k indices and values\n                top_logits, top_indices = torch.topk(router_logits, k, dim=-1)\n                \n                # Compute softmax weights\n                weights = F.softmax(top_logits, dim=-1)\n                \n                # Store results\n                expert_indices[:, :, :k] = top_indices\n                expert_weights[:, :, :k] = weights\n        else:\n            # Variable k - need to handle token by token\n            for b in range(batch_size):\n                for s in range(seq_len):\n                    k = int(num_experts_per_token[b, s].item())\n                    if k > 0:\n                        token_logits = router_logits[b, s]\n                        top_logits, top_indices = torch.topk(token_logits, k)\n                        weights = F.softmax(top_logits, dim=0)\n                        \n                        expert_indices[b, s, :k] = top_indices\n                        expert_weights[b, s, :k] = weights\n        \n        return expert_indices, expert_weights\n    \n    def _threshold_selection_torch(\n        self,\n        router_logits: torch.Tensor,\n        num_experts_per_token: torch.Tensor,\n        expert_indices: torch.Tensor,\n        expert_weights: torch.Tensor\n    ) -> tuple:\n        \"\"\"Threshold-based selection for PyTorch.\"\"\"\n        batch_size, seq_len, num_experts = router_logits.shape\n        \n        # This is more complex to vectorize, so we use a loop\n        for b in range(batch_size):\n            for s in range(seq_len):\n                k = int(num_experts_per_token[b, s].item())\n                token_logits = router_logits[b, s]\n                \n                if k > 0:\n                    # Set threshold to select approximately k experts\n                    if k >= num_experts:\n                        threshold = float('-inf')\n                    else:\n                        sorted_logits, _ = torch.sort(token_logits, descending=True)\n                        threshold = sorted_logits[k-1].item()\n                    \n                    # Select experts above threshold\n                    selected_mask = token_logits >= threshold\n                    selected_indices = torch.where(selected_mask)[0]\n                    \n                    # Limit to max_k experts\n                    if len(selected_indices) > self.max_experts:\n                        # Take the highest scoring ones\n                        selected_logits = token_logits[selected_indices]\n                        _, top_idx = torch.topk(selected_logits, self.max_experts)\n                        selected_indices = selected_indices[top_idx]\n                    \n                    # Compute weights\n                    if len(selected_indices) > 0:\n                        selected_logits = token_logits[selected_indices]\n                        weights = F.softmax(selected_logits, dim=0)\n                        \n                        # Store results\n                        num_selected = len(selected_indices)\n                        expert_indices[b, s, :num_selected] = selected_indices\n                        expert_weights[b, s, :num_selected] = weights\n        \n        return expert_indices, expert_weights\n    \n    def _apply_load_balancing_torch(\n        self,\n        expert_indices: torch.Tensor,\n        expert_weights: torch.Tensor,\n        router_logits: torch.Tensor\n    ) -> torch.Tensor:\n        \"\"\"Apply load balancing using PyTorch operations.\"\"\"\n        # Convert to numpy for compatibility with base class logic\n        expert_indices_np = expert_indices.detach().cpu().numpy()\n        expert_weights_np = expert_weights.detach().cpu().numpy()\n        router_logits_np = router_logits.detach().cpu().numpy()\n        \n        # Apply base class load balancing\n        balanced_weights_np = super()._apply_load_balancing(\n            expert_indices_np, expert_weights_np, router_logits_np\n        )\n        \n        # Convert back to PyTorch\n        return torch.from_numpy(balanced_weights_np).to(expert_weights.device)\n    \n    def _compute_routing_stats_torch(\n        self,\n        expert_indices: torch.Tensor,\n        num_experts_per_token: torch.Tensor,\n        complexity_scores: torch.Tensor\n    ) -> Dict[str, Any]:\n        \"\"\"Compute routing statistics for PyTorch tensors.\"\"\"\n        batch_size, seq_len = complexity_scores.shape\n        total_tokens = batch_size * seq_len\n        \n        # Average experts per token\n        avg_experts = float(torch.mean(num_experts_per_token.float()).item())\n        \n        # Expert utilization distribution\n        valid_indices = expert_indices[expert_indices >= 0]\n        expert_counts = torch.bincount(valid_indices, minlength=self.num_experts)\n        total_expert_calls = torch.sum(expert_counts).item()\n        expert_utilization = (expert_counts.float() / total_expert_calls).tolist() if total_expert_calls > 0 else [0.0] * self.num_experts\n        \n        # FLOP reduction estimate\n        static_flops = total_tokens * self.num_experts\n        dynamic_flops = torch.sum(num_experts_per_token).item()\n        flop_reduction = 1.0 - (dynamic_flops / static_flops) if static_flops > 0 else 0.0\n        \n        return {\n            'avg_experts_per_token': avg_experts,\n            'total_expert_calls': int(dynamic_flops),\n            'flop_reduction': flop_reduction,\n            'expert_utilization': expert_utilization,\n            'complexity_stats': {\n                'mean': float(torch.mean(complexity_scores).item()),\n                'std': float(torch.std(complexity_scores).item()),\n                'min': float(torch.min(complexity_scores).item()),\n                'max': float(torch.max(complexity_scores).item())\n            }\n        }\n\n\nclass TorchAdaptiveRouter(TorchDynamicRouter):\n    \"\"\"PyTorch implementation of adaptive router.\"\"\"\n    \n    def __init__(self, adaptation_rate: float = 0.01, **kwargs):\n        super().__init__(**kwargs)\n        self.adaptation_rate = adaptation_rate\n        \n        # Store thresholds as learnable parameters\n        initial_thresholds = torch.linspace(0.0, 1.0, self.max_experts + 1)\n        self.register_parameter(\n            'complexity_thresholds', \n            nn.Parameter(initial_thresholds)\n        )\n        self.performance_history = []\n    \n    def update_thresholds(self, performance_score: float):\n        \"\"\"Update complexity thresholds based on performance feedback.\"\"\"\n        self.performance_history.append(performance_score)\n        \n        if len(self.performance_history) > 1:\n            # Simple gradient-based update\n            perf_gradient = performance_score - self.performance_history[-2]\n            \n            with torch.no_grad():\n                # Adjust thresholds to encourage better performance\n                if perf_gradient > 0:\n                    # Performance improved\n                    self.complexity_thresholds[1:-1] *= (1 + self.adaptation_rate)\n                else:\n                    # Performance degraded\n                    self.complexity_thresholds[1:-1] *= (1 - self.adaptation_rate)\n                \n                # Keep thresholds sorted and bounded\n                self.complexity_thresholds.clamp_(0.0, 1.0)\n                self.complexity_thresholds.data = torch.sort(self.complexity_thresholds)[0]\n    \n    def _compute_expert_counts_torch(self, complexity_scores: torch.Tensor) -> torch.Tensor:\n        \"\"\"Compute expert counts using adaptive thresholds.\"\"\"\n        expert_counts = torch.ones_like(complexity_scores, dtype=torch.long) * self.min_experts\n        \n        for i in range(self.min_experts, self.max_experts):\n            threshold = self.complexity_thresholds[i]\n            mask = complexity_scores >= threshold\n            expert_counts[mask] = i + 1\n        \n        return expert_counts"
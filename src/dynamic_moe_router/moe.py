"""DynamicMoE: full model combining Expert pool + DynamicRouter."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .experts import Expert
from .router import DynamicRouter, RouterOutput


class DynamicMoE(nn.Module):
    """Mixture-of-Experts layer with dynamic routing.

    Each forward pass:
    1. Routes every token to ``top_k`` experts via ``DynamicRouter``.
    2. Each selected expert processes the token.
    3. Outputs are combined as a weighted sum.
    4. An auxiliary load-balancing loss is returned alongside the output.

    Parameters
    ----------
    input_dim:
        Token hidden dimension (input and output).
    expert_hidden_dim:
        Hidden dimension *inside* each FFN expert.
    num_experts:
        Number of parallel experts.
    top_k:
        How many experts process each token.
    capacity_factor:
        Expert capacity headroom (see ``DynamicRouter``).
    aux_loss_coeff:
        Coefficient for the load-balancing auxiliary loss.
    dropout:
        Dropout rate inside each expert.
    """

    def __init__(
        self,
        input_dim: int,
        expert_hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_coeff: float = 1e-2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = DynamicRouter(
            input_dim=input_dim,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
            aux_loss_coeff=aux_loss_coeff,
        )
        self.experts = nn.ModuleList(
            [Expert(input_dim, expert_hidden_dim, input_dim, dropout=dropout)
             for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, RouterOutput]:
        """Forward pass.

        Parameters
        ----------
        x:
            Shape (batch, seq_len, input_dim) or (S, input_dim).

        Returns
        -------
        output:
            Same shape as input  (experts map input_dim → input_dim).
        router_output:
            ``RouterOutput`` containing aux_loss and routing stats.
        """
        original_shape = x.shape
        if x.dim() == 3:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
        else:
            x_flat = x
        S, D = x_flat.shape

        # Route tokens
        router_out = self.router(x_flat)
        indices = router_out.dispatch_indices   # (S, k)
        weights = router_out.dispatch_weights   # (S, k)

        # Compute expert outputs and combine
        output = torch.zeros_like(x_flat)
        for k_slot in range(self.top_k):
            expert_ids = indices[:, k_slot]    # (S,)
            slot_weights = weights[:, k_slot]  # (S,)

            # Group tokens by expert for efficient batched processing
            for expert_idx in range(self.num_experts):
                token_mask = (expert_ids == expert_idx)  # (S,)
                if not token_mask.any():
                    continue
                tokens = x_flat[token_mask]                           # (n, D)
                expert_out = self.experts[expert_idx](tokens)         # (n, D)
                output[token_mask] += slot_weights[token_mask].unsqueeze(-1) * expert_out

        output = output.reshape(original_shape)
        return output, router_out

    def set_capacity_factor(self, factor: float) -> None:
        """Propagate capacity factor change to the router."""
        self.router.set_capacity_factor(factor)

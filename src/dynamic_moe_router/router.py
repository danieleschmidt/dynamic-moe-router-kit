"""Dynamic Router with capacity limits, load balancing, and overflow handling.

Key behaviours
--------------
* Top-k gating  – every token picks its k favourite experts.
* Capacity limits – each expert can serve at most ``capacity_factor × (tokens / num_experts)``
  tokens per batch.  Tokens that exceed an expert's capacity are redirected to their
  next-best available expert (*overflow routing*).
* Load-balancing loss – auxiliary loss that penalises routing imbalance so the network
  learns to distribute tokens evenly across experts over training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class RouterOutput:
    """All outputs produced by one forward pass of DynamicRouter."""
    # Per-token selected expert indices, shape (S, top_k)
    dispatch_indices: torch.Tensor
    # Per-token routing weights (softmax over selected experts), shape (S, top_k)
    dispatch_weights: torch.Tensor
    # Scalar auxiliary load-balancing loss
    aux_loss: torch.Tensor
    # How many token-slots were dropped due to overflow
    overflow_count: int


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class DynamicRouter(nn.Module):
    """Learnable router with top-k gating, capacity limits, and overflow handling.

    Parameters
    ----------
    input_dim:
        Token hidden dimension.
    num_experts:
        Total number of experts.
    top_k:
        Number of experts every token is routed to (default 2).
    capacity_factor:
        Multiplier on the ideal per-expert capacity.  1.0 → each expert
        processes exactly ``tokens / num_experts`` on average; 1.25 gives
        25 % extra headroom before overflow kicks in.
    aux_loss_coeff:
        Weight for the load-balancing auxiliary loss.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_coeff: float = 1e-2,
    ):
        super().__init__()
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) must be ≤ num_experts ({num_experts})")

        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_coeff = aux_loss_coeff

        # Learned gate: maps every token to a logit per expert
        self.gate = nn.Linear(input_dim, num_experts, bias=False)

    # ------------------------------------------------------------------
    # Core routing
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """Route tokens.

        Parameters
        ----------
        x:
            Token representations, shape (batch, seq_len, input_dim)
            *or* (S, input_dim) where S = batch × seq_len.

        Returns
        -------
        RouterOutput
        """
        # Flatten to (S, D) for simplicity
        if x.dim() == 3:
            S = x.shape[0] * x.shape[1]
            x_flat = x.reshape(S, x.shape[2])
        else:
            S = x.shape[0]
            x_flat = x

        # ----- Gate logits ------------------------------------------------
        logits = self.gate(x_flat)        # (S, E)
        probs = F.softmax(logits, dim=-1)  # (S, E)

        # ----- Top-k selection --------------------------------------------
        # Sort ALL experts by prob so we can do overflow fallbacks cheaply
        sorted_weights, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        # (S, E) — all experts ranked best→worst per token

        topk_indices = sorted_indices[:, :self.top_k]   # (S, k)
        topk_weights = sorted_weights[:, :self.top_k]   # (S, k)

        # Re-normalise so weights over selected k experts sum to 1
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # ----- Capacity enforcement + overflow ----------------------------
        # capacity = ceil(capacity_factor * S * top_k / num_experts)
        # This is total token-slot budget per expert across all k slots.
        capacity = max(self.top_k, math.ceil(self.capacity_factor * S * self.top_k / self.num_experts))
        dispatch_indices, dispatch_weights, overflow_count = self._enforce_capacity(
            sorted_indices, topk_weights, capacity
        )

        # ----- Auxiliary load-balancing loss ------------------------------
        aux_loss = self._load_balance_loss(probs, topk_indices)

        return RouterOutput(
            dispatch_indices=dispatch_indices,
            dispatch_weights=dispatch_weights,
            aux_loss=aux_loss,
            overflow_count=overflow_count,
        )

    # ------------------------------------------------------------------
    # Capacity / overflow
    # ------------------------------------------------------------------

    def _enforce_capacity(
        self,
        sorted_indices: torch.Tensor,  # (S, E) — all experts ranked best→worst
        topk_weights: torch.Tensor,    # (S, k) — weights for the initial top-k
        capacity: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Redirect overflowing tokens to their next-best available expert.

        Algorithm
        ---------
        1. Process tokens in descending order of their top-1 score (highest
           confidence first), so the most certain assignments get priority.
        2. For each token, iterate through its ranked expert list until we
           find ``top_k`` experts that still have capacity.
        3. If no expert with capacity is found for a slot, that slot's
           weight is zeroed (token is dropped for that slot).
        """
        S, E = sorted_indices.shape
        k = self.top_k
        device = sorted_indices.device

        expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        final_indices = torch.zeros(S, k, dtype=torch.long, device=device)
        final_weights = torch.zeros(S, k, device=device)

        # Sort tokens by their top-1 routing weight (best-confidence first)
        top1_weight = topk_weights[:, 0]  # (S,)
        token_order = top1_weight.argsort(descending=True).tolist()

        overflow_count = 0

        for token_idx in token_order:
            assigned = 0          # slots filled for this token
            raw_weights = []      # accumulate raw probs for assigned experts

            for rank in range(E):
                if assigned == k:
                    break
                expert = sorted_indices[token_idx, rank].item()
                if expert_counts[expert] < capacity:
                    expert_counts[expert] += 1
                    final_indices[token_idx, assigned] = expert
                    # Use the original sorted weight as the raw score
                    # (we'll renormalise below)
                    mag = topk_weights[token_idx, assigned].detach().item() if assigned < k else 0.0
                    raw_weights.append(mag)
                    assigned += 1
                else:
                    if assigned < k:
                        overflow_count += 1  # this rank slot was overflowed

            # Fill weights using the magnitudes we collected (detached floats)
            for slot in range(assigned):
                final_weights[token_idx, slot] = raw_weights[slot]

        # Re-normalise per-token weights in case some slots were dropped
        weight_sum = final_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        final_weights = final_weights / weight_sum

        return final_indices, final_weights, overflow_count

    # ------------------------------------------------------------------
    # Load-balancing loss  (Switch Transformer style)
    # ------------------------------------------------------------------

    def _load_balance_loss(
        self,
        probs: torch.Tensor,        # (S, E)  — full softmax probs
        topk_indices: torch.Tensor  # (S, k)
    ) -> torch.Tensor:
        """Compute auxiliary loss that encourages uniform expert utilisation.

        L_aux = num_experts × Σ_e  f_e × p_e

        where
          f_e = fraction of tokens dispatched to expert e  (not differentiable)
          p_e = mean routing probability assigned to expert e  (differentiable)

        Minimising L_aux pushes the network to spread tokens evenly.
        """
        S, E = probs.shape

        # f_e — routing fraction (straight-through, no gradient)
        one_hot = torch.zeros(S, E, device=probs.device)
        for k_idx in range(self.top_k):
            one_hot.scatter_add_(
                1,
                topk_indices[:, k_idx:k_idx + 1],
                torch.ones(S, 1, device=probs.device),
            )
        f = one_hot.mean(dim=0)  # (E,)

        # p_e — mean gate probability (has gradient)
        p = probs.mean(dim=0)  # (E,)

        aux_loss = self.aux_loss_coeff * E * (f.detach() * p).sum()
        return aux_loss

    # ------------------------------------------------------------------
    # Utility: dynamic capacity adjustment
    # ------------------------------------------------------------------

    def set_capacity_factor(self, factor: float) -> None:
        """Adjust capacity factor at runtime (e.g. based on observed overflow)."""
        if factor < 1.0:
            raise ValueError("capacity_factor must be ≥ 1.0")
        self.capacity_factor = factor

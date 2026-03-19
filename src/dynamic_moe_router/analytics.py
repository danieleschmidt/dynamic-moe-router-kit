"""RouterAnalytics: tracks per-expert load, utilisation, and overflow over training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import torch


@dataclass
class StepStats:
    """Stats captured at a single training step."""
    step: int
    # Fraction of tokens each expert received  (E,)
    expert_load: List[float]
    # Number of token-slots dropped due to capacity overflow
    overflow_count: int
    # Ideal uniform load per expert (= top_k / num_experts)
    ideal_load: float

    @property
    def load_imbalance(self) -> float:
        """Max deviation from ideal load (lower is better, 0 = perfectly balanced)."""
        return max(abs(l - self.ideal_load) for l in self.expert_load)

    @property
    def utilisation(self) -> List[float]:
        """Per-expert utilisation relative to ideal  (1.0 = ideal)."""
        ideal = self.ideal_load or 1e-9
        return [l / ideal for l in self.expert_load]


class RouterAnalytics:
    """Lightweight analytics collector attached to a DynamicRouter.

    Usage::

        analytics = RouterAnalytics(num_experts=8, top_k=2, window=100)
        # Inside training loop after router.forward():
        analytics.record(step, router_output)
        # Print a summary every N steps:
        analytics.summary()

    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        window: int = 200,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.ideal_load = top_k / num_experts   # fraction per expert if balanced

        # Rolling window of StepStats
        self._history: Deque[StepStats] = deque(maxlen=window)

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def record(self, step: int, router_output) -> StepStats:
        """Ingest one RouterOutput and store stats.

        Parameters
        ----------
        step:
            Current training step number.
        router_output:
            A ``RouterOutput`` as returned by ``DynamicRouter.forward``.
        """
        indices = router_output.dispatch_indices   # (S, k)
        weights = router_output.dispatch_weights   # (S, k)
        S = indices.shape[0]

        # Compute per-expert token fraction (weighted)
        load = torch.zeros(self.num_experts, device=indices.device)
        for k_slot in range(indices.shape[1]):
            weight_col = weights[:, k_slot]             # (S,)
            load.scatter_add_(0, indices[:, k_slot], weight_col)

        # Normalise so they sum to top_k (one load-unit per assigned slot)
        total = load.sum().item()
        if total > 1e-9:
            load_frac = (load / total * self.top_k).tolist()
        else:
            load_frac = [0.0] * self.num_experts

        stats = StepStats(
            step=step,
            expert_load=load_frac,
            overflow_count=router_output.overflow_count,
            ideal_load=self.ideal_load * self.top_k,   # scale to match load_frac
        )
        self._history.append(stats)
        return stats

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def summary(self, last_n: Optional[int] = None) -> Dict:
        """Return a dict summarising recent statistics.

        Parameters
        ----------
        last_n:
            Summarise only the last N steps (default: all in window).
        """
        history = list(self._history)
        if last_n is not None:
            history = history[-last_n:]
        if not history:
            return {"error": "no data recorded yet"}

        num_steps = len(history)
        E = self.num_experts

        # Aggregate per-expert load across recorded steps
        avg_load = [
            sum(s.expert_load[e] for s in history) / num_steps
            for e in range(E)
        ]
        ideal = history[-1].ideal_load

        # Utilisation: avg_load / ideal
        utilisation = [l / ideal if ideal > 0 else 0.0 for l in avg_load]

        # Overflow
        total_overflow = sum(s.overflow_count for s in history)
        avg_overflow = total_overflow / num_steps

        # Imbalance: std dev of avg_load (0 = perfectly balanced)
        mean_load = sum(avg_load) / E
        load_std = (sum((l - mean_load) ** 2 for l in avg_load) / E) ** 0.5

        return {
            "steps_recorded": num_steps,
            "per_expert_avg_load": [round(l, 4) for l in avg_load],
            "per_expert_utilisation": [round(u, 4) for u in utilisation],
            "load_std": round(load_std, 4),
            "avg_overflow_per_step": round(avg_overflow, 2),
            "most_loaded_expert": int(avg_load.index(max(avg_load))),
            "least_loaded_expert": int(avg_load.index(min(avg_load))),
        }

    def print_summary(self, last_n: Optional[int] = None) -> None:
        """Pretty-print a utilisation table."""
        s = self.summary(last_n)
        if "error" in s:
            print(s["error"])
            return

        print(f"\n{'Expert':<8} {'Avg Load':>10} {'Utilisation':>13}")
        print("-" * 34)
        for e, (load, util) in enumerate(
            zip(s["per_expert_avg_load"], s["per_expert_utilisation"])
        ):
            bar = "█" * int(util * 10)
            print(f"  E{e:<5d} {load:>10.4f}  {util:>6.2f}x  {bar}")
        print("-" * 34)
        print(
            f"  Load std:  {s['load_std']:.4f}  "
            f"(0 = perfect)  |  Avg overflow/step: {s['avg_overflow_per_step']:.1f}"
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def load_history(self) -> List[List[float]]:
        """Return list of per-expert load vectors over recorded steps."""
        return [s.expert_load for s in self._history]

    def overflow_history(self) -> List[int]:
        """Return list of overflow counts over recorded steps."""
        return [s.overflow_count for s in self._history]

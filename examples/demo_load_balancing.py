"""Demo: train a DynamicMoE layer and watch expert utilisation improve.

Task
----
Regression on random data.  The MoE layer sits between two linear
projections.  We train with and without the auxiliary load-balancing
loss and compare expert utilisation at the end.

Run:
    python examples/demo_load_balancing.py
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dynamic_moe_router import DynamicMoE, RouterAnalytics


# ------------------------------------------------------------------ #
# Toy model: Linear → MoE → Linear                                    #
# ------------------------------------------------------------------ #

class ToyModel(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, top_k: int, aux_coeff: float):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, input_dim)
        self.moe = DynamicMoE(
            input_dim=input_dim,
            expert_hidden_dim=input_dim * 4,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=1.5,
            aux_loss_coeff=aux_coeff,
        )
        self.proj_out = nn.Linear(input_dim, 1)

    def forward(self, x):
        h = torch.relu(self.proj_in(x))
        h, router_out = self.moe(h)
        pred = self.proj_out(h).squeeze(-1)
        return pred, router_out


# ------------------------------------------------------------------ #
# Training loop                                                        #
# ------------------------------------------------------------------ #

def train(label: str, aux_coeff: float, steps: int = 400, seed: int = 42):
    torch.manual_seed(seed)
    random.seed(seed)

    INPUT_DIM = 32
    NUM_EXPERTS = 8
    TOP_K = 2
    BATCH = 64
    SEQ_LEN = 16

    model = ToyModel(INPUT_DIM, NUM_EXPERTS, TOP_K, aux_coeff)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    analytics = RouterAnalytics(num_experts=NUM_EXPERTS, top_k=TOP_K, window=steps)

    task_losses = []

    for step in range(steps):
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        target = torch.randn(BATCH, SEQ_LEN)

        pred, router_out = model(x)
        task_loss = nn.functional.mse_loss(pred, target)
        total_loss = task_loss + router_out.aux_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        analytics.record(step, router_out)
        task_losses.append(task_loss.item())

        if (step + 1) % 100 == 0:
            recent_loss = sum(task_losses[-50:]) / 50
            print(f"  [{label}] step {step+1:4d}  task_loss={recent_loss:.4f}  "
                  f"aux_loss={router_out.aux_loss.item():.5f}  "
                  f"overflow={router_out.overflow_count}")

    return analytics


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    print("=" * 60)
    print("  dynamic-moe-router-kit  |  Load Balancing Demo")
    print("=" * 60)

    print("\n--- Training WITHOUT load-balancing (aux_coeff=0.0) ---")
    analytics_no_lb = train("no-lb", aux_coeff=0.0)

    print("\n--- Training WITH load-balancing (aux_coeff=0.01) ---")
    analytics_lb = train("lb  ", aux_coeff=0.01)

    # ------ Compare utilisation ---------
    print("\n" + "=" * 60)
    print("  Final expert utilisation (last 100 steps)")
    print("=" * 60)

    print("\n[Without load balancing]")
    analytics_no_lb.print_summary(last_n=100)

    print("\n[With load balancing]")
    analytics_lb.print_summary(last_n=100)

    s_no = analytics_no_lb.summary(last_n=100)
    s_lb = analytics_lb.summary(last_n=100)
    print("\n  Load std (lower = more balanced):")
    print(f"    no-lb : {s_no['load_std']:.4f}")
    print(f"    with-lb : {s_lb['load_std']:.4f}")
    improved = s_lb['load_std'] < s_no['load_std']
    print(f"\n  ✅ Load balancing improved utilisation: {improved}")


if __name__ == "__main__":
    main()

# dynamic-moe-router-kit

A PyTorch drop-in Mixture-of-Experts layer with **dynamic routing** that adapts
to input load and enforces expert capacity constraints.

## Features

| Feature | Details |
|---------|---------|
| **Top-k gating** | Each token is routed to exactly `k` experts via learned gating weights |
| **Capacity limits** | No expert is overloaded — overflow tokens are redirected to their next-best available expert |
| **Load-balancing loss** | Auxiliary loss (Switch Transformer style) drives uniform expert utilisation |
| **RouterAnalytics** | Tracks per-expert load, utilisation, and overflow rate over training |

## Quick start

```python
import torch
from dynamic_moe_router import DynamicMoE, RouterAnalytics

# Drop-in MoE layer: same input/output dim as a standard FFN
moe = DynamicMoE(
    input_dim=512,
    expert_hidden_dim=2048,
    num_experts=8,
    top_k=2,
    capacity_factor=1.25,   # 25% headroom per expert
    aux_loss_coeff=1e-2,    # load-balancing loss weight
)

analytics = RouterAnalytics(num_experts=8, top_k=2)

x = torch.randn(4, 32, 512)   # (batch, seq_len, dim)
output, router_out = moe(x)

# Combine task loss + auxiliary load-balancing loss
task_loss = my_loss_fn(output, target)
total_loss = task_loss + router_out.aux_loss

analytics.record(step=0, router_output=router_out)
analytics.print_summary()
```

## Architecture

```
DynamicMoE
├── DynamicRouter        — learned gate (Linear) → top-k + capacity enforcement
│   ├── top-k gating     — softmax over E experts, pick best k
│   ├── capacity limits  — max capacity_factor × S/E tokens per expert
│   ├── overflow routing — overloaded tokens go to next-best expert
│   └── aux_loss         — Switch-Transformer load-balancing loss
└── Expert × N           — independent FFN: Linear → GELU → Linear
```

### DynamicRouter

```python
from dynamic_moe_router import DynamicRouter

router = DynamicRouter(
    input_dim=512,
    num_experts=8,
    top_k=2,
    capacity_factor=1.25,
    aux_loss_coeff=1e-2,
)

from dynamic_moe_router import RouterOutput
router_out: RouterOutput = router(x)

router_out.dispatch_indices  # (S, k)  — expert assignments
router_out.dispatch_weights  # (S, k)  — combination weights
router_out.aux_loss          # scalar  — backprop through this
router_out.overflow_count    # int     — tokens rerouted this step
```

### RouterAnalytics

```python
from dynamic_moe_router import RouterAnalytics

analytics = RouterAnalytics(num_experts=8, top_k=2, window=200)

# Inside training loop:
analytics.record(step, router_out)

# Print a utilisation table:
analytics.print_summary(last_n=100)

# Programmatic access:
summary = analytics.summary()
# {
#   "per_expert_avg_load": [...],
#   "per_expert_utilisation": [...],
#   "load_std": 0.0017,
#   "avg_overflow_per_step": 0.0,
#   ...
# }
```

## Demo

```
python examples/demo_load_balancing.py
```

Sample output (8 experts, top-k=2, 400 training steps):

```
[Without load balancing]
  E0         0.2517    0.50x  █████
  E3         0.3911    0.78x  ███████
  E2         0.0728    0.15x  █
  Load std:  0.1042  |  Avg overflow/step: 73.0

[With load balancing]
  All experts: ~0.50x utilisation
  Load std:  0.0017  |  Avg overflow/step: 0.0

✅ Load balancing improved utilisation: True
```

## Install

```bash
pip install -e .                          # from source
```

Requirements: Python ≥ 3.8, PyTorch ≥ 1.9.

## Tests

```bash
pytest tests/
```

30 tests, 92% coverage.

## License

MIT

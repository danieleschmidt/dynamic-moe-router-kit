# Getting Started Guide

## Overview

This guide will help you get up and running with dynamic-moe-router-kit in just a few minutes.

## Prerequisites

- Python 3.8 or higher
- One of: PyTorch, JAX/Flax, or TensorFlow
- Basic familiarity with MoE concepts

## Quick Installation

```bash
# Install with your preferred backend
pip install dynamic-moe-router-kit[torch]   # PyTorch
pip install dynamic-moe-router-kit[jax]     # JAX/Flax  
pip install dynamic-moe-router-kit[tf]      # TensorFlow
```

## 5-Minute Example

### 1. Import and Setup

```python
import torch
from dynamic_moe_router import DynamicRouter, MoELayer

# Create a dynamic router
router = DynamicRouter(
    input_dim=768,
    num_experts=8,
    min_experts=1,
    max_experts=4
)
```

### 2. Create MoE Layer

```python
# Define your expert function
def create_expert():
    return torch.nn.Sequential(
        torch.nn.Linear(768, 3072),
        torch.nn.ReLU(),
        torch.nn.Linear(3072, 768)
    )

# Create the MoE layer
moe_layer = MoELayer(
    router=router,
    expert_fn=create_expert,
    num_experts=8
)
```

### 3. Use It

```python
# Sample input: [batch_size, sequence_length, hidden_dim]
inputs = torch.randn(4, 64, 768)

# Forward pass with routing info
outputs, routing_info = moe_layer(inputs, return_router_logits=True)

print(f"Output shape: {outputs.shape}")
print(f"Average experts per token: {routing_info['avg_experts_per_token']:.2f}")
print(f"FLOP reduction: {routing_info['flop_reduction']:.1%}")
```

## Next Steps

- [Integration Guide](integration.md) - Add to existing models
- [Complexity Estimators](complexity-estimators.md) - Customize routing behavior
- [Performance Tuning](performance-tuning.md) - Optimize for your use case
- [API Reference](../../api/index.md) - Complete API documentation

## Common Issues

### Installation Problems

**Error**: "No module named 'dynamic_moe_router'"
**Solution**: Make sure you installed with the correct backend: `pip install dynamic-moe-router-kit[torch]`

**Error**: CUDA out of memory
**Solution**: Reduce batch size or use gradient checkpointing

### Runtime Issues

**Error**: "ComplexityEstimator requires gradients"
**Solution**: Ensure model is in training mode: `model.train()`

**Error**: Expert load imbalance
**Solution**: Adjust load balancing parameters or use auxiliary loss

## Getting Help

- 📖 [Documentation](https://dynamic-moe-router.readthedocs.io)
- 💬 [GitHub Discussions](https://github.com/yourusername/dynamic-moe-router-kit/discussions)
- 🐛 [Issue Tracker](https://github.com/yourusername/dynamic-moe-router-kit/issues)
- 📧 [Email Support](mailto:support@example.com)
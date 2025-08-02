# ADR-001: Multi-Backend Architecture

## Status

Accepted

## Context

The dynamic MoE routing system needs to support multiple deep learning frameworks (PyTorch, JAX/Flax, TensorFlow) to reach the widest possible audience. Each framework has different APIs, computation graphs, and optimization strategies.

## Decision

Implement a multi-backend architecture with:
- Shared core abstractions for routing logic
- Framework-specific implementations in separate modules
- Common interfaces for complexity estimators and routers
- Backend-agnostic high-level API for users

Structure:
```
src/dynamic_moe_router/
├── core/           # Framework-agnostic logic
├── torch/          # PyTorch implementations  
├── jax/            # JAX/Flax implementations
└── tf/             # TensorFlow implementations
```

## Consequences

### Positive
- Wider adoption across ML community
- Framework-specific optimizations possible
- Users can stay in their preferred ecosystem
- Code reuse through shared abstractions

### Negative
- Increased maintenance burden
- Need expertise in multiple frameworks
- Testing complexity across backends
- Potential API inconsistencies

### Neutral
- More complex project structure
- Need clear backend selection mechanism
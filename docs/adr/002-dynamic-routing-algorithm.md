# ADR-002: Dynamic Routing Algorithm

## Status

Accepted

## Context

Traditional MoE models use static top-k routing, always activating the same number of experts. Research shows that input complexity varies significantly, suggesting adaptive expert selection could improve efficiency without sacrificing quality.

## Decision

Implement dynamic routing that:
1. Estimates input complexity per token
2. Maps complexity to expert count (k) between min/max bounds
3. Selects top-k experts based on router scores
4. Maintains load balancing across experts

Algorithm:
```python
complexity = estimate_complexity(inputs)
k = min_experts + (max_experts - min_experts) * complexity
expert_indices = router_logits.topk(k.round().int())
```

## Consequences

### Positive
- Significant FLOP reduction (30-40% observed)
- Better resource utilization
- Maintained or improved quality
- Configurable complexity/performance trade-offs

### Negative
- Additional complexity estimation overhead
- Variable computation makes batching harder
- Load balancing becomes more complex
- Requires tuning of min/max expert bounds

### Neutral
- Need robust complexity estimation methods
- Requires careful evaluation methodology
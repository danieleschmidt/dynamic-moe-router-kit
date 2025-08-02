# ADR-003: Complexity Estimation Strategy

## Status

Accepted

## Context

Dynamic routing requires accurate complexity estimation to determine appropriate expert count. The estimator must be fast, differentiable, and correlate well with actual computation requirements.

## Decision

Provide multiple complexity estimation strategies:

1. **Gradient Norm**: Uses gradient magnitude as complexity proxy
2. **Attention Entropy**: Leverages attention distribution entropy
3. **Perplexity Proxy**: Estimates model confidence via hidden state distributions
4. **Custom**: Extensible interface for user-defined estimators

Default to gradient norm for speed and general applicability.

Interface:
```python
class ComplexityEstimator:
    def estimate(self, hidden_states, **kwargs) -> Tensor:
        # Returns complexity scores in [0, 1] range
        pass
```

## Consequences

### Positive
- Multiple estimation strategies available
- Extensible for domain-specific needs
- Fast default option (gradient norm)
- Normalized output simplifies routing logic

### Negative
- Each estimator has different compute costs
- Gradient norm requires backward pass
- Need validation of complexity-quality correlation
- Hyperparameter tuning per estimator

### Neutral
- Strategy selection impacts overall performance
- May need ensemble estimation for best results
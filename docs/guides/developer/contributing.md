# Developer Contributing Guide

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/yourusername/dynamic-moe-router-kit.git
cd dynamic-moe-router-kit
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install development dependencies
pip install -e ".[dev,torch,jax,tf]"

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Installation

```bash
# Run tests
pytest

# Check code style
make lint

# Run type checking
make type-check
```

## Project Structure

```
src/dynamic_moe_router/
├── core/              # Framework-agnostic logic
│   ├── router.py      # Base router classes
│   ├── estimators.py  # Complexity estimators
│   └── utils.py       # Shared utilities
├── torch/             # PyTorch backend
├── jax/               # JAX/Flax backend
├── tf/                # TensorFlow backend
└── integrations/      # Framework integrations
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/my-awesome-feature
```

### 2. Make Changes

- Follow existing code style and patterns
- Add tests for new functionality
- Update documentation as needed
- Keep commits atomic and well-described

### 3. Run Quality Checks

```bash
# Format code
make format

# Run all tests
make test

# Check coverage
make coverage

# Run benchmarks (if applicable)
make benchmark
```

### 4. Submit Pull Request

1. Push your branch to your fork
2. Create a pull request with descriptive title
3. Fill out the PR template completely
4. Request review from maintainers

## Code Style Guidelines

### Python Style

We follow [Black](https://black.readthedocs.io/) formatting with these conventions:

- Line length: 88 characters
- Use type hints for all public APIs
- Docstrings follow Google style
- Import sorting via isort

### Example Function

```python
def estimate_complexity(
    hidden_states: torch.Tensor,
    attention_weights: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Estimate input complexity for dynamic routing.
    
    Args:
        hidden_states: Input hidden states [batch, seq, dim].
        attention_weights: Optional attention weights for estimation.
        **kwargs: Additional backend-specific parameters.
        
    Returns:
        Complexity scores in range [0, 1] with shape [batch, seq].
        
    Raises:
        ValueError: If hidden_states has wrong dimensions.
    """
    if hidden_states.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {hidden_states.ndim}D")
    
    # Implementation here...
    return complexity_scores
```

## Testing Guidelines

### Test Structure

```python
import pytest
import torch
from dynamic_moe_router import DynamicRouter

class TestDynamicRouter:
    """Test suite for DynamicRouter class."""
    
    @pytest.fixture
    def router(self):
        """Create router for testing."""
        return DynamicRouter(
            input_dim=768,
            num_experts=8,
            min_experts=1,
            max_experts=4
        )
    
    def test_forward_pass(self, router):
        """Test basic forward pass."""
        inputs = torch.randn(2, 10, 768)
        indices, weights = router(inputs)
        
        assert indices.shape == (2, 10, 4)  # max_experts
        assert weights.shape == (2, 10, 4)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 10))
```

### Test Categories

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test component interactions
3. **Backend Tests**: Framework-specific behavior
4. **Performance Tests**: Benchmark critical paths

### Running Tests

```bash
# All tests
pytest

# Specific backend
pytest tests/torch/

# With coverage
pytest --cov=dynamic_moe_router

# Performance tests
pytest tests/performance/ --benchmark-only
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def my_function(param1: str, param2: int = 0) -> bool:
    """One-line summary of the function.
    
    Longer description if needed, explaining the purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of first parameter.
        param2: Description of second parameter with default.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: When param1 is empty.
        RuntimeError: When computation fails.
        
    Example:
        Basic usage example:
        
        >>> result = my_function("hello", 42)
        >>> print(result)
        True
    """
```

### Building Docs

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html

# Serve locally
python -m http.server 8000 --directory _build/html
```

## Performance Considerations

### Profiling

```python
from dynamic_moe_router import FLOPProfiler

# Profile your changes
profiler = FLOPProfiler()
with profiler:
    outputs = model(inputs)

print(profiler.summary())
```

### Benchmarking

```python
# Add benchmarks for new features
def test_routing_performance(benchmark):
    router = DynamicRouter(...)
    inputs = torch.randn(32, 128, 768)
    
    result = benchmark(router, inputs)
    assert result.shape == inputs.shape
```

## Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):
- MAJOR: Incompatible API changes
- MINOR: New functionality, backwards compatible
- PATCH: Bug fixes, backwards compatible

### Changelog

Update `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
## [0.2.0] - 2024-02-01

### Added
- New complexity estimator: attention entropy
- JAX backend performance optimizations

### Changed  
- Router API now requires explicit min/max experts

### Fixed
- Memory leak in TensorFlow backend
```

## Getting Help

### Development Questions

1. Check existing [GitHub Discussions](https://github.com/yourusername/dynamic-moe-router-kit/discussions)
2. Search [closed issues](https://github.com/yourusername/dynamic-moe-router-kit/issues?q=is%3Aissue+is%3Aclosed)
3. Ask in the `#development` channel on Discord
4. Tag maintainers in your PR for specific questions

### Debugging Tips

- Use `logging.DEBUG` for detailed router behavior
- Enable `torch.autograd.detect_anomaly()` for gradient issues
- Profile with `torch.profiler` for performance bottlenecks
- Test with different batch sizes and sequence lengths

## Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes for significant contributions
- Annual contributor spotlight blog posts

Thank you for contributing to dynamic-moe-router-kit! 🚀
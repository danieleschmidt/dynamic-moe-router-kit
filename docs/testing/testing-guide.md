# Testing Guide

## Overview

This guide covers the comprehensive testing strategy for dynamic-moe-router-kit, including unit tests, integration tests, property-based tests, and performance benchmarks.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── fixtures/                # Test data and model fixtures
│   ├── models.py           # Model configurations for testing
│   └── data.py             # Synthetic data generators
├── unit/                   # Fast, isolated unit tests
├── integration/            # Multi-component integration tests
├── e2e/                    # End-to-end workflow tests
├── performance/            # Performance and benchmark tests
└── property/               # Property-based tests
```

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Fast unit tests only
pytest -m integration   # Integration tests
pytest -m performance   # Performance benchmarks
pytest -m slow          # Long-running tests

# Run tests for specific backends
pytest -m torch         # PyTorch tests only
pytest -m jax           # JAX/Flax tests only
pytest -m tf            # TensorFlow tests only
```

### Advanced Options

```bash
# Run with coverage
pytest --cov=dynamic_moe_router --cov-report=html

# Parallel execution
pytest -n auto

# Stop on first failure
pytest -x

# Verbose output
pytest -v -s

# Run specific test file
pytest tests/unit/test_router.py

# Run specific test function
pytest tests/unit/test_router.py::test_basic_routing
```

## Test Categories

### Unit Tests (`tests/unit/`)

Fast, isolated tests for individual components:

- **Router components**: Dynamic routing logic, complexity estimators
- **Backend implementations**: Framework-specific code
- **Utilities**: Helper functions and data processing
- **Error handling**: Edge cases and error conditions

**Characteristics:**
- Fast execution (< 1 second per test)
- No external dependencies
- High code coverage
- Deterministic results

### Integration Tests (`tests/integration/`)

Tests for component interactions:

- **Multi-backend compatibility**: Ensure consistent behavior
- **Model integration**: Integration with existing architectures
- **Data flow**: End-to-end data processing
- **Configuration**: Various parameter combinations

**Characteristics:**
- Moderate execution time (1-10 seconds)
- May use multiple backends
- Focus on interfaces and contracts

### End-to-End Tests (`tests/e2e/`)

Complete workflow tests:

- **Training pipelines**: Full training loops with dynamic routing
- **Inference pipelines**: Complete inference workflows
- **Performance validation**: Real-world performance characteristics
- **Error recovery**: System behavior under failure conditions

**Characteristics:**
- Slower execution (10+ seconds)
- Realistic data and scenarios
- Full system validation

### Property-Based Tests (`tests/property/`)

Automated testing using Hypothesis:

- **Routing invariants**: Properties that must always hold
- **Numerical stability**: Behavior under various inputs
- **Boundary conditions**: Edge cases and limits
- **Mathematical properties**: Correctness of algorithms

**Characteristics:**
- Generates many test cases automatically
- Finds edge cases humans might miss
- Verifies fundamental properties

### Performance Tests (`tests/performance/`)

Benchmarks and performance validation:

- **FLOP counting**: Computational efficiency
- **Memory usage**: Memory consumption patterns
- **Latency**: Inference and training speed
- **Scalability**: Performance under load

**Characteristics:**
- Quantitative measurements
- Baseline comparisons
- Performance regression detection

## Test Fixtures

### Model Configurations

```python
# Small model for fast tests
small_model_config = ModelConfig(
    num_experts=4,
    hidden_dim=128,
    min_experts=1,
    max_experts=2
)

# Medium model for integration tests
medium_model_config = ModelConfig(
    num_experts=8, 
    hidden_dim=768,
    min_experts=1,
    max_experts=4
)
```

### Data Generators

```python
# Synthetic text data
synthetic_data = synthetic_text_data(
    batch_size=4,
    seq_len=128,
    vocab_size=1000
)

# Complexity patterns
patterns = complexity_patterns(batch_size=4, seq_len=128)
# Returns: uniform, increasing, spiky, sinusoidal, bimodal
```

## Backend-Specific Testing

### PyTorch Tests

```python
@pytest.mark.torch
def test_pytorch_router():
    torch = pytest.importorskip("torch")
    # PyTorch-specific test implementation
```

### JAX Tests

```python
@pytest.mark.jax  
def test_jax_router():
    jax = pytest.importorskip("jax")
    # JAX-specific test implementation
```

### TensorFlow Tests

```python
@pytest.mark.tf
def test_tensorflow_router():
    tf = pytest.importorskip("tensorflow")
    # TensorFlow-specific test implementation
```

## Property-Based Testing

Using Hypothesis for automated test case generation:

```python
from hypothesis import given, strategies as st

@given(
    batch_size=st.integers(min_value=1, max_value=16),
    seq_len=st.integers(min_value=1, max_value=256),
    complexity=st.lists(
        st.floats(min_value=0.0, max_value=1.0),
        min_size=1, max_size=1000
    )
)
def test_routing_bounds(batch_size, seq_len, complexity):
    # Test that routing respects bounds for any valid input
    pass
```

## Performance Testing

### Benchmark Structure

```python
def test_flop_reduction(benchmark):
    def routing_operation():
        # Implementation of routing
        return result
    
    result = benchmark(routing_operation)
    
    # Verify performance characteristics
    assert result.flop_reduction > 0.2  # At least 20% reduction
```

### Memory Profiling

```python
import tracemalloc

def test_memory_usage():
    tracemalloc.start()
    
    # Run test operation
    result = run_dynamic_routing()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Verify memory constraints
    assert peak < 1024 * 1024 * 100  # < 100MB
```

## Test Configuration

### Markers

Configure test markers in `pytest.ini`:

```ini
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests
    performance: Performance benchmarks  
    slow: Long-running tests
    torch: PyTorch backend tests
    jax: JAX/Flax backend tests
    tf: TensorFlow backend tests
    gpu: GPU-required tests
    smoke: Basic functionality tests
```

### Coverage Configuration

Coverage settings in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "--cov=dynamic_moe_router --cov-report=html --cov-report=term"

[tool.coverage.run]
source = ["src/dynamic_moe_router"]
omit = [
    "*/tests/*",
    "*/venv/*", 
    "setup.py"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError"
]
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
        
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Install dependencies
      run: |
        pip install -e ".[dev,torch,jax,tf]"
        
    - name: Run tests
      run: |
        pytest -m "not slow" --cov=dynamic_moe_router
        
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Best Practices

### Writing Good Tests

1. **Test one thing at a time**: Each test should verify a single behavior
2. **Use descriptive names**: Test names should explain what is being tested
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Use fixtures**: Reuse common test data and configurations
5. **Mock external dependencies**: Keep tests isolated and fast

### Test Data Management

1. **Use synthetic data**: Generate test data programmatically
2. **Keep tests deterministic**: Use fixed random seeds
3. **Test edge cases**: Include boundary conditions and error scenarios
4. **Validate assumptions**: Use property-based testing for invariants

### Performance Testing

1. **Establish baselines**: Compare against known performance metrics
2. **Test different scales**: Verify scalability characteristics
3. **Monitor regressions**: Track performance over time
4. **Profile bottlenecks**: Identify performance-critical paths

## Debugging Tests

### Common Issues

1. **Flaky tests**: Use fixed random seeds, avoid timing dependencies
2. **Slow tests**: Profile and optimize, move to appropriate category
3. **Backend unavailability**: Use `pytest.importorskip()` for optional dependencies
4. **Memory leaks**: Monitor memory usage in long-running tests

### Debugging Tools

```bash
# Run with debugger
pytest --pdb

# Capture output
pytest -s

# Show local variables on failure
pytest --tb=long

# Profile test performance
pytest --profile
```

## Contributing Tests

When adding new features:

1. **Write tests first**: Follow TDD principles
2. **Maintain coverage**: Aim for >90% code coverage
3. **Add property tests**: Include property-based tests for new algorithms
4. **Update benchmarks**: Add performance tests for new features
5. **Document test cases**: Explain complex test scenarios

For more information, see the [Contributing Guide](../guides/developer/contributing.md).
# Development Guide

## Quick Start

1. **Clone and Setup**:
   ```bash
   git clone https://github.com/yourusername/dynamic-moe-router-kit.git
   cd dynamic-moe-router-kit
   pip install -e ".[dev]"
   ```

2. **Run Tests**:
   ```bash
   pytest
   ```

3. **Code Quality**:
   ```bash
   black .
   isort .
   ruff check .
   mypy .
   ```

## Project Structure

```
dynamic-moe-router-kit/
├── src/dynamic_moe_router/    # Main package
├── tests/                     # Test suite
├── docs/                      # Documentation
├── examples/                  # Usage examples
└── benchmarks/               # Performance benchmarks
```

## Development Workflow

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement with tests and documentation
3. Run full test suite: `make test`
4. Create pull request

### Backend Support

The project supports three backends:
- **PyTorch**: Primary backend (`src/dynamic_moe_router/torch/`)
- **JAX/Flax**: Secondary backend (`src/dynamic_moe_router/jax/`)
- **TensorFlow**: Tertiary backend (`src/dynamic_moe_router/tf/`)

### Testing Strategy

- **Unit Tests**: Core routing logic
- **Integration Tests**: Full model integration
- **Performance Tests**: FLOP counting accuracy
- **Backend Tests**: Cross-framework compatibility

### Code Quality Standards

- **Type Hints**: All public APIs must be typed
- **Docstrings**: Google-style docstrings required
- **Test Coverage**: Minimum 90% coverage
- **Performance**: No regressions allowed

## Debugging

### Common Issues

1. **CUDA/GPU Issues**:
   ```bash
   export CUDA_LAUNCH_BLOCKING=1
   python debug_script.py
   ```

2. **Memory Profiling**:
   ```bash
   python -m memory_profiler your_script.py
   ```

3. **FLOP Counting**:
   ```python
   from dynamic_moe_router import FLOPProfiler
   with FLOPProfiler() as profiler:
       # Your code here
   print(profiler.summary())
   ```

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release PR
4. Tag release: `git tag v0.1.0`
5. Build and publish: `make release`
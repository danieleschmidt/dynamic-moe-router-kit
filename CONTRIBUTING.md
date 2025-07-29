# Contributing to dynamic-moe-router-kit

We welcome contributions! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/dynamic-moe-router-kit.git
   cd dynamic-moe-router-kit
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/new-feature
   ```

2. Make your changes and add tests
3. Run the test suite:
   ```bash
   pytest
   ```

4. Check code formatting:
   ```bash
   black .
   isort .
   ruff check .
   ```

5. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```

6. Push to your fork and create a pull request

## Code Standards

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write docstrings for public APIs
- Include tests for new functionality
- Keep line length to 88 characters

## Testing

- Write unit tests for all new code
- Ensure tests pass on all supported Python versions
- Aim for high test coverage (>90%)

## Pull Request Process

1. Update documentation if needed
2. Add entry to CHANGELOG.md
3. Ensure all tests pass
4. Request review from maintainers

## Bug Reports

Please include:
- Python version
- Package versions
- Minimal reproduction example
- Expected vs actual behavior

## Questions?

Open an issue or start a discussion on GitHub.
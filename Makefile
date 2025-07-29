.PHONY: help dev-setup test lint format type-check clean build docs benchmark install

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

dev-setup:  ## Set up development environment
	pip install -e ".[dev]"
	pre-commit install

install:  ## Install package
	pip install -e .

test:  ## Run test suite
	pytest --cov=dynamic_moe_router --cov-report=html --cov-report=term

test-all:  ## Run tests on all backends
	pytest tests/ -k "torch" --cov=dynamic_moe_router.torch
	pytest tests/ -k "jax" --cov=dynamic_moe_router.jax  
	pytest tests/ -k "tf" --cov=dynamic_moe_router.tf

lint:  ## Run all linting checks
	ruff check .
	black --check .
	isort --check-only .

format:  ## Format code
	black .
	isort .
	ruff check . --fix

type-check:  ## Run type checking
	mypy src/dynamic_moe_router

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build package
	python -m build

docs:  ## Build documentation
	cd docs && make html

docs-serve:  ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

benchmark:  ## Run performance benchmarks
	python -m dynamic_moe_router.benchmark --model mixtral-8x7b --tasks bbh,mmlu

profile:  ## Profile model performance
	python -m dynamic_moe_router.profile --model-path ./examples/models

security:  ## Run security checks
	safety check
	bandit -r src/

pre-commit:  ## Run pre-commit hooks
	pre-commit run --all-files

release:  ## Build and upload to PyPI
	$(MAKE) clean
	$(MAKE) build
	twine check dist/*
	twine upload dist/*

docker-build:  ## Build Docker image
	docker build -t dynamic-moe-router:latest .

docker-test:  ## Test in Docker container
	docker run --rm dynamic-moe-router:latest pytest

all-checks: lint type-check test security  ## Run all quality checks
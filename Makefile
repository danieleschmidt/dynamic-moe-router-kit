.PHONY: test demo install

install:
	pip install -e .

test:
	pytest tests/

demo:
	python examples/demo_load_balancing.py

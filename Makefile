.PHONY: install lint format test demo ci clean

PYTHON ?= python
PKG := gpforge

install:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m ruff check $(PKG)/ examples/ tests/

format:
	$(PYTHON) -m ruff format $(PKG)/ examples/ tests/

test:
	$(PYTHON) -m pytest tests/ -q

demo:
	PYTHONPATH=. $(PYTHON) -m examples.run_demo

ci: lint format test demo

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +

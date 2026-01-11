# FlowShield-UDRL Makefile
# Convenience commands for development and experiments

.PHONY: install test lint format clean train evaluate docs help

# Python environment
PYTHON := python
PIP := pip

# Default target
help:
	@echo "FlowShield-UDRL - Available Commands"
	@echo "====================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install dependencies"
	@echo "  make install-dev   Install with development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run all tests"
	@echo "  make test-fast     Run tests without slow markers"
	@echo "  make lint          Run linting (ruff)"
	@echo "  make format        Format code (black, isort)"
	@echo "  make typecheck     Run type checking (mypy)"
	@echo ""
	@echo "Training:"
	@echo "  make train         Train Flow shield on LunarLander"
	@echo "  make train-all     Train all components (policy + shields)"
	@echo "  make collect-data  Collect expert data"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate      Run evaluation"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          Build documentation"
	@echo "  make docs-serve    Serve documentation locally"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Remove generated files"
	@echo "  make clean-all     Remove all generated files including checkpoints"

# Installation
install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

# Testing
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-fast:
	$(PYTHON) -m pytest tests/ -v --tb=short -m "not slow"

test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Code quality
lint:
	$(PYTHON) -m ruff check src/ scripts/ tests/

format:
	$(PYTHON) -m black src/ scripts/ tests/
	$(PYTHON) -m isort src/ scripts/ tests/

typecheck:
	$(PYTHON) -m mypy src/ --ignore-missing-imports

# Training (argparse-based)
train:
	$(PYTHON) scripts/train_flow.py --data data/lunarlander_expert.npz --epochs 100

train-policy:
	$(PYTHON) scripts/train_policy.py --data data/lunarlander_expert.npz --epochs 100

train-flow:
	$(PYTHON) scripts/train_flow.py --data data/lunarlander_expert.npz --epochs 100

train-quantile:
	$(PYTHON) scripts/train_quantile.py --data data/lunarlander_expert.npz --epochs 100

train-diffusion:
	$(PYTHON) scripts/train_diffusion.py --data data/lunarlander_expert.npz --epochs 100

train-all: train-policy train-flow train-quantile train-diffusion
	@echo "All models trained successfully!"

collect-data:
	$(PYTHON) scripts/collect_expert_data.py --env lunarlander --n-episodes 500 --output data/lunarlander_expert.npz

collect-expert:
	$(PYTHON) scripts/collect_expert_data.py --train-expert --timesteps 500000

# Evaluation
evaluate:
	$(PYTHON) scripts/evaluate_models.py --env lunarlander --data data/lunarlander_expert.npz

evaluate-offline:
	$(PYTHON) scripts/evaluate_models.py --offline --data data/lunarlander_expert.npz

# Full experiment
experiment:
	$(PYTHON) scripts/run_experiments.py --env lunarlander

experiment-quick:
	$(PYTHON) scripts/run_experiments.py --env lunarlander --quick

# Documentation
docs:
	cd docs && sphinx-build -b html . _build/html

docs-serve:
	cd docs/_build/html && $(PYTHON) -m http.server 8000

docs-clean:
	rm -rf docs/_build

# Notebooks
notebook:
	jupyter notebook notebooks/

lab:
	jupyter lab notebooks/

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ 2>/dev/null || true

clean-all: clean
	rm -rf checkpoints/* logs/* 2>/dev/null || true

# Quick verification
quick-test:
	$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')"
	$(PYTHON) -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"
	$(PYTHON) -c "from pathlib import Path; assert Path('data/lunarlander_expert.npz').exists(), 'Dataset missing'"
	@echo "All checks passed!"

# Docker (optional)
docker-build:
	docker build -t flowshield-udrl .

docker-run:
	docker run -it --gpus all -v $(PWD):/workspace flowshield-udrl

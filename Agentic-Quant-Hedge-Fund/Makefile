.PHONY: all clean install test format lint type-check docker-build docker-run help data

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
DOCKER_IMAGE := agentic-quant
DOCKER_TAG := latest

all: install test ## Install dependencies and run tests

install: ## Install the package in development mode
	$(PIP) install -e .

clean: ## Clean up cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage .pytest_cache/

test: ## Run the test suite
	$(PYTHON) -m pytest src/agentic_quant/tests/ -v --cov=src/agentic_quant/

format: ## Format code with black and isort
	$(PYTHON) -m black src/agentic_quant/
	$(PYTHON) -m isort src/agentic_quant/

lint: ## Lint code with flake8
	$(PYTHON) -m flake8 src/agentic_quant/ --max-line-length=88 --extend-ignore=E203,W503

type-check: ## Run type checking with mypy
	$(PYTHON) -m mypy src/agentic_quant/ --ignore-missing-imports

docker-build: ## Build Docker image
	docker build -f docker/Dockerfile -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run: ## Run Docker container
	docker run --rm -it -v $(PWD)/data:/app/data $(DOCKER_IMAGE):$(DOCKER_TAG)

data: ## Create data directory structure
	mkdir -p data/{raw,processed,features}
	mkdir -p artifacts/{ideas,alphas}

dev-setup: install data ## Set up development environment
	$(PIP) install -e ".[dev]"

ci: format lint type-check test ## Run all CI checks

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'


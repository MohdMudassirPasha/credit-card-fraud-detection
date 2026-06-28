# ===========================================================================
# Developer command shortcuts. Override the interpreter with e.g.:
#     make train PYTHON=.venv/bin/python
# ===========================================================================

PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.DEFAULT_GOAL := help
.PHONY: help install install-dev data train tune evaluate api mlflow test lint format docker docker-up clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install runtime + development dependencies
	$(PIP) install -r requirements-dev.txt

data: ## Download the Kaggle dataset into data/raw/
	$(PYTHON) -m src.data.download

train: ## Run the full training/evaluation pipeline
	$(PYTHON) main.py

tune: ## Run the pipeline with Optuna hyperparameter tuning
	$(PYTHON) main.py --tune

evaluate: ## Alias for the full pipeline (training also evaluates)
	$(PYTHON) main.py

api: ## Serve the FastAPI app locally on port 8000
	$(PYTHON) -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

mlflow: ## Launch the MLflow tracking UI on port 5000
	$(PYTHON) -m mlflow ui --backend-store-uri file:./mlruns --port 5000

test: ## Run the test suite
	$(PYTHON) -m pytest

lint: ## Run ruff + black/isort checks (no changes)
	$(PYTHON) -m ruff check src app tests main.py
	$(PYTHON) -m black --check src app tests main.py
	$(PYTHON) -m isort --check-only src app tests main.py

format: ## Auto-format the codebase with black + isort
	$(PYTHON) -m isort src app tests main.py
	$(PYTHON) -m black src app tests main.py

docker: ## Build the Docker image
	docker build -t credit-card-fraud-detection:latest .

docker-up: ## Start the full stack (API + MLflow) via docker compose
	docker compose up --build

clean: ## Remove caches and generated artifacts (keeps source)
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ __pycache__
	rm -rf reports/*.png reports/shap reports/optuna
	rm -f reports/metrics_summary.csv reports/metrics_summary.json
	rm -f reports/classification_report_*.txt
	rm -f models/*.joblib models/model_metadata.json
	rm -f logs/*.log

# ===========================================================================
# Developer command shortcuts. Override the interpreter with e.g.:
#     make train PYTHON=.venv/bin/python
# ===========================================================================

PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.DEFAULT_GOAL := help
.PHONY: help install install-dev data train tune evaluate api dashboard mlflow \
        test coverage lint typecheck format docker docker-up clean

# Source trees passed to the linters/formatters/type-checker.
LINT_PATHS := src app dashboard tests main.py

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
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

dashboard: ## Serve the Dash dashboard locally on port 8050 (API must be running)
	$(PYTHON) -m dashboard.app

mlflow: ## Launch the MLflow tracking UI on port 5000
	$(PYTHON) -m mlflow ui --backend-store-uri file:./mlruns --port 5000

test: ## Run the test suite
	$(PYTHON) -m pytest

coverage: ## Run the test suite with a coverage report
	$(PYTHON) -m pytest --cov=src --cov=app --cov=dashboard --cov-report=term-missing

lint: ## Run ruff + black/isort checks (no changes)
	$(PYTHON) -m ruff check $(LINT_PATHS)
	$(PYTHON) -m black --check $(LINT_PATHS)
	$(PYTHON) -m isort --check-only $(LINT_PATHS)

typecheck: ## Run mypy static type checking on app + dashboard
	$(PYTHON) -m mypy app dashboard

format: ## Auto-format the codebase with black + isort
	$(PYTHON) -m isort $(LINT_PATHS)
	$(PYTHON) -m black $(LINT_PATHS)

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

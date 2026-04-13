# Makefile — development convenience commands for options-research.
#
# Prerequisites: Docker Desktop, Python 3.11+, Node 20+
#
# Usage:
#   make up          Start all services (Docker Compose)
#   make down        Stop all services
#   make logs        Tail logs from all containers
#   make migrate     Apply pending Alembic migrations
#   make test        Run the full backend test suite
#   make lint        Run all linters (no changes)
#   make format      Auto-format Python files

.DEFAULT_GOAL := help
BACKEND_DIR   := backend
FRONTEND_DIR  := frontend

# ── Docker Compose ────────────────────────────────────────────────────────────

.PHONY: up
up:           ## Start all services
	docker compose up --build

.PHONY: up-d
up-d:         ## Start all services (detached)
	docker compose up --build -d

.PHONY: down
down:         ## Stop and remove containers
	docker compose down

.PHONY: down-v
down-v:       ## Stop containers and delete volumes (DESTRUCTIVE — clears DB)
	docker compose down -v

.PHONY: restart
restart:      ## Restart all services
	docker compose restart

.PHONY: logs
logs:         ## Tail logs from all containers
	docker compose logs -f

.PHONY: logs-backend
logs-backend: ## Tail backend logs only
	docker compose logs -f backend

.PHONY: shell-backend
shell-backend: ## Open a shell in the running backend container
	docker compose exec backend bash

.PHONY: shell-db
shell-db:     ## Open psql in the running postgres container
	docker compose exec postgres psql -U postgres options_research

# ── Database migrations (Alembic) ─────────────────────────────────────────────

.PHONY: migrate
migrate:      ## Apply all pending migrations
	cd $(BACKEND_DIR) && python -m alembic upgrade head

.PHONY: migrate-new
migrate-new:  ## Create a new blank migration (NAME= required)
	@[ -n "$(NAME)" ] || (echo "Usage: make migrate-new NAME=your_description" && exit 1)
	cd $(BACKEND_DIR) && python -m alembic revision --autogenerate -m "$(NAME)"

.PHONY: migrate-rollback
migrate-rollback: ## Roll back the most recent migration
	cd $(BACKEND_DIR) && python -m alembic downgrade -1

.PHONY: migrate-history
migrate-history: ## Show migration history
	cd $(BACKEND_DIR) && python -m alembic history --verbose

.PHONY: migrate-current
migrate-current: ## Show current migration head
	cd $(BACKEND_DIR) && python -m alembic current

# ── Backend tests ─────────────────────────────────────────────────────────────

.PHONY: test
test:         ## Run the full backend test suite (excludes integration)
	cd $(BACKEND_DIR) && python -m pytest tests/ -v --tb=short \
	  --ignore=tests/integration

.PHONY: test-leakage
test-leakage: ## Run leakage regression tests only (hard gate)
	cd $(BACKEND_DIR) && python -m pytest tests/test_leakage.py -v

.PHONY: test-calibration
test-calibration: ## Run calibration regression tests
	cd $(BACKEND_DIR) && python -m pytest tests/calibration/ -v

.PHONY: test-risk
test-risk:    ## Run risk-critical unit tests
	cd $(BACKEND_DIR) && python -m pytest tests/ -m risk_critical -v \
	  --ignore=tests/integration

.PHONY: test-integration
test-integration: ## Run integration tests (requires live DB/Redis)
	cd $(BACKEND_DIR) && INTEGRATION_TESTS=1 python -m pytest tests/integration/ -v

.PHONY: test-cov
test-cov:     ## Run tests with coverage report
	cd $(BACKEND_DIR) && python -m pytest tests/ --cov=app \
	  --cov-report=term-missing --ignore=tests/integration

# ── Linting & formatting ──────────────────────────────────────────────────────

.PHONY: lint
lint:         ## Lint Python (flake8 + ruff) — no changes
	cd $(BACKEND_DIR) && python -m flake8 app/ --max-line-length=120 --ignore=E203,W503
	cd $(BACKEND_DIR) && python -m ruff check app/

.PHONY: typecheck
typecheck:    ## Run mypy on the backend
	cd $(BACKEND_DIR) && python -m mypy app/ --ignore-missing-imports

.PHONY: format
format:       ## Auto-format Python with black + isort
	cd $(BACKEND_DIR) && python -m black app/ tests/
	cd $(BACKEND_DIR) && python -m isort --profile black app/ tests/

.PHONY: format-check
format-check: ## Check formatting without applying changes
	cd $(BACKEND_DIR) && python -m black --check app/ tests/
	cd $(BACKEND_DIR) && python -m isort --check --profile black app/ tests/

# ── Frontend ──────────────────────────────────────────────────────────────────

.PHONY: frontend-install
frontend-install: ## Install frontend dependencies
	cd $(FRONTEND_DIR) && npm install

.PHONY: frontend-dev
frontend-dev: ## Start frontend dev server (standalone, no Docker)
	cd $(FRONTEND_DIR) && npm run dev

.PHONY: frontend-build
frontend-build: ## Production build of the frontend
	cd $(FRONTEND_DIR) && npm run build

.PHONY: frontend-lint
frontend-lint: ## Lint frontend TypeScript
	cd $(FRONTEND_DIR) && npx tsc --noEmit && npx next lint

# ── Pre-commit ────────────────────────────────────────────────────────────────

.PHONY: hooks-install
hooks-install: ## Install pre-commit hooks
	pre-commit install

.PHONY: hooks-run
hooks-run:    ## Run all pre-commit hooks against staged files
	pre-commit run

.PHONY: hooks-run-all
hooks-run-all: ## Run all pre-commit hooks against every file
	pre-commit run --all-files

# ── Health checks ─────────────────────────────────────────────────────────────

.PHONY: health
health:       ## Check backend liveness
	curl -s http://localhost:8000/health | python -m json.tool

.PHONY: ready
ready:        ## Check backend readiness (DB + Redis)
	curl -s http://localhost:8000/ready | python -m json.tool

# ── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help:         ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

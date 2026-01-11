.PHONY: setup install dev lint format typecheck test test-cov run clean help check-ollama pull-models notebook

# =============================================================================
# ATS Resume Analyzer - Makefile
# =============================================================================

# Default target
help:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║           ATS Resume Analyzer - Available Commands             ║"
	@echo "╠════════════════════════════════════════════════════════════════╣"
	@echo "║  SETUP                                                         ║"
	@echo "║    make setup        - Create venv and install all deps        ║"
	@echo "║    make install      - Install production dependencies only    ║"
	@echo "║    make dev          - Install development dependencies        ║"
	@echo "║    make pull-models  - Pull required Ollama models             ║"
	@echo "║                                                                ║"
	@echo "║  DEVELOPMENT                                                   ║"
	@echo "║    make lint         - Run ruff linter                         ║"
	@echo "║    make format       - Format code with ruff                   ║"
	@echo "║    make typecheck    - Run mypy type checker                   ║"
	@echo "║    make test         - Run pytest tests                        ║"
	@echo "║    make test-cov     - Run tests with coverage report          ║"
	@echo "║                                                                ║"
	@echo "║  RUN                                                           ║"
	@echo "║    make run          - Start Streamlit web app                 ║"
	@echo "║    make notebook     - Start Jupyter notebook server           ║"
	@echo "║    make cli          - Show CLI help                           ║"
	@echo "║    make check-ollama - Check if Ollama is running              ║"
	@echo "║                                                                ║"
	@echo "║  MAINTENANCE                                                   ║"
	@echo "║    make clean        - Remove build artifacts and cache        ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"

# -----------------------------------------------------------------------------
# Setup Commands
# -----------------------------------------------------------------------------

# Setup environment with uv
setup:
	uv venv
	uv sync --all-extras
	@echo "✅ Setup complete! Run 'make run' to start the app."
	@echo "💡 Don't forget to: make pull-models"

# Install production dependencies
install:
	uv sync

# Install dev dependencies
dev:
	uv sync --all-extras

# Pull required Ollama models
pull-models:
	@echo "📥 Pulling required Ollama models..."
	ollama pull qwen2.5:7b-instruct
	ollama pull nomic-embed-text
	@echo "✅ Models ready!"

# -----------------------------------------------------------------------------
# Development Commands
# -----------------------------------------------------------------------------

# Linting with ruff
lint:
	uv run ruff check src tests

# Format code with ruff
format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

# Type checking with mypy
typecheck:
	uv run mypy src

# Run tests with pytest
test:
	uv run pytest

# Run tests with coverage
test-cov:
	uv run pytest --cov=ats_checker --cov-report=html
	@echo "📊 Coverage report: htmlcov/index.html"

# -----------------------------------------------------------------------------
# Run Commands
# -----------------------------------------------------------------------------

# Start Streamlit web app
run:
	uv run streamlit run main.py

# Start Jupyter notebook server
notebook:
	uv run jupyter notebook notebooks/

# Show CLI help
cli:
	uv run python -m ats_checker.cli --help

# Analyze resume (usage: make analyze RESUME=resume.pdf JOB=job.txt)
analyze:
	uv run python -m ats_checker.cli analyze $(RESUME) $(JOB)

# Check if Ollama is running
check-ollama:
	@curl -s http://localhost:11434/api/tags > /dev/null && echo "✅ Ollama is running" || echo "❌ Ollama not running. Start with: ollama serve"

# -----------------------------------------------------------------------------
# Maintenance Commands
# -----------------------------------------------------------------------------

# Clean build artifacts
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	rm -rf src/*.egg-info build dist
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "🧹 Clean complete!"

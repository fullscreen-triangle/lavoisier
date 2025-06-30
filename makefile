.PHONY: help install dev-install test lint format clean build docs docker-build docker-run setup-dev

# Default target
help:
	@echo "Available targets:"
	@echo "  help          Show this help message"
	@echo "  install       Install package for production use"
	@echo "  dev-install   Install package in development mode"
	@echo "  setup-dev     Set up development environment"
	@echo "  test          Run tests"
	@echo "  test-cov      Run tests with coverage"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code"
	@echo "  clean         Remove build artifacts"
	@echo "  build         Build package"
	@echo "  build-rust    Build Rust components"
	@echo "  docs          Build documentation"
	@echo "  docker-build  Build Docker images"
	@echo "  docker-run    Run with Docker Compose"
	@echo "  release       Create a release"

# Installation targets
install:
	pip install -e .

dev-install:
	pip install -e ".[dev,ai,docs]"
	pre-commit install

setup-dev: dev-install
	@echo "Setting up development environment..."
	@if [ ! -d ".venv" ]; then python -m venv .venv; fi
	@echo "Activate virtual environment with: source .venv/bin/activate"
	@echo "Development environment setup complete!"

# Testing targets
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=lavoisier --cov-report=html --cov-report=term-missing

test-integration:
	pytest tests/ -v -m integration

test-slow:
	pytest tests/ -v -m slow

# Code quality targets
lint:
	black --check lavoisier tests
	isort --check-only lavoisier tests
	flake8 lavoisier tests
	mypy lavoisier
	bandit -r lavoisier

format:
	black lavoisier tests
	isort lavoisier tests
	cargo fmt --all

# Rust targets
build-rust:
	cargo build --release
	cargo test

clippy:
	cargo clippy --all-targets --all-features -- -D warnings

# Build targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf target/debug
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

build-wheel:
	python -m build --wheel

# Documentation targets
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8080

# Docker targets
docker-build:
	docker build -t lavoisier:latest .
	docker build -f Dockerfile.dev -t lavoisier:dev .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Release targets
release-check:
	@echo "Pre-release checks..."
	$(MAKE) test
	$(MAKE) lint
	$(MAKE) build
	@echo "Release checks passed!"

release: release-check
	@echo "Creating release..."
	git tag v$(shell python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
	git push origin --tags
	python -m build
	@echo "Release created! Upload to PyPI with: twine upload dist/*"

# Utility targets
requirements:
	pip-compile pyproject.toml --output-file requirements.txt
	pip-compile pyproject.toml --extra dev --output-file requirements-dev.txt

update-deps:
	pip-compile --upgrade pyproject.toml --output-file requirements.txt
	pip-compile --upgrade pyproject.toml --extra dev --output-file requirements-dev.txt

benchmark:
	python -m pytest tests/performance/ -v --benchmark-only

profile:
	python -m cProfile -o profile.stats scripts/profile_analysis.py

# CI/CD targets
ci-test: test lint
	@echo "CI tests completed"

# Environment info
env-info:
	@echo "Python version: $(shell python --version)"
	@echo "Rust version: $(shell rustc --version)"
	@echo "Cargo version: $(shell cargo --version)"
	@echo "Git branch: $(shell git branch --show-current)"
	@echo "Git commit: $(shell git rev-parse --short HEAD)" 
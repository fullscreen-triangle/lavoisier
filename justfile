# Lavoisier Development Commands
# Use `just --list` to see all available commands

# Default recipe
default:
    @just --list

# Environment setup
setup-dev:
    python -m venv .venv
    .venv/bin/pip install -e ".[dev,ai,docs]"
    .venv/bin/pre-commit install
    @echo "Development environment setup complete!"
    @echo "Activate with: source .venv/bin/activate"

# Install package
install:
    pip install -e .

install-dev:
    pip install -e ".[dev,ai,docs]"
    pre-commit install

# Testing
test:
    pytest tests/ -v

test-cov:
    pytest tests/ -v --cov=lavoisier --cov-report=html --cov-report=term-missing

test-integration:
    pytest tests/ -v -m integration

test-slow:
    pytest tests/ -v -m slow

# Code quality
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

# Rust operations
build-rust:
    cargo build --release

test-rust:
    cargo test

clippy:
    cargo clippy --all-targets --all-features -- -D warnings

# Documentation
docs:
    cd docs && make html

docs-serve:
    cd docs/_build/html && python -m http.server 8080

# Cleaning
clean:
    rm -rf build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/ target/debug
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# Building
build: clean
    python -m build

# Docker operations
docker-build:
    docker build -t lavoisier:latest .
    docker build -f Dockerfile.dev -t lavoisier:dev .

docker-run:
    docker-compose up -d

docker-stop:
    docker-compose down

# Benchmarking
benchmark:
    pytest tests/performance/ -v --benchmark-only

# Release
release-check:
    just test
    just lint
    just build
    @echo "Release checks passed!"

# Utilities
env-info:
    @echo "Python: $(python --version)"
    @echo "Rust: $(rustc --version)"
    @echo "Cargo: $(cargo --version)"
    @echo "Git branch: $(git branch --show-current)"
    @echo "Git commit: $(git rev-parse --short HEAD)" 
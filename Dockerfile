# Multi-stage build for Python + Rust hybrid package
FROM rust:1.75-slim as rust-builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Rust source
COPY Cargo.toml Cargo.lock ./
COPY lavoisier-core ./lavoisier-core/
COPY lavoisier-io ./lavoisier-io/
COPY lavoisier-buhera ./lavoisier-buhera/

# Build Rust components
RUN cargo build --release

FROM python:3.11-slim as python-builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libhdf5-dev \
    libopencv-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (needed for setuptools-rust)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy Python requirements
COPY requirements.txt pyproject.toml ./
COPY lavoisier ./lavoisier/

# Copy Rust artifacts from previous stage
COPY --from=rust-builder /app/target ./target/
COPY --from=rust-builder /app/Cargo.toml /app/Cargo.lock ./
COPY --from=rust-builder /app/lavoisier-core ./lavoisier-core/
COPY --from=rust-builder /app/lavoisier-io ./lavoisier-io/
COPY --from=rust-builder /app/lavoisier-buhera ./lavoisier-buhera/

# Install Python dependencies and package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

FROM python:3.11-slim as runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-103 \
    libopencv-core406 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash lavoisier
USER lavoisier
WORKDIR /home/lavoisier

# Copy installed packages from builder
COPY --from=python-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin
COPY --from=python-builder /app /home/lavoisier/app

# Set environment variables
ENV PYTHONPATH=/home/lavoisier/app
ENV PYTHONUNBUFFERED=1
ENV RUST_LOG=info

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "lavoisier.api:app", "--host", "0.0.0.0", "--port", "8000"] 
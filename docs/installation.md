---
layout: default
title: Installation
nav_order: 2
---

# Installation Guide

## Prerequisites

Before installing Lavoisier, ensure you have:

- Python 3.8 or higher
- pip package manager
- Git (for development installation)

## Quick Installation

Install the latest stable version from PyPI:

```bash
pip install lavoisier
```

## Development Installation

For the latest development version:

```bash
git clone https://github.com/username/lavoisier.git
cd lavoisier
pip install -e ".[dev]"
```

## Optional Dependencies

### GPU Support
For GPU acceleration:
```bash
pip install lavoisier[gpu]
```

### Visualization Dependencies
For the visual pipeline:
```bash
pip install lavoisier[visual]
```

### Full Installation
For all features:
```bash
pip install lavoisier[all]
```

## Configuration

After installation, configure your environment:

1. Set up API keys (if using commercial LLMs):
```bash
export OPENAI_API_KEY="your-key-here"
```

2. Configure compute resources:
```bash
lavoisier config --cores 8 --memory 16
```

## Verification

Verify your installation:

```bash
lavoisier --version
lavoisier test
```

## Troubleshooting

Common issues and solutions:

### Missing Dependencies
If you encounter missing dependencies:
```bash
pip install -r requirements.txt
```

### GPU Issues
For CUDA-related issues:
1. Verify CUDA installation
2. Check GPU compatibility
3. Update GPU drivers

## Next Steps

- Read the [Quick Start Guide](quickstart.html)
- Explore [Example Workflows](examples.html)
- Review [API Documentation](api.html) 
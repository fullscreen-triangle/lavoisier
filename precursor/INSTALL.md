# Installation Guide

Detailed installation instructions for Lavoisier Precursor.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
- [Verifying Installation](#verifying-installation)
- [Optional Dependencies](#optional-dependencies)
- [Platform-Specific Notes](#platform-specific-notes)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Python**: ≥ 3.8 (tested on 3.8, 3.9, 3.10, 3.11)
- **Operating System**: Windows, macOS, Linux
- **Memory**: ≥ 8 GB RAM recommended
- **Disk Space**: ≥ 2 GB free space

### Required Software

1. **Python** (≥ 3.8)

   ```bash
   python --version  # Should show 3.8 or higher
   ```

2. **pip** (Python package installer)

   ```bash
   pip --version
   ```

3. **git** (for cloning repository)

   ```bash
   git --version
   ```

## Installation Methods

### Method 1: Development Installation (Recommended)

For active development or testing:

```bash
# Clone the repository
git clone https://github.com/lavoisier-project/precursor.git
cd precursor

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install in editable mode
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"
```

### Method 2: User Installation

For regular use without development:

```bash
# Clone and install
git clone https://github.com/lavoisier-project/precursor.git
cd precursor
pip install .
```

### Method 3: From PyPI (when published)

```bash
# Install stable release
pip install lavoisier-precursor

# With GPU support
pip install lavoisier-precursor[gpu]

# With development tools
pip install lavoisier-precursor[dev]

# With everything
pip install lavoisier-precursor[all]
```

## Verifying Installation

### Quick Test

```python
# Test imports
python -c "from precursor.pipeline import Theatre; print('✓ Pipeline module OK')"
python -c "from precursor.analysis import QualityBundle; print('✓ Analysis module OK')"
python -c "from precursor.core import SEntropyTransformer; print('✓ Core module OK')"
python -c "from precursor.hardware import ResonantComputationEngine; print('✓ Hardware module OK')"
```

### Run Example

```python
from precursor.pipeline import Theatre, create_stage, NavigationMode

theatre = Theatre("Test", navigation_mode=NavigationMode.LINEAR)
stage = create_stage("Test Stage", "stage_01", ['data_loading'])
theatre.add_stage(stage)

print("✓ Precursor installed successfully!")
```

### Run Tests (if development install)

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# View coverage report
# Open htmlcov/index.html in browser
```

## Optional Dependencies

### GPU Support (NVIDIA GPUs only)

For GPU-accelerated hardware harvesting:

```bash
pip install pynvml

# Verify
python -c "import pynvml; pynvml.nvmlInit(); print('✓ GPU support OK')"
```

### Development Tools

For contributing or development:

```bash
pip install -e ".[dev]"

# This installs:
# - pytest (testing)
# - pytest-cov (coverage)
# - pytest-asyncio (async testing)
# - black (code formatting)
# - flake8 (linting)
# - mypy (type checking)
```

### Mass Spectrometry File Formats

For reading various MS file formats:

```bash
# Already included in main dependencies:
# - pymzml (mzML files)
# - pyteomics (various formats)

# Additional formats:
pip install pyopenms  # For vendor formats
```

## Platform-Specific Notes

### Windows

1. **Visual C++ Redistributable** may be required for some dependencies:
   - Download from [Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)

2. **PowerShell** is recommended over Command Prompt:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Long Path Support** (Windows 10/11):
   - May be needed for deep directory structures
   - Enable in Group Policy or Registry

### macOS

1. **Xcode Command Line Tools** (for compiling dependencies):

   ```bash
   xcode-select --install
   ```

2. **Homebrew** packages (optional but recommended):

   ```bash
   brew install python@3.11
   brew install git
   ```

### Linux

1. **System packages** (Ubuntu/Debian):

   ```bash
   sudo apt-get update
   sudo apt-get install python3-dev python3-pip python3-venv
   sudo apt-get install build-essential
   ```

2. **System packages** (CentOS/RHEL):

   ```bash
   sudo yum install python3-devel python3-pip
   sudo yum groupinstall "Development Tools"
   ```

## Dependency Installation Order

If you encounter issues, install dependencies in this order:

```bash
# 1. Core scientific computing
pip install numpy pandas scipy

# 2. Machine learning frameworks
pip install scikit-learn torch

# 3. Mass spectrometry libraries
pip install pymzml pyteomics

# 4. Visualization
pip install matplotlib seaborn

# 5. Computer vision
pip install opencv-python pillow

# 6. Everything else
pip install -e .
```

## Troubleshooting

### Issue: PyTorch Installation Fails

**Solution**: Install PyTorch separately first:

```bash
# CPU-only version
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU version (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Then install precursor
pip install -e .
```

### Issue: OpenCV Import Error

**Solution**: Try headless version:

```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

### Issue: NumPy/SciPy Compilation Errors

**Solution**: Use pre-compiled wheels:

```bash
# Upgrade pip first
pip install --upgrade pip

# Install with binary wheels
pip install --only-binary :all: numpy scipy
```

### Issue: Permission Denied

**Solution**: Use user installation:

```bash
pip install --user -e .
```

### Issue: SSL Certificate Error

**Solution**: Upgrade certifi:

```bash
pip install --upgrade certifi
```

### Issue: Module Not Found After Installation

**Solution**: Verify installation and Python path:

```bash
# Check installation
pip show lavoisier-precursor

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Reinstall in development mode
pip install -e .
```

## Uninstallation

```bash
# If installed normally
pip uninstall lavoisier-precursor

# If installed in development mode
cd precursor
pip uninstall lavoisier-precursor

# Clean up virtual environment
deactivate  # If in virtual environment
rm -rf venv  # Remove virtual environment directory
```

## Advanced Installation

### Custom Installation Location

```bash
pip install --prefix=/custom/path .
```

### Installing from Specific Branch

```bash
pip install git+https://github.com/lavoisier-project/precursor.git@develop
```

### Installing without Dependencies

```bash
pip install --no-deps .
```

## Post-Installation

### Configuration

1. **Create config directory** (optional):

   ```bash
   mkdir -p ~/.precursor
   ```

2. **Set environment variables** (optional):

   ```bash
   export PRECURSOR_DATA_DIR=/path/to/data
   export PRECURSOR_CACHE_DIR=/path/to/cache
   ```

### Download Example Data

```bash
# If example data repository exists
git clone https://github.com/lavoisier-project/precursor-examples.git
cd precursor-examples
```

## Getting Help

- **Documentation**: [ReadTheDocs](https://lavoisier-precursor.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/lavoisier-project/precursor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/lavoisier-project/precursor/discussions)

## Next Steps

After installation:

1. Read the [README](README.md) for quick start
2. Try [example scripts](src/pipeline/PIPELINE_EXAMPLE.md)
3. Explore [documentation](src/)
4. Join the community discussions

---

**Installation complete!** 🎉

You're ready to use Lavoisier Precursor for advanced mass spectrometry analysis.

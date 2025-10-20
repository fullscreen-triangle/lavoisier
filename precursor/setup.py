#!/usr/bin/env python
"""
Lavoisier Precursor - Advanced Mass Spectrometry Analysis
==========================================================

Setup configuration for the Precursor package.

Features:
- Finite observer architecture for pipeline stages
- Resonant computation engine with hardware oscillation harvesting
- S-Entropy coordinate system for mass spectrometry
- Phase-lock network analysis
- Analysis bundles for surgical injection
- Experiment-to-LLM generation
- Theatre-based non-linear pipeline navigation

Installation:
    pip install -e .                    # Development install
    pip install -e ".[dev]"             # With development tools
    pip install -e ".[gpu]"             # With GPU support
    pip install -e ".[all]"             # All optional dependencies

Author: Lavoisier Project Team
License: MIT
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]
else:
    requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "networkx>=2.6.0",
        "pymzml>=2.5.0",
        "pyteomics>=4.5.0",
        "torch>=1.10.0",
        "transformers>=4.20.0",
        "opencv-python>=4.5.0",
        "pillow>=8.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "h5py>=3.1.0",
        "tables>=3.6.0",
        "statsmodels>=0.13.0",
        "psutil>=5.8.0",
    ]

setup(
    name="lavoisier-precursor",
    version="1.0.0",
    author="Lavoisier Project Team",
    author_email="lavoisier@project.org",
    description="Advanced mass spectrometry analysis with finite observer architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lavoisier-project/precursor",
    project_urls={
        "Documentation": "https://lavoisier-precursor.readthedocs.io",
        "Source": "https://github.com/lavoisier-project/precursor",
        "Bug Tracker": "https://github.com/lavoisier-project/precursor/issues",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "gpu": [
            "pynvml>=11.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lavoisier-precursor=precursor.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "mass-spectrometry",
        "metabolomics",
        "proteomics",
        "bioinformatics",
        "phase-lock",
        "s-entropy",
        "finite-observers",
        "resonant-computation",
        "pipeline",
        "theatre",
        "analysis-bundles",
    ],
)

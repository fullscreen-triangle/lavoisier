"""
Setup script for Lavoisier Validation Framework

Standalone validation package for comparing mass spectrometry analysis methods.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Lavoisier Validation Framework - Comprehensive validation for mass spectrometry methods"

# Read requirements
def read_requirements():
    try:
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []

setup(
    name="lavoisier-validation",
    version="1.0.0",
    description="Comprehensive validation framework for mass spectrometry analysis methods",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    
    # Author information
    author="Lavoisier Research Team",
    author_email="validation@lavoisier-research.org",
    
    # Project URLs
    url="https://github.com/lavoisier-research/validation-framework",
    project_urls={
        "Documentation": "https://lavoisier-validation.readthedocs.io/",
        "Source": "https://github.com/lavoisier-research/validation-framework",
        "Tracker": "https://github.com/lavoisier-research/validation-framework/issues",
    },
    
    # Package discovery
    packages=find_packages(),
    include_package_data=True,
    
    # Requirements
    install_requires=read_requirements(),
    
    # Extra requirements for different use cases
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
        ],
        "all": [
            # Includes all optional dependencies
        ]
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Package data
    package_data={
        "validation": [
            "public/*.mzML",
            "config/*.yaml",
            "templates/*.html",
        ],
    },
    
    # Entry points for command line tools
    entry_points={
        "console_scripts": [
            "lavoisier-validate=validation.cli:main",
            "lavoisier-benchmark=validation.experiments.experiment_runner:main",
            "lavoisier-compare=validation.experiments.comparison_study:main",
        ],
    },
    
    # Classification
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
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    
    # Keywords for discoverability
    keywords=[
        "mass-spectrometry", "metabolomics", "bioinformatics", 
        "validation", "benchmarking", "machine-learning",
        "computer-vision", "s-entropy", "stellas"
    ],
    
    # License
    license="MIT",
    
    # Zip safety
    zip_safe=False,
)

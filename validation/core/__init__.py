"""
Lavoisier Validation Framework - Core Components

This module provides the core validation framework for comparing traditional mass spectrometry
methods, computer vision approaches, and S-Stellas framework transformations.
"""

from .base_validator import BaseValidator, ValidationResult
from .data_loader import MZMLDataLoader, DatasetInfo
from .metrics import PerformanceMetrics, ValidationMetrics
from .benchmarking import BenchmarkRunner, ComparisonStudy


__all__ = [
    'BaseValidator',
    'ValidationResult', 
    'MZMLDataLoader',
    'DatasetInfo',
    'PerformanceMetrics',
    'ValidationMetrics',
    'BenchmarkRunner',
    'ComparisonStudy',

]

__version__ = '1.0.0'

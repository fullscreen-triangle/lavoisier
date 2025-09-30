"""
Lavoisier Validation Framework - Core Components

This module provides the core validation framework for comparing traditional mass spectrometry
methods, computer vision approaches, and S-Stellas framework transformations.
"""

from .base_validator import BaseValidator, ValidationResult
from .data_loader import MZMLDataLoader, DatasetInfo
from .metrics import PerformanceMetrics, ValidationMetrics
from .benchmarking import BenchmarkRunner, ComparisonStudy
from .report_generator import ResultsReport, ComparisonReport

__all__ = [
    'BaseValidator',
    'ValidationResult', 
    'MZMLDataLoader',
    'DatasetInfo',
    'PerformanceMetrics',
    'ValidationMetrics',
    'BenchmarkRunner',
    'ComparisonStudy',
    'ResultsReport',
    'ComparisonReport'
]

__version__ = '1.0.0'

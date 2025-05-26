"""
Statistical Validation Module

Provides comprehensive statistical testing and validation tools for comparing
the numerical and visual pipelines.
"""

from .hypothesis_testing import HypothesisTestSuite
from .effect_size import EffectSizeCalculator
from .statistical_validator import StatisticalValidator
from .bias_detection import BiasDetector

__all__ = [
    "HypothesisTestSuite",
    "EffectSizeCalculator",
    "StatisticalValidator", 
    "BiasDetector"
] 
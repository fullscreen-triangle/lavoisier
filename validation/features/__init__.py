"""
Feature Extraction Comparison Module

Provides comprehensive tools for comparing feature extraction between
numerical and visual pipelines, including PCA, t-SNE, clustering,
and information content analysis.
"""

from .feature_comparator import FeatureExtractorComparator
from .information_content import InformationContentAnalyzer
from .dimensionality_reduction import DimensionalityReducer
from .clustering_validator import ClusteringValidator

__all__ = [
    "FeatureExtractorComparator",
    "InformationContentAnalyzer",
    "DimensionalityReducer",
    "ClusteringValidator"
] 
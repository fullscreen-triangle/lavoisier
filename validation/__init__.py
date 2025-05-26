"""
Lavoisier Pipeline Validation Module

This module provides comprehensive validation and comparison tools for evaluating
the performance of Lavoisier's visual pipeline against the traditional numerical pipeline.

The validation framework includes:
- Statistical validation and hypothesis testing
- Performance benchmarking and efficiency analysis
- Data quality and completeness assessment
- Feature extraction comparison
- Computer vision method evaluation
- Annotation and identification performance analysis
"""

from .statistics import (
    HypothesisTestSuite,
    EffectSizeCalculator,
    StatisticalValidator,
    BiasDetector
)

from .performance import (
    PerformanceBenchmark,
    EfficiencyAnalyzer,
    ScalabilityTester,
    ResourceMonitor
)

from .quality import (
    DataQualityAssessor,
    FidelityAnalyzer,
    IntegrityChecker,
    QualityMetrics
)

from .completeness import (
    CompletenessAnalyzer,
    CoverageAssessment,
    MissingDataDetector,
    ProcessingValidator
)

from .features import (
    FeatureExtractorComparator,
    InformationContentAnalyzer,
    DimensionalityReducer,
    ClusteringValidator
)

from .vision import (
    ComputerVisionValidator,
    ImageQualityAssessor,
    VideoAnalyzer
)

from .annotation import (
    AnnotationPerformanceEvaluator,
    CompoundIdentificationValidator,
    DatabaseSearchAnalyzer,
    ConfidenceScoreValidator
)

__version__ = "1.0.0"
__author__ = "Lavoisier Development Team"

__all__ = [
    # Statistics
    "HypothesisTestSuite",
    "EffectSizeCalculator", 
    "StatisticalValidator",
    "BiasDetector",
    
    # Performance
    "PerformanceBenchmark",
    "EfficiencyAnalyzer",
    "ScalabilityTester",
    "ResourceMonitor",
    
    # Quality
    "DataQualityAssessor",
    "FidelityAnalyzer",
    "IntegrityChecker",
    "QualityMetrics",
    
    # Completeness
    "CompletenessAnalyzer",
    "CoverageAssessment",
    "MissingDataDetector",
    "ProcessingValidator",
    
    # Features
    "FeatureExtractorComparator",
    "InformationContentAnalyzer",
    "DimensionalityReducer",
    "ClusteringValidator",
    
    # Vision
    "ComputerVisionValidator",
    "ImageQualityAssessor",
    "VideoAnalyzer",
    
    # Annotation
    "AnnotationPerformanceEvaluator",
    "CompoundIdentificationValidator",
    "DatabaseSearchAnalyzer",
    "ConfidenceScoreValidator"
]

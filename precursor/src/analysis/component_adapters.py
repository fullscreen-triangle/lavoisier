"""
Component Adapters
==================

Adapters that wrap existing analysis scripts to conform to the AnalysisComponent interface.
These adapters enable surgical injection of legacy code into pipelines without modification.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from pathlib import Path

from .analysis_component import (
    AnalysisComponent,
    AnalysisResult,
    AnalysisCategory,
    ComponentStatus,
    register_component
)

# Import existing analysis modules
from .annotation.annotation_performance_evaluator import AnnotationPerformanceEvaluator
from .annotation.compound_identification import CompoundIdentificationAnalyzer
from .annotation.confidence_score_validator import ConfidenceScoreValidator
from .annotation.database_search_analyzer import DatabaseSearchAnalyzer

from .features.clustering_validator import ClusteringValidator
from .features.dimensionality_reducer import DimensionalityReducer
from .features.feature_comparator import FeatureComparator
from .features.information_content_analyzer import InformationContentAnalyzer

from .quality.data_quality import DataQualityAssessor
from .quality.fidelity_analyzer import FidelityAnalyzer
from .quality.integrity_checker import IntegrityChecker
from .quality.quality_metrics import QualityMetricsCalculator

from .completeness.completeness_analyzer import CompletenessAnalyzer
from .completeness.coverage_assessment import CoverageAssessor
from .completeness.missing_data_detector import MissingDataDetector
from .completeness.processing_validator import ProcessingValidator

from .statistical.statistical_validator import StatisticalValidator
from .statistical.hypothesis_testing import HypothesisTestSuite
from .statistical.effect_size import EffectSizeCalculator
from .statistical.bias_detection import BiasDetector


# ============================================================================
# ANNOTATION COMPONENTS
# ============================================================================

@register_component("annotation_performance")
class AnnotationPerformanceComponent(AnalysisComponent):
    """Adapter for AnnotationPerformanceEvaluator"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="annotation_performance",
            category=AnalysisCategory.ANNOTATION,
            description="Evaluate annotation performance with classification metrics",
            config=config
        )
        self.evaluator = AnnotationPerformanceEvaluator()

    def execute(self, data: Any, **kwargs) -> AnalysisResult:
        start_time = time.perf_counter()

        try:
            # Expect data to be dict with 'true_labels' and 'predicted_labels'
            if isinstance(data, dict):
                true_labels = data.get('true_labels')
                predicted_labels = data.get('predicted_labels')
            else:
                true_labels = kwargs.get('true_labels')
                predicted_labels = kwargs.get('predicted_labels')

            # Execute original evaluator
            results = self.evaluator.evaluate_classification_performance(
                true_labels, predicted_labels
            )

            # Convert to standardized format
            metrics = {r.metric_name: r.value for r in results}
            interpretations = {r.metric_name: r.interpretation for r in results}

            self._execution_time = time.perf_counter() - start_time

            self.result = self._create_result(
                status=ComponentStatus.COMPLETED,
                metrics=metrics,
                interpretations=interpretations,
                metadata={'num_samples': len(true_labels)}
            )

        except Exception as e:
            self._execution_time = time.perf_counter() - start_time
            self.result = self._create_result(
                status=ComponentStatus.FAILED,
                metrics={},
                interpretations={},
                error_message=str(e)
            )

        return self.result


@register_component("compound_identification")
class CompoundIdentificationComponent(AnalysisComponent):
    """Adapter for CompoundIdentificationAnalyzer"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="compound_identification",
            category=AnalysisCategory.ANNOTATION,
            description="Analyze compound identification results",
            config=config
        )
        self.analyzer = CompoundIdentificationAnalyzer()

    def execute(self, data: Any, **kwargs) -> AnalysisResult:
        start_time = time.perf_counter()

        try:
            # Execute analyzer
            results = self.analyzer.analyze_identification_results(data, **kwargs)

            # Convert to standardized format
            metrics = {
                'total_identified': results.get('total_identified', 0),
                'identification_rate': results.get('identification_rate', 0.0),
                'average_confidence': results.get('average_confidence', 0.0)
            }

            interpretations = {
                'summary': results.get('summary', 'Identification analysis complete')
            }

            self._execution_time = time.perf_counter() - start_time

            self.result = self._create_result(
                status=ComponentStatus.COMPLETED,
                metrics=metrics,
                interpretations=interpretations,
                metadata=results
            )

        except Exception as e:
            self._execution_time = time.perf_counter() - start_time
            self.result = self._create_result(
                status=ComponentStatus.FAILED,
                metrics={},
                interpretations={},
                error_message=str(e)
            )

        return self.result


# ============================================================================
# FEATURE COMPONENTS
# ============================================================================

@register_component("clustering_validation")
class ClusteringValidationComponent(AnalysisComponent):
    """Adapter for ClusteringValidator"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="clustering_validation",
            category=AnalysisCategory.FEATURES,
            description="Validate clustering quality and find optimal clusters",
            config=config
        )
        self.validator = ClusteringValidator()

    def execute(self, data: Any, **kwargs) -> AnalysisResult:
        start_time = time.perf_counter()

        try:
            # Expect features array
            features = data if isinstance(data, np.ndarray) else data.get('features')

            n_clusters_range = self.config.get('n_clusters_range', (2, 10))

            # Execute validator
            results = self.validator.validate_kmeans_clustering(
                features, n_clusters_range
            )

            # Find optimal
            optimal = self.validator.find_optimal_clusters(features)

            # Convert to standardized format
            metrics = {
                'optimal_clusters': optimal.n_clusters,
                'optimal_silhouette': optimal.silhouette_score,
                'optimal_inertia': optimal.inertia
            }

            interpretations = {
                'optimal': optimal.interpretation,
                'all_results': [r.interpretation for r in results]
            }

            self._execution_time = time.perf_counter() - start_time

            self.result = self._create_result(
                status=ComponentStatus.COMPLETED,
                metrics=metrics,
                interpretations=interpretations,
                metadata={'all_clustering_results': [r.__dict__ for r in results]}
            )

        except Exception as e:
            self._execution_time = time.perf_counter() - start_time
            self.result = self._create_result(
                status=ComponentStatus.FAILED,
                metrics={},
                interpretations={},
                error_message=str(e)
            )

        return self.result


@register_component("feature_comparison")
class FeatureComparisonComponent(AnalysisComponent):
    """Adapter for FeatureComparator"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="feature_comparison",
            category=AnalysisCategory.FEATURES,
            description="Compare features across different methods",
            config=config
        )
        self.comparator = FeatureComparator()

    def execute(self, data: Any, **kwargs) -> AnalysisResult:
        start_time = time.perf_counter()

        try:
            # Expect dict with 'method1_features' and 'method2_features'
            method1_features = data.get('method1_features')
            method2_features = data.get('method2_features')

            # Execute comparison
            comparison = self.comparator.compare_features(
                method1_features, method2_features
            )

            metrics = {
                'correlation': comparison.get('correlation', 0.0),
                'similarity': comparison.get('similarity', 0.0),
                'difference': comparison.get('difference', 0.0)
            }

            interpretations = {
                'summary': comparison.get('interpretation', 'Feature comparison complete')
            }

            self._execution_time = time.perf_counter() - start_time

            self.result = self._create_result(
                status=ComponentStatus.COMPLETED,
                metrics=metrics,
                interpretations=interpretations,
                metadata=comparison
            )

        except Exception as e:
            self._execution_time = time.perf_counter() - start_time
            self.result = self._create_result(
                status=ComponentStatus.FAILED,
                metrics={},
                interpretations={},
                error_message=str(e)
            )

        return self.result


# ============================================================================
# QUALITY COMPONENTS
# ============================================================================

@register_component("data_quality")
class DataQualityComponent(AnalysisComponent):
    """Adapter for DataQualityAssessor"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="data_quality",
            category=AnalysisCategory.QUALITY,
            description="Assess data quality across multiple dimensions",
            config=config
        )
        quality_thresholds = self.config.get('quality_thresholds')
        self.assessor = DataQualityAssessor(quality_thresholds=quality_thresholds)

    def execute(self, data: Any, **kwargs) -> AnalysisResult:
        start_time = time.perf_counter()

        try:
            # Assess completeness
            completeness = self.assessor.assess_completeness(data)

            # Assess consistency
            consistency = self.assessor.assess_consistency(data)

            # Assess accuracy (if reference provided)
            reference = kwargs.get('reference_data')
            if reference is not None:
                accuracy = self.assessor.assess_accuracy(data, reference)
            else:
                accuracy = None

            # Compile metrics
            metrics = {
                'completeness_score': completeness.score,
                'completeness_passed': float(completeness.passed),
                'consistency_score': consistency.score,
                'consistency_passed': float(consistency.passed)
            }

            if accuracy:
                metrics['accuracy_score'] = accuracy.score
                metrics['accuracy_passed'] = float(accuracy.passed)

            interpretations = {
                'completeness': completeness.interpretation,
                'consistency': consistency.interpretation
            }

            if accuracy:
                interpretations['accuracy'] = accuracy.interpretation

            self._execution_time = time.perf_counter() - start_time

            self.result = self._create_result(
                status=ComponentStatus.COMPLETED,
                metrics=metrics,
                interpretations=interpretations,
                metadata={
                    'completeness_details': completeness.details,
                    'consistency_details': consistency.details
                }
            )

        except Exception as e:
            self._execution_time = time.perf_counter() - start_time
            self.result = self._create_result(
                status=ComponentStatus.FAILED,
                metrics={},
                interpretations={},
                error_message=str(e)
            )

        return self.result


# ============================================================================
# STATISTICAL COMPONENTS
# ============================================================================

@register_component("statistical_validation")
class StatisticalValidationComponent(AnalysisComponent):
    """Adapter for StatisticalValidator"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="statistical_validation",
            category=AnalysisCategory.STATISTICAL,
            description="Comprehensive statistical validation with hypothesis testing",
            config=config
        )
        alpha = self.config.get('alpha', 0.05)
        confidence_level = self.config.get('confidence_level', 0.95)
        self.validator = StatisticalValidator(alpha=alpha, confidence_level=confidence_level)

    def execute(self, data: Any, **kwargs) -> AnalysisResult:
        start_time = time.perf_counter()

        try:
            # Expect dict with 'numerical_data' and 'visual_data'
            numerical_data = data.get('numerical_data')
            visual_data = data.get('visual_data')
            combined_data = data.get('combined_data')
            performance_metrics = data.get('performance_metrics')

            # Execute validation
            report = self.validator.validate_pipelines(
                numerical_data,
                visual_data,
                combined_data,
                performance_metrics
            )

            # Extract metrics
            metrics = {
                'num_hypothesis_tests': len(report.hypothesis_results),
                'num_significant': sum(1 for r in report.hypothesis_results if r.significant),
                'num_effect_sizes': len(report.effect_size_results),
                'num_biases_detected': len([b for b in report.bias_results if b.bias_detected])
            }

            interpretations = {
                'overall_conclusion': report.overall_conclusion,
                'recommendations': '; '.join(report.recommendations)
            }

            self._execution_time = time.perf_counter() - start_time

            self.result = self._create_result(
                status=ComponentStatus.COMPLETED,
                metrics=metrics,
                interpretations=interpretations,
                metadata={
                    'summary_statistics': report.summary_statistics,
                    'hypothesis_results': [r.__dict__ for r in report.hypothesis_results],
                    'effect_size_results': [r.__dict__ for r in report.effect_size_results],
                    'bias_results': [r.__dict__ for r in report.bias_results]
                }
            )

        except Exception as e:
            self._execution_time = time.perf_counter() - start_time
            self.result = self._create_result(
                status=ComponentStatus.FAILED,
                metrics={},
                interpretations={},
                error_message=str(e)
            )

        return self.result


# ============================================================================
# COMPLETENESS COMPONENTS
# ============================================================================

@register_component("completeness_analysis")
class CompletenessAnalysisComponent(AnalysisComponent):
    """Adapter for CompletenessAnalyzer"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="completeness_analysis",
            category=AnalysisCategory.COMPLETENESS,
            description="Analyze data completeness and coverage",
            config=config
        )
        self.analyzer = CompletenessAnalyzer()

    def execute(self, data: Any, **kwargs) -> AnalysisResult:
        start_time = time.perf_counter()

        try:
            # Execute analyzer
            results = self.analyzer.analyze_completeness(data, **kwargs)

            metrics = {
                'completeness_score': results.get('completeness_score', 0.0),
                'coverage_percentage': results.get('coverage_percentage', 0.0),
                'missing_count': results.get('missing_count', 0)
            }

            interpretations = {
                'summary': results.get('interpretation', 'Completeness analysis complete')
            }

            self._execution_time = time.perf_counter() - start_time

            self.result = self._create_result(
                status=ComponentStatus.COMPLETED,
                metrics=metrics,
                interpretations=interpretations,
                metadata=results
            )

        except Exception as e:
            self._execution_time = time.perf_counter() - start_time
            self.result = self._create_result(
                status=ComponentStatus.FAILED,
                metrics={},
                interpretations={},
                error_message=str(e)
            )

        return self.result


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_standard_pipeline() -> 'AnalysisPipeline':
    """
    Create a standard analysis pipeline with commonly used components.

    Returns:
        Pre-configured AnalysisPipeline
    """
    from .analysis_component import AnalysisPipeline

    pipeline = AnalysisPipeline(name="StandardAnalysisPipeline")

    # Add standard components in order
    pipeline.add_components([
        DataQualityComponent(),
        CompletenessAnalysisComponent(),
        ClusteringValidationComponent(),
        FeatureComparisonComponent(),
        AnnotationPerformanceComponent()
    ])

    return pipeline


def create_statistical_pipeline() -> 'AnalysisPipeline':
    """
    Create a statistical validation pipeline.

    Returns:
        Pre-configured AnalysisPipeline for statistical validation
    """
    from .analysis_component import AnalysisPipeline

    pipeline = AnalysisPipeline(name="StatisticalValidationPipeline")

    pipeline.add_components([
        DataQualityComponent(),
        StatisticalValidationComponent()
    ])

    return pipeline


def inject_quality_checks(pipeline: 'AnalysisPipeline') -> 'AnalysisPipeline':
    """
    Surgically inject quality check components into existing pipeline.

    Args:
        pipeline: Target pipeline

    Returns:
        Modified pipeline with quality checks
    """
    # Inject at beginning
    pipeline.inject_at(0, DataQualityComponent())

    # Inject at end
    pipeline.add_component(CompletenessAnalysisComponent())

    return pipeline

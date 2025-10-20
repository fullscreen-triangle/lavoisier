"""
Lavoisier Analysis Bundles
===========================

Reusable, composable analysis components for surgical injection into pipelines.

Public API:
-----------

Base Classes:
    - AnalysisComponent: Base class for all components
    - AnalysisResult: Standardized result container
    - AnalysisPipeline: Composer for chaining components
    - ComponentRegistry: Discovery and injection system

Bundles:
    - QualityBundle: Data quality assessment
    - AnnotationBundle: Annotation analysis
    - FeatureBundle: Feature analysis
    - StatisticalBundle: Statistical validation
    - CompletenessBundle: Completeness analysis
    - CompleteBundle: All categories combined
    - CustomBundle: Build your own

Components (Auto-registered):
    - DataQualityComponent
    - AnnotationPerformanceComponent
    - CompoundIdentificationComponent
    - ClusteringValidationComponent
    - FeatureComparisonComponent
    - StatisticalValidationComponent
    - CompletenessAnalysisComponent

Utilities:
    - PipelineInjector: Surgical injection system
    - global_registry: Global component registry
    - quick_quality_check: Quick quality assessment
    - quick_annotation_eval: Quick annotation evaluation
    - quick_statistical_validation: Quick statistical validation
    - create_injectable_bundle: Create bundle from categories

Enums:
    - AnalysisCategory: Component categories
    - ComponentStatus: Execution status

Usage:
------

    from precursor.src.analysis import QualityBundle, PipelineInjector

    # Create and run bundle
    bundle = QualityBundle()
    pipeline = bundle.build_pipeline()
    results = pipeline.execute_sequential(data)

    # Surgical injection
    injector = PipelineInjector(your_pipeline)
    injector.inject_bundle(bundle)
    modified_pipeline = injector.get_pipeline()

See ANALYSIS_BUNDLES_README.md for comprehensive documentation.
"""

# Base classes and interfaces
from .analysis_component import (
    AnalysisComponent,
    AnalysisResult,
    AnalysisPipeline,
    ComponentRegistry,
    AnalysisCategory,
    ComponentStatus,
    global_registry,
    register_component
)

# Component adapters (auto-registers components)
from .component_adapters import (
    # Annotation
    AnnotationPerformanceComponent,
    CompoundIdentificationComponent,

    # Features
    ClusteringValidationComponent,
    FeatureComparisonComponent,

    # Quality
    DataQualityComponent,

    # Statistical
    StatisticalValidationComponent,

    # Completeness
    CompletenessAnalysisComponent,

    # Helper functions
    create_standard_pipeline,
    create_statistical_pipeline,
    inject_quality_checks
)

# Bundles and injection
from .bundles import (
    # Bundles
    AnalysisBundle,
    QualityBundle,
    AnnotationBundle,
    FeatureBundle,
    StatisticalBundle,
    CompletenessBundle,
    CompleteBundle,
    CustomBundle,

    # Injector
    PipelineInjector,

    # Convenience functions
    quick_quality_check,
    quick_annotation_eval,
    quick_statistical_validation,
    create_injectable_bundle
)

# Version
__version__ = "1.0.0"

# Public API
__all__ = [
    # Base classes
    "AnalysisComponent",
    "AnalysisResult",
    "AnalysisPipeline",
    "ComponentRegistry",
    "AnalysisCategory",
    "ComponentStatus",

    # Registry
    "global_registry",
    "register_component",

    # Bundles
    "AnalysisBundle",
    "QualityBundle",
    "AnnotationBundle",
    "FeatureBundle",
    "StatisticalBundle",
    "CompletenessBundle",
    "CompleteBundle",
    "CustomBundle",

    # Components
    "AnnotationPerformanceComponent",
    "CompoundIdentificationComponent",
    "ClusteringValidationComponent",
    "FeatureComparisonComponent",
    "DataQualityComponent",
    "StatisticalValidationComponent",
    "CompletenessAnalysisComponent",

    # Utilities
    "PipelineInjector",
    "create_standard_pipeline",
    "create_statistical_pipeline",
    "inject_quality_checks",
    "quick_quality_check",
    "quick_annotation_eval",
    "quick_statistical_validation",
    "create_injectable_bundle",
]


def get_version():
    """Get version string"""
    return __version__


def list_all_components():
    """List all registered components"""
    return global_registry.list_components()


def list_all_bundles():
    """List all available bundles"""
    return [
        "QualityBundle",
        "AnnotationBundle",
        "FeatureBundle",
        "StatisticalBundle",
        "CompletenessBundle",
        "CompleteBundle",
        "CustomBundle"
    ]


def get_bundle_info(bundle_name: str) -> str:
    """Get information about a bundle"""
    bundle_info = {
        "QualityBundle": "Comprehensive data quality assessment",
        "AnnotationBundle": "Complete annotation analysis",
        "FeatureBundle": "Complete feature analysis",
        "StatisticalBundle": "Complete statistical validation",
        "CompletenessBundle": "Complete completeness analysis",
        "CompleteBundle": "All categories combined",
        "CustomBundle": "Build your own bundle"
    }
    return bundle_info.get(bundle_name, "Unknown bundle")


# Initialize module
print(f"[Lavoisier Analysis Bundles v{__version__}] Loaded")
print(f"  Registered components: {len(list_all_components())}")
print(f"  Available bundles: {len(list_all_bundles())}")

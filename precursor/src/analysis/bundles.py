"""
Analysis Bundles
================

Pre-configured bundles of analysis components for common use cases.
Enables surgical injection of complete analysis workflows into pipelines.

Bundles:
- QualityBundle: Data quality and integrity checks
- AnnotationBundle: Annotation performance and identification
- FeatureBundle: Feature analysis and comparison
- StatisticalBundle: Statistical validation and hypothesis testing
- CompleteBundle: Comprehensive analysis across all categories
- CustomBundle: Build your own bundle
"""

from typing import List, Dict, Any, Optional
from .analysis_component import AnalysisPipeline, AnalysisComponent, AnalysisCategory, global_registry
from .component_adapters import *


class AnalysisBundle:
    """
    Base class for analysis bundles.

    A bundle is a pre-configured collection of components for a specific purpose.
    """

    def __init__(self, name: str, description: str):
        """
        Initialize analysis bundle.

        Args:
            name: Bundle name
            description: Bundle description
        """
        self.name = name
        self.description = description
        self.components: List[AnalysisComponent] = []
        self.pipeline: Optional[AnalysisPipeline] = None

    def add_component(self, component: AnalysisComponent):
        """Add component to bundle"""
        self.components.append(component)
        return self

    def build_pipeline(self) -> AnalysisPipeline:
        """Build pipeline from components"""
        self.pipeline = AnalysisPipeline(name=f"{self.name}_Pipeline")
        self.pipeline.add_components(self.components)
        return self.pipeline

    def inject_into(self, target_pipeline: AnalysisPipeline, position: str = 'end') -> AnalysisPipeline:
        """
        Surgically inject this bundle into a target pipeline.

        Args:
            target_pipeline: Target pipeline
            position: Where to inject ('start', 'end', or int index)

        Returns:
            Modified pipeline
        """
        if position == 'start':
            for i, component in enumerate(self.components):
                target_pipeline.inject_at(i, component)
        elif position == 'end':
            target_pipeline.add_components(self.components)
        elif isinstance(position, int):
            for i, component in enumerate(self.components):
                target_pipeline.inject_at(position + i, component)

        return target_pipeline

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', components={len(self.components)})>"


class QualityBundle(AnalysisBundle):
    """
    Quality Assessment Bundle

    Comprehensive data quality checks including:
    - Data quality assessment
    - Integrity checking
    - Fidelity analysis
    - Quality metrics calculation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize quality bundle.

        Args:
            config: Configuration for quality thresholds
        """
        super().__init__(
            name="QualityBundle",
            description="Comprehensive data quality assessment"
        )

        config = config or {}

        # Add quality components
        self.add_component(DataQualityComponent(config=config.get('data_quality')))

        # Note: IntegrityChecker, FidelityAnalyzer, QualityMetricsCalculator adapters would go here
        # Simplified for now with DataQualityComponent


class AnnotationBundle(AnalysisBundle):
    """
    Annotation Analysis Bundle

    Complete annotation assessment including:
    - Performance evaluation
    - Compound identification
    - Confidence score validation
    - Database search analysis
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize annotation bundle.

        Args:
            config: Configuration for annotation analysis
        """
        super().__init__(
            name="AnnotationBundle",
            description="Comprehensive annotation analysis"
        )

        config = config or {}

        # Add annotation components
        self.add_component(AnnotationPerformanceComponent(config=config.get('performance')))
        self.add_component(CompoundIdentificationComponent(config=config.get('identification')))


class FeatureBundle(AnalysisBundle):
    """
    Feature Analysis Bundle

    Complete feature analysis including:
    - Clustering validation
    - Dimensionality reduction
    - Feature comparison
    - Information content analysis
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature bundle.

        Args:
            config: Configuration for feature analysis
        """
        super().__init__(
            name="FeatureBundle",
            description="Comprehensive feature analysis"
        )

        config = config or {}

        # Add feature components
        self.add_component(ClusteringValidationComponent(config=config.get('clustering')))
        self.add_component(FeatureComparisonComponent(config=config.get('comparison')))


class StatisticalBundle(AnalysisBundle):
    """
    Statistical Validation Bundle

    Complete statistical analysis including:
    - Hypothesis testing
    - Effect size calculation
    - Bias detection
    - Comprehensive validation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize statistical bundle.

        Args:
            config: Configuration for statistical analysis
        """
        super().__init__(
            name="StatisticalBundle",
            description="Comprehensive statistical validation"
        )

        config = config or {}

        # Add statistical components
        self.add_component(StatisticalValidationComponent(config=config.get('validation')))


class CompletenessBundle(AnalysisBundle):
    """
    Completeness Analysis Bundle

    Complete completeness assessment including:
    - Completeness analysis
    - Coverage assessment
    - Missing data detection
    - Processing validation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize completeness bundle.

        Args:
            config: Configuration for completeness analysis
        """
        super().__init__(
            name="CompletenessBundle",
            description="Comprehensive completeness analysis"
        )

        config = config or {}

        # Add completeness components
        self.add_component(CompletenessAnalysisComponent(config=config.get('completeness')))


class CompleteBundle(AnalysisBundle):
    """
    Complete Analysis Bundle

    Combines all analysis categories for comprehensive validation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize complete bundle.

        Args:
            config: Configuration for all analysis types
        """
        super().__init__(
            name="CompleteBundle",
            description="Complete analysis across all categories"
        )

        config = config or {}

        # Add components from all categories in logical order
        # 1. Quality checks first
        self.add_component(DataQualityComponent(config=config.get('quality')))

        # 2. Completeness assessment
        self.add_component(CompletenessAnalysisComponent(config=config.get('completeness')))

        # 3. Feature analysis
        self.add_component(ClusteringValidationComponent(config=config.get('clustering')))
        self.add_component(FeatureComparisonComponent(config=config.get('features')))

        # 4. Annotation evaluation
        self.add_component(AnnotationPerformanceComponent(config=config.get('annotation')))
        self.add_component(CompoundIdentificationComponent(config=config.get('identification')))

        # 5. Statistical validation last
        self.add_component(StatisticalValidationComponent(config=config.get('statistical')))


class CustomBundle(AnalysisBundle):
    """
    Custom Analysis Bundle

    Build your own bundle from registered components.
    """

    def __init__(
        self,
        name: str,
        component_names: List[str],
        configs: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize custom bundle.

        Args:
            name: Bundle name
            component_names: List of component names from registry
            configs: Configurations for each component
        """
        super().__init__(
            name=name,
            description=f"Custom bundle with {len(component_names)} components"
        )

        configs = configs or {}

        # Add components from registry
        for comp_name in component_names:
            instance = global_registry.create_instance(comp_name, configs.get(comp_name))
            if instance:
                self.add_component(instance)


# ============================================================================
# PIPELINE INJECTOR
# ============================================================================

class PipelineInjector:
    """
    Surgical injection system for adding analysis components to pipelines.

    Enables:
    - Inject before/after specific steps
    - Inject at arbitrary positions
    - Conditional injection
    - Batch injection
    """

    def __init__(self, target_pipeline: AnalysisPipeline):
        """
        Initialize pipeline injector.

        Args:
            target_pipeline: Pipeline to inject into
        """
        self.pipeline = target_pipeline
        self.injection_log: List[Dict[str, Any]] = []

    def inject_bundle(
        self,
        bundle: AnalysisBundle,
        position: str = 'end'
    ) -> 'PipelineInjector':
        """
        Inject entire bundle into pipeline.

        Args:
            bundle: Bundle to inject
            position: Where to inject ('start', 'end', or component name)

        Returns:
            Self for chaining
        """
        bundle.inject_into(self.pipeline, position)

        self.injection_log.append({
            'type': 'bundle',
            'bundle_name': bundle.name,
            'position': position,
            'components_added': len(bundle.components)
        })

        return self

    def inject_component(
        self,
        component: AnalysisComponent,
        before: Optional[str] = None,
        after: Optional[str] = None,
        at: Optional[int] = None
    ) -> 'PipelineInjector':
        """
        Inject single component into pipeline.

        Args:
            component: Component to inject
            before: Inject before this component name
            after: Inject after this component name
            at: Inject at this index

        Returns:
            Self for chaining
        """
        if before:
            self.pipeline.inject_before(before, component)
            position = f"before:{before}"
        elif after:
            self.pipeline.inject_after(after, component)
            position = f"after:{after}"
        elif at is not None:
            self.pipeline.inject_at(at, component)
            position = f"at:{at}"
        else:
            self.pipeline.add_component(component)
            position = "end"

        self.injection_log.append({
            'type': 'component',
            'component_name': component.name,
            'position': position
        })

        return self

    def inject_if(
        self,
        condition: bool,
        component: AnalysisComponent,
        position: str = 'end'
    ) -> 'PipelineInjector':
        """
        Conditionally inject component.

        Args:
            condition: Inject only if True
            component: Component to inject
            position: Where to inject

        Returns:
            Self for chaining
        """
        if condition:
            if position == 'end':
                self.pipeline.add_component(component)
            else:
                self.pipeline.inject_at(int(position), component)

            self.injection_log.append({
                'type': 'conditional',
                'component_name': component.name,
                'condition_met': True
            })
        else:
            self.injection_log.append({
                'type': 'conditional',
                'component_name': component.name,
                'condition_met': False
            })

        return self

    def inject_quality_gates(self) -> 'PipelineInjector':
        """
        Inject quality gates at strategic positions.

        Returns:
            Self for chaining
        """
        # Quality check at start
        self.inject_component(
            DataQualityComponent(config={'name': 'input_quality'}),
            at=0
        )

        # Quality check at end
        self.inject_component(
            DataQualityComponent(config={'name': 'output_quality'}),
            at=len(self.pipeline.components)
        )

        return self

    def get_pipeline(self) -> AnalysisPipeline:
        """Get modified pipeline"""
        return self.pipeline

    def get_injection_log(self) -> List[Dict[str, Any]]:
        """Get log of all injections"""
        return self.injection_log


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_quality_check(data: Any, **kwargs) -> Dict[str, Any]:
    """
    Quick quality check using QualityBundle.

    Args:
        data: Data to check
        **kwargs: Additional parameters

    Returns:
        Dictionary of quality metrics
    """
    bundle = QualityBundle()
    pipeline = bundle.build_pipeline()
    results = pipeline.execute_sequential(data, **kwargs)

    return {
        'passed': all(r.status == ComponentStatus.COMPLETED for r in results),
        'metrics': {r.component_name: r.metrics for r in results},
        'interpretations': {r.component_name: r.interpretations for r in results}
    }


def quick_annotation_eval(true_labels, predicted_labels) -> Dict[str, Any]:
    """
    Quick annotation evaluation using AnnotationBundle.

    Args:
        true_labels: Ground truth labels
        predicted_labels: Predicted labels

    Returns:
        Dictionary of annotation metrics
    """
    bundle = AnnotationBundle()
    pipeline = bundle.build_pipeline()

    data = {
        'true_labels': true_labels,
        'predicted_labels': predicted_labels
    }

    results = pipeline.execute_sequential(data)

    return {
        'metrics': {r.component_name: r.metrics for r in results},
        'interpretations': {r.component_name: r.interpretations for r in results}
    }


def quick_statistical_validation(numerical_data, visual_data, **kwargs) -> Dict[str, Any]:
    """
    Quick statistical validation using StatisticalBundle.

    Args:
        numerical_data: Numerical pipeline data
        visual_data: Visual pipeline data
        **kwargs: Additional parameters

    Returns:
        Dictionary of statistical validation results
    """
    bundle = StatisticalBundle()
    pipeline = bundle.build_pipeline()

    data = {
        'numerical_data': numerical_data,
        'visual_data': visual_data
    }
    data.update(kwargs)

    results = pipeline.execute_sequential(data)

    return {
        'metrics': {r.component_name: r.metrics for r in results},
        'interpretations': {r.component_name: r.interpretations for r in results},
        'metadata': {r.component_name: r.metadata for r in results}
    }


def create_injectable_bundle(
    categories: List[AnalysisCategory],
    config: Optional[Dict[str, Any]] = None
) -> AnalysisBundle:
    """
    Create a bundle from specified categories.

    Args:
        categories: List of analysis categories to include
        config: Configuration for components

    Returns:
        Custom bundle with specified categories
    """
    bundle = AnalysisBundle(
        name="InjectableBundle",
        description=f"Bundle with categories: {[c.value for c in categories]}"
    )

    config = config or {}

    for category in categories:
        if category == AnalysisCategory.QUALITY:
            bundle.add_component(DataQualityComponent(config=config.get('quality')))
        elif category == AnalysisCategory.ANNOTATION:
            bundle.add_component(AnnotationPerformanceComponent(config=config.get('annotation')))
        elif category == AnalysisCategory.FEATURES:
            bundle.add_component(ClusteringValidationComponent(config=config.get('features')))
        elif category == AnalysisCategory.STATISTICAL:
            bundle.add_component(StatisticalValidationComponent(config=config.get('statistical')))
        elif category == AnalysisCategory.COMPLETENESS:
            bundle.add_component(CompletenessAnalysisComponent(config=config.get('completeness')))

    return bundle

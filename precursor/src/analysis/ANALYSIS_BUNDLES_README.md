## Analysis Bundles System

### Overview

The **Analysis Bundles System** provides reusable, composable analysis components that can be **surgically injected** into any pipeline. This system transforms the 25+ individual analysis scripts into modular, plug-and-play components.

### Key Concepts

#### 1. Analysis Component

Base abstraction for all analysis modules. Every component:

- Has standardized `execute()` interface
- Returns `AnalysisResult` with metrics, interpretations, and metadata
- Can be configured independently
- Tracks execution time and status

#### 2. Analysis Bundle

Pre-configured collection of components for specific purposes:

- `QualityBundle`: Data quality and integrity checks
- `AnnotationBundle`: Annotation performance and identification
- `FeatureBundle`: Feature analysis and comparison
- `StatisticalBundle`: Statistical validation and hypothesis testing
- `CompletenessBundle`: Completeness and coverage assessment
- `CompleteBundle`: All categories combined
- `CustomBundle`: Build your own

#### 3. Analysis Pipeline

Composer for chaining components with:

- Sequential execution
- Parallel execution (planned)
- Conditional execution
- Result aggregation

#### 4. Pipeline Injector

Surgical injection system for adding analysis to existing pipelines:

- Inject before/after specific steps
- Inject at arbitrary positions
- Conditional injection
- Batch injection

### Quick Start

#### Basic Usage

```python
from precursor.src.analysis import QualityBundle, AnnotationBundle

# Create a bundle
quality_bundle = QualityBundle()

# Build pipeline from bundle
pipeline = quality_bundle.build_pipeline()

# Execute on your data
results = pipeline.execute_sequential(your_data)

# Access results
for result in results:
    print(f"{result.component_name}: {result.metrics}")
```

#### Surgical Injection

```python
from precursor.src.analysis import PipelineInjector, QualityBundle
from your_project import YourPipeline

# Your existing pipeline
your_pipeline = YourPipeline()

# Create injector
injector = PipelineInjector(your_pipeline)

# Inject quality checks
injector.inject_quality_gates()

# Inject feature analysis before annotation step
from precursor.src.analysis import ClusteringValidationComponent
injector.inject_component(
    ClusteringValidationComponent(),
    before="annotation_step"
)

# Get modified pipeline
modified_pipeline = injector.get_pipeline()

# Run with analysis
results = modified_pipeline.execute()
```

#### Quick Functions

```python
from precursor.src.analysis import (
    quick_quality_check,
    quick_annotation_eval,
    quick_statistical_validation
)

# Quick quality check
quality_results = quick_quality_check(data)
print(f"Quality passed: {quality_results['passed']}")

# Quick annotation evaluation
annotation_results = quick_annotation_eval(true_labels, predicted_labels)
print(f"Accuracy: {annotation_results['metrics']['annotation_performance']['Accuracy']}")

# Quick statistical validation
stat_results = quick_statistical_validation(numerical_data, visual_data)
print(f"Significant tests: {stat_results['metrics']['statistical_validation']['num_significant']}")
```

### Component Categories

#### Quality Components

- `DataQualityComponent`: Comprehensive data quality assessment
- `IntegrityCheckerComponent`: Data integrity verification
- `FidelityAnalyzerComponent`: Fidelity analysis
- `QualityMetricsComponent`: Quality metrics calculation

#### Annotation Components

- `AnnotationPerformanceComponent`: Classification performance metrics
- `CompoundIdentificationComponent`: Compound identification analysis
- `ConfidenceScoreComponent`: Confidence score validation
- `DatabaseSearchComponent`: Database search analysis

#### Feature Components

- `ClusteringValidationComponent`: Clustering quality and optimal k
- `DimensionalityReductionComponent`: Dimensionality reduction analysis
- `FeatureComparisonComponent`: Feature comparison across methods
- `InformationContentComponent`: Information content analysis

#### Statistical Components

- `StatisticalValidationComponent`: Comprehensive statistical validation
- `HypothesisTestingComponent`: Hypothesis testing suite
- `EffectSizeComponent`: Effect size calculation
- `BiasDetectionComponent`: Bias detection analysis

#### Completeness Components

- `CompletenessAnalysisComponent`: Completeness analysis
- `CoverageAssessmentComponent`: Coverage assessment
- `MissingDataDetectorComponent`: Missing data detection
- `ProcessingValidatorComponent`: Processing validation

### Advanced Usage

#### Custom Bundles

```python
from precursor.src.analysis import CustomBundle, AnalysisCategory

# Create custom bundle by component names
custom = CustomBundle(
    name="MyCustomBundle",
    component_names=[
        "data_quality",
        "clustering_validation",
        "annotation_performance"
    ],
    configs={
        "data_quality": {"quality_thresholds": {"completeness": 0.99}},
        "clustering_validation": {"n_clusters_range": (3, 15)}
    }
)

pipeline = custom.build_pipeline()
results = pipeline.execute_sequential(data)
```

#### Category-Based Bundles

```python
from precursor.src.analysis import create_injectable_bundle, AnalysisCategory

# Create bundle from categories
bundle = create_injectable_bundle(
    categories=[
        AnalysisCategory.QUALITY,
        AnalysisCategory.FEATURES,
        AnalysisCategory.STATISTICAL
    ],
    config={
        "quality": {"quality_thresholds": {"completeness": 0.95}},
        "features": {"n_clusters_range": (2, 10)},
        "statistical": {"alpha": 0.01}
    }
)
```

#### Conditional Execution

```python
from precursor.src.analysis import AnalysisPipeline, ComponentStatus

pipeline = AnalysisPipeline()
pipeline.add_components([...])

# Execute with condition
def continue_if_quality_passed(result):
    return result.status == ComponentStatus.COMPLETED and \
           result.metrics.get('completeness_score', 0) >= 0.95

results = pipeline.execute_conditional(
    data,
    condition=continue_if_quality_passed
)
```

#### Result Aggregation

```python
# Execute pipeline
results = pipeline.execute_sequential(data)

# Get summary
summary = pipeline.get_summary()
print(f"Total execution time: {summary['total_execution_time']:.2f}s")
print(f"Components executed: {summary['executed_components']}")
print(f"Success rate: {summary['status_counts']['completed'] / summary['executed_components']:.1%}")

# Export results
pipeline.export_results("analysis_results.json")
```

### Integration with Existing Pipelines

#### Lavoisier Validation Pipeline

```python
from validation.run_complete_validation import ValidationOrchestrator
from precursor.src.analysis import CompleteBundle, PipelineInjector

# Create validator
validator = ValidationOrchestrator()

# Inject complete analysis bundle
injector = PipelineInjector(validator.pipeline)
injector.inject_bundle(CompleteBundle(), position='end')

# Run validation with analysis
results = validator.run_validation()
```

#### Resonant Computation Engine

```python
from precursor.src.hardware.resonant_computation_engine import ResonantComputationEngine
from precursor.src.analysis import StatisticalBundle, FeatureBundle

engine = ResonantComputationEngine()

# Process with analysis
spectrum_data = {...}
results = await engine.process_experiment_as_bayesian_network(spectrum_data, process_fn)

# Inject analysis on results
injector = PipelineInjector(AnalysisPipeline())
injector.inject_bundle(StatisticalBundle())
injector.inject_bundle(FeatureBundle())

analysis_pipeline = injector.get_pipeline()
analysis_results = analysis_pipeline.execute_sequential(results['evidence_network'])
```

#### Custom Pipeline Integration

```python
class MyCustomPipeline:
    def __init__(self):
        self.steps = []
        self.analysis_pipeline = None

    def add_analysis(self, bundle):
        """Add analysis bundle to pipeline"""
        self.analysis_pipeline = bundle.build_pipeline()

    def run(self, data):
        # Run main pipeline
        main_results = self._run_main_steps(data)

        # Run analysis if configured
        if self.analysis_pipeline:
            analysis_results = self.analysis_pipeline.execute_sequential(main_results)
            return {
                'main_results': main_results,
                'analysis_results': analysis_results
            }

        return {'main_results': main_results}

# Usage
pipeline = MyCustomPipeline()
pipeline.add_analysis(CompleteBundle())
results = pipeline.run(data)
```

### Component Registry

All components are automatically registered in the global registry:

```python
from precursor.src.analysis import global_registry

# List all components
all_components = global_registry.list_components()

# List by category
quality_components = global_registry.list_components(AnalysisCategory.QUALITY)

# Create instance by name
component = global_registry.create_instance(
    "data_quality",
    config={"quality_thresholds": {...}}
)

# Inject into pipeline from registry
global_registry.inject_into_pipeline(
    pipeline,
    component_names=["data_quality", "clustering_validation"],
    configs={...}
)
```

### Creating Custom Components

#### Minimal Component

```python
from precursor.src.analysis import AnalysisComponent, AnalysisCategory, ComponentStatus, register_component

@register_component("my_custom_component")
class MyCustomComponent(AnalysisComponent):
    def __init__(self, config=None):
        super().__init__(
            name="my_custom_component",
            category=AnalysisCategory.CUSTOM,
            description="My custom analysis",
            config=config
        )

    def execute(self, data, **kwargs):
        import time
        start_time = time.perf_counter()

        try:
            # Your analysis logic
            result_value = self._my_analysis(data)

            self._execution_time = time.perf_counter() - start_time

            return self._create_result(
                status=ComponentStatus.COMPLETED,
                metrics={"my_metric": result_value},
                interpretations={"summary": f"Analysis complete: {result_value}"}
            )
        except Exception as e:
            self._execution_time = time.perf_counter() - start_time
            return self._create_result(
                status=ComponentStatus.FAILED,
                metrics={},
                interpretations={},
                error_message=str(e)
            )

    def _my_analysis(self, data):
        # Your custom logic
        return 42.0
```

#### Advanced Component with Visualization

```python
@register_component("advanced_component")
class AdvancedComponent(AnalysisComponent):
    def execute(self, data, **kwargs):
        import time
        import matplotlib.pyplot as plt

        start_time = time.perf_counter()

        # Analysis
        metrics = self._compute_metrics(data)

        # Visualization
        fig, ax = plt.subplots()
        ax.plot(data)

        self._execution_time = time.perf_counter() - start_time

        return self._create_result(
            status=ComponentStatus.COMPLETED,
            metrics=metrics,
            interpretations={"summary": "Advanced analysis complete"},
            visualizations={"plot": fig},
            metadata={"data_shape": data.shape}
        )
```

### Best Practices

1. **Always use bundles for common patterns**: Don't create individual components when a bundle exists
2. **Configure before building**: Set config before calling `build_pipeline()`
3. **Check status**: Always check `ComponentStatus` in results
4. **Use surgical injection**: Inject at specific points rather than rebuilding entire pipelines
5. **Export results**: Use `export_results()` for reproducibility
6. **Conditional execution**: Use conditions to skip unnecessary analysis
7. **Registry for discovery**: Use registry to discover available components
8. **Custom components**: Register custom components for reusability

### Performance Considerations

- Components track execution time automatically
- Parallel execution (coming soon) will speed up independent components
- Conditional execution skips unnecessary components
- Registry enables lazy loading of components

### Integration Points

The analysis bundle system integrates with:

- **Validation Pipeline**: Quality gates and completeness checks
- **Resonant Computation Engine**: Statistical validation of evidence networks
- **LLM Generation**: Feature analysis for training data
- **Database Search**: Annotation performance evaluation
- **Hardware Harvesting**: Quality checks on hardware measurements

### Troubleshooting

#### Component Not Found

```python
# Check registry
print(global_registry.list_components())

# Import adapter module to register
from precursor.src.analysis.component_adapters import *
```

#### Execution Failed

```python
# Check result status
for result in results:
    if result.status == ComponentStatus.FAILED:
        print(f"Failed: {result.component_name}")
        print(f"Error: {result.error_message}")
```

#### Configuration Issues

```python
# Verify config structure
component = DataQualityComponent(config={
    "quality_thresholds": {
        "completeness": 0.95,
        "consistency": 0.90
    }
})
```

### Examples

See `precursor/src/analysis/examples/` for complete examples:

- `quality_gates_example.py`: Adding quality gates to pipeline
- `statistical_validation_example.py`: Statistical validation workflow
- `custom_bundle_example.py`: Creating custom bundles
- `surgical_injection_example.py`: Surgical injection techniques

### Architecture

```
analysis/
├── analysis_component.py       # Base classes
├── component_adapters.py       # Adapters for existing scripts
├── bundles.py                  # Pre-configured bundles
├── __init__.py                 # Public API
├── annotation/                 # Original scripts
├── features/                   # Original scripts
├── quality/                    # Original scripts
├── statistical/                # Original scripts
└── completeness/               # Original scripts
```

The bundling system wraps original scripts without modification, providing a clean abstraction layer for pipeline integration.

### Future Enhancements

- Parallel execution of independent components
- Asynchronous execution support
- Distributed execution across machines
- Real-time streaming analysis
- Auto-tuning of configurations
- ML-based component recommendation
- Visual pipeline builder
- Integration with experiment LLMs

### Authors

Lavoisier Project Team
October 2025

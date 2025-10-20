"""
Analysis Bundles Usage Example
===============================

Demonstrates surgical injection of analysis components into pipelines.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import analysis bundles
from precursor.src.analysis import (
    # Bundles
    QualityBundle,
    AnnotationBundle,
    FeatureBundle,
    StatisticalBundle,
    CompleteBundle,
    CustomBundle,

    # Components
    DataQualityComponent,
    ClusteringValidationComponent,

    # Utilities
    PipelineInjector,
    AnalysisPipeline,
    quick_quality_check,
    quick_annotation_eval,
    quick_statistical_validation,
    create_injectable_bundle,
    AnalysisCategory,
    global_registry
)


def example_1_basic_bundle():
    """Example 1: Basic bundle usage"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Bundle Usage")
    print("="*70)

    # Create mock data
    data = pd.DataFrame({
        'feature_1': np.random.rand(100),
        'feature_2': np.random.rand(100),
        'feature_3': np.random.rand(100)
    })

    # Create quality bundle
    quality_bundle = QualityBundle()

    # Build pipeline from bundle
    pipeline = quality_bundle.build_pipeline()

    # Execute
    results = pipeline.execute_sequential(data)

    # Display results
    print(f"\nExecuted {len(results)} components:")
    for result in results:
        print(f"\n  {result.component_name}:")
        print(f"    Status: {result.status.value}")
        print(f"    Execution time: {result.execution_time:.3f}s")
        print(f"    Metrics: {result.metrics}")
        print(f"    Interpretation: {list(result.interpretations.values())[0] if result.interpretations else 'N/A'}")

    # Get summary
    summary = pipeline.get_summary()
    print(f"\nPipeline Summary:")
    print(f"  Total time: {summary['total_execution_time']:.3f}s")
    print(f"  Components executed: {summary['executed_components']}/{summary['total_components']}")


def example_2_surgical_injection():
    """Example 2: Surgical injection into existing pipeline"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Surgical Injection")
    print("="*70)

    # Create a mock existing pipeline
    existing_pipeline = AnalysisPipeline(name="ExistingPipeline")

    # Add some existing components
    existing_pipeline.add_component(
        DataQualityComponent(config={'name': 'existing_quality'})
    )

    print(f"\nOriginal pipeline: {len(existing_pipeline.components)} components")

    # Create injector
    injector = PipelineInjector(existing_pipeline)

    # Inject quality gates
    print("  Injecting quality gates...")
    injector.inject_quality_gates()

    # Inject feature analysis
    print("  Injecting feature analysis...")
    injector.inject_component(
        ClusteringValidationComponent(),
        at=2
    )

    # Inject bundle conditionally
    print("  Conditionally injecting annotation bundle...")
    has_labels = True  # Mock condition
    if has_labels:
        injector.inject_bundle(AnnotationBundle(), position='end')

    # Get modified pipeline
    modified_pipeline = injector.get_pipeline()

    print(f"\nModified pipeline: {len(modified_pipeline.components)} components")
    print(f"Injection log: {len(injector.get_injection_log())} operations")

    for log_entry in injector.get_injection_log():
        print(f"  - {log_entry['type']}: {log_entry.get('component_name', log_entry.get('bundle_name'))}")


def example_3_custom_bundle():
    """Example 3: Creating custom bundles"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Bundle")
    print("="*70)

    # Method 1: Custom bundle by component names
    print("\nMethod 1: By component names")
    custom1 = CustomBundle(
        name="MyAnalysisWorkflow",
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

    print(f"  Created '{custom1.name}' with {len(custom1.components)} components")

    # Method 2: Category-based bundle
    print("\nMethod 2: By categories")
    custom2 = create_injectable_bundle(
        categories=[
            AnalysisCategory.QUALITY,
            AnalysisCategory.FEATURES
        ],
        config={
            "quality": {"quality_thresholds": {"completeness": 0.95}},
            "features": {"n_clusters_range": (2, 10)}
        }
    )

    print(f"  Created bundle with {len(custom2.components)} components")


def example_4_quick_functions():
    """Example 4: Quick convenience functions"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Quick Functions")
    print("="*70)

    # Quick quality check
    print("\nQuick Quality Check:")
    data = pd.DataFrame(np.random.rand(100, 5))
    quality_results = quick_quality_check(data)
    print(f"  Quality passed: {quality_results['passed']}")
    print(f"  Metrics: {list(quality_results['metrics'].keys())}")

    # Quick annotation evaluation
    print("\nQuick Annotation Evaluation:")
    true_labels = np.random.randint(0, 3, 100)
    predicted_labels = np.random.randint(0, 3, 100)
    annotation_results = quick_annotation_eval(true_labels, predicted_labels)
    if 'annotation_performance' in annotation_results['metrics']:
        metrics = annotation_results['metrics']['annotation_performance']
        print(f"  Accuracy: {metrics.get('Accuracy', 0):.3f}")
        print(f"  Precision: {metrics.get('Precision', 0):.3f}")
        print(f"  Recall: {metrics.get('Recall', 0):.3f}")

    # Quick statistical validation
    print("\nQuick Statistical Validation:")
    numerical_data = {'method1': np.random.rand(50)}
    visual_data = {'method2': np.random.rand(50)}
    stat_results = quick_statistical_validation(numerical_data, visual_data)
    print(f"  Components executed: {len(stat_results['metrics'])}")


def example_5_complete_workflow():
    """Example 5: Complete analysis workflow"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Complete Workflow")
    print("="*70)

    # Create complete bundle
    complete = CompleteBundle(
        config={
            "quality": {"quality_thresholds": {"completeness": 0.95}},
            "features": {"n_clusters_range": (2, 8)},
            "statistical": {"alpha": 0.05}
        }
    )

    # Build pipeline
    pipeline = complete.build_pipeline()

    print(f"\nComplete pipeline created:")
    print(f"  Total components: {len(pipeline.components)}")
    print(f"  Component names:")
    for i, comp in enumerate(pipeline.components, 1):
        print(f"    {i}. {comp.name} ({comp.category.value})")

    # Mock execution (would fail with real data)
    print(f"\n  (Ready to execute on real data)")


def example_6_registry_usage():
    """Example 6: Using the component registry"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Component Registry")
    print("="*70)

    # List all components
    all_components = global_registry.list_components()
    print(f"\nTotal registered components: {len(all_components)}")

    # List by category
    quality_components = global_registry.list_components(AnalysisCategory.QUALITY)
    print(f"Quality components: {len(quality_components)}")
    for name in quality_components:
        print(f"  - {name}")

    # Create instance from registry
    print(f"\nCreating component from registry:")
    component = global_registry.create_instance(
        "data_quality",
        config={"quality_thresholds": {"completeness": 0.98}}
    )
    if component:
        print(f"  Created: {component.name}")
        print(f"  Category: {component.category.value}")
        print(f"  Config: {component.config}")


def example_7_conditional_execution():
    """Example 7: Conditional execution"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Conditional Execution")
    print("="*70)

    # Create pipeline
    pipeline = AnalysisPipeline(name="ConditionalPipeline")
    pipeline.add_components([
        DataQualityComponent(),
        ClusteringValidationComponent()
    ])

    # Define condition
    from precursor.src.analysis import ComponentStatus

    def continue_if_quality_passed(result):
        """Only continue if previous component passed"""
        return result.status == ComponentStatus.COMPLETED and \
               result.metrics.get('completeness_score', 0) >= 0.95

    # Mock data
    data = pd.DataFrame(np.random.rand(100, 5))

    # Execute with condition
    print(f"\nExecuting {len(pipeline.components)} components conditionally...")
    results = pipeline.execute_conditional(
        data,
        condition=continue_if_quality_passed
    )

    print(f"\nResults:")
    for result in results:
        print(f"  {result.component_name}: {result.status.value}")


# Run all examples
if __name__ == "__main__":
    print("\n" + "="*70)
    print("LAVOISIER ANALYSIS BUNDLES - USAGE EXAMPLES")
    print("="*70)

    try:
        example_1_basic_bundle()
        example_2_surgical_injection()
        example_3_custom_bundle()
        example_4_quick_functions()
        example_5_complete_workflow()
        example_6_registry_usage()
        example_7_conditional_execution()

        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70)

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()

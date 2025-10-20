# Pipeline Stage/Theatre System - Complete Example

## Overview

Demonstrates the finite observer architecture where:

- **Theatre** (transcendent observer) observes **Stages** (finite observers)
- **Stages** observe **Processes** (lowest-level observers)
- **Results saved at EVERY stage** (.tab and .json) for bidirectional navigation
- **No inherent sequence** - navigate in any direction

## Complete Working Example

```python
from precursor.src.pipeline import (
    Theatre,
    create_stage,
    NavigationMode,
    ProcessObserver,
    ProcessResult,
    StageStatus
)
from pathlib import Path
import pandas as pd
import time


# ============================================================================
# STEP 1: Create Custom Process Observers
# ============================================================================

class MSDataLoadingProcess(ProcessObserver):
    """Custom process for MS data loading"""

    def __init__(self, config=None):
        super().__init__("ms_data_loading", config)

    def observe(self, input_data, **kwargs):
        start_time = time.perf_counter()

        try:
            # Load mzML file
            file_path = input_data

            # Simulate loading spectra
            data = pd.DataFrame({
                'mz': [100, 200, 300, 400, 500],
                'intensity': [1000, 800, 1200, 600, 900],
                'rt': [10.5, 11.2, 11.8, 12.1, 12.5]
            })

            execution_time = time.perf_counter() - start_time

            return ProcessResult(
                process_name=self.name,
                status=StageStatus.COMPLETED,
                execution_time=execution_time,
                data=data,
                metrics={
                    'spectra_count': len(data),
                    'mz_range': f"{data['mz'].min()}-{data['mz'].max()}"
                },
                metadata={'file_path': str(file_path)}
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ProcessResult(
                process_name=self.name,
                status=StageStatus.FAILED,
                execution_time=execution_time,
                data=None,
                error_message=str(e)
            )


class SEntropyTransformProcess(ProcessObserver):
    """Custom process for S-Entropy transformation"""

    def __init__(self, config=None):
        super().__init__("s_entropy_transform", config)

    def observe(self, input_data, **kwargs):
        start_time = time.perf_counter()

        try:
            # Transform to S-Entropy coordinates
            if isinstance(input_data, pd.DataFrame):
                # Add S-entropy coordinates (mock)
                input_data['s_knowledge'] = input_data['mz'] / 1000
                input_data['s_time'] = input_data['rt'] / 60
                input_data['s_entropy'] = input_data['intensity'] / input_data['intensity'].max()

                transformed_data = input_data
            else:
                transformed_data = input_data

            execution_time = time.perf_counter() - start_time

            return ProcessResult(
                process_name=self.name,
                status=StageStatus.COMPLETED,
                execution_time=execution_time,
                data=transformed_data,
                metrics={
                    'transformed_count': len(transformed_data) if isinstance(transformed_data, pd.DataFrame) else 0
                },
                metadata={'transformation': 'S-Entropy 3D coordinates'}
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ProcessResult(
                process_name=self.name,
                status=StageStatus.FAILED,
                execution_time=execution_time,
                data=None,
                error_message=str(e)
            )


class PhaseLockDetectionProcess(ProcessObserver):
    """Custom process for phase-lock detection"""

    def __init__(self, config=None):
        super().__init__("phase_lock_detection", config)

    def observe(self, input_data, **kwargs):
        start_time = time.perf_counter()

        try:
            # Detect phase-locks (mock)
            phase_locks = {
                'total_locks': 15,
                'strong_locks': 8,
                'weak_locks': 7,
                'coherence_threshold': 0.3
            }

            execution_time = time.perf_counter() - start_time

            return ProcessResult(
                process_name=self.name,
                status=StageStatus.COMPLETED,
                execution_time=execution_time,
                data=phase_locks,
                metrics=phase_locks,
                metadata={'method': 'Enhanced phase-lock measurement device'}
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ProcessResult(
                process_name=self.name,
                status=StageStatus.FAILED,
                execution_time=execution_time,
                data=None,
                error_message=str(e)
            )


# ============================================================================
# STEP 2: Create Custom Stages
# ============================================================================

def create_data_loading_stage(save_dir):
    """Create data loading stage"""
    from .stages import StageObserver

    stage = StageObserver(
        stage_name="Data Loading and QC",
        stage_id="stage_01_data_loading",
        process_observers=[MSDataLoadingProcess()],
        save_dir=save_dir / "stage_01_data_loading"
    )

    return stage


def create_transformation_stage(save_dir):
    """Create S-Entropy transformation stage"""
    from .stages import StageObserver

    stage = StageObserver(
        stage_name="S-Entropy Transformation",
        stage_id="stage_02_transformation",
        process_observers=[SEntropyTransformProcess()],
        save_dir=save_dir / "stage_02_transformation"
    )

    return stage


def create_phase_lock_stage(save_dir):
    """Create phase-lock detection stage"""
    from .stages import StageObserver

    stage = StageObserver(
        stage_name="Phase-Lock Detection",
        stage_id="stage_03_phase_locks",
        process_observers=[PhaseLockDetectionProcess()],
        save_dir=save_dir / "stage_03_phase_locks"
    )

    return stage


# ============================================================================
# STEP 3: Build Theatre with Multiple Navigation Modes
# ============================================================================

def example_1_linear_navigation():
    """Example 1: Traditional linear navigation"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Linear Navigation")
    print("="*70)

    theatre = Theatre(
        theatre_name="Linear_MS_Pipeline",
        output_dir=Path("./example_results/linear"),
        navigation_mode=NavigationMode.LINEAR
    )

    # Add stages
    theatre.add_stage(create_data_loading_stage(Path("./example_results/linear")))
    theatre.add_stage(create_transformation_stage(Path("./example_results/linear")))
    theatre.add_stage(create_phase_lock_stage(Path("./example_results/linear")))

    # Execute
    result = theatre.observe_all_stages("experiment_001.mzml")

    print(f"\nResult: {result.status.value}")
    print(f"Execution order: {' → '.join(result.execution_order)}")
    print(f"Total time: {result.execution_time:.3f}s")


def example_2_dependency_navigation():
    """Example 2: Dependency-based navigation"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Dependency Navigation")
    print("="*70)

    theatre = Theatre(
        theatre_name="Dependency_MS_Pipeline",
        output_dir=Path("./example_results/dependency"),
        navigation_mode=NavigationMode.DEPENDENCY
    )

    # Add stages
    stage1 = create_data_loading_stage(Path("./example_results/dependency"))
    stage2 = create_transformation_stage(Path("./example_results/dependency"))
    stage3 = create_phase_lock_stage(Path("./example_results/dependency"))

    theatre.add_stage(stage1)
    theatre.add_stage(stage2)
    theatre.add_stage(stage3)

    # Define dependencies
    theatre.add_stage_dependency("stage_01_data_loading", "stage_02_transformation")
    theatre.add_stage_dependency("stage_02_transformation", "stage_03_phase_locks")

    # Execute
    result = theatre.observe_all_stages("experiment_001.mzml")

    print(f"\nResult: {result.status.value}")
    print(f"Execution order: {' → '.join(result.execution_order)}")


def example_3_gear_ratio_navigation():
    """Example 3: Gear ratio navigation (O(1) hierarchical jumps)"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Gear Ratio Navigation")
    print("="*70)

    theatre = Theatre(
        theatre_name="GearRatio_MS_Pipeline",
        output_dir=Path("./example_results/gear_ratio"),
        navigation_mode=NavigationMode.GEAR_RATIO
    )

    # Add stages
    stage1 = create_data_loading_stage(Path("./example_results/gear_ratio"))
    stage2 = create_transformation_stage(Path("./example_results/gear_ratio"))
    stage3 = create_phase_lock_stage(Path("./example_results/gear_ratio"))

    theatre.add_stage(stage1)
    theatre.add_stage(stage2)
    theatre.add_stage(stage3)

    # Define dependencies (creates hierarchy)
    theatre.add_stage_dependency("stage_01_data_loading", "stage_02_transformation")
    theatre.add_stage_dependency("stage_02_transformation", "stage_03_phase_locks")

    # Execute with gear ratio navigation
    result = theatre.observe_all_stages("experiment_001.mzml")

    print(f"\nResult: {result.status.value}")
    print(f"Navigation history: {len(result.metadata['navigation_history'])} transitions")

    for from_stage, to_stage, gear_ratio in result.metadata['navigation_history']:
        print(f"  {from_stage} → {to_stage}: ratio = {gear_ratio:.3f}")


def example_4_bidirectional_navigation():
    """Example 4: Bidirectional navigation (forward and backward)"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Bidirectional Navigation")
    print("="*70)

    theatre = Theatre(
        theatre_name="Bidirectional_MS_Pipeline",
        output_dir=Path("./example_results/bidirectional"),
        navigation_mode=NavigationMode.BIDIRECTIONAL
    )

    # Add stages
    stage1 = create_data_loading_stage(Path("./example_results/bidirectional"))
    stage2 = create_transformation_stage(Path("./example_results/bidirectional"))
    stage3 = create_phase_lock_stage(Path("./example_results/bidirectional"))

    theatre.add_stage(stage1)
    theatre.add_stage(stage2)
    theatre.add_stage(stage3)

    # Define dependencies
    theatre.add_stage_dependency("stage_01_data_loading", "stage_02_transformation")
    theatre.add_stage_dependency("stage_02_transformation", "stage_03_phase_locks")

    # Define observations (stage 3 observes stage 1)
    theatre.add_stage_observation("stage_03_phase_locks", "stage_01_data_loading")

    # Execute
    result = theatre.observe_all_stages("experiment_001.mzml")

    print(f"\nResult: {result.status.value}")
    print(f"Execution order: {' → '.join(result.execution_order)}")
    print(f"Note: Stage 3 observed Stage 1 results during execution")


def example_5_resume_from_saved():
    """Example 5: Resume from saved stage results"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Resume from Saved Results")
    print("="*70)

    # First execution
    print("First execution...")
    theatre1 = Theatre(
        theatre_name="Resume_Test_Pipeline",
        output_dir=Path("./example_results/resume"),
        navigation_mode=NavigationMode.DEPENDENCY
    )

    stage1 = create_data_loading_stage(Path("./example_results/resume"))
    stage2 = create_transformation_stage(Path("./example_results/resume"))

    theatre1.add_stage(stage1)
    theatre1.add_stage(stage2)
    theatre1.add_stage_dependency("stage_01_data_loading", "stage_02_transformation")

    result1 = theatre1.observe_all_stages("experiment_001.mzml")
    print(f"  First run: {result1.status.value}")
    print(f"  Results saved in: {theatre1.output_dir}")

    # Resume from saved results
    print("\nResuming from saved results...")
    theatre2 = Theatre(
        theatre_name="Resume_Test_Pipeline",
        output_dir=Path("./example_results/resume"),
        navigation_mode=NavigationMode.DEPENDENCY
    )

    # Load saved stage results
    stage1_loaded = create_data_loading_stage(Path("./example_results/resume"))
    saved_result = stage1_loaded.load_stage_result()

    if saved_result:
        print(f"  Successfully loaded stage result from previous run")
        print(f"  Can continue from any saved stage!")


# ============================================================================
# Run All Examples
# ============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "="*70)
    print("PIPELINE STAGE/THEATRE SYSTEM - COMPLETE EXAMPLES")
    print("="*70)

    example_1_linear_navigation()
    example_2_dependency_navigation()
    example_3_gear_ratio_navigation()
    example_4_bidirectional_navigation()
    example_5_resume_from_saved()

    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. Results saved at EVERY stage (.json + .tab)")
    print("  2. Can navigate in ANY direction")
    print("  3. Resume from any saved stage")
    print("  4. Multiple navigation modes available")
    print("  5. Gear ratio navigation for O(1) hierarchical jumps")
    print("  6. Theatre observes stages, stages observe processes")
```

## Integration with Existing Systems

### With Analysis Bundles

```python
from precursor.src.pipeline import Theatre, StageObserver
from precursor.src.analysis import QualityBundle, AnnotationBundle

# Create stage with analysis bundle
class AnalysisStage(StageObserver):
    def __init__(self, bundle, save_dir):
        super().__init__(
            stage_name=f"{bundle.name} Stage",
            stage_id=f"stage_{bundle.name.lower()}",
            save_dir=save_dir
        )

        # Integrate bundle as processes
        self.analysis_bundle = bundle
        self.analysis_pipeline = bundle.build_pipeline()

    def observe(self, input_data, **kwargs):
        # Run analysis bundle
        analysis_results = self.analysis_pipeline.execute_sequential(input_data)

        # Convert to stage result
        return super().observe(input_data, **kwargs)

# Use in theatre
theatre = Theatre("MS_with_Analysis")
theatre.add_stage(AnalysisStage(QualityBundle(), save_dir))
```

### With Resonant Computation Engine

```python
from precursor.src.pipeline import Theatre, ProcessObserver
from precursor.src.hardware.resonant_computation_engine import ResonantComputationEngine

class ResonantProcess(ProcessObserver):
    def __init__(self):
        super().__init__("resonant_computation")
        self.engine = ResonantComputationEngine()

    def observe(self, input_data, **kwargs):
        # Run resonant computation
        results = await self.engine.process_experiment_as_bayesian_network(
            input_data,
            process_fn
        )

        return ProcessResult(
            process_name=self.name,
            status=StageStatus.COMPLETED,
            execution_time=results['experiment_metadata']['total_time'],
            data=results,
            metrics={'evidence_nodes': len(results['evidence_network'])}
        )
```

## File Outputs at Each Stage

### JSON Output (`stage_id_result.json`)

```json
{
  "stage_name": "Data Loading and QC",
  "stage_id": "stage_01_data_loading",
  "status": "completed",
  "execution_time": 0.123,
  "process_results": [...],
  "metrics": {
    "total_processes": 1,
    "completed_processes": 1
  }
}
```

### TAB Output (`stage_id_data.tab`)

```
mz      intensity   rt      s_knowledge s_time  s_entropy
100     1000        10.5    0.100       0.175   1.000
200     800         11.2    0.200       0.187   0.800
300     1200        11.8    0.300       0.197   1.000
```

### Process Summary (`stage_id_processes.tab`)

```
process_name        status      execution_time  metrics
ms_data_loading     completed   0.123           {"spectra_count": 5}
```

## Benefits

1. **Bidirectional Navigation**: Can move forward or backward through stages
2. **Resume Capability**: Resume from any saved stage
3. **Parallel Execution**: Execute independent stages in parallel
4. **Dynamic Re-arrangement**: Change stage order without re-running
5. **Gear Ratio Navigation**: O(1) hierarchical jumps between stages
6. **Observer Hierarchy**: Clear separation of concerns (Theatre → Stages → Processes)
7. **Complete Audit Trail**: All results saved with full provenance

## Architecture Summary

```
Theatre (Transcendent Observer)
    ↓ observes
StageObserver (Finite Observer)
    ↓ observes
ProcessObserver (Lowest Level)
    ↓ produces
Results (.json + .tab)
    ↓ enables
Bidirectional Navigation
```

**Key Innovation: Results saved at EVERY stage enable movement in ANY direction!**

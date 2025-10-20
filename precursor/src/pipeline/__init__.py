"""
Lavoisier Pipeline - Finite Observer Architecture
==================================================

Stage-based pipeline system where:
- Each stage is a finite observer watching lower-level process observers
- Theatre is the transcendent observer coordinating all stages
- Results saved at EVERY stage (.tab and .json) for bidirectional navigation
- No inherent execution sequence - navigate in any direction

Public API:
-----------

Stages:
    - ProcessObserver: Base class for process-level observations
    - StageObserver: Finite observer for a collection of processes
    - ProcessResult: Result from process observation
    - StageResult: Result from stage observation
    - create_stage: Factory for creating stages

Theatre:
    - Theatre: Transcendent observer coordinating all stages
    - TheatreResult: Complete theatre execution result
    - NavigationMode: Execution modes (LINEAR, DEPENDENCY, OPPORTUNISTIC, BIDIRECTIONAL, GEAR_RATIO)

Enums:
    - ObserverLevel: PROCESS, STAGE, THEATRE
    - StageStatus: PENDING, RUNNING, COMPLETED, FAILED, SKIPPED
    - TheatreStatus: IDLE, RUNNING, COMPLETED, FAILED, PAUSED

Usage:
------

    from precursor.src.pipeline import Theatre, create_stage, NavigationMode

    # Create theatre
    theatre = Theatre("MS_Pipeline", navigation_mode=NavigationMode.GEAR_RATIO)

    # Create and add stages
    stage1 = create_stage("Data Loading", "stage_01", ['data_loading'])
    stage2 = create_stage("Annotation", "stage_02", ['annotation'])

    theatre.add_stage(stage1)
    theatre.add_stage(stage2)
    theatre.add_stage_dependency("stage_01", "stage_02")

    # Execute with non-linear navigation
    result = theatre.observe_all_stages(input_data)

    # Results saved at every stage - can navigate in any direction!

Key Innovation:
---------------

Results saved at EVERY stage enable:
- Resume from any point
- Bidirectional navigation
- Parallel stage execution
- Dynamic re-arrangement
- Experiment replay with different orderings

Author: Lavoisier Project
Date: October 2025
Version: 1.0.0
"""

from .stages import (
    # Base classes
    ProcessObserver,
    StageObserver,

    # Results
    ProcessResult,
    StageResult,

    # Enums
    ObserverLevel,
    StageStatus,

    # Factory
    create_stage,

    # Concrete processes
    DataLoadingProcess,
    FeatureExtractionProcess,
    AnnotationProcess
)

from .theatre import (
    # Theatre
    Theatre,
    TheatreResult,

    # Enums
    TheatreStatus,
    NavigationMode
)

__version__ = "1.0.0"

__all__ = [
    # Stages
    "ProcessObserver",
    "StageObserver",
    "ProcessResult",
    "StageResult",
    "ObserverLevel",
    "StageStatus",
    "create_stage",
    "DataLoadingProcess",
    "FeatureExtractionProcess",
    "AnnotationProcess",

    # Theatre
    "Theatre",
    "TheatreResult",
    "TheatreStatus",
    "NavigationMode",
]


def get_version():
    """Get version string"""
    return __version__


print(f"[Lavoisier Pipeline v{__version__}] Finite Observer Architecture")
print("  Theatre → Stages → Processes")
print("  Results saved at every stage for bidirectional navigation")

"""
Pipeline Theatre - Transcendent Observer
=========================================

The Theatre is the transcendent observer that coordinates all stage observers.
It arranges stages in a coherent way and enables non-linear navigation through
saved stage results.

Key Principles:
1. No inherent sequence - stages can execute in any order
2. Bidirectional navigation through saved results
3. Gear ratio navigation between hierarchical stages
4. Observer hierarchy: Theatre → Stages → Processes
5. Dynamic stage arrangement based on dependencies

Architecture:
- Theatre observes all StageObservers
- Stages observe other Stages (through saved results)
- Enables navigation in any direction
- Coordinates entire experiment workflow
"""

import json
import time
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from .stages import (
    StageObserver,
    StageResult,
    StageStatus,
    ObserverLevel
)

# Import BMD components
try:
    from ..bmd import (
        BiologicalMaxwellDemonReference,
        HardwareBMDStream,
        BMDState,
        compute_stream_divergence
    )
    BMD_AVAILABLE = True
except ImportError:
    BMD_AVAILABLE = False


class TheatreStatus(Enum):
    """Execution status of the theatre"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class NavigationMode(Enum):
    """Navigation mode through stages"""
    LINEAR = "linear"              # Traditional sequential execution
    DEPENDENCY = "dependency"      # Execute based on dependencies
    OPPORTUNISTIC = "opportunistic"  # Execute what's ready
    BIDIRECTIONAL = "bidirectional"  # Move forward and backward
    GEAR_RATIO = "gear_ratio"      # Hierarchical navigation with O(1) jumps


@dataclass
class TheatreResult:
    """
    Complete theatre execution result.

    Contains all stage results and navigation metadata.
    """
    theatre_name: str
    status: TheatreStatus
    execution_time: float
    stage_results: Dict[str, StageResult] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    navigation_mode: NavigationMode = NavigationMode.DEPENDENCY
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'theatre_name': self.theatre_name,
            'status': self.status.value,
            'execution_time': self.execution_time,
            'stage_results': {k: v.to_dict() for k, v in self.stage_results.items()},
            'execution_order': self.execution_order,
            'navigation_mode': self.navigation_mode.value,
            'metrics': self.metrics,
            'metadata': self.metadata
        }

    def save(self, output_path: Path):
        """Save theatre result"""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class Theatre:
    """
    Theatre - Transcendent Observer

    Coordinates all stage observers, enabling non-linear navigation through
    saved stage results. The theatre observes stages, which observe processes.

    Key Features:
    - No inherent execution sequence
    - Bidirectional navigation
    - Gear ratio navigation between hierarchical levels
    - Dynamic stage arrangement
    - Checkpoint/resume capability
    """

    def __init__(
        self,
        theatre_name: str,
        output_dir: Optional[Path] = None,
        navigation_mode: NavigationMode = NavigationMode.DEPENDENCY,
        config: Optional[Dict[str, Any]] = None,
        enable_bmd_grounding: bool = True,
        bmd_reference: Optional['BiologicalMaxwellDemonReference'] = None
    ):
        """
        Initialize theatre.

        Args:
            theatre_name: Theatre name
            output_dir: Directory for theatre outputs
            navigation_mode: How to navigate through stages
            config: Theatre configuration
            enable_bmd_grounding: Enable BMD grounding for reality checking
            bmd_reference: Optional pre-configured BMD reference
        """
        self.theatre_name = theatre_name
        self.output_dir = Path(output_dir) if output_dir else Path(f"./theatre_results/{theatre_name}")
        self.navigation_mode = navigation_mode
        self.config = config or {}
        self.logger = logging.getLogger(f"Theatre.{theatre_name}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Stage management
        self.stages: Dict[str, StageObserver] = {}
        self.stage_graph = nx.DiGraph()

        # Execution state
        self.status = TheatreStatus.IDLE
        self.current_stage: Optional[str] = None
        self.completed_stages: Set[str] = set()
        self.failed_stages: Set[str] = set()
        self.stage_results: Dict[str, StageResult] = {}

        # Navigation state
        self.execution_order: List[str] = []
        self.navigation_history: List[Tuple[str, str, float]] = []  # (from_stage, to_stage, gear_ratio)

        # BMD grounding
        self.enable_bmd_grounding = enable_bmd_grounding and BMD_AVAILABLE
        self.bmd_reference: Optional['BiologicalMaxwellDemonReference'] = None
        self.hardware_bmd_stream: Optional['HardwareBMDStream'] = None
        self.network_bmd: Optional['BMDState'] = None
        self.stream_divergence_threshold = 5.0  # Alert if divergence exceeds this

        if self.enable_bmd_grounding:
            if bmd_reference:
                self.bmd_reference = bmd_reference
            else:
                try:
                    from ..bmd import BiologicalMaxwellDemonReference
                    self.bmd_reference = BiologicalMaxwellDemonReference(enable_all_harvesters=True)
                    self.logger.info("  BMD grounding enabled with hardware reference")
                except Exception as e:
                    self.logger.warning(f"  Could not initialize BMD reference: {e}")
                    self.enable_bmd_grounding = False

        self.logger.info(f"[Theatre '{theatre_name}'] Initialized with mode: {navigation_mode.value}")
        if self.enable_bmd_grounding:
            self.logger.info("  BMD Grounding: ENABLED")

    # ========================================================================
    # STAGE MANAGEMENT
    # ========================================================================

    def add_stage(self, stage: StageObserver):
        """
        Add stage to theatre.

        Args:
            stage: StageObserver to add
        """
        self.stages[stage.stage_id] = stage
        self.stage_graph.add_node(stage.stage_id, name=stage.stage_name, stage=stage)

        self.logger.info(f"  Added stage: {stage.stage_name} (id: {stage.stage_id})")

    def add_stage_dependency(self, from_stage_id: str, to_stage_id: str):
        """
        Add dependency between stages.

        Args:
            from_stage_id: Stage that must complete first
            to_stage_id: Stage that depends on from_stage
        """
        if from_stage_id not in self.stages or to_stage_id not in self.stages:
            raise ValueError(f"Invalid stage IDs: {from_stage_id} or {to_stage_id}")

        self.stage_graph.add_edge(from_stage_id, to_stage_id)
        self.stages[to_stage_id].add_dependency(from_stage_id)

        self.logger.info(f"  Added dependency: {from_stage_id} → {to_stage_id}")

    def add_stage_observation(self, observer_stage_id: str, observed_stage_id: str):
        """
        Configure one stage to observe another.

        Args:
            observer_stage_id: Stage that observes
            observed_stage_id: Stage being observed
        """
        if observer_stage_id not in self.stages or observed_stage_id not in self.stages:
            raise ValueError(f"Invalid stage IDs: {observer_stage_id} or {observed_stage_id}")

        self.stages[observer_stage_id].add_observer(observed_stage_id)

        self.logger.info(f"  Added observation: {observer_stage_id} observes {observed_stage_id}")

    def get_stage_hierarchy(self) -> Dict[int, List[str]]:
        """
        Get hierarchical levels of stages based on dependencies.

        Returns:
            Dictionary mapping level to stage IDs
        """
        hierarchy = {}

        try:
            # Topological sort gives levels
            for level, stage_ids in enumerate(nx.topological_generations(self.stage_graph)):
                hierarchy[level] = list(stage_ids)
        except nx.NetworkXError:
            # Cyclic graph - fallback to all at level 0
            hierarchy[0] = list(self.stages.keys())

        return hierarchy

    def compute_gear_ratios(self) -> Dict[Tuple[str, str], float]:
        """
        Compute gear ratios between stages for O(1) navigation.

        Gear ratio = complexity_target / complexity_source

        Returns:
            Dictionary mapping (from_stage, to_stage) to gear ratio
        """
        gear_ratios = {}

        hierarchy = self.get_stage_hierarchy()

        # Compute complexity for each stage (simplified: use process count)
        stage_complexity = {}
        for stage_id, stage in self.stages.items():
            stage_complexity[stage_id] = len(stage.process_observers) if stage.process_observers else 1

        # Compute ratios between adjacent levels
        for level in range(len(hierarchy) - 1):
            current_level = hierarchy[level]
            next_level = hierarchy.get(level + 1, [])

            for stage1 in current_level:
                for stage2 in next_level:
                    ratio = stage_complexity[stage2] / (stage_complexity[stage1] + 1e-9)
                    gear_ratios[(stage1, stage2)] = ratio

        return gear_ratios

    # ========================================================================
    # EXECUTION MODES
    # ========================================================================

    def observe_all_stages(
        self,
        input_data: Any,
        mode: Optional[NavigationMode] = None,
        **kwargs
    ) -> TheatreResult:
        """
        Observe all stages using specified navigation mode.

        Args:
            input_data: Initial input data
            mode: Navigation mode (overrides default)
            **kwargs: Additional parameters

        Returns:
            TheatreResult with all stage results
        """
        mode = mode or self.navigation_mode

        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"THEATRE: {self.theatre_name}")
        self.logger.info(f"Navigation Mode: {mode.value}")
        self.logger.info(f"Stages: {len(self.stages)}")
        self.logger.info(f"{'='*70}\n")

        self.status = TheatreStatus.RUNNING
        start_time = time.perf_counter()

        try:
            if mode == NavigationMode.LINEAR:
                self._execute_linear(input_data, **kwargs)
            elif mode == NavigationMode.DEPENDENCY:
                self._execute_dependency(input_data, **kwargs)
            elif mode == NavigationMode.OPPORTUNISTIC:
                self._execute_opportunistic(input_data, **kwargs)
            elif mode == NavigationMode.BIDIRECTIONAL:
                self._execute_bidirectional(input_data, **kwargs)
            elif mode == NavigationMode.GEAR_RATIO:
                self._execute_gear_ratio(input_data, **kwargs)

            execution_time = time.perf_counter() - start_time
            self.status = TheatreStatus.COMPLETED

            # Compute theatre metrics
            metrics = {
                'total_stages': len(self.stages),
                'completed_stages': len(self.completed_stages),
                'failed_stages': len(self.failed_stages),
                'total_execution_time': execution_time,
                'average_stage_time': execution_time / len(self.completed_stages) if self.completed_stages else 0
            }

            result = TheatreResult(
                theatre_name=self.theatre_name,
                status=self.status,
                execution_time=execution_time,
                stage_results=self.stage_results,
                execution_order=self.execution_order,
                navigation_mode=mode,
                metrics=metrics,
                metadata={
                    'navigation_history': self.navigation_history,
                    'config': self.config
                }
            )

            # Save theatre result
            result.save(self.output_dir / "theatre_result.json")

            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"THEATRE COMPLETE")
            self.logger.info(f"  Total time: {execution_time:.2f}s")
            self.logger.info(f"  Completed stages: {len(self.completed_stages)}/{len(self.stages)}")
            self.logger.info(f"  Failed stages: {len(self.failed_stages)}")
            self.logger.info(f"{'='*70}\n")

            return result

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            self.status = TheatreStatus.FAILED
            self.logger.error(f"Theatre execution failed: {str(e)}")

            return TheatreResult(
                theatre_name=self.theatre_name,
                status=self.status,
                execution_time=execution_time,
                stage_results=self.stage_results,
                execution_order=self.execution_order,
                navigation_mode=mode
            )

    def _execute_linear(self, input_data: Any, **kwargs):
        """Execute stages in linear order (as added)"""
        self.logger.info("[Linear Mode] Executing stages sequentially...")

        current_data = input_data

        for stage_id, stage in self.stages.items():
            self._execute_stage(stage_id, current_data, **kwargs)

            # Update data for next stage
            if stage_id in self.stage_results:
                result = self.stage_results[stage_id]
                if result.output_data is not None:
                    current_data = result.output_data

    def _execute_dependency(self, input_data: Any, **kwargs):
        """Execute stages based on dependency graph (topological order)"""
        self.logger.info("[Dependency Mode] Executing stages by dependencies...")

        try:
            # Topological sort
            execution_order = list(nx.topological_sort(self.stage_graph))
        except nx.NetworkXError:
            self.logger.warning("  Cyclic dependencies detected, falling back to linear order")
            execution_order = list(self.stages.keys())

        current_data = input_data

        for stage_id in execution_order:
            self._execute_stage(stage_id, current_data, **kwargs)

            # Update data
            if stage_id in self.stage_results:
                result = self.stage_results[stage_id]
                if result.output_data is not None:
                    current_data = result.output_data

    def _execute_opportunistic(self, input_data: Any, **kwargs):
        """Execute stages opportunistically (as dependencies are met)"""
        self.logger.info("[Opportunistic Mode] Executing ready stages...")

        current_data = input_data
        remaining_stages = set(self.stages.keys())

        while remaining_stages:
            # Find stages ready to execute
            ready_stages = []
            for stage_id in remaining_stages:
                stage = self.stages[stage_id]

                # Check if all dependencies are completed
                dependencies_met = all(
                    dep in self.completed_stages for dep in stage.dependencies
                )

                if dependencies_met:
                    ready_stages.append(stage_id)

            if not ready_stages:
                self.logger.warning("  No ready stages, breaking (possible cyclic dependency)")
                break

            # Execute ready stages
            for stage_id in ready_stages:
                self._execute_stage(stage_id, current_data, **kwargs)
                remaining_stages.remove(stage_id)

    def _execute_bidirectional(self, input_data: Any, **kwargs):
        """Execute stages bidirectionally (forward and backward)"""
        self.logger.info("[Bidirectional Mode] Executing with forward/backward navigation...")

        # First, execute forward
        self._execute_dependency(input_data, **kwargs)

        # Then, optionally revisit stages (using saved results)
        self.logger.info("  Revisiting stages with updated information...")

        # Example: Re-execute stages that observe completed stages
        for stage_id, stage in self.stages.items():
            if stage.observers:
                self.logger.info(f"  Re-observing stage: {stage.stage_name}")
                # Load previous result
                previous_result = stage.load_stage_result()

                # Re-execute with observer information
                self._execute_stage(stage_id, previous_result or input_data, **kwargs)

    def _execute_gear_ratio(self, input_data: Any, **kwargs):
        """Execute stages using gear ratio navigation (O(1) hierarchical jumps)"""
        self.logger.info("[Gear Ratio Mode] Executing with hierarchical navigation...")

        # Get hierarchical levels
        hierarchy = self.get_stage_hierarchy()
        gear_ratios = self.compute_gear_ratios()

        self.logger.info(f"  Hierarchy levels: {len(hierarchy)}")
        self.logger.info(f"  Gear ratios computed: {len(gear_ratios)}")

        current_data = input_data

        # Execute level by level, using gear ratios for navigation
        for level, stage_ids in hierarchy.items():
            self.logger.info(f"\n  [Level {level}] Executing {len(stage_ids)} stages...")

            for stage_id in stage_ids:
                # Check gear ratios to previous level
                if level > 0:
                    prev_level = hierarchy[level - 1]
                    for prev_stage_id in prev_level:
                        ratio = gear_ratios.get((prev_stage_id, stage_id), 1.0)
                        self.navigation_history.append((prev_stage_id, stage_id, ratio))
                        self.logger.info(f"    Gear ratio: {prev_stage_id} → {stage_id} = {ratio:.3f}")

                # Execute stage
                self._execute_stage(stage_id, current_data, **kwargs)

    def _execute_stage(self, stage_id: str, input_data: Any, **kwargs):
        """
        Execute a single stage.

        Args:
            stage_id: Stage to execute
            input_data: Input data
            **kwargs: Additional parameters
        """
        if stage_id not in self.stages:
            self.logger.error(f"Stage not found: {stage_id}")
            return

        stage = self.stages[stage_id]
        self.current_stage = stage_id
        self.execution_order.append(stage_id)

        self.logger.info(f"\n[Stage: {stage.stage_name}]")

        try:
            # Check if BMD grounding is enabled
            if self.enable_bmd_grounding and self.bmd_reference:
                result = self._execute_stage_with_bmd_grounding(stage, input_data, **kwargs)
            else:
                # Execute stage observation normally
                result = stage.observe(
                    input_data=input_data,
                    previous_stage_results=self.stage_results,
                    **kwargs
                )

            # Store result
            self.stage_results[stage_id] = result

            if result.status == StageStatus.COMPLETED:
                self.completed_stages.add(stage_id)
            else:
                self.failed_stages.add(stage_id)

        except Exception as e:
            self.logger.error(f"  Stage execution error: {str(e)}")
            self.failed_stages.add(stage_id)

    def _execute_stage_with_bmd_grounding(self, stage: StageObserver,
                                         input_data: Any, **kwargs) -> StageResult:
        """
        Execute stage with BMD grounding for reality checking.

        This implements hardware-constrained categorical completion:
        1. Measure hardware BMD stream (reality reference)
        2. Execute stage with BMD filtering
        3. Check stream divergence
        4. Update network BMD through hierarchical integration

        Args:
            stage: Stage to execute
            input_data: Input data
            **kwargs: Additional parameters

        Returns:
            StageResult with BMD grounding metadata
        """
        # 1. Update hardware BMD stream (continuous measurement)
        if self.bmd_reference:
            self.hardware_bmd_stream = self.bmd_reference.measure_stream()

            # Initialize network BMD on first measurement
            if self.network_bmd is None:
                self.network_bmd = self.hardware_bmd_stream.unified_bmd
                self.logger.info("  Initialized network BMD from hardware stream")

            # Log stream quality
            if self.hardware_bmd_stream.is_coherent():
                self.logger.info(f"  Hardware BMD stream: COHERENT (quality={self.hardware_bmd_stream.phase_lock_quality:.3f})")
            else:
                self.logger.warning(f"  Hardware BMD stream: INCOHERENT (quality={self.hardware_bmd_stream.phase_lock_quality:.3f})")

        # 2. Execute stage with BMD grounding (if stage supports it)
        if hasattr(stage, 'observe_with_bmd_grounding'):
            result = stage.observe_with_bmd_grounding(
                input_data=input_data,
                hardware_bmd=self.hardware_bmd_stream.unified_bmd if self.hardware_bmd_stream else None,
                network_bmd=self.network_bmd,
                previous_stage_results=self.stage_results,
                **kwargs
            )
        else:
            # Fallback to normal observation
            result = stage.observe(
                input_data=input_data,
                previous_stage_results=self.stage_results,
                **kwargs
            )

        # 3. Check stream divergence (reality drift detection)
        if self.hardware_bmd_stream and self.network_bmd:
            stream_div = compute_stream_divergence(
                self.network_bmd,
                self.hardware_bmd_stream.unified_bmd
            )

            self.logger.info(f"  Stream divergence: {stream_div:.3f}")

            if stream_div > self.stream_divergence_threshold:
                self.logger.warning(f"  ⚠ Network BMD drifting from hardware reality (D={stream_div:.3f} > {self.stream_divergence_threshold})")

            # Store in result metadata
            if result.metadata is None:
                result.metadata = {}
            result.metadata['stream_divergence'] = stream_div
            result.metadata['stream_coherent'] = self.hardware_bmd_stream.is_coherent()

        # 4. Update network BMD if stage generated new BMD
        if hasattr(result, 'generated_bmd') and result.generated_bmd is not None:
            from ..bmd import integrate_hierarchical

            self.network_bmd = integrate_hierarchical(
                network_bmd=self.network_bmd,
                new_bmd=result.generated_bmd,
                processing_sequence=self.execution_order
            )

            self.logger.info(f"  Updated network BMD (richness={self.network_bmd.categorical_richness})")

        return result

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    def visualize_stage_graph(self, output_path: Optional[Path] = None):
        """
        Visualize stage dependency graph.

        Args:
            output_path: Path to save visualization
        """
        if not self.stage_graph.nodes:
            self.logger.warning("No stages to visualize")
            return

        plt.figure(figsize=(12, 8))

        # Get hierarchical layout
        pos = nx.spring_layout(self.stage_graph, seed=42)

        # Color nodes by status
        node_colors = []
        for stage_id in self.stage_graph.nodes:
            if stage_id in self.completed_stages:
                node_colors.append('green')
            elif stage_id in self.failed_stages:
                node_colors.append('red')
            else:
                node_colors.append('gray')

        # Draw graph
        nx.draw(
            self.stage_graph,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=2000,
            font_size=8,
            font_weight='bold',
            arrows=True,
            arrowsize=20
        )

        plt.title(f"Theatre: {self.theatre_name}\nStage Dependency Graph")

        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            self.logger.info(f"Saved stage graph: {output_path}")

        plt.close()

    def get_summary(self) -> Dict[str, Any]:
        """Get theatre execution summary"""
        return {
            'theatre_name': self.theatre_name,
            'status': self.status.value,
            'navigation_mode': self.navigation_mode.value,
            'total_stages': len(self.stages),
            'completed_stages': len(self.completed_stages),
            'failed_stages': len(self.failed_stages),
            'execution_order': self.execution_order,
            'navigation_history': self.navigation_history,
            'output_dir': str(self.output_dir)
        }

    def __repr__(self) -> str:
        return f"<Theatre(name='{self.theatre_name}', stages={len(self.stages)}, mode={self.navigation_mode.value})>"


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    from .stages import create_stage

    print("\n" + "="*70)
    print("THEATRE - Transcendent Observer")
    print("="*70)

    # Create theatre
    theatre = Theatre(
        theatre_name="MS_Analysis_Pipeline",
        output_dir=Path("./test_theatre_results"),
        navigation_mode=NavigationMode.DEPENDENCY
    )

    # Create stages
    stage1 = create_stage(
        stage_name="Data Loading",
        stage_id="stage_01_data_loading",
        process_names=['data_loading'],
        save_dir=Path("./test_theatre_results/stage_01")
    )

    stage2 = create_stage(
        stage_name="Feature Extraction",
        stage_id="stage_02_features",
        process_names=['feature_extraction'],
        save_dir=Path("./test_theatre_results/stage_02")
    )

    stage3 = create_stage(
        stage_name="Annotation",
        stage_id="stage_03_annotation",
        process_names=['annotation'],
        save_dir=Path("./test_theatre_results/stage_03")
    )

    # Add stages to theatre
    theatre.add_stage(stage1)
    theatre.add_stage(stage2)
    theatre.add_stage(stage3)

    # Define dependencies
    theatre.add_stage_dependency("stage_01_data_loading", "stage_02_features")
    theatre.add_stage_dependency("stage_02_features", "stage_03_annotation")

    # Define observations
    theatre.add_stage_observation("stage_03_annotation", "stage_01_data_loading")

    # Execute theatre
    result = theatre.observe_all_stages(
        input_data="test_experiment.mzml",
        mode=NavigationMode.GEAR_RATIO
    )

    # Display summary
    summary = theatre.get_summary()
    print("\nTheatre Summary:")
    print(f"  Status: {summary['status']}")
    print(f"  Completed: {summary['completed_stages']}/{summary['total_stages']}")
    print(f"  Execution order: {' → '.join(summary['execution_order'])}")
    print(f"  Navigation history: {len(summary['navigation_history'])} transitions")

    # Visualize
    theatre.visualize_stage_graph(Path("./test_theatre_results/stage_graph.png"))

    print("\n" + "="*70)

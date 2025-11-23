"""
Pipeline Stages - Finite Observers Architecture
================================================

Each stage is a finite observer watching lower-level process observers.
A stage is a coherent collection of processes used together (e.g., annotation
algorithms and their imports).

Key Principles:
1. No inherent sequence - stages can be executed in any order
2. Each stage saves results (.tab/.json) for bidirectional movement
3. Stages are composable units of computation
4. Stages observe other stages through result artifacts

Architecture:
- ProcessObserver: Observes a single computational process
- StageObserver: Observes multiple ProcessObservers (finite observer)
- Results saved at every stage enable navigation in any direction
"""

import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging

# Import BMD components
try:
    from ..bmd import (
        BMDState,
        CategoricalState,
        compute_ambiguity,
        generate_bmd_from_comparison
    )
    BMD_AVAILABLE = True
except ImportError:
    BMD_AVAILABLE = False


class ObserverLevel(Enum):
    """Hierarchical observer levels"""
    PROCESS = "process"           # Lowest level - individual computation
    STAGE = "stage"               # Middle level - collection of processes
    THEATRE = "theatre"           # Highest level - transcendent observer


class StageStatus(Enum):
    """Execution status of a stage"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProcessResult:
    """
    Result from a single process observation.

    Represents the output of a ProcessObserver.
    """
    process_name: str
    status: StageStatus
    execution_time: float
    data: Any
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large data)"""
        return {
            'process_name': self.process_name,
            'status': self.status.value,
            'execution_time': self.execution_time,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'error_message': self.error_message,
            'data_type': str(type(self.data).__name__),
            'data_shape': getattr(self.data, 'shape', None) if hasattr(self.data, 'shape') else None
        }


@dataclass
class StageResult:
    """
    Result from a stage observation.

    Represents the output of a StageObserver, containing all process results.
    """
    stage_name: str
    stage_id: str
    observer_level: ObserverLevel
    status: StageStatus
    execution_time: float
    process_results: List[ProcessResult] = field(default_factory=list)
    output_data: Any = None
    output_files: List[Path] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    # BMD grounding fields
    generated_bmd: Optional['BMDState'] = None  # BMD generated through categorical completion
    input_filter_count: int = 0  # Number of inputs filtered
    output_filter_count: int = 0  # Number of outputs filtered
    ambiguity: float = 0.0  # Final ambiguity measure

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'stage_name': self.stage_name,
            'stage_id': self.stage_id,
            'observer_level': self.observer_level.value,
            'status': self.status.value,
            'execution_time': self.execution_time,
            'process_results': [pr.to_dict() for pr in self.process_results],
            'output_files': [str(f) for f in self.output_files],
            'metrics': self.metrics,
            'metadata': self.metadata,
            'error_message': self.error_message
        }

    def save_json(self, output_path: Union[str, Path]):
        """Save result as JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        self.output_files.append(output_path)

    def save_tab(self, output_path: Union[str, Path]):
        """Save result as tab-delimited file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame if possible
        if isinstance(self.output_data, pd.DataFrame):
            self.output_data.to_csv(output_path, sep='\t', index=False)
        elif isinstance(self.output_data, dict):
            df = pd.DataFrame([self.output_data])
            df.to_csv(output_path, sep='\t', index=False)
        elif isinstance(self.output_data, list):
            df = pd.DataFrame(self.output_data)
            df.to_csv(output_path, sep='\t', index=False)
        else:
            # Fallback: save metrics and metadata
            data = {
                'stage_name': [self.stage_name],
                'status': [self.status.value],
                'execution_time': [self.execution_time],
                **{f'metric_{k}': [v] for k, v in self.metrics.items()}
            }
            df = pd.DataFrame(data)
            df.to_csv(output_path, sep='\t', index=False)

        self.output_files.append(output_path)


class ProcessObserver(ABC):
    """
    Process Observer - Lowest level finite observer.

    Observes a single computational process (e.g., data loading, feature extraction).
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize process observer.

        Args:
            name: Process name
            config: Process configuration
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"ProcessObserver.{name}")

    @abstractmethod
    def observe(self, input_data: Any, **kwargs) -> ProcessResult:
        """
        Observe the process execution.

        Args:
            input_data: Input data for process
            **kwargs: Additional parameters

        Returns:
            ProcessResult
        """
        pass

    def __repr__(self) -> str:
        return f"<ProcessObserver(name='{self.name}')>"


class StageObserver:
    """
    Stage Observer - Middle level finite observer.

    Observes multiple ProcessObservers. A stage is a coherent collection of
    processes used together (e.g., annotation algorithms and their imports).

    Key Features:
    - Watches lower-level ProcessObservers
    - Saves results at completion (.tab and .json)
    - Enables bidirectional navigation
    - No inherent execution order
    """

    def __init__(
        self,
        stage_name: str,
        stage_id: str,
        process_observers: Optional[List[ProcessObserver]] = None,
        save_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize stage observer.

        Args:
            stage_name: Stage name
            stage_id: Unique stage identifier
            process_observers: List of ProcessObservers in this stage
            save_dir: Directory to save stage results
            config: Stage configuration
        """
        self.stage_name = stage_name
        self.stage_id = stage_id
        self.process_observers = process_observers or []
        self.save_dir = Path(save_dir) if save_dir else Path(f"./stage_results/{stage_id}")
        self.config = config or {}
        self.logger = logging.getLogger(f"StageObserver.{stage_name}")

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Execution state
        self.result: Optional[StageResult] = None
        self.dependencies: List[str] = []  # Stage IDs this stage depends on
        self.observers: List[str] = []  # Stage IDs this stage observes

    def add_process_observer(self, observer: ProcessObserver):
        """Add process observer to this stage"""
        self.process_observers.append(observer)

    def add_dependency(self, stage_id: str):
        """Add stage dependency"""
        self.dependencies.append(stage_id)

    def add_observer(self, stage_id: str):
        """Add stage to observe"""
        self.observers.append(stage_id)

    def observe(
        self,
        input_data: Any,
        previous_stage_results: Optional[Dict[str, StageResult]] = None,
        **kwargs
    ) -> StageResult:
        """
        Observe all processes in this stage.

        Args:
            input_data: Input data for stage
            previous_stage_results: Results from previous stages (for observation)
            **kwargs: Additional parameters

        Returns:
            StageResult with all process results
        """
        self.logger.info(f"[{self.stage_name}] Starting stage observation...")
        start_time = time.perf_counter()

        process_results = []
        stage_status = StageStatus.COMPLETED
        stage_error = None
        current_data = input_data

        try:
            # Observe previous stages if configured
            if previous_stage_results and self.observers:
                self.logger.info(f"  Observing {len(self.observers)} previous stages...")
                for observed_stage_id in self.observers:
                    if observed_stage_id in previous_stage_results:
                        observed_result = previous_stage_results[observed_stage_id]
                        self.logger.info(f"    Observed: {observed_result.stage_name} "
                                       f"(status: {observed_result.status.value})")

            # Execute each process observer
            for i, process_observer in enumerate(self.process_observers, 1):
                self.logger.info(f"  [{i}/{len(self.process_observers)}] Observing process: {process_observer.name}")

                try:
                    process_result = process_observer.observe(current_data, **kwargs)
                    process_results.append(process_result)

                    # Update current data for next process
                    if process_result.data is not None:
                        current_data = process_result.data

                    # Track failures
                    if process_result.status == StageStatus.FAILED:
                        stage_status = StageStatus.FAILED
                        stage_error = process_result.error_message
                        self.logger.error(f"    Process failed: {process_result.error_message}")
                        break

                except Exception as e:
                    self.logger.error(f"    Process observation error: {str(e)}")
                    process_results.append(ProcessResult(
                        process_name=process_observer.name,
                        status=StageStatus.FAILED,
                        execution_time=0.0,
                        data=None,
                        error_message=str(e)
                    ))
                    stage_status = StageStatus.FAILED
                    stage_error = str(e)
                    break

            execution_time = time.perf_counter() - start_time

            # Compute stage metrics
            stage_metrics = {
                'total_processes': len(self.process_observers),
                'completed_processes': sum(1 for pr in process_results if pr.status == StageStatus.COMPLETED),
                'failed_processes': sum(1 for pr in process_results if pr.status == StageStatus.FAILED),
                'average_process_time': np.mean([pr.execution_time for pr in process_results]) if process_results else 0.0
            }

            # Create stage result
            self.result = StageResult(
                stage_name=self.stage_name,
                stage_id=self.stage_id,
                observer_level=ObserverLevel.STAGE,
                status=stage_status,
                execution_time=execution_time,
                process_results=process_results,
                output_data=current_data,
                metrics=stage_metrics,
                metadata={
                    'dependencies': self.dependencies,
                    'observers': self.observers,
                    'config': self.config
                },
                error_message=stage_error
            )

            # CRITICAL: Save results at every stage
            self._save_stage_results()

            self.logger.info(f"[{self.stage_name}] Stage complete: {stage_status.value} "
                           f"in {execution_time:.2f}s")

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            self.logger.error(f"[{self.stage_name}] Stage failed: {str(e)}")

            self.result = StageResult(
                stage_name=self.stage_name,
                stage_id=self.stage_id,
                observer_level=ObserverLevel.STAGE,
                status=StageStatus.FAILED,
                execution_time=execution_time,
                process_results=process_results,
                error_message=str(e)
            )

            # Save even on failure
            self._save_stage_results()

        return self.result

    def observe_with_bmd_grounding(
        self,
        input_data: Any,
        hardware_bmd: Optional['BMDState'] = None,
        network_bmd: Optional['BMDState'] = None,
        previous_stage_results: Optional[Dict[str, StageResult]] = None,
        **kwargs
    ) -> StageResult:
        """
        Observe stage with BMD dual filtering (input + output).

        Implements Biological Maxwell Demon operations:
        1. Input Filter (ℑ_input): Select signal from noise via phase-lock coherence
        2. Process with BMD comparison and generation
        3. Output Filter (ℑ_output): Target physically-grounded interpretations

        Args:
            input_data: Input data
            hardware_bmd: Hardware BMD stream (reality reference)
            network_bmd: Network BMD from processing history
            previous_stage_results: Results from previous stages
            **kwargs: Additional parameters

        Returns:
            StageResult with BMD grounding metadata
        """
        self.logger.info(f"[{self.stage_name}] Starting BMD-grounded observation...")
        start_time = time.perf_counter()

        process_results = []
        stage_status = StageStatus.COMPLETED
        stage_error = None
        current_data = input_data

        # Initialize BMD for this stage
        current_bmd = hardware_bmd if hardware_bmd else None
        input_filtered_count = 0
        output_filtered_count = 0

        try:
            # STEP 1: INPUT FILTER - Select signal from noise
            if current_bmd and hasattr(input_data, '__iter__') and not isinstance(input_data, (str, dict)):
                self.logger.info("  [INPUT FILTER] Applying phase-lock filtering...")

                # Convert input to list if needed
                input_candidates = list(input_data) if not isinstance(input_data, list) else input_data
                original_count = len(input_candidates)

                # Apply BMD input filter
                filtered_inputs = current_bmd.input_filter(
                    candidates=input_candidates,
                    criterion='phase_lock'
                )

                input_filtered_count = original_count - len(filtered_inputs)
                self.logger.info(f"  Filtered {input_filtered_count}/{original_count} inputs (noise rejection)")

                # Use filtered data
                current_data = filtered_inputs if filtered_inputs else input_data

            # STEP 2: Process observations with BMD comparison
            for i, process_observer in enumerate(self.process_observers, 1):
                self.logger.info(f"  [{i}/{len(self.process_observers)}] Observing process: {process_observer.name}")

                # Compute ambiguity before processing
                if current_bmd:
                    try:
                        ambiguity = compute_ambiguity(current_bmd, current_data)
                        self.logger.info(f"    Ambiguity: {ambiguity:.3f}")
                    except:
                        ambiguity = 0.0

                try:
                    # Execute process
                    process_result = process_observer.observe(current_data, **kwargs)
                    process_results.append(process_result)

                    # Generate new BMD through categorical completion
                    if current_bmd and process_result.data is not None:
                        try:
                            current_bmd = generate_bmd_from_comparison(
                                bmd=current_bmd,
                                target=process_result.data,
                                hardware_bmd=hardware_bmd if hardware_bmd else current_bmd
                            )
                            self.logger.info(f"    Generated new BMD (richness={current_bmd.categorical_richness})")
                        except Exception as e:
                            self.logger.warning(f"    Could not generate BMD: {e}")

                    # Update current data
                    if process_result.data is not None:
                        current_data = process_result.data

                    # Track failures
                    if process_result.status == StageStatus.FAILED:
                        stage_status = StageStatus.FAILED
                        stage_error = process_result.error_message
                        self.logger.error(f"    Process failed: {process_result.error_message}")
                        break

                except Exception as e:
                    self.logger.error(f"    Process observation error: {str(e)}")
                    process_results.append(ProcessResult(
                        process_name=process_observer.name,
                        status=StageStatus.FAILED,
                        execution_time=0.0,
                        data=None,
                        error_message=str(e)
                    ))
                    stage_status = StageStatus.FAILED
                    stage_error = str(e)
                    break

            # STEP 3: OUTPUT FILTER - Select physically-grounded interpretations
            if current_bmd and hasattr(current_data, '__iter__') and not isinstance(current_data, (str, dict)):
                self.logger.info("  [OUTPUT FILTER] Applying stream coherence filtering...")

                # Convert output to list if needed
                output_candidates = list(current_data) if not isinstance(current_data, list) else current_data
                original_count = len(output_candidates)

                # Apply BMD output filter
                filtered_outputs = current_bmd.output_filter(
                    interpretations=output_candidates,
                    hardware_bmd=hardware_bmd if hardware_bmd else current_bmd
                )

                output_filtered_count = original_count - len(filtered_outputs)
                self.logger.info(f"  Filtered {output_filtered_count}/{original_count} outputs (unphysical interpretations)")

                # Use filtered data
                current_data = filtered_outputs if filtered_outputs else current_data

            execution_time = time.perf_counter() - start_time

            # Compute final ambiguity
            final_ambiguity = 0.0
            if current_bmd:
                try:
                    final_ambiguity = compute_ambiguity(current_bmd, current_data)
                except:
                    pass

            # Compute stage metrics
            stage_metrics = {
                'total_processes': len(self.process_observers),
                'completed_processes': sum(1 for pr in process_results if pr.status == StageStatus.COMPLETED),
                'failed_processes': sum(1 for pr in process_results if pr.status == StageStatus.FAILED),
                'average_process_time': np.mean([pr.execution_time for pr in process_results]) if process_results else 0.0,
                'input_filtered': input_filtered_count,
                'output_filtered': output_filtered_count,
                'final_ambiguity': final_ambiguity,
                'bmd_categorical_richness': current_bmd.categorical_richness if current_bmd else 0
            }

            # Create stage result with BMD fields
            self.result = StageResult(
                stage_name=self.stage_name,
                stage_id=self.stage_id,
                observer_level=ObserverLevel.STAGE,
                status=stage_status,
                execution_time=execution_time,
                process_results=process_results,
                output_data=current_data,
                metrics=stage_metrics,
                metadata={
                    'dependencies': self.dependencies,
                    'observers': self.observers,
                    'config': self.config,
                    'bmd_grounding': True,
                    'hardware_bmd_available': hardware_bmd is not None,
                    'network_bmd_available': network_bmd is not None
                },
                error_message=stage_error,
                generated_bmd=current_bmd,
                input_filter_count=input_filtered_count,
                output_filter_count=output_filtered_count,
                ambiguity=final_ambiguity
            )

            # Save results
            self._save_stage_results()

            self.logger.info(f"[{self.stage_name}] BMD-grounded stage complete: {stage_status.value} in {execution_time:.2f}s")
            self.logger.info(f"  Input filtering: {input_filtered_count} rejected")
            self.logger.info(f"  Output filtering: {output_filtered_count} rejected")
            self.logger.info(f"  Final ambiguity: {final_ambiguity:.3f}")

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            self.logger.error(f"[{self.stage_name}] BMD-grounded stage failed: {str(e)}")

            self.result = StageResult(
                stage_name=self.stage_name,
                stage_id=self.stage_id,
                observer_level=ObserverLevel.STAGE,
                status=StageStatus.FAILED,
                execution_time=execution_time,
                process_results=process_results,
                error_message=str(e),
                generated_bmd=current_bmd
            )

            self._save_stage_results()

        return self.result

    def _save_stage_results(self):
        """
        CRITICAL: Save stage results in both .json and .tab formats.

        This enables bidirectional navigation - can move in any direction.
        """
        if not self.result:
            return

        # Save JSON (complete result)
        json_path = self.save_dir / f"{self.stage_id}_result.json"
        self.result.save_json(json_path)
        self.logger.info(f"  Saved JSON: {json_path}")

        # Save TAB (data table)
        tab_path = self.save_dir / f"{self.stage_id}_data.tab"
        try:
            self.result.save_tab(tab_path)
            self.logger.info(f"  Saved TAB: {tab_path}")
        except Exception as e:
            self.logger.warning(f"  Could not save TAB: {str(e)}")

        # Save process-level results
        process_summary = pd.DataFrame([pr.to_dict() for pr in self.result.process_results])
        process_path = self.save_dir / f"{self.stage_id}_processes.tab"
        process_summary.to_csv(process_path, sep='\t', index=False)
        self.logger.info(f"  Saved process summary: {process_path}")

        # SAVE ACTUAL MS DATA - Stage-specific outputs
        self._save_ms_data()

    def _save_ms_data(self):
        """
        Save actual mass spectrometry data based on stage type.

        This saves the REAL data: spectra, peaks, annotations, etc.
        """
        if not self.result or not self.result.output_data:
            return

        data = self.result.output_data

        # Safety check: data must be a dictionary
        if not isinstance(data, dict):
            return

        try:
            # SPECTRAL DATA (Stage 1)
            if 'spectra' in data and 'scan_info' in data:
                self._save_spectral_data(data)

            # FEATURE DATA (Stage 2)
            if 'sentropy_features' in data or 'categorical_states' in data:
                self._save_feature_data(data)

            # COMPUTER VISION DATA (Stage 2 CV)
            if 'cv_images' in data or 'cv_features' in data or 'ion_droplets' in data:
                self._save_cv_data(data)

            # BMD/GROUNDING DATA (Stage 3)
            if 'hardware_bmd_stream' in data or 'divergences' in data:
                self._save_bmd_data(data)

            # ANNOTATION DATA (Stage 4)
            if 'annotations' in data or 'cv_matches' in data:
                self._save_annotation_data(data)

        except Exception as e:
            self.logger.warning(f"  Could not save MS-specific data: {e}")

    def _save_spectral_data(self, data):
        """Save raw spectral data: scan info, peak lists, XIC"""
        # Save scan information
        if 'scan_info' in data:
            scan_path = self.save_dir / "scan_info.tsv"
            data['scan_info'].to_csv(scan_path, sep='\t', index=False)
            self.logger.info(f"  Saved scan info: {scan_path} ({len(data['scan_info'])} scans)")

        # Save XIC data
        if 'xic' in data:
            xic_path = self.save_dir / "xic_data.tsv"
            data['xic'].to_csv(xic_path, sep='\t', index=False)
            self.logger.info(f"  Saved XIC: {xic_path}")

        # Save individual spectra
        if 'spectra' in data:
            spectra_dir = self.save_dir / "spectra"
            spectra_dir.mkdir(exist_ok=True)

            count = 0
            for spec_id, spec_df in data['spectra'].items():
                if isinstance(spec_df, pd.DataFrame) and not spec_df.empty:
                    spec_path = spectra_dir / f"spectrum_{spec_id}.tsv"
                    spec_df.to_csv(spec_path, sep='\t', index=False)
                    count += 1

                    if count <= 5:  # Log first few
                        self.logger.info(f"    Saved spectrum {spec_id}: {len(spec_df)} peaks")

            self.logger.info(f"  Saved {count} spectra to: {spectra_dir}/")

    def _save_feature_data(self, data):
        """Save S-Entropy features and categorical states"""
        # Save S-Entropy features
        if 'sentropy_features' in data:
            features_list = []
            for scan_id, features in data['sentropy_features'].items():
                if hasattr(features, '__dict__'):
                    row = {'scan_id': scan_id}
                    row.update(features.__dict__)
                    features_list.append(row)
                elif isinstance(features, dict):
                    row = {'scan_id': scan_id}
                    row.update(features)
                    features_list.append(row)

            if features_list:
                features_df = pd.DataFrame(features_list)
                features_path = self.save_dir / "sentropy_features.tsv"
                features_df.to_csv(features_path, sep='\t', index=False)
                self.logger.info(f"  Saved S-Entropy features: {features_path} ({len(features_df)} spectra)")

        # Save categorical states
        if 'categorical_states' in data:
            states_list = []
            for scan_id, state in data['categorical_states'].items():
                if isinstance(state, dict):
                    row = {'scan_id': scan_id}
                    row.update(state)
                    states_list.append(row)
                elif hasattr(state, '__dict__'):
                    row = {'scan_id': scan_id}
                    row.update(state.__dict__)
                    states_list.append(row)

            if states_list:
                states_df = pd.DataFrame(states_list)
                states_path = self.save_dir / "categorical_states.tsv"
                states_df.to_csv(states_path, sep='\t', index=False)
                self.logger.info(f"  Saved categorical states: {states_path} ({len(states_df)} states)")

    def _save_bmd_data(self, data):
        """Save BMD grounding and coherence data"""
        if 'divergences' in data:
            div_list = [{'scan_id': k, 'divergence': v} for k, v in data['divergences'].items()]
            div_df = pd.DataFrame(div_list)
            div_path = self.save_dir / "stream_divergences.tsv"
            div_df.to_csv(div_path, sep='\t', index=False)
            self.logger.info(f"  Saved divergences: {div_path}")

        if 'coherence_scores' in data:
            coh_list = [{'scan_id': k, 'coherence': v} for k, v in data['coherence_scores'].items()]
            coh_df = pd.DataFrame(coh_list)
            coh_path = self.save_dir / "coherence_scores.tsv"
            coh_df.to_csv(coh_path, sep='\t', index=False)
            self.logger.info(f"  Saved coherence: {coh_path}")

    def _save_cv_data(self, data):
        """Save computer vision data: images, features, ion droplets"""
        try:
            import cv2
        except ImportError:
            self.logger.warning("  OpenCV not available, skipping CV image saving")
            cv2 = None

        # Save CV images as PNG
        if 'cv_images' in data and data['cv_images'] and cv2 is not None:
            cv_images_dir = self.save_dir / "cv_images"
            cv_images_dir.mkdir(exist_ok=True)

            count = 0
            for scan_id, image in data['cv_images'].items():
                image_path = cv_images_dir / f"spectrum_{scan_id}_droplet.png"
                # Ensure uint8
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
                cv2.imwrite(str(image_path), image)
                count += 1

            self.logger.info(f"  Saved {count} CV droplet images: {cv_images_dir}/")

        # Save CV features summary
        if 'cv_features' in data and data['cv_features']:
            cv_features_path = self.save_dir / "cv_features.tsv"

            cv_records = []
            for scan_id, features in data['cv_features'].items():
                record = {
                    'scan_id': scan_id,
                    'n_keypoints': features.get('n_keypoints', 0),
                    'n_droplets': features.get('n_droplets', 0)
                }
                cv_records.append(record)

            cv_df = pd.DataFrame(cv_records)
            cv_df.to_csv(cv_features_path, sep='\t', index=False)
            self.logger.info(f"  Saved CV features: {cv_features_path} ({len(cv_df)} spectra)")

        # Save ion droplet details
        if 'ion_droplets' in data and data['ion_droplets']:
            droplets_path = self.save_dir / "ion_droplets.tsv"

            droplet_records = []
            for scan_id, droplets in data['ion_droplets'].items():
                if droplets and len(droplets) > 0:
                    for i, droplet in enumerate(droplets):
                        if hasattr(droplet, 'mz'):  # Check it's an IonDroplet object
                            record = {
                                'scan_id': scan_id,
                                'droplet_idx': i,
                                'mz': droplet.mz,
                                'intensity': droplet.intensity,
                                's_knowledge': droplet.s_entropy_coords.s_knowledge,
                                's_time': droplet.s_entropy_coords.s_time,
                                's_entropy': droplet.s_entropy_coords.s_entropy,
                                'velocity': droplet.droplet_params.velocity,
                                'radius': droplet.droplet_params.radius,
                                'phase_coherence': droplet.droplet_params.phase_coherence,
                                'categorical_state': droplet.categorical_state
                            }
                            droplet_records.append(record)

            if droplet_records:
                droplet_df = pd.DataFrame(droplet_records)
                droplet_df.to_csv(droplets_path, sep='\t', index=False)
                self.logger.info(f"  Saved ion droplets: {droplets_path} ({len(droplet_records)} droplets)")

    def _save_annotation_data(self, data):
        """Save metabolite/compound annotations with CV matching details"""
        # Save main annotations
        if 'annotations' in data and data['annotations']:
            annot_list = []
            for scan_id, annot in data['annotations'].items():
                if isinstance(annot, dict):
                    row = {'scan_id': scan_id}
                    row.update(annot)
                    annot_list.append(row)

            if annot_list:
                annot_df = pd.DataFrame(annot_list)
                annot_path = self.save_dir / "annotations.tsv"
                annot_df.to_csv(annot_path, sep='\t', index=False)
                self.logger.info(f"  Saved annotations: {annot_path} ({len(annot_df)} annotations)")

        # Save detailed CV matches
        if 'cv_matches' in data and data['cv_matches']:
            cv_matches_path = self.save_dir / "cv_matches_detailed.tsv"

            match_records = []
            for scan_id, match_data in data['cv_matches'].items():
                if isinstance(match_data, dict) and 'top_matches' in match_data:
                    for i, match in enumerate(match_data['top_matches']):
                        record = {
                            'scan_id': scan_id,
                            'match_rank': i + 1,
                            'database_id': match.get('database_id', ''),
                            'similarity': match.get('similarity', 0.0),
                            'structural_similarity': match.get('structural_similarity', 0.0),
                            'phase_lock_similarity': match.get('phase_lock_similarity', 0.0),
                            'categorical_match': match.get('categorical_match', 0.0),
                            's_entropy_distance': match.get('s_entropy_distance', 0.0),
                            'n_matched_features': match.get('n_matched_features', 0)
                        }
                        match_records.append(record)

            if match_records:
                match_df = pd.DataFrame(match_records)
                match_df.to_csv(cv_matches_path, sep='\t', index=False)
                self.logger.info(f"  Saved CV match details: {cv_matches_path} ({len(match_records)} matches)")

    def load_stage_result(self, stage_result_path: Optional[Path] = None) -> Optional[StageResult]:
        """
        Load previously saved stage result.

        Enables resumption and bidirectional navigation.

        Args:
            stage_result_path: Path to JSON result file (defaults to standard location)

        Returns:
            StageResult if found, None otherwise
        """
        if stage_result_path is None:
            stage_result_path = self.save_dir / f"{self.stage_id}_result.json"

        if not Path(stage_result_path).exists():
            self.logger.warning(f"No saved result found at {stage_result_path}")
            return None

        try:
            with open(stage_result_path, 'r') as f:
                result_dict = json.load(f)

            self.logger.info(f"Loaded stage result from {stage_result_path}")
            return result_dict  # Simplified - would reconstruct StageResult in full implementation

        except Exception as e:
            self.logger.error(f"Failed to load stage result: {str(e)}")
            return None

    def can_observe_stage(self, stage_id: str, available_results: Dict[str, StageResult]) -> bool:
        """
        Check if this stage can observe another stage (results available).

        Args:
            stage_id: Stage ID to observe
            available_results: Currently available stage results

        Returns:
            True if observation possible
        """
        return stage_id in available_results and \
               available_results[stage_id].status == StageStatus.COMPLETED

    def __repr__(self) -> str:
        return f"<StageObserver(name='{self.stage_name}', id='{self.stage_id}', processes={len(self.process_observers)})>"


# ============================================================================
# CONCRETE PROCESS OBSERVERS
# ============================================================================

class DataLoadingProcess(ProcessObserver):
    """Process observer for data loading"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("data_loading", config)

    def observe(self, input_data: Any, **kwargs) -> ProcessResult:
        start_time = time.perf_counter()

        try:
            # Mock data loading (replace with actual implementation)
            file_path = kwargs.get('file_path', input_data)

            # Simulate loading
            loaded_data = pd.DataFrame({'mz': [100, 200, 300], 'intensity': [1000, 800, 1200]})

            execution_time = time.perf_counter() - start_time

            return ProcessResult(
                process_name=self.name,
                status=StageStatus.COMPLETED,
                execution_time=execution_time,
                data=loaded_data,
                metrics={'rows_loaded': len(loaded_data)},
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


class FeatureExtractionProcess(ProcessObserver):
    """Process observer for feature extraction"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("feature_extraction", config)

    def observe(self, input_data: Any, **kwargs) -> ProcessResult:
        start_time = time.perf_counter()

        try:
            # Extract features from input data
            if isinstance(input_data, pd.DataFrame):
                features = input_data.describe().to_dict()
            else:
                features = {'extracted': True}

            execution_time = time.perf_counter() - start_time

            return ProcessResult(
                process_name=self.name,
                status=StageStatus.COMPLETED,
                execution_time=execution_time,
                data=features,
                metrics={'feature_count': len(features)},
                metadata={'input_type': str(type(input_data).__name__)}
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


class AnnotationProcess(ProcessObserver):
    """Process observer for annotation"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("annotation", config)

    def observe(self, input_data: Any, **kwargs) -> ProcessResult:
        start_time = time.perf_counter()

        try:
            # Perform annotation
            annotations = {
                'annotated_count': 10,
                'confidence_threshold': 0.8
            }

            execution_time = time.perf_counter() - start_time

            return ProcessResult(
                process_name=self.name,
                status=StageStatus.COMPLETED,
                execution_time=execution_time,
                data=annotations,
                metrics={'annotated': annotations['annotated_count']},
                metadata={'config': self.config}
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
# STAGE FACTORY
# ============================================================================

def create_stage(
    stage_name: str,
    stage_id: str,
    process_names: List[str],
    save_dir: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None
) -> StageObserver:
    """
    Factory function to create pre-configured stages.

    Args:
        stage_name: Stage name
        stage_id: Unique stage identifier
        process_names: List of process names to include
        save_dir: Directory to save results
        config: Stage configuration

    Returns:
        Configured StageObserver
    """
    process_map = {
        'data_loading': DataLoadingProcess,
        'feature_extraction': FeatureExtractionProcess,
        'annotation': AnnotationProcess
    }

    process_observers = []
    for name in process_names:
        if name in process_map:
            process_observers.append(process_map[name](config=config))
        else:
            logging.warning(f"Unknown process: {name}")

    return StageObserver(
        stage_name=stage_name,
        stage_id=stage_id,
        process_observers=process_observers,
        save_dir=save_dir,
        config=config
    )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*70)
    print("STAGE OBSERVERS - Finite Observer Architecture")
    print("="*70)

    # Create a stage
    stage = create_stage(
        stage_name="Data Processing Stage",
        stage_id="stage_01_data_processing",
        process_names=['data_loading', 'feature_extraction'],
        save_dir=Path("./test_stage_results"),
        config={'test': True}
    )

    # Observe the stage
    result = stage.observe(input_data="test_data.mzml")

    print(f"\nStage Result:")
    print(f"  Status: {result.status.value}")
    print(f"  Execution time: {result.execution_time:.3f}s")
    print(f"  Processes: {result.metrics['completed_processes']}/{result.metrics['total_processes']}")
    print(f"  Output files: {len(result.output_files)}")

    for output_file in result.output_files:
        print(f"    - {output_file}")

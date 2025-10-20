from typing import Dict, List, Optional, Union, Any, Tuple
import os
from pathlib import Path
import logging
import time
import json
import threading
from enum import Enum
from dataclasses import dataclass

from lavoisier.core.config import GlobalConfig
from lavoisier.core.logging import get_logger, ProgressLogger


class PipelineType(Enum):
    """Types of analysis pipelines"""
    NUMERIC = "numeric"
    VISUAL = "visual"
    COMPARISON = "comparison"
    HYBRID = "hybrid"


class AnalysisStatus(Enum):
    """Status of an analysis task"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class AnalysisTask:
    """Represents an analysis task to be executed"""
    id: str
    type: PipelineType
    input_files: List[str]
    output_dir: str
    parameters: Dict[str, Any]
    status: AnalysisStatus = AnalysisStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class Orchestrator:
    """
    Metacognitive orchestration layer that manages multiple pipelines,
    coordinates their execution, and integrates LLM capabilities.
    """

    def __init__(self, config: GlobalConfig):
        """
        Initialize the orchestrator with the given configuration

        Args:
            config: Lavoisier configuration
        """
        self.config = config
        self.logger = get_logger("orchestrator")
        self.tasks: Dict[str, AnalysisTask] = {}
        self.running_tasks: Dict[str, threading.Thread] = {}
        self.lock = threading.Lock()

        # Will be initialized lazily
        self._numeric_pipeline = None
        self._visual_pipeline = None
        self._llm_service = None
        self._model_repository = None

        self.logger.info("Orchestrator initialized")

    def _init_numeric_pipeline(self):
        """Initialize the numeric pipeline if not already initialized"""
        if self._numeric_pipeline is None:
            from lavoisier.numerical.pipeline import NumericPipeline
            self._numeric_pipeline = NumericPipeline(self.config)
            self.logger.info("Numeric pipeline initialized")

    def _init_visual_pipeline(self):
        """Initialize the visual pipeline if not already initialized"""
        if self._visual_pipeline is None:
            from lavoisier.visual.pipeline import VisualPipeline
            self._visual_pipeline = VisualPipeline(self.config)
            self.logger.info("Visual pipeline initialized")

    def _init_llm_service(self):
        """Initialize the LLM service if not already initialized"""
        if self._llm_service is None and self.config.llm.enabled:
            from lavoisier.llm.service import LLMService
            self._llm_service = LLMService(self.config.llm)
            self.logger.info("LLM service initialized")

    def _init_model_repository(self):
        """Initialize the model repository if not already initialized"""
        if self._model_repository is None:
            from lavoisier.models.repository import ModelRepository
            self._model_repository = ModelRepository(self.config)
            self.logger.info("Model repository initialized")

    def create_task(
        self,
        pipeline_type: PipelineType,
        input_files: List[str],
        output_dir: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new analysis task

        Args:
            pipeline_type: Type of pipeline to run
            input_files: List of input files to process
            output_dir: Directory to store the results
            parameters: Additional parameters for the pipeline

        Returns:
            Task ID
        """
        import uuid

        task_id = str(uuid.uuid4())
        parameters = parameters or {}

        task = AnalysisTask(
            id=task_id,
            type=pipeline_type,
            input_files=input_files,
            output_dir=output_dir,
            parameters=parameters
        )

        with self.lock:
            self.tasks[task_id] = task

        self.logger.info(f"Created task {task_id} of type {pipeline_type.value}")
        return task_id

    def start_task(self, task_id: str) -> bool:
        """
        Start executing the specified task

        Args:
            task_id: ID of the task to start

        Returns:
            True if the task was started, False otherwise
        """
        with self.lock:
            if task_id not in self.tasks:
                self.logger.error(f"Task {task_id} not found")
                return False

            task = self.tasks[task_id]
            if task.status != AnalysisStatus.PENDING:
                self.logger.error(f"Task {task_id} is already {task.status.value}")
                return False

            # Update task status
            task.status = AnalysisStatus.RUNNING
            task.start_time = time.time()

            # Start task in a separate thread
            thread = threading.Thread(
                target=self._execute_task,
                args=(task_id,),
                daemon=True
            )
            self.running_tasks[task_id] = thread
            thread.start()

            self.logger.info(f"Started task {task_id}")
            return True

    def _execute_task(self, task_id: str) -> None:
        """
        Execute the specified task

        Args:
            task_id: ID of the task to execute
        """
        try:
            with self.lock:
                task = self.tasks[task_id]

            self.logger.info(f"Executing task {task_id} of type {task.type.value}")

            # Execute the task based on the pipeline type
            if task.type == PipelineType.NUMERIC:
                self._run_numeric_pipeline(task)
            elif task.type == PipelineType.VISUAL:
                self._run_visual_pipeline(task)
            elif task.type == PipelineType.COMPARISON:
                self._run_comparison_pipeline(task)
            elif task.type == PipelineType.HYBRID:
                self._run_hybrid_pipeline(task)
            else:
                raise ValueError(f"Unknown pipeline type: {task.type}")

            # Update task status
            with self.lock:
                task.status = AnalysisStatus.COMPLETED
                task.end_time = time.time()
                task.progress = 100.0

            self.logger.info(f"Task {task_id} completed successfully")

            # Optionally trigger continuous learning
            if self.config.metacognitive.enable_continuous_learning:
                self._trigger_continuous_learning(task)

        except Exception as e:
            self.logger.error(f"Error executing task {task_id}: {str(e)}", exc_info=True)

            # Update task status
            with self.lock:
                task = self.tasks[task_id]
                task.status = AnalysisStatus.FAILED
                task.end_time = time.time()
                task.error = str(e)

        finally:
            # Remove from running tasks
            with self.lock:
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]

    def _run_numeric_pipeline(self, task: AnalysisTask) -> None:
        """
        Run the numeric pipeline

        Args:
            task: Task to execute
        """
        self._init_numeric_pipeline()

        # Create a progress logger for the task
        with ProgressLogger(f"numeric_task_{task.id}", description="Running numeric pipeline") as progress:
            # Create a progress callback
            def update_progress(percent: float, message: str):
                with self.lock:
                    task.progress = percent
                progress.update(description=message)
                progress.info(message)

            # Run the pipeline
            result = self._numeric_pipeline.process_files(
                input_files=task.input_files,
                output_dir=task.output_dir,
                parameters=task.parameters,
                progress_callback=update_progress
            )

            # Store the result
            with self.lock:
                task.result = result

    def _run_visual_pipeline(self, task: AnalysisTask) -> None:
        """
        Run the visual pipeline

        Args:
            task: Task to execute
        """
        self._init_visual_pipeline()

        # Create a progress logger for the task
        with ProgressLogger(f"visual_task_{task.id}", description="Running visual pipeline") as progress:
            # Create a progress callback
            def update_progress(percent: float, message: str):
                with self.lock:
                    task.progress = percent
                progress.update(description=message)
                progress.info(message)

            # Run the pipeline
            result = self._visual_pipeline.process_files(
                input_files=task.input_files,
                output_dir=task.output_dir,
                parameters=task.parameters,
                progress_callback=update_progress
            )

            # Store the result
            with self.lock:
                task.result = result

    def _run_comparison_pipeline(self, task: AnalysisTask) -> None:
        """
        Run both pipelines and compare results

        Args:
            task: Task to execute
        """
        self._init_numeric_pipeline()
        self._init_visual_pipeline()

        # Create a progress logger for the task
        with ProgressLogger(f"comparison_task_{task.id}", description="Running comparison") as progress:
            # Create a progress callback
            def update_progress(percent: float, message: str):
                with self.lock:
                    task.progress = percent
                progress.update(description=message)
                progress.info(message)

            # Run numeric pipeline (50% of total progress)
            progress.info("Running numeric pipeline...")
            numeric_result = self._numeric_pipeline.process_files(
                input_files=task.input_files,
                output_dir=f"{task.output_dir}/numeric",
                parameters=task.parameters,
                progress_callback=lambda pct, msg: update_progress(pct * 0.5, f"Numeric: {msg}")
            )

            # Run visual pipeline (50% of total progress)
            progress.info("Running visual pipeline...")
            visual_result = self._visual_pipeline.process_files(
                input_files=task.input_files,
                output_dir=f"{task.output_dir}/visual",
                parameters=task.parameters,
                progress_callback=lambda pct, msg: update_progress(50 + pct * 0.5, f"Visual: {msg}")
            )

            # Compare results
            progress.info("Comparing results...")
            from lavoisier.core.comparison import compare_results
            comparison_result = compare_results(numeric_result, visual_result)

            # Store all results
            with self.lock:
                task.result = {
                    "numeric": numeric_result,
                    "visual": visual_result,
                    "comparison": comparison_result
                }

    def _run_hybrid_pipeline(self, task: AnalysisTask) -> None:
        """
        Run both pipelines together using hybrid approach

        Args:
            task: Task to execute
        """
        self._init_numeric_pipeline()
        self._init_visual_pipeline()
        self._init_llm_service()

        # Create a progress logger for the task
        with ProgressLogger(f"hybrid_task_{task.id}", description="Running hybrid analysis") as progress:
            # Create a progress callback
            def update_progress(percent: float, message: str):
                with self.lock:
                    task.progress = percent
                progress.update(description=message)
                progress.info(message)

            # Run numeric and visual pipelines in parallel
            progress.info("Running both pipelines...")

            # This would be implemented to coordinate both pipelines
            # with LLM assistance for integrating results
            # For now, we just run them sequentially

            numeric_result = self._numeric_pipeline.process_files(
                input_files=task.input_files,
                output_dir=f"{task.output_dir}/numeric",
                parameters=task.parameters,
                progress_callback=lambda pct, msg: update_progress(pct * 0.4, f"Numeric: {msg}")
            )

            visual_result = self._visual_pipeline.process_files(
                input_files=task.input_files,
                output_dir=f"{task.output_dir}/visual",
                parameters=task.parameters,
                progress_callback=lambda pct, msg: update_progress(40 + pct * 0.4, f"Visual: {msg}")
            )

            # Use LLM to integrate and analyze results if enabled
            combined_result = {
                "numeric": numeric_result,
                "visual": visual_result
            }

            if self._llm_service is not None:
                progress.info("Using LLM to analyze results...")
                llm_analysis = self._llm_service.analyze_results(
                    numeric_result,
                    visual_result,
                    progress_callback=lambda pct, msg: update_progress(80 + pct * 0.2, f"LLM: {msg}")
                )
                combined_result["llm_analysis"] = llm_analysis

            # Store results
            with self.lock:
                task.result = combined_result

    def _trigger_continuous_learning(self, task: AnalysisTask) -> None:
        """
        Trigger continuous learning based on task results

        Args:
            task: Completed task with results
        """
        self.logger.info(f"Triggering continuous learning for task {task.id}")

        # Initialize model repository
        self._init_model_repository()

        # Run continuous learning in a separate thread
        threading.Thread(
            target=self._perform_continuous_learning,
            args=(task,),
            daemon=True
        ).start()

    def _perform_continuous_learning(self, task: AnalysisTask) -> None:
        """
        Perform continuous learning based on task results

        Args:
            task: Completed task with results
        """
        try:
            self.logger.info(f"Performing continuous learning for task {task.id}")

            # Initialize LLM service if needed
            self._init_llm_service()

            # Extract training data from task results
            if not task.result:
                self.logger.warning(f"Task {task.id} has no results for continuous learning")
                return

            # Initialize model repository if needed
            if self._model_repository is None:
                self._init_model_repository()

            # 1. Extract training data from task results
            training_data = self._extract_training_data(task)

            # 2. Update models with new data if we have training data
            if training_data and self._model_repository is not None:
                # Update the models using the training data
                self._model_repository.update_models_from_data(training_data)
                self.logger.info(f"Updated models with training data from task {task.id}")

                # 3. Perform knowledge distillation if configured
                if self.config.metacognitive.knowledge_distillation and self._llm_service is not None:
                    self._llm_service.perform_knowledge_distillation(
                        task.result,
                        self._model_repository
                    )
                    self.logger.info(f"Performed knowledge distillation for task {task.id}")

                # 4. Validate updated models
                validation_result = self._model_repository.validate_models()
                if validation_result:
                    self.logger.info(f"Validated updated models: {validation_result}")
                else:
                    self.logger.warning("Failed to validate updated models")
            else:
                self.logger.info(f"No suitable training data found in task {task.id} for model updates")

        except Exception as e:
            self.logger.error(f"Error in continuous learning for task {task.id}: {str(e)}", exc_info=True)

    def _extract_training_data(self, task: AnalysisTask) -> Dict[str, Any]:
        """
        Extract training data from task results

        Args:
            task: Completed task

        Returns:
            Dictionary with training data for different model types
        """
        training_data = {}

        try:
            # Skip if task has no results
            if not task.result:
                return training_data

            # Extract data depending on the pipeline type
            if task.type == PipelineType.NUMERIC or task.type == PipelineType.HYBRID:
                if "numeric" in task.result:
                    numeric_results = task.result["numeric"]

                    # Extract MS2 spectra and possible labels from numerical results
                    spectra_data = []
                    annotation_data = []

                    # Process each file's results
                    for file_path, file_result in numeric_results.items():
                        if isinstance(file_result, dict) and "ml_enhanced" in file_result:
                            ml_results = file_result["ml_enhanced"]

                            # If there are annotations in the ML results, extract them as training data
                            if isinstance(ml_results, dict) and "annotations" in ml_results:
                                annotations = ml_results["annotations"]
                                scan_indices = annotations.get("scan_indices", [])
                                results = annotations.get("results", [])

                                # Extract original spectra from file_result["original"]
                                if "original" in file_result and isinstance(file_result["original"], tuple) and len(file_result["original"]) == 3:
                                    _, spec_dict, _ = file_result["original"]

                                    # Match spectra with annotations
                                    for idx, scan_idx in enumerate(scan_indices):
                                        if scan_idx in spec_dict and idx < len(results):
                                            spectrum = spec_dict[scan_idx]
                                            annotation = results[idx]

                                            # Add to training data if the annotation is confident enough
                                            high_confidence_ids = [
                                                id_obj for id_obj in annotation.get("identifications", [])
                                                if id_obj.get("confidence") == "high" and id_obj.get("score", 0) > 0.8
                                            ]

                                            if high_confidence_ids:
                                                spectra_data.append({
                                                    "spectrum": spectrum,
                                                    "identification": high_confidence_ids[0]
                                                })
                                                annotation_data.append({
                                                    "spectrum": spectrum,
                                                    "identifications": high_confidence_ids
                                                })

                    # Add the extracted data to the training data
                    if spectra_data:
                        training_data["classifier"] = spectra_data

                    if annotation_data:
                        training_data["annotation"] = annotation_data

            # Add visual data if available
            if task.type == PipelineType.VISUAL or task.type == PipelineType.HYBRID:
                if "visual" in task.result:
                    visual_results = task.result["visual"]

                    # Extract visual features if available
                    if isinstance(visual_results, dict) and "features" in visual_results:
                        training_data["visual"] = visual_results["features"]

            # Add LLM analysis data if available
            if "llm_analysis" in task.result:
                llm_results = task.result["llm_analysis"]

                if isinstance(llm_results, dict):
                    training_data["llm"] = llm_results

            return training_data

        except Exception as e:
            self.logger.error(f"Error extracting training data: {str(e)}", exc_info=True)
            return {}

    def get_task(self, task_id: str) -> Optional[AnalysisTask]:
        """
        Get a task by ID

        Args:
            task_id: ID of the task

        Returns:
            Task object or None if not found
        """
        with self.lock:
            return self.tasks.get(task_id)

    def list_tasks(self) -> List[AnalysisTask]:
        """
        List all tasks

        Returns:
            List of all tasks
        """
        with self.lock:
            return list(self.tasks.values())

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task

        Args:
            task_id: ID of the task to cancel

        Returns:
            True if the task was canceled, False otherwise
        """
        with self.lock:
            if task_id not in self.tasks:
                self.logger.error(f"Task {task_id} not found")
                return False

            task = self.tasks[task_id]
            if task.status != AnalysisStatus.RUNNING:
                self.logger.error(f"Task {task_id} is not running (status: {task.status.value})")
                return False

            # Mark as canceled
            task.status = AnalysisStatus.CANCELED
            task.end_time = time.time()

            # Note: We can't actually force the thread to stop
            # as Python doesn't support thread termination
            # The task implementation should check for cancellation periodically

            self.logger.info(f"Task {task_id} marked as canceled")
            return True

    def get_task_progress(self, task_id: str) -> Optional[Tuple[float, AnalysisStatus]]:
        """
        Get the progress of a task

        Args:
            task_id: ID of the task

        Returns:
            Tuple of (progress percentage, status) or None if task not found
        """
        with self.lock:
            if task_id not in self.tasks:
                return None

            task = self.tasks[task_id]
            return (task.progress, task.status)

    def cleanup(self) -> None:
        """Clean up resources used by the orchestrator"""
        self.logger.info("Cleaning up orchestrator resources")

        # Cancel all running tasks
        with self.lock:
            for task_id, task in self.tasks.items():
                if task.status == AnalysisStatus.RUNNING:
                    task.status = AnalysisStatus.CANCELED
                    task.end_time = time.time()

        # Clean up pipelines
        if self._numeric_pipeline is not None:
            self._numeric_pipeline.cleanup()

        if self._visual_pipeline is not None:
            self._visual_pipeline.cleanup()

        if self._llm_service is not None:
            self._llm_service.cleanup()

        if self._model_repository is not None:
            self._model_repository.cleanup()

        self.logger.info("Orchestrator resources cleaned up")


# Factory function to create an orchestrator instance
def create_orchestrator(config: GlobalConfig) -> Orchestrator:
    """
    Create an orchestrator instance

    Args:
        config: Lavoisier configuration

    Returns:
        Orchestrator instance
    """
    return Orchestrator(config)

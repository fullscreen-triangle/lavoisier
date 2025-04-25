"""
Batch processing functionality for handling multiple files at once.

This module implements advanced batch processing capabilities for mass spectrometry 
files, with features like parallel processing, checkpointing, and resuming interrupted 
operations.
"""
import os
import sys
import time
import json
import traceback
import logging
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
import uuid
import shutil
import glob
import csv
import datetime
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.prompt import Confirm

from lavoisier.core.config import LavoisierConfig
from lavoisier.core.progress import get_progress_tracker, ProgressTracker
from lavoisier.core.logging import get_logger
from lavoisier.numerical.pipeline import NumericPipeline
from lavoisier.visual.visual import MSImageProcessor


logger = get_logger("batch_processor")
console = Console()


class BatchProcessor:
    """
    Process multiple files in batches with advanced features like 
    checkpointing and resuming interrupted operations.
    """
    
    def __init__(self, config: LavoisierConfig):
        """
        Initialize the batch processor
        
        Args:
            config: Lavoisier configuration
        """
        self.config = config
        self.logger = get_logger("batch_processor")
        self.batch_id = None
        self.output_dir = None
        self.state_file = None
        self.progress_tracker = get_progress_tracker()
        self.progress_id = None
        
        # Pipelines to use (lazy loaded)
        self._numeric_pipeline = None
        self._visual_pipeline = None
    
    def _init_numeric_pipeline(self) -> NumericPipeline:
        """Initialize and return the numeric pipeline"""
        if self._numeric_pipeline is None:
            self.logger.info("Initializing numeric pipeline")
            self._numeric_pipeline = NumericPipeline(self.config)
        return self._numeric_pipeline
    
    def _init_visual_pipeline(self) -> MSImageProcessor:
        """Initialize and return the visual pipeline"""
        if self._visual_pipeline is None:
            self.logger.info("Initializing visual pipeline")
            self._visual_pipeline = MSImageProcessor(self.config)
        return self._visual_pipeline
    
    def process_batch(
        self,
        input_files: List[str],
        output_dir: str,
        pipeline_type: str = "numeric",
        max_workers: Optional[int] = None,
        resume: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        checkpointing: bool = True,
        processing_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a batch of files
        
        Args:
            input_files: List of input files to process
            output_dir: Output directory
            pipeline_type: Type of pipeline to use (numeric or visual)
            max_workers: Maximum number of worker processes (None = auto)
            resume: Whether to resume a previous batch operation
            progress_callback: Optional callback for progress updates
            checkpointing: Whether to use checkpointing
            processing_params: Additional processing parameters
            
        Returns:
            Batch processing results
        """
        start_time = time.time()
        
        # Initialize output directory and batch ID
        os.makedirs(output_dir, exist_ok=True)
        
        # Create or load batch state
        if resume:
            # Find the most recent batch state file
            state_files = glob.glob(os.path.join(output_dir, "batch_state_*.json"))
            if not state_files:
                self.logger.error("No batch state files found in the output directory")
                return {"success": False, "error": "No batch state files found for resuming"}
            
            # Sort by modification time (most recent first)
            state_files.sort(key=os.path.getmtime, reverse=True)
            self.state_file = state_files[0]
            
            # Load state
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.batch_id = state.get("batch_id", f"batch_{uuid.uuid4().hex[:8]}")
                pipeline_type = state.get("pipeline_type", pipeline_type)
                completed_files = state.get("completed_files", [])
                failed_files = state.get("failed_files", {})
                
                # Filter out completed and failed files from input_files
                pending_files = [f for f in input_files if f not in completed_files and f not in failed_files]
                
                self.logger.info(f"Resuming batch {self.batch_id} with {len(pending_files)} pending files")
                self.logger.info(f"Previously completed: {len(completed_files)}, failed: {len(failed_files)}")
                
                # Use the pending files as our input files
                input_files = pending_files
                
            except Exception as e:
                self.logger.error(f"Error loading batch state: {str(e)}")
                return {"success": False, "error": f"Error loading batch state: {str(e)}"}
        else:
            # Create new batch ID and state file
            self.batch_id = f"batch_{uuid.uuid4().hex[:8]}"
            self.state_file = os.path.join(output_dir, f"batch_state_{self.batch_id}.json")
            
            # Initialize state
            state = {
                "batch_id": self.batch_id,
                "pipeline_type": pipeline_type,
                "start_time": time.time(),
                "input_files": input_files,
                "output_dir": output_dir,
                "completed_files": [],
                "failed_files": {},
                "processing_params": processing_params or {}
            }
            
            # Save initial state
            self._save_state(state)
        
        # Set output directory
        self.output_dir = output_dir
        
        # Initialize progress tracker
        self.progress_id = f"batch_{self.batch_id}"
        progress = self.progress_tracker.create_tracker(
            self.progress_id,
            total_steps=len(input_files),
            name=f"Batch Processing {len(input_files)} files",
            history_size=20
        )
        
        # Register the progress callback
        def internal_progress_callback(percent: float, message: str, status: Dict[str, Any]) -> None:
            if progress_callback:
                progress_callback(percent, message)
        
        progress.callback = internal_progress_callback
        
        # Determine the maximum number of workers
        if max_workers is None:
            max_workers = min(32, os.cpu_count() + 4)
        
        # Process files based on pipeline type
        try:
            if pipeline_type.lower() == "numeric":
                return self._process_with_numeric_pipeline(
                    input_files=input_files,
                    output_dir=output_dir,
                    state=state,
                    progress=progress,
                    max_workers=max_workers,
                    checkpointing=checkpointing,
                    processing_params=processing_params or {}
                )
            elif pipeline_type.lower() == "visual":
                return self._process_with_visual_pipeline(
                    input_files=input_files,
                    output_dir=output_dir,
                    state=state,
                    progress=progress,
                    max_workers=max_workers,
                    checkpointing=checkpointing,
                    processing_params=processing_params or {}
                )
            else:
                error_msg = f"Unknown pipeline type: {pipeline_type}"
                self.logger.error(error_msg)
                progress.complete(message=f"Error: {error_msg}")
                return {"success": False, "error": error_msg}
        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
            self.logger.error(traceback.format_exc())
            progress.complete(message=f"Error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _process_with_numeric_pipeline(
        self,
        input_files: List[str],
        output_dir: str,
        state: Dict[str, Any],
        progress: ProgressTracker,
        max_workers: int,
        checkpointing: bool,
        processing_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process files with the numeric pipeline
        
        Args:
            input_files: List of input files to process
            output_dir: Output directory
            state: Current batch state
            progress: Progress tracker
            max_workers: Maximum number of worker processes
            checkpointing: Whether to use checkpointing
            processing_params: Additional processing parameters
            
        Returns:
            Processing results
        """
        # Ensure we have input files to process
        if not input_files:
            progress.complete(message="No files to process")
            return {
                "success": True,
                "batch_id": self.batch_id,
                "output_dir": output_dir,
                "completed_files": state.get("completed_files", []),
                "failed_files": state.get("failed_files", {}),
                "processing_time": 0,
                "message": "No files to process (all files were already processed)"
            }
        
        # Initialize the numeric pipeline
        pipeline = self._init_numeric_pipeline()
        
        # Create batch results directory
        batch_dir = os.path.join(output_dir, f"batch_{self.batch_id}")
        os.makedirs(batch_dir, exist_ok=True)
        
        # Process files in parallel
        completed_files = state.get("completed_files", [])
        failed_files = state.get("failed_files", {})
        start_time = time.time()
        
        # Create a summary file
        summary_file = os.path.join(batch_dir, "summary.csv")
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "File", "Status", "Processing Time (s)", "File Size (MB)",
                "Start Time", "End Time", "Error"
            ])
        
        # Define the file processor function
        def process_file(file_path: str) -> Tuple[str, bool, float, Optional[str]]:
            file_name = os.path.basename(file_path)
            file_start_time = time.time()
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            try:
                # Create a file-specific output directory
                file_dir = os.path.join(batch_dir, os.path.splitext(file_name)[0])
                os.makedirs(file_dir, exist_ok=True)
                
                self.logger.info(f"Processing file: {file_path}")
                
                # Process the file
                result = pipeline.process_files(
                    input_files=[file_path],
                    output_dir=file_dir,
                    parameters=processing_params
                )
                
                processing_time = time.time() - file_start_time
                
                if result.get("success", False):
                    status = "Completed"
                    error = None
                else:
                    status = "Failed"
                    error = result.get("error", "Unknown error")
                
                # Update the summary file
                with open(summary_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        file_path,
                        status,
                        f"{processing_time:.2f}",
                        f"{file_size_mb:.2f}",
                        datetime.datetime.fromtimestamp(file_start_time).strftime("%Y-%m-%d %H:%M:%S"),
                        datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"),
                        error or ""
                    ])
                
                return file_path, status == "Completed", processing_time, error
                
            except Exception as e:
                processing_time = time.time() - file_start_time
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                self.logger.error(f"Error processing {file_path}: {error_msg}")
                
                # Update the summary file
                with open(summary_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        file_path,
                        "Failed",
                        f"{processing_time:.2f}",
                        f"{file_size_mb:.2f}",
                        datetime.datetime.fromtimestamp(file_start_time).strftime("%Y-%m-%d %H:%M:%S"),
                        datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"),
                        str(e)
                    ])
                
                return file_path, False, processing_time, error_msg
        
        # Process files in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            self.logger.info(f"Starting batch processing with {max_workers} workers")
            progress.update(message=f"Processing {len(input_files)} files with {max_workers} workers")
            
            # Submit all files for processing
            future_to_file = {
                executor.submit(process_file, file_path): file_path
                for file_path in input_files
            }
            
            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
                file_path = future_to_file[future]
                file_name = os.path.basename(file_path)
                
                try:
                    file_path, success, processing_time, error = future.result()
                    
                    if success:
                        completed_files.append(file_path)
                        progress.update(
                            increment=1,
                            message=f"Processed {file_name} in {processing_time:.2f}s ({i+1}/{len(input_files)})"
                        )
                    else:
                        failed_files[file_path] = error
                        progress.update(
                            increment=1,
                            message=f"Failed to process {file_name}: {error}"
                        )
                    
                    # Update and save state if checkpointing is enabled
                    if checkpointing:
                        state["completed_files"] = completed_files
                        state["failed_files"] = failed_files
                        self._save_state(state)
                        
                except Exception as e:
                    self.logger.error(f"Error handling result for {file_path}: {str(e)}")
                    failed_files[file_path] = str(e)
                    progress.update(
                        increment=1,
                        message=f"Error handling result for {file_name}: {str(e)}"
                    )
                    
                    # Update and save state if checkpointing is enabled
                    if checkpointing:
                        state["failed_files"] = failed_files
                        self._save_state(state)
        
        # Calculate total processing time
        processing_time = time.time() - start_time
        
        # Create final state
        state.update({
            "completed_files": completed_files,
            "failed_files": failed_files,
            "end_time": time.time(),
            "processing_time": processing_time
        })
        
        # Save final state
        self._save_state(state)
        
        # Complete progress tracking
        completed_count = len(completed_files)
        failed_count = len(failed_files)
        total_count = len(state.get("input_files", []))
        
        message = (
            f"Batch processing completed in {processing_time:.2f}s: "
            f"{completed_count}/{total_count} completed, {failed_count}/{total_count} failed"
        )
        progress.complete(message=message)
        
        # Return results
        return {
            "success": True,
            "batch_id": self.batch_id,
            "output_dir": batch_dir,
            "completed_files": completed_files,
            "failed_files": failed_files,
            "processing_time": processing_time,
            "total_files": total_count,
            "completed_count": completed_count,
            "failed_count": failed_count,
            "summary_file": summary_file,
            "message": message
        }
    
    def _process_with_visual_pipeline(
        self,
        input_files: List[str],
        output_dir: str,
        state: Dict[str, Any],
        progress: ProgressTracker,
        max_workers: int,
        checkpointing: bool,
        processing_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process files with the visual pipeline
        
        Args:
            input_files: List of input files to process
            output_dir: Output directory
            state: Current batch state
            progress: Progress tracker
            max_workers: Maximum number of worker processes
            checkpointing: Whether to use checkpointing
            processing_params: Additional processing parameters
            
        Returns:
            Processing results
        """
        # Ensure we have input files to process
        if not input_files:
            progress.complete(message="No files to process")
            return {
                "success": True,
                "batch_id": self.batch_id,
                "output_dir": output_dir,
                "completed_files": state.get("completed_files", []),
                "failed_files": state.get("failed_files", {}),
                "processing_time": 0,
                "message": "No files to process (all files were already processed)"
            }
        
        # Initialize the visual pipeline
        pipeline = self._init_visual_pipeline()
        
        # Create batch results directory
        batch_dir = os.path.join(output_dir, f"batch_{self.batch_id}")
        os.makedirs(batch_dir, exist_ok=True)
        
        # Process files in parallel
        completed_files = state.get("completed_files", [])
        failed_files = state.get("failed_files", {})
        start_time = time.time()
        
        # Create a summary file
        summary_file = os.path.join(batch_dir, "summary.csv")
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "File", "Status", "Processing Time (s)", "File Size (MB)",
                "Start Time", "End Time", "Error"
            ])
        
        # Define the file processor function
        def process_file(file_path: str) -> Tuple[str, bool, float, Optional[str]]:
            file_name = os.path.basename(file_path)
            file_start_time = time.time()
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            try:
                # Create a file-specific output directory
                file_dir = os.path.join(batch_dir, os.path.splitext(file_name)[0])
                os.makedirs(file_dir, exist_ok=True)
                
                self.logger.info(f"Processing file: {file_path}")
                
                # Process the file
                result = pipeline.process_spectrum(
                    file_path=file_path,
                    output_dir=file_dir,
                    **processing_params
                )
                
                processing_time = time.time() - file_start_time
                
                if result.get("success", False):
                    status = "Completed"
                    error = None
                else:
                    status = "Failed"
                    error = result.get("error", "Unknown error")
                
                # Update the summary file
                with open(summary_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        file_path,
                        status,
                        f"{processing_time:.2f}",
                        f"{file_size_mb:.2f}",
                        datetime.datetime.fromtimestamp(file_start_time).strftime("%Y-%m-%d %H:%M:%S"),
                        datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"),
                        error or ""
                    ])
                
                return file_path, status == "Completed", processing_time, error
                
            except Exception as e:
                processing_time = time.time() - file_start_time
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                self.logger.error(f"Error processing {file_path}: {error_msg}")
                
                # Update the summary file
                with open(summary_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        file_path,
                        "Failed",
                        f"{processing_time:.2f}",
                        f"{file_size_mb:.2f}",
                        datetime.datetime.fromtimestamp(file_start_time).strftime("%Y-%m-%d %H:%M:%S"),
                        datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"),
                        str(e)
                    ])
                
                return file_path, False, processing_time, error_msg
        
        # Process files in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            self.logger.info(f"Starting batch processing with {max_workers} workers")
            progress.update(message=f"Processing {len(input_files)} files with {max_workers} workers")
            
            # Submit all files for processing
            future_to_file = {
                executor.submit(process_file, file_path): file_path
                for file_path in input_files
            }
            
            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
                file_path = future_to_file[future]
                file_name = os.path.basename(file_path)
                
                try:
                    file_path, success, processing_time, error = future.result()
                    
                    if success:
                        completed_files.append(file_path)
                        progress.update(
                            increment=1,
                            message=f"Processed {file_name} in {processing_time:.2f}s ({i+1}/{len(input_files)})"
                        )
                    else:
                        failed_files[file_path] = error
                        progress.update(
                            increment=1,
                            message=f"Failed to process {file_name}: {error}"
                        )
                    
                    # Update and save state if checkpointing is enabled
                    if checkpointing:
                        state["completed_files"] = completed_files
                        state["failed_files"] = failed_files
                        self._save_state(state)
                        
                except Exception as e:
                    self.logger.error(f"Error handling result for {file_path}: {str(e)}")
                    failed_files[file_path] = str(e)
                    progress.update(
                        increment=1,
                        message=f"Error handling result for {file_name}: {str(e)}"
                    )
                    
                    # Update and save state if checkpointing is enabled
                    if checkpointing:
                        state["failed_files"] = failed_files
                        self._save_state(state)
        
        # Calculate total processing time
        processing_time = time.time() - start_time
        
        # Create final state
        state.update({
            "completed_files": completed_files,
            "failed_files": failed_files,
            "end_time": time.time(),
            "processing_time": processing_time
        })
        
        # Save final state
        self._save_state(state)
        
        # Complete progress tracking
        completed_count = len(completed_files)
        failed_count = len(failed_files)
        total_count = len(state.get("input_files", []))
        
        message = (
            f"Batch processing completed in {processing_time:.2f}s: "
            f"{completed_count}/{total_count} completed, {failed_count}/{total_count} failed"
        )
        progress.complete(message=message)
        
        # Return results
        return {
            "success": True,
            "batch_id": self.batch_id,
            "output_dir": batch_dir,
            "completed_files": completed_files,
            "failed_files": failed_files,
            "processing_time": processing_time,
            "total_files": total_count,
            "completed_count": completed_count,
            "failed_count": failed_count,
            "summary_file": summary_file,
            "message": message
        }
    
    def _save_state(self, state: Dict[str, Any]) -> None:
        """Save batch state to disk"""
        if self.state_file:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
    
    def get_batch_status(self, batch_id: Optional[str] = None, output_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the status of a batch
        
        Args:
            batch_id: Batch ID (if None, uses the current batch ID)
            output_dir: Output directory (if None, uses the current output directory)
            
        Returns:
            Batch status dictionary, or None if the batch is not found
        """
        # Use current batch ID and output directory if not specified
        batch_id = batch_id or self.batch_id
        output_dir = output_dir or self.output_dir
        
        if not batch_id or not output_dir:
            return None
        
        # Find the batch state file
        if batch_id:
            state_file = os.path.join(output_dir, f"batch_state_{batch_id}.json")
        else:
            # Find the most recent batch state file
            state_files = glob.glob(os.path.join(output_dir, "batch_state_*.json"))
            if not state_files:
                return None
            
            # Sort by modification time (most recent first)
            state_files.sort(key=os.path.getmtime, reverse=True)
            state_file = state_files[0]
        
        # Load and return the batch state
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading batch state: {str(e)}")
            return None
    
    def list_batches(self, output_dir: str) -> List[Dict[str, Any]]:
        """
        List all batches in the specified output directory
        
        Args:
            output_dir: Output directory
            
        Returns:
            List of batch status dictionaries
        """
        # Find all batch state files
        state_files = glob.glob(os.path.join(output_dir, "batch_state_*.json"))
        if not state_files:
            return []
        
        # Sort by modification time (most recent first)
        state_files.sort(key=os.path.getmtime, reverse=True)
        
        # Load and return the batch states
        batches = []
        for state_file in state_files:
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                batches.append(state)
            except Exception as e:
                self.logger.error(f"Error loading batch state from {state_file}: {str(e)}")
        
        return batches 

# CLI COMMANDS

@click.group(name="batch", help="Commands for batch processing multiple files")
def batch_group():
    """Batch processing commands for handling multiple files"""
    pass


@batch_group.command(name="process", help="Process multiple files in batch mode")
@click.argument("input_pattern", type=str)
@click.option("--output-dir", "-o", type=click.Path(file_okay=False), required=True, 
              help="Directory where to store the processing results")
@click.option("--pipeline", "-p", type=click.Choice(["numeric", "visual"]), default="numeric",
              help="Pipeline type to use for processing (default: numeric)")
@click.option("--max-workers", "-w", type=int, default=None,
              help="Maximum number of worker processes (default: auto)")
@click.option("--resume/--no-resume", default=False,
              help="Resume a previous batch operation")
@click.option("--checkpoint/--no-checkpoint", default=True,
              help="Enable checkpointing (saving progress)")
@click.option("--force/--no-force", default=False,
              help="Force processing even if files were already processed")
@click.option("--config", "-c", type=click.Path(exists=True, dir_okay=False),
              help="Path to custom configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def process_batch(input_pattern: str, output_dir: str, pipeline: str, max_workers: Optional[int], 
                resume: bool, checkpoint: bool, force: bool, config: Optional[str], verbose: bool):
    """
    Process multiple files in batch mode.
    
    INPUT_PATTERN: Glob pattern for input files (e.g., "data/*.mzML" or "data/sample*.imzML")
    """
    # Set up logging
    if verbose:
        logging.getLogger("batch_processor").setLevel(logging.DEBUG)
    
    # Load configuration
    cfg = LavoisierConfig.load(config_path=config) if config else LavoisierConfig.load()
    
    # Expand input pattern to get file list
    input_files = glob.glob(input_pattern)
    if not input_files:
        console.print(f"[bold red]Error:[/] No files matching pattern '{input_pattern}'")
        sys.exit(1)
    
    # Show summary of what will be processed
    console.print(f"[bold]Batch Processing Summary[/]")
    console.print(f"Pipeline type: [cyan]{pipeline}[/]")
    console.print(f"Files to process: [cyan]{len(input_files)}[/]")
    console.print(f"Output directory: [cyan]{output_dir}[/]")
    console.print(f"Max workers: [cyan]{max_workers or 'auto'}[/]")
    
    # Display first few files
    max_files_to_show = 5
    if len(input_files) > max_files_to_show:
        for i, file in enumerate(input_files[:max_files_to_show]):
            console.print(f"  {i+1}. [cyan]{file}[/]")
        console.print(f"  ... and {len(input_files) - max_files_to_show} more files")
    else:
        for i, file in enumerate(input_files):
            console.print(f"  {i+1}. [cyan]{file}[/]")
    
    # Confirm before proceeding
    if not Confirm.ask("Proceed with batch processing?"):
        console.print("[yellow]Operation cancelled[/]")
        return
    
    # Initialize batch processor
    processor = BatchProcessor(cfg)
    
    # Set up progress tracking
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[bold green]Processing files...", total=len(input_files))
        
        def progress_callback(percent: float, message: str) -> None:
            progress.update(task, completed=int(percent * len(input_files)), description=f"[bold green]{message}")
        
        # Process the batch
        result = processor.process_batch(
            input_files=input_files,
            output_dir=output_dir,
            pipeline_type=pipeline,
            max_workers=max_workers,
            resume=resume,
            progress_callback=progress_callback,
            checkpointing=checkpoint,
            processing_params={"force": force}
        )
    
    # Display results
    if result.get("success", False):
        console.print("\n[bold green]Batch processing completed successfully![/]")
        
        # Create result table
        table = Table(title=f"Batch Processing Results - {result.get('batch_id', 'N/A')}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Files", str(result.get("total_files", 0)))
        table.add_row("Completed", str(result.get("completed_count", 0)))
        table.add_row("Failed", str(result.get("failed_count", 0)))
        table.add_row("Processing Time", f"{result.get('processing_time', 0):.2f} seconds")
        table.add_row("Output Directory", result.get("output_dir", "N/A"))
        
        console.print(table)
        
        # Show summary file location
        if "summary_file" in result:
            console.print(f"Summary file: [bold]{result['summary_file']}[/]")
        
        # Show failures if any
        if result.get("failed_count", 0) > 0:
            console.print("\n[bold yellow]Failed Files:[/]")
            for i, (file, error) in enumerate(result.get("failed_files", {}).items()):
                if i < 5:  # Show only first 5 failures
                    console.print(f"  [cyan]{file}[/]: [red]{error.split('\n')[0]}[/]")
                else:
                    remaining = len(result.get("failed_files", {})) - 5
                    console.print(f"  ... and {remaining} more failures (see summary file)")
                    break
    else:
        console.print(f"\n[bold red]Batch processing failed:[/] {result.get('error', 'Unknown error')}")


@batch_group.command(name="status", help="Get the status of a batch operation")
@click.argument("output_dir", type=click.Path(file_okay=False, exists=True))
@click.option("--batch-id", "-b", type=str, default=None,
              help="Batch ID (if not provided, shows the most recent batch)")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed status information")
def batch_status(output_dir: str, batch_id: Optional[str], verbose: bool):
    """
    Get the status of a batch operation.
    
    OUTPUT_DIR: Directory containing the batch operations
    """
    # Initialize processor with default config
    processor = BatchProcessor(LavoisierConfig.load())
    
    # Get batch status
    batch = processor.get_batch_status(batch_id=batch_id, output_dir=output_dir)
    
    if not batch:
        console.print(f"[bold red]Error:[/] No batch {'with ID ' + batch_id if batch_id else ''} found in {output_dir}")
        return
    
    # Display batch information
    console.print(f"[bold]Batch Status: {batch.get('batch_id', 'N/A')}[/]")
    
    # Create status table
    table = Table(title=f"Batch {batch.get('batch_id', 'N/A')} Details")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    start_time = datetime.datetime.fromtimestamp(batch.get("start_time", 0)).strftime("%Y-%m-%d %H:%M:%S")
    end_time = datetime.datetime.fromtimestamp(batch.get("end_time", 0)).strftime("%Y-%m-%d %H:%M:%S") if "end_time" in batch else "In Progress"
    
    table.add_row("Pipeline", batch.get("pipeline_type", "N/A"))
    table.add_row("Start Time", start_time)
    table.add_row("End Time", end_time)
    
    total_files = len(batch.get("input_files", []))
    completed = len(batch.get("completed_files", []))
    failed = len(batch.get("failed_files", {}))
    
    table.add_row("Total Files", str(total_files))
    table.add_row("Completed", f"{completed} ({completed/total_files*100:.1f}%)" if total_files else "0 (0%)")
    table.add_row("Failed", f"{failed} ({failed/total_files*100:.1f}%)" if total_files else "0 (0%)")
    table.add_row("Pending", f"{total_files - completed - failed} ({(total_files - completed - failed)/total_files*100:.1f}%)" if total_files else "0 (0%)")
    
    if "processing_time" in batch:
        hours, remainder = divmod(batch["processing_time"], 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        table.add_row("Processing Time", time_str)
    
    console.print(table)
    
    # Show file details if verbose
    if verbose:
        if batch.get("completed_files", []):
            console.print("\n[bold green]Completed Files:[/]")
            for i, file in enumerate(batch.get("completed_files", [])[:10]):  # Show first 10
                console.print(f"  {i+1}. [cyan]{file}[/]")
            
            if len(batch.get("completed_files", [])) > 10:
                console.print(f"  ... and {len(batch.get('completed_files', [])) - 10} more files")
        
        if batch.get("failed_files", {}):
            console.print("\n[bold red]Failed Files:[/]")
            for i, (file, error) in enumerate(list(batch.get("failed_files", {}).items())[:10]):  # Show first 10
                console.print(f"  {i+1}. [cyan]{file}[/]: [red]{error.split('\n')[0]}[/]")
            
            if len(batch.get("failed_files", {})) > 10:
                console.print(f"  ... and {len(batch.get('failed_files', {})) - 10} more files")


@batch_group.command(name="list", help="List all batch operations in a directory")
@click.argument("output_dir", type=click.Path(file_okay=False, exists=True))
def list_batches(output_dir: str):
    """
    List all batch operations in a directory.
    
    OUTPUT_DIR: Directory containing the batch operations
    """
    # Initialize processor with default config
    processor = BatchProcessor(LavoisierConfig.load())
    
    # Get all batches
    batches = processor.list_batches(output_dir)
    
    if not batches:
        console.print(f"[bold yellow]No batch operations found in {output_dir}[/]")
        return
    
    # Display batch information
    console.print(f"[bold]Found {len(batches)} batch operations in {output_dir}[/]")
    
    # Create table
    table = Table(title="Batch Operations")
    table.add_column("ID", style="cyan")
    table.add_column("Pipeline", style="green")
    table.add_column("Start Time", style="blue")
    table.add_column("Status", style="yellow")
    table.add_column("Progress", style="magenta")
    table.add_column("Duration", style="green")
    
    for batch in batches:
        batch_id = batch.get("batch_id", "N/A")
        pipeline = batch.get("pipeline_type", "N/A")
        
        start_time = datetime.datetime.fromtimestamp(batch.get("start_time", 0)).strftime("%Y-%m-%d %H:%M:%S")
        
        total_files = len(batch.get("input_files", []))
        completed = len(batch.get("completed_files", []))
        failed = len(batch.get("failed_files", {}))
        
        if "end_time" in batch:
            status = "Completed"
            if failed > 0:
                status = f"Completed with {failed} failures"
        else:
            status = "In Progress"
        
        progress = f"{completed}/{total_files} ({completed/total_files*100:.1f}%)" if total_files else "0/0 (0%)"
        
        if "processing_time" in batch:
            duration_seconds = batch["processing_time"]
        elif "end_time" in batch and "start_time" in batch:
            duration_seconds = batch["end_time"] - batch["start_time"]
        else:
            duration_seconds = time.time() - batch.get("start_time", time.time())
            
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        
        table.add_row(batch_id, pipeline, start_time, status, progress, duration)
    
    console.print(table)
    
    # Display how to get more details
    console.print("\nTo see more details about a specific batch, run:")
    console.print(f"  [bold]lavoisier batch status {output_dir} --batch-id <BATCH_ID> --verbose[/]")


@batch_group.command(name="clean", help="Delete batch data to free up disk space")
@click.argument("output_dir", type=click.Path(file_okay=False, exists=True))
@click.option("--batch-id", "-b", type=str, default=None,
              help="Batch ID to clean (if not provided, asks for confirmation for all)")
@click.option("--force", "-f", is_flag=True, help="Don't ask for confirmation")
@click.option("--keep-latest", "-k", type=int, default=0,
              help="Keep the latest N batch operations (0 to delete all)")
def clean_batches(output_dir: str, batch_id: Optional[str], force: bool, keep_latest: int):
    """
    Delete batch data to free up disk space.
    
    OUTPUT_DIR: Directory containing the batch operations
    """
    # Initialize processor with default config
    processor = BatchProcessor(LavoisierConfig.load())
    
    if batch_id:
        # Delete specific batch
        batch = processor.get_batch_status(batch_id=batch_id, output_dir=output_dir)
        if not batch:
            console.print(f"[bold red]Error:[/] No batch with ID {batch_id} found in {output_dir}")
            return
        
        # Get batch directory
        batch_dir = os.path.join(output_dir, f"batch_{batch_id}")
        state_file = os.path.join(output_dir, f"batch_state_{batch_id}.json")
        
        # Check if files exist
        state_exists = os.path.exists(state_file)
        dir_exists = os.path.exists(batch_dir)
        
        if not state_exists and not dir_exists:
            console.print(f"[bold red]Error:[/] Batch {batch_id} files not found in {output_dir}")
            return
        
        # Calculate size
        total_size = 0
        if dir_exists:
            for dirpath, dirnames, filenames in os.walk(batch_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
        
        if state_exists:
            total_size += os.path.getsize(state_file)
        
        # Format size
        if total_size < 1024:
            size_str = f"{total_size} bytes"
        elif total_size < 1024 * 1024:
            size_str = f"{total_size / 1024:.2f} KB"
        elif total_size < 1024 * 1024 * 1024:
            size_str = f"{total_size / (1024 * 1024):.2f} MB"
        else:
            size_str = f"{total_size / (1024 * 1024 * 1024):.2f} GB"
        
        # Confirm deletion
        if not force and not Confirm.ask(f"Delete batch {batch_id} ({size_str})?"):
            console.print("[yellow]Operation cancelled[/]")
            return
        
        # Delete files
        if dir_exists:
            shutil.rmtree(batch_dir)
        if state_exists:
            os.remove(state_file)
        
        console.print(f"[bold green]Successfully deleted batch {batch_id} ({size_str})[/]")
    
    else:
        # Delete multiple batches
        batches = processor.list_batches(output_dir)
        
        if not batches:
            console.print(f"[bold yellow]No batch operations found in {output_dir}[/]")
            return
        
        # Keep latest N batches if requested
        if keep_latest > 0:
            # Sort by start time (newest first)
            batches.sort(key=lambda b: b.get("start_time", 0), reverse=True)
            # Keep only the latest N
            batches_to_delete = batches[keep_latest:]
        else:
            batches_to_delete = batches
        
        if not batches_to_delete:
            console.print(f"[bold yellow]No batches to delete (keeping the latest {keep_latest})[/]")
            return
        
        # Calculate total size
        total_size = 0
        batch_info = []
        
        for batch in batches_to_delete:
            batch_id = batch.get("batch_id", "N/A")
            batch_dir = os.path.join(output_dir, f"batch_{batch_id}")
            state_file = os.path.join(output_dir, f"batch_state_{batch_id}.json")
            
            batch_size = 0
            if os.path.exists(batch_dir):
                for dirpath, dirnames, filenames in os.walk(batch_dir):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        batch_size += os.path.getsize(fp)
            
            if os.path.exists(state_file):
                batch_size += os.path.getsize(state_file)
            
            total_size += batch_size
            batch_info.append((batch_id, batch_size))
        
        # Format size
        if total_size < 1024:
            size_str = f"{total_size} bytes"
        elif total_size < 1024 * 1024:
            size_str = f"{total_size / 1024:.2f} KB"
        elif total_size < 1024 * 1024 * 1024:
            size_str = f"{total_size / (1024 * 1024):.2f} MB"
        else:
            size_str = f"{total_size / (1024 * 1024 * 1024):.2f} GB"
        
        # Confirm deletion
        if not force:
            console.print(f"[bold]Will delete {len(batches_to_delete)} batches ({size_str})[/]")
            
            # Show batches to delete
            table = Table(title="Batches to Delete")
            table.add_column("ID", style="cyan")
            table.add_column("Start Time", style="blue")
            table.add_column("Size", style="yellow")
            
            for batch_id, batch_size in batch_info:
                batch = next((b for b in batches_to_delete if b.get("batch_id") == batch_id), None)
                start_time = datetime.datetime.fromtimestamp(
                    batch.get("start_time", 0)
                ).strftime("%Y-%m-%d %H:%M:%S") if batch else "N/A"
                
                # Format batch size
                if batch_size < 1024:
                    b_size_str = f"{batch_size} bytes"
                elif batch_size < 1024 * 1024:
                    b_size_str = f"{batch_size / 1024:.2f} KB"
                elif batch_size < 1024 * 1024 * 1024:
                    b_size_str = f"{batch_size / (1024 * 1024):.2f} MB"
                else:
                    b_size_str = f"{batch_size / (1024 * 1024 * 1024):.2f} GB"
                
                table.add_row(batch_id, start_time, b_size_str)
            
            console.print(table)
            
            if not Confirm.ask(f"Delete these {len(batches_to_delete)} batches?"):
                console.print("[yellow]Operation cancelled[/]")
                return
        
        # Delete batches
        for batch_id, _ in batch_info:
            batch_dir = os.path.join(output_dir, f"batch_{batch_id}")
            state_file = os.path.join(output_dir, f"batch_state_{batch_id}.json")
            
            if os.path.exists(batch_dir):
                shutil.rmtree(batch_dir)
            if os.path.exists(state_file):
                os.remove(state_file)
        
        console.print(f"[bold green]Successfully deleted {len(batches_to_delete)} batches ({size_str})[/]")

# Command group to be registered with the main CLI
commands = [batch_group] 
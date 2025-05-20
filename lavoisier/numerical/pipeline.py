"""
Enhanced numeric pipeline implementation that builds on the original MS analysis code.
"""
from typing import List, Dict, Any, Optional, Callable, Tuple
import os
import time
from pathlib import Path
import logging
import ray
import psutil
import pandas as pd
import dask.dataframe as dd
import dask.array as da
from distributed import Client, LocalCluster
import zarr
import numpy as np
import gc
import uuid
import h5py

from lavoisier.core.config import GlobalConfig
from lavoisier.core.logging import get_logger
from lavoisier.utils.cache import get_cache, cached
from lavoisier.core.progress import get_progress_tracker, ProgressTracker


class NumericPipeline:
    """
    Enhanced numeric pipeline for MS analysis that incorporates deep learning
    and leverages distributed computing.
    """
    
    def __init__(self, config: GlobalConfig):
        """
        Initialize the numeric pipeline
        
        Args:
            config: Lavoisier configuration
        """
        self.config = config
        self.logger = get_logger("numeric_pipeline")
        
        # Will be initialized lazily
        self._cluster = None
        self._client = None
        self._reader = None
        self._ml_models = {}
        
        # Initialize cache for intermediate results
        self._cache = get_cache(config)
        
        # Memory tracking
        self._memory_tracker = MemoryTracker(logger=self.logger)
        
        # Progress tracking
        self._progress_tracker = get_progress_tracker()
        self._current_task_id = None
        
        self.logger.info("Numeric pipeline initialized")
    
    def _setup_distributed_environment(self):
        """Set up distributed computing environment based on configuration"""
        if not self.config.distributed.use_ray and not self.config.distributed.use_dask:
            self.logger.warning("Both Ray and Dask are disabled. Using local processing only.")
            return
        
        # Get available resources
        total_memory = psutil.virtual_memory().total
        cpu_count = (self.config.distributed.n_workers if self.config.distributed.n_workers > 0 
                     else psutil.cpu_count())
        
        # Calculate memory allocation
        available_memory = int(total_memory * self.config.distributed.memory_fraction)
        memory_per_worker = available_memory // cpu_count
        
        self.logger.info(f"Setting up distributed environment with {cpu_count} workers")
        self.logger.info(f"Memory per worker: {memory_per_worker / (1024**3):.2f} GB")
        
        # Initialize Ray if enabled
        if self.config.distributed.use_ray:
            self.logger.info("Initializing Ray")
            # Only initialize Ray if it's not already initialized
            if not ray.is_initialized():
                ray.init(
                    num_cpus=cpu_count,
                    _memory=memory_per_worker,
                    object_store_memory=memory_per_worker,
                    # Log to file
                    logging_level=logging.getLevelName(self.config.logging.level),
                    log_to_driver=True
                )
            
        # Initialize Dask if enabled
        if self.config.distributed.use_dask:
            self.logger.info("Initializing Dask")
            self._cluster = LocalCluster(
                n_workers=cpu_count,
                threads_per_worker=self.config.distributed.threads_per_worker,
                memory_limit=f"{memory_per_worker}B"
            )
            self._client = Client(self._cluster)
            self.logger.info(f"Dask dashboard available at: {self._client.dashboard_link}")
    
    def _init_reader(self):
        """Initialize the MS reader"""
        try:
            # Import reader from original codebase
            from spectra_reader.MZMLReader import MSParameters, MZMLReader
            
            # Create parameters from config
            params = MSParameters(
                ms1_threshold=self.config.ms_parameters.ms1_threshold,
                ms2_threshold=self.config.ms_parameters.ms2_threshold,
                mz_tolerance=self.config.ms_parameters.mz_tolerance,
                rt_tolerance=self.config.ms_parameters.rt_tolerance,
                min_intensity=self.config.ms_parameters.min_intensity,
                output_dir=self.config.paths.output_dir,
                n_workers=self.config.distributed.n_workers
            )
            
            self._reader = MZMLReader(params)
            self.logger.info("MZMLReader initialized")
            
        except ImportError:
            self.logger.error("Failed to import MZMLReader. Make sure the original codebase is in PYTHONPATH.")
            raise
    
    def _init_ml_models(self):
        """Initialize machine learning models for enhanced analysis"""
        from lavoisier.numerical.ml.models import load_models
        
        self.logger.info("Loading ML models")
        self._ml_models = load_models(
            model_dir=self.config.paths.model_dir,
            resolution=self.config.processing.resolution,
            feature_dim=self.config.processing.feature_dimension
        )
        
        self.logger.info(f"Loaded {len(self._ml_models)} ML models")
    
    @staticmethod
    @ray.remote
    def _process_file_chunk(
        files: List[str],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a chunk of files using Ray
        
        Args:
            files: List of file paths to process
            config: Configuration dictionary
            
        Returns:
            Dictionary of results for each file
        """
        # This would normally import from the original package
        # and process the files using the original code
        from spectra_reader.MZMLReader import MSParameters, MZMLReader
        
        # Convert config dict back to parameters
        params = MSParameters(
            ms1_threshold=config['ms1_threshold'],
            ms2_threshold=config['ms2_threshold'],
            mz_tolerance=config['mz_tolerance'],
            rt_tolerance=config['rt_tolerance'],
            min_intensity=config['min_intensity'],
            output_dir=config['output_dir'],
            n_workers=1  # Use 1 worker per task since we're already parallelized
        )
        
        # Create reader
        reader = MZMLReader(params)
        
        # Process each file
        results = {}
        for file_path in files:
            try:
                print(f"Processing file: {file_path}")
                result = reader.extract_spectra(str(file_path))
                results[str(file_path)] = result
                print(f"Successfully processed: {file_path}")
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                print(error_msg)
                # Log error but continue with other files
                results[str(file_path)] = {"error": error_msg}
                
        return results
    
    @cached(level="raw")
    def _read_spectra(self, file_path: str) -> Dict[str, Any]:
        """
        Read spectra from file with caching
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            Dictionary with extracted spectra
        """
        self.logger.debug(f"Reading spectra from {file_path}")
        
        if self._reader is None:
            self._init_reader()
            
        return self._reader.extract_spectra(file_path)
    
    @cached(level="processed")
    def _process_spectra(self, spectra: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process spectra with caching
        
        Args:
            spectra: Raw spectra data
            parameters: Processing parameters
            
        Returns:
            Processed spectra
        """
        self.logger.debug("Processing spectra")
        
        # Processing logic here...
        # This is a placeholder - the actual implementation would process the spectra
        
        return spectra
    
    @cached(level="analyzed")
    def _analyze_spectra(self, processed_spectra: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze processed spectra with caching
        
        Args:
            processed_spectra: Processed spectra data
            
        Returns:
            Analysis results
        """
        self.logger.debug("Analyzing spectra")
        
        # Analysis logic here...
        # This is a placeholder - the actual implementation would analyze the spectra
        
        return processed_spectra
    
    def process_files(
        self,
        input_files: List[str],
        output_dir: str,
        parameters: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Process MS files using the numeric pipeline
        
        Args:
            input_files: List of input files to process
            output_dir: Directory to store output files
            parameters: Additional parameters to override defaults
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        self.logger.info(f"Starting processing of {len(input_files)} files")
        
        # Initialize progress tracking
        total_files = len(input_files)
        processed_files = 0
        
        # Create progress tracker for this task
        self._current_task_id = f"numeric_pipeline_{uuid.uuid4().hex[:8]}"
        progress = self._progress_tracker.create_tracker(
            self._current_task_id,
            total_steps=total_files * 3,  # 3 steps per file: read, process, analyze
            name=f"Processing {total_files} files",
            history_size=20
        )
        
        # Define a callback that forwards to the provided callback
        def internal_progress_callback(percent: float, message: str, status: Dict[str, Any]) -> None:
            if progress_callback:
                progress_callback(percent, message)
        
        # Set the callback on the tracker
        progress.callback = internal_progress_callback
        
        # Track initial memory usage
        self._memory_tracker.track("start_processing")
        
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            progress.update(message="Created output directory")
            
            # Initialize distributed environment if needed
            if not ray.is_initialized() and self.config.distributed.use_ray:
                self._setup_distributed_environment()
                progress.update(message="Initialized distributed environment")
            
            # Initialize reader if not already done
            if self._reader is None:
                self._init_reader()
                progress.update(message="Initialized MS reader")
            
            # Initialize ML models if not already done
            if not self._ml_models and self.config.processing.use_ml:
                self._init_ml_models()
                progress.update(message="Loaded ML models")
            
            # Track memory after initialization
            self._memory_tracker.track("after_initialization")
            
            # Split files into batches for memory efficiency
            batch_size = min(
                self.config.processing.batch_size, 
                max(1, total_files // (psutil.cpu_count() * 2))
            )
            
            batches = [input_files[i:i + batch_size] for i in range(0, total_files, batch_size)]
            self.logger.info(f"Split {total_files} files into {len(batches)} batches of size {batch_size}")
            progress.update(message=f"Split into {len(batches)} batches")
            
            # Process files in batches with caching
            all_results = {}
            for batch_idx, batch in enumerate(batches):
                self.logger.info(f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch)} files")
                progress.update(message=f"Starting batch {batch_idx+1}/{len(batches)}")
                
                # Track memory before batch processing
                self._memory_tracker.track(f"before_batch_{batch_idx}")
                
                # Process each file in the batch
                batch_results = {}
                for file_idx, file_path in enumerate(batch):
                    try:
                        file_name = os.path.basename(file_path)
                        
                        # Step 1: Read spectra with caching
                        progress.update(message=f"Reading {file_name} ({file_idx+1}/{len(batch)})")
                        raw_spectra = self._read_spectra(file_path)
                        progress.update(increment=1)
                        
                        # Step 2: Process spectra with caching
                        progress.update(message=f"Processing {file_name} ({file_idx+1}/{len(batch)})")
                        proc_params = parameters.copy() if parameters else {}
                        processed_spectra = self._process_spectra(raw_spectra, proc_params)
                        progress.update(increment=1)
                        
                        # Step 3: Analyze processed spectra with caching
                        progress.update(message=f"Analyzing {file_name} ({file_idx+1}/{len(batch)})")
                        analyzed_spectra = self._analyze_spectra(processed_spectra)
                        progress.update(increment=1)
                        
                        # Store the final results
                        batch_results[file_path] = analyzed_spectra
                        
                        # Update processed files count
                        processed_files += 1
                        
                        # Log estimated time remaining
                        status = progress.get_status()
                        if "remaining_formatted" in status:
                            remaining = status["remaining_formatted"]
                            self.logger.info(f"Processed {processed_files}/{total_files} files, estimated time remaining: {remaining}")
                        else:
                            self.logger.info(f"Processed {processed_files}/{total_files} files")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing file {file_path}: {str(e)}")
                        progress.update(increment=3, message=f"Error processing {file_name}: {str(e)}")
                
                # Update the overall results
                all_results.update(batch_results)
                
                # Force garbage collection between batches
                gc.collect()
                
                # Track memory after batch
                self._memory_tracker.track(f"after_batch_{batch_idx}")
                
                # Log progress and estimated completion time
                status = progress.get_status()
                if "eta" in status:
                    eta = status["eta"]
                    self.logger.info(f"Batch {batch_idx+1} completed, ETA: {eta}")
                else:
                    self.logger.info(f"Batch {batch_idx+1} completed")
            
            # Apply ML enhancements if configured
            if self.config.processing.use_ml and self._ml_models:
                self.logger.info("Applying ML enhancements")
                progress.update(message="Applying ML enhancements")
                all_results = self._apply_ml_enhancements(all_results)
                
            # Track memory before saving
            self._memory_tracker.track("before_saving")
                
            # Save results
            self.logger.info(f"Saving results to {output_dir}")
            progress.update(message="Saving results")
            output_path = self._save_results(all_results, output_dir)
            
            # Track final memory usage
            self._memory_tracker.track("end_processing")
            
            # Complete progress tracking
            processing_time = time.time() - start_time
            progress.complete(message=f"Processing completed in {processing_time:.2f} seconds")
            
            # Log memory usage summary
            self._memory_tracker.log_summary()
            
            # Calculate processing time
            self.logger.info(f"Processing completed in {processing_time:.2f} seconds")
            
            return {
                "success": True,
                "output_path": output_path,
                "processed_files": processed_files,
                "processing_time": processing_time,
                "memory_peak": self._memory_tracker.peak_memory
            }
            
        except Exception as e:
            self.logger.error(f"Error in processing pipeline: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Update progress with error
            if progress:
                progress.complete(message=f"Error: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "processed_files": processed_files,
                "memory_peak": self._memory_tracker.peak_memory
            }
        finally:
            # Cleanup to free memory resources
            self.cleanup()
    
    def _apply_ml_enhancements(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply machine learning enhancements to the extracted spectra
        
        Args:
            results: Dictionary of extraction results
            
        Returns:
            Enhanced results
        """
        self.logger.info("Applying ML enhancements to results")
        
        # Initialize ML models if not already
        if not self._ml_models:
            self._init_ml_models()
        
        enhanced_results = {}
        
        for file_path, result in results.items():
            # Skip error results
            if isinstance(result, dict) and "error" in result:
                enhanced_results[file_path] = result
                continue
                
            # For real results, apply ML enhancement
            try:
                # Unpack the original result tuple if it exists
                if isinstance(result, tuple) and len(result) == 3:
                    scan_info, spec_dict, ms1_xic = result
                    
                    # Apply ML models to the extracted spectra
                    ml_results = {}
                    
                    # Get all MS2 spectra data and convert to numpy arrays for ML processing
                    ms2_spectra = []
                    scan_indices = []
                    precursor_mzs = []
                    
                    for scan_idx, spectrum in spec_dict.items():
                        # Each spectrum is a pandas DataFrame with mz and intensity columns
                        if spectrum is not None and not spectrum.empty:
                            # Find precursor m/z from scan_info
                            scan_row = scan_info[scan_info['scan_number'] == int(scan_idx)]
                            precursor_mz = scan_row['precursor_mz'].values[0] if not scan_row.empty else None
                            
                            # Get the spectrum as a flattened vector
                            # For real implementation, you would apply proper featurization
                            if 'intensity' in spectrum.columns:
                                # Use intensity values as features
                                # In real implementation, you'd want to align m/z values or use more sophisticated featurization
                                spectrum_vector = spectrum['intensity'].values
                                
                                # Pad or truncate to a fixed length for batch processing
                                max_length = 2000  # Example fixed length
                                if len(spectrum_vector) > max_length:
                                    spectrum_vector = spectrum_vector[:max_length]
                                else:
                                    spectrum_vector = np.pad(spectrum_vector, (0, max_length - len(spectrum_vector)))
                                
                                ms2_spectra.append(spectrum_vector)
                                scan_indices.append(scan_idx)
                                
                                if precursor_mz is not None:
                                    precursor_mzs.append(precursor_mz)
                    
                    # Only proceed if we have MS2 spectra
                    if ms2_spectra:
                        # Convert to numpy array for ML processing
                        ms2_array = np.array(ms2_spectra)
                        precursor_mzs_array = np.array(precursor_mzs) if precursor_mzs else None
                        
                        # Apply classification if available
                        if 'ms2_classifier' in self._ml_models:
                            try:
                                class_ids, probabilities = self._ml_models['ms2_classifier'].predict(ms2_array)
                                ml_results['classification'] = {
                                    'scan_indices': scan_indices,
                                    'class_ids': class_ids.tolist(),
                                    'probabilities': probabilities.tolist()
                                }
                                self.logger.info(f"Applied classification to {len(ms2_array)} spectra from {file_path}")
                            except Exception as e:
                                self.logger.error(f"Error applying classifier to {file_path}: {str(e)}")
                        
                        # Apply embedding if available
                        if 'ms2_embedding' in self._ml_models:
                            try:
                                embeddings = self._ml_models['ms2_embedding'].embed(ms2_array)
                                ml_results['embeddings'] = {
                                    'scan_indices': scan_indices,
                                    'vectors': embeddings.tolist()
                                }
                                self.logger.info(f"Generated embeddings for {len(ms2_array)} spectra from {file_path}")
                            except Exception as e:
                                self.logger.error(f"Error generating embeddings for {file_path}: {str(e)}")
                        
                        # Apply annotation if available
                        if 'ms2_annotation' in self._ml_models and precursor_mzs_array is not None:
                            try:
                                annotations = self._ml_models['ms2_annotation'].annotate(ms2_array, precursor_mzs_array)
                                ml_results['annotations'] = {
                                    'scan_indices': scan_indices,
                                    'results': annotations
                                }
                                self.logger.info(f"Generated annotations for {len(ms2_array)} spectra from {file_path}")
                            except Exception as e:
                                self.logger.error(f"Error generating annotations for {file_path}: {str(e)}")
                    
                    enhanced_results[file_path] = {
                        "original": result,
                        "ml_enhanced": ml_results
                    }
                else:
                    # Unknown result format, just pass through
                    enhanced_results[file_path] = {
                        "original": result,
                        "ml_enhanced": None,
                        "error": "Unknown result format"
                    }
            except Exception as e:
                self.logger.error(f"Error enhancing results for {file_path}: {str(e)}")
                enhanced_results[file_path] = {
                    "original": result,
                    "ml_enhanced": None,
                    "error": str(e)
                }
        
        return enhanced_results
    
    def _save_results(self, results: Dict[str, Any], output_dir: str) -> str:
        """
        Save results to disk in multiple formats for easier access
        
        Args:
            results: Dictionary of results
            output_dir: Directory to save results to
            
        Returns:
            Path to saved results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save in zarr format (original)
        zarr_path = os.path.join(output_dir, "results.zarr")
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.group(store=store)
        
        # Also save in HDF5 format for easier viewing
        h5_path = os.path.join(output_dir, "results.h5")
        h5_file = h5py.File(h5_path, 'w')
        
        # Create CSV directory for even easier viewing
        csv_dir = os.path.join(output_dir, "csv_results")
        os.makedirs(csv_dir, exist_ok=True)
        
        # Use chunked storage for large arrays
        for file_path, result in results.items():
            file_name = os.path.basename(file_path)
            group_name = os.path.splitext(file_name)[0]
            
            # Create group in zarr
            zarr_group = root.create_group(group_name)
            
            # Create group in HDF5
            h5_group = h5_file.create_group(group_name)
            
            # Create directory for CSV
            file_csv_dir = os.path.join(csv_dir, group_name)
            os.makedirs(file_csv_dir, exist_ok=True)
            
            # Process each file's results in all formats
            try:
                self._save_result_in_all_formats(zarr_group, h5_group, file_csv_dir, result)
            except Exception as e:
                self.logger.error(f"Error saving results for {file_path}: {str(e)}")
        
        # Close the HDF5 file
        h5_file.close()
        
        self.logger.info(f"Results saved to multiple formats:")
        self.logger.info(f"  - Zarr: {zarr_path}")
        self.logger.info(f"  - HDF5: {h5_path}")
        self.logger.info(f"  - CSV: {csv_dir}")
        
        return output_dir
    
    def _save_result_in_all_formats(self, zarr_group, h5_group, csv_dir, result):
        """Helper method to save result in all formats"""
        try:
            # Original result format is (scan_info, spec_dict, ms1_xic)
            if isinstance(result, tuple) and len(result) == 3:
                scan_info, spec_dict, ms1_xic = result
                
                # Save scan info
                if isinstance(scan_info, pd.DataFrame):
                    # Save to zarr
                    dask_arr = da.from_array(scan_info.values)
                    scan_info_ds = zarr_group.create_dataset("scan_info", 
                                                    data=dask_arr,
                                                    chunks=True, 
                                                    compression='lz4')
                    scan_info_ds.attrs["columns"] = list(scan_info.columns)
                    
                    # Save to HDF5
                    h5_group.create_dataset("scan_info", data=scan_info.values)
                    h5_group.create_dataset("scan_info_columns", data=np.array(list(scan_info.columns), dtype='S'))
                    
                    # Save to CSV
                    scan_info.to_csv(os.path.join(csv_dir, "scan_info.csv"), index=False)
                
                # Save MS1 XIC
                if isinstance(ms1_xic, pd.DataFrame):
                    # Save to zarr
                    dask_arr = da.from_array(ms1_xic.values)
                    ms1_xic_ds = zarr_group.create_dataset("ms1_xic", 
                                                   data=dask_arr,
                                                   chunks=True, 
                                                   compression='lz4')
                    ms1_xic_ds.attrs["columns"] = list(ms1_xic.columns)
                    
                    # Save to HDF5
                    h5_group.create_dataset("ms1_xic", data=ms1_xic.values)
                    h5_group.create_dataset("ms1_xic_columns", data=np.array(list(ms1_xic.columns), dtype='S'))
                    
                    # Save to CSV
                    ms1_xic.to_csv(os.path.join(csv_dir, "ms1_xic.csv"), index=False)
                
                # Save spectrum dictionary
                if spec_dict:
                    # Create spectrum groups
                    zarr_spec_group = zarr_group.create_group("spectra")
                    h5_spec_group = h5_group.create_group("spectra")
                    spec_csv_dir = os.path.join(csv_dir, "spectra")
                    os.makedirs(spec_csv_dir, exist_ok=True)
                    
                    for spec_idx, spec_df in spec_dict.items():
                        if isinstance(spec_df, pd.DataFrame):
                            # Save to zarr
                            dask_arr = da.from_array(spec_df.values)
                            spec_ds = zarr_spec_group.create_dataset(str(spec_idx), 
                                                           data=dask_arr,
                                                           chunks=True, 
                                                           compression='lz4')
                            spec_ds.attrs["columns"] = list(spec_df.columns)
                            
                            # Save to HDF5
                            h5_spec_group.create_dataset(str(spec_idx), data=spec_df.values)
                            h5_spec_group.create_dataset(f"{spec_idx}_columns", data=np.array(list(spec_df.columns), dtype='S'))
                            
                            # Save to CSV
                            spec_df.to_csv(os.path.join(spec_csv_dir, f"spec_{spec_idx}.csv"), index=False)
        
        except Exception as e:
            self.logger.error(f"Error saving result in multiple formats: {str(e)}")
    
    def _save_original_result(self, group, result):
        """Helper method to save original result format (keeping for backward compatibility)"""
        try:
            # Original result format is (scan_info, spec_dict, ms1_xic)
            if isinstance(result, tuple) and len(result) == 3:
                scan_info, spec_dict, ms1_xic = result
                
                # Save scan info
                if isinstance(scan_info, pd.DataFrame):
                    dask_arr = da.from_array(scan_info.values)
                    scan_info_ds = group.create_dataset("scan_info", 
                                                     data=dask_arr,
                                                     chunks=True, 
                                                     compression='lz4')
                    scan_info_ds.attrs["columns"] = list(scan_info.columns)
                
                # Save MS1 XIC
                if isinstance(ms1_xic, pd.DataFrame):
                    dask_arr = da.from_array(ms1_xic.values)
                    ms1_xic_ds = group.create_dataset("ms1_xic", 
                                                   data=dask_arr,
                                                   chunks=True, 
                                                   compression='lz4')
                    ms1_xic_ds.attrs["columns"] = list(ms1_xic.columns)
        except Exception as e:
            group.attrs["save_error"] = str(e)
    
    def cleanup(self):
        """Clean up resources to free memory"""
        self.logger.info("Cleaning up resources")
        
        # Clear ML models from memory if they exist
        if hasattr(self, '_ml_models') and self._ml_models:
            self._ml_models.clear()
        
        # Close Dask client and cluster if they exist
        if hasattr(self, '_client') and self._client:
            self._client.close()
            self._client = None
        
        if hasattr(self, '_cluster') and self._cluster:
            self._cluster.close()
            self._cluster = None
            
        # Force garbage collection
        gc.collect()
        
        self.logger.info("Cleanup completed")
    
    def clear_cache(self, level: Optional[str] = None):
        """
        Clear the cache for intermediate results
        
        Args:
            level: Specific level to clear, or None for all levels
        """
        if hasattr(self, '_cache'):
            self.logger.info(f"Clearing cache{f' for level {level}' if level else ''}")
            self._cache.clear(level)


class MemoryTracker:
    """Utility class to track memory usage during processing"""
    
    def __init__(self, logger=None):
        self.memory_usage = {}
        self.peak_memory = 0
        self.logger = logger or logging.getLogger(__name__)
    
    def track(self, label: str):
        """Track memory usage at a specific point"""
        memory_info = psutil.Process(os.getpid()).memory_info()
        memory_usage_mb = memory_info.rss / (1024 * 1024)
        self.memory_usage[label] = memory_usage_mb
        
        # Update peak memory
        self.peak_memory = max(self.peak_memory, memory_usage_mb)
        
        self.logger.debug(f"Memory usage at {label}: {memory_usage_mb:.2f} MB")
    
    def log_summary(self):
        """Log a summary of memory usage"""
        self.logger.info("Memory usage summary:")
        for label, usage in self.memory_usage.items():
            self.logger.info(f"  {label}: {usage:.2f} MB")
        self.logger.info(f"  Peak memory usage: {self.peak_memory:.2f} MB") 
import logging
import json
from pathlib import Path
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import dask.dataframe as dd
import dask.array as da
from distributed import Client, LocalCluster
import ray
import psutil
from typing import List, Dict, Tuple, Optional
import time
import os
import pymzml
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Hardcoded configuration - no more imports!
@dataclass
class MSParameters:
    """Parameters for MS data processing"""
    ms1_threshold: float = 1000.0
    ms2_threshold: float = 100.0
    mz_tolerance: float = 0.01
    rt_tolerance: float = 0.5
    min_intensity: float = 500.0
    output_dir: str = "output"
    n_workers: int = -1  # Use all available CPUs

class MZMLReader:
    """Class to handle mzML file reading and processing"""

    def __init__(self, params: MSParameters):
        self.params = params

    def _init_spectrum_reader(self, mzml_path: str):
        """Initialize the pymzML reader"""
        try:
            return pymzml.run.Reader(mzml_path, MS1_Precision=5e-6, obo_version="4.1.33")
        except Exception as e:
            logger.error(f"Failed to initialize mzML reader: {str(e)}")
            raise

    def _validate_spectrum(self, spec) -> bool:
        """Validate spectrum data"""
        try:
            return hasattr(spec, 'ID') and hasattr(spec, 'ms_level')
        except:
            return False

    def _extract_polarity(self, filename: str) -> str:
        """Extract polarity from filename"""
        filename = filename.lower()
        if 'pos' in filename:
            return 'positive'
        elif 'neg' in filename:
            return 'negative'
        else:
            return 'unknown'

    def extract_spectra(self, mzml_path: str) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
        """Main method to extract spectra from mzML file"""
        if not Path(mzml_path).is_file():
            raise FileNotFoundError(f"File not found: {mzml_path}")

        logger.info(f"Extracting: {mzml_path}")
        spec_obj = self._init_spectrum_reader(mzml_path)
        polarity = self._extract_polarity(Path(mzml_path).name)
        
        scan_info = {
            "spec_index": [],
            "scan_time": [],
            "dda_event_idx": [],
            "DDA_rank": [],
            "scan_number": [],
            "polarity": []
        }
        spec_dict = {}
        ms1_data = []
        current_dda_event = 0

        for spec in spec_obj:
            if not self._validate_spectrum(spec):
                continue

            try:
                spec_idx = int(spec.ID)
                ms_level = spec.ms_level

                if ms_level == 1:
                    # Fix the peak extraction - use the correct pymzml API
                    try:
                        mz_array, int_array = [], []
                        
                        if hasattr(spec, 'peaks') and callable(spec.peaks):
                            peaks_data = spec.peaks("raw")
                            if isinstance(peaks_data, tuple) and len(peaks_data) == 2:
                                mz_array, int_array = peaks_data
                            elif hasattr(peaks_data, '__len__') and len(peaks_data) > 0:
                                # Handle if peaks_data is a list/array of [mz, intensity] pairs
                                peaks_array = np.array(peaks_data)
                                if peaks_array.ndim == 2 and peaks_array.shape[1] >= 2:
                                    mz_array = peaks_array[:, 0]
                                    int_array = peaks_array[:, 1]
                        elif hasattr(spec, 'peaks'):
                            peaks = spec.peaks
                            if hasattr(peaks, '__len__') and len(peaks) > 0:
                                peaks_array = np.array(peaks)
                                if peaks_array.ndim == 2 and peaks_array.shape[1] >= 2:
                                    mz_array = peaks_array[:, 0]
                                    int_array = peaks_array[:, 1]
                        elif hasattr(spec, 'mz') and hasattr(spec, 'i'):
                            mz_array = np.array(spec.mz)
                            int_array = np.array(spec.i)
                        
                        # Convert to numpy arrays and ensure they're the same length
                        mz_array = np.array(mz_array)
                        int_array = np.array(int_array)
                        
                        if len(mz_array) != len(int_array):
                            print(f"Mismatch in array lengths for spectrum {spec_idx}: mz={len(mz_array)}, int={len(int_array)}")
                            continue
                            
                        if len(mz_array) == 0:
                            continue
                            
                    except Exception as e:
                        print(f"Error extracting peaks from spectrum {spec_idx}: {e}")
                        continue
                        
                    scan_time = spec.scan_time_in_minutes()

                    if len(mz_array) > 0:
                        ms1_data.append({
                            'scan_time': scan_time,
                            'mz_array': mz_array,
                            'int_array': int_array
                        })

                        scan_info["spec_index"].append(spec_idx)
                        scan_info["scan_time"].append(scan_time)
                        scan_info["dda_event_idx"].append(current_dda_event)
                        scan_info["DDA_rank"].append(0)
                        scan_info["scan_number"].append(spec_idx)
                        scan_info["polarity"].append(polarity)

                        if self.params.ms1_threshold > 0:
                            mask = int_array >= self.params.ms1_threshold
                            mz_array = mz_array[mask]
                            int_array = int_array[mask]

                        spec_dict[spec_idx] = pd.DataFrame({
                            'mz': mz_array,
                            'intensity': int_array
                        })

                        current_dda_event += 1

                elif ms_level == 2:
                    if hasattr(spec, 'selected_precursors') and spec.selected_precursors:
                        precursor_mz = spec.selected_precursors[0].get('mz', 0)
                        precursor_intensity = spec.selected_precursors[0].get('i', 0)

                        if precursor_mz > 0:
                            # Fix the peak extraction for MS2 as well
                            try:
                                mz_array, int_array = [], []
                                
                                if hasattr(spec, 'peaks') and callable(spec.peaks):
                                    peaks_data = spec.peaks("raw")
                                    if isinstance(peaks_data, tuple) and len(peaks_data) == 2:
                                        mz_array, int_array = peaks_data
                                    elif hasattr(peaks_data, '__len__') and len(peaks_data) > 0:
                                        peaks_array = np.array(peaks_data)
                                        if peaks_array.ndim == 2 and peaks_array.shape[1] >= 2:
                                            mz_array = peaks_array[:, 0]
                                            int_array = peaks_array[:, 1]
                                elif hasattr(spec, 'peaks'):
                                    peaks = spec.peaks
                                    if hasattr(peaks, '__len__') and len(peaks) > 0:
                                        peaks_array = np.array(peaks)
                                        if peaks_array.ndim == 2 and peaks_array.shape[1] >= 2:
                                            mz_array = peaks_array[:, 0]
                                            int_array = peaks_array[:, 1]
                                elif hasattr(spec, 'mz') and hasattr(spec, 'i'):
                                    mz_array = np.array(spec.mz)
                                    int_array = np.array(spec.i)
                                
                                # Convert to numpy arrays and ensure they're the same length
                                mz_array = np.array(mz_array)
                                int_array = np.array(int_array)
                                
                                if len(mz_array) != len(int_array):
                                    print(f"Mismatch in array lengths for MS2 spectrum {spec_idx}")
                                    continue
                                    
                                if len(mz_array) == 0:
                                    continue
                                    
                            except Exception as e:
                                print(f"Error extracting peaks from MS2 spectrum {spec_idx}: {e}")
                                continue
                                
                            scan_time = spec.scan_time_in_minutes()

                            if len(mz_array) > 0:
                                scan_info["spec_index"].append(spec_idx)
                                scan_info["scan_time"].append(scan_time)
                                scan_info["dda_event_idx"].append(current_dda_event - 1)
                                scan_info["DDA_rank"].append(len([x for x in scan_info["dda_event_idx"] if x == current_dda_event - 1]))
                                scan_info["scan_number"].append(spec_idx)
                                scan_info["polarity"].append(polarity)

                                spec_dict[spec_idx] = pd.DataFrame({
                                    'mz': mz_array,
                                    'intensity': int_array,
                                    'precursor_mz': precursor_mz,
                                    'precursor_intensity': precursor_intensity
                                })
            except Exception as e:
                logger.error(f"Error processing spectrum {spec_idx}: {str(e)}")
                continue

        # Create DataFrames
        scan_info_df = pd.DataFrame(scan_info)
        ms1_xic_df = pd.DataFrame(ms1_data) if ms1_data else pd.DataFrame()
        
        return scan_info_df, spec_dict, ms1_xic_df

class MSAnalysisPipeline:
    def __init__(self, config_path: str = None):
        # Hardcoded config - no more JSON loading!
        self.config = {
            'ms_parameters': {
                'ms1_threshold': 1000,
                'ms2_threshold': 100,
                'mz_tolerance': 0.01,
                'rt_tolerance': 0.5,
                'min_intensity': 500,
                'output_dir': 'output',
                'n_workers': -1
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(levelname)s - %(message)s',
                'file': None  # Start with no file logging
            }
        }
        
        self.setup_logging()
        self.params = MSParameters(**self.config['ms_parameters'])
        self.reader = MZMLReader(self.params)

        print("Initializing MSAnalysisPipeline...")
        print(f"Output directory: {self.params.output_dir}")
        print(f"Number of workers: {self.params.n_workers}")

        # Initialize distributed computing
        self._setup_compute_environment()

    def setup_logging(self):
        log_file = self.config['logging']['file']
        
        handlers = [logging.StreamHandler()]  # Always have console output
        
        # Add file handler only if we have a proper log file path
        if log_file and os.path.dirname(log_file):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            handlers.append(logging.FileHandler(log_file))

        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=handlers
        )
        logger.info("Logging initialized")

    def _setup_compute_environment(self):
        """Setup distributed computing environment based on available resources"""
        total_memory = psutil.virtual_memory().total
        cpu_count = self.params.n_workers if self.params.n_workers > 0 else mp.cpu_count()

        # Reserve some memory for system operations
        available_memory = int(total_memory * 0.8)
        memory_per_worker = available_memory // cpu_count

        # Initialize Ray for distributed computing
        ray.init(
            num_cpus=cpu_count,
            _memory=memory_per_worker,
            object_store_memory=memory_per_worker
        )

        # Setup Dask cluster for large data processing
        self.cluster = LocalCluster(
            n_workers=cpu_count,
            threads_per_worker=2,
            memory_limit=f"{memory_per_worker}B"
        )
        self.client = Client(self.cluster)

    @staticmethod
    @ray.remote
    def _process_chunk(file_chunk: List[Path], params: MSParameters) -> Dict:
        """Process a chunk of files using Ray"""
        results = {}
        reader = MZMLReader(params)

        for file_path in file_chunk:
            try:
                print(f"Processing file: {file_path}")
                result = reader.extract_spectra(str(file_path))
                results[str(file_path)] = result
                print(f"Successfully processed: {file_path}")
            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                print(error_msg)
        return results

    def process_files(self, input_dir: str):
        """Main method to process MS files"""
        start_time = time.time()
        input_path = Path(input_dir)
        print(f"Searching for mzML files in: {input_path}")

        if not input_path.exists():
            error_msg = f"Input directory does not exist: {input_path}"
            logger.error(error_msg)
            print(error_msg)
            return

        mzml_files = list(input_path.glob('*.mzML'))

        print(f"Found {len(mzml_files)} mzML files:")
        for file in mzml_files:
            print(f"- {file}")

        if not mzml_files:
            logger.error(f"No mzML files found in {input_dir}")
            print(f"No mzML files found in {input_dir}")
            return

        try:
            # Calculate optimal chunk size based on available CPUs
            n_workers = self.params.n_workers if self.params.n_workers > 0 else mp.cpu_count()
            chunk_size = max(1, len(mzml_files) // n_workers)

            print(f"Using {n_workers} workers")
            print(f"Chunk size: {chunk_size}")

            file_chunks = [mzml_files[i:i + chunk_size] for i in range(0, len(mzml_files), chunk_size)]

            print(f"Created {len(file_chunks)} chunks for processing")

            # Pass params instead of self
            futures = [self._process_chunk.remote(chunk, self.params) for chunk in file_chunks]
            print("Processing chunks...")
            results = ray.get(futures)

            combined_results = {}
            for result_dict in results:
                combined_results.update(result_dict)

            print(f"Successfully processed {len(combined_results)} files")

            output_dir = Path(self.params.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving results to: {output_dir}")
            self._save_results_distributed(combined_results, output_dir)

        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            logger.error(error_msg)
            print(error_msg)
            raise

        finally:
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"Processing completed in {processing_time:.2f} seconds")
            print(f"Processing completed in {processing_time:.2f} seconds")

            # Clean up
            self.client.close()
            self.cluster.close()
            ray.shutdown()

    def _save_results_distributed(self, results: Dict, output_dir: Path):
        """Save results using simple file formats - no more zarr complexity!"""
        print("Saving results...")
        
        for file_path, (scan_info, spec_dict, ms1_xic) in results.items():
            file_name = Path(file_path).stem
            file_output_dir = output_dir / file_name
            file_output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Save scan info as CSV
                scan_info.to_csv(file_output_dir / "scan_info.csv", index=False)
                print(f"Saved scan info for {file_name}")
                
                # Save MS1 XIC as CSV if not empty
                if not ms1_xic.empty:
                    ms1_xic.to_csv(file_output_dir / "ms1_xic.csv", index=False)
                    print(f"Saved MS1 XIC for {file_name}")
                
                # Save individual spectra as CSV files
                spectra_dir = file_output_dir / "spectra"
                spectra_dir.mkdir(exist_ok=True)
                
                for spec_idx, spec_data in spec_dict.items():
                    spec_data.to_csv(spectra_dir / f"spectrum_{spec_idx}.csv", index=False)
                
                print(f"Saved {len(spec_dict)} spectra for {file_name}")
                
            except Exception as e:
                print(f"Error saving results for {file_name}: {e}")
                logger.error(f"Error saving results for {file_name}: {e}")

def main():
    # Hardcoded paths - no more config files!
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "../../public/spectra")  # Point to public/spectra
    output_dir = os.path.join(base_dir, "../../output")
    log_dir = os.path.join(base_dir, "../../logs")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Log directory: {log_dir}")

    # Create pipeline and run
    pipeline = MSAnalysisPipeline()
    pipeline.params.output_dir = output_dir
    
    # Fix the log file path
    pipeline.config['logging']['file'] = os.path.join(log_dir, 'ms_processing.log')
    
    # Re-setup logging with correct path
    pipeline.setup_logging()
    
    pipeline.process_files(input_dir)

if __name__ == "__main__":
    main()




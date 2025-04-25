import logging
import yaml
from pathlib import Path
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import dask.dataframe as dd
import dask.array as da
from distributed import Client, LocalCluster
import ray
import psutil
from typing import List, Dict
import zarr
import time
import os

from spectra_reader.MZMLReader import MSParameters, MZMLReader

logger = logging.getLogger(__name__)


class MSAnalysisPipeline:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.params = MSParameters(**self.config['ms_parameters'])
        self.reader = MZMLReader(self.params)

        print("Initializing MSAnalysisPipeline...")
        print(f"Output directory: {self.params.output_dir}")
        print(f"Number of workers: {self.params.n_workers}")

        # Initialize distributed computing
        self._setup_compute_environment()

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        log_file = self.config['logging']['file']
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
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
        reader = MZMLReader(params)  # Create reader instance inside the static method

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

            file_chunks = [mzml_files[i:i + chunk_size]
                           for i in range(0, len(mzml_files), chunk_size)]

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
        """Save results using parallel I/O operations"""
        store = zarr.DirectoryStore(str(output_dir / 'results.zarr'))
        root = zarr.group(store=store)

        def save_dataset(name, data, group):
            try:
                if isinstance(data, (pd.DataFrame, dd.DataFrame)):
                    dask_arr = da.from_array(data.values)
                    group.create_dataset(name, data=dask_arr,
                                      chunks=True, compression='lz4')
            except Exception as e:
                logger.error(f"Error saving dataset {name}: {e}")

        with ThreadPoolExecutor() as executor:
            futures = []
            for file_path, (scan_info, spec_dict, ms1_xic) in results.items():
                group_name = Path(file_path).stem
                group = root.create_group(group_name)

                futures.extend([
                    executor.submit(save_dataset, 'scan_info', scan_info, group),
                    executor.submit(save_dataset, 'ms1_xic', ms1_xic, group),
                    executor.submit(self._save_spec_dict, spec_dict, group)
                ])

            for future in futures:
                future.result()

    def _save_spec_dict(self, spec_dict: Dict, group):
        """Save spectrum dictionary efficiently"""
        spec_group = group.create_group('spectra')
        for spec_idx, spec_data in spec_dict.items():
            try:
                dask_arr = da.from_array(spec_data.values)
                spec_group.create_dataset(str(spec_idx), data=dask_arr,
                                       chunks=True, compression='lz4')
            except Exception as e:
                logger.error(f"Error saving spectrum {spec_idx}: {e}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "../spectra")
    config_path = os.path.join(base_dir, "../config/numeric_config.yaml")

    # Create output and log directories if they don't exist
    output_dir = os.path.join(base_dir, "../output")
    log_dir = os.path.join(base_dir, "../logs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Modify the config with absolute paths
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update paths in config
    config['ms_parameters']['output_dir'] = output_dir
    config['logging']['file'] = os.path.join(log_dir, 'ms_processing.log')

    # Write updated config
    temp_config_path = os.path.join(base_dir, "../config/temp_config.yaml")
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Log directory: {log_dir}")

    pipeline = MSAnalysisPipeline(temp_config_path)
    pipeline.process_files(input_dir)

    # Clean up temporary config
    os.remove(temp_config_path)

if __name__ == "__main__":
    main()




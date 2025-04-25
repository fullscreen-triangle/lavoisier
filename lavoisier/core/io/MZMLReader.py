from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np
import pymzml
import logging
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing


logger = logging.getLogger(__name__)


@dataclass
class MSParameters:
    """Parameters for MS data processing"""
    ms1_threshold: float
    ms2_threshold: float
    mz_tolerance: float
    rt_tolerance: float
    min_intensity: float
    output_dir: str
    n_workers: int = multiprocessing.cpu_count()


class MZMLReader:
    """Class to handle mzML file reading and processing"""

    def __init__(self, params: MSParameters):
        self.params = params

    def process_files_parallel(self, mzml_files: List[str]) -> None:
        """Process multiple mzML files in parallel"""
        logger.info(f"Processing {len(mzml_files)} files using {self.params.n_workers} workers")

        with ProcessPoolExecutor(max_workers=self.params.n_workers) as executor:
            futures = [executor.submit(self.process_single_file, mzml_file)
                       for mzml_file in mzml_files]

            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in parallel processing: {str(e)}")

    def process_single_file(self, mzml_path: str) -> None:
        """Process a single mzML file"""
        try:
            logger.info(f"Processing file: {mzml_path}")
            scan_info_df, spec_dict, ms1_xic_df = self.extract_spectra(mzml_path)

            # Create output directory
            output_dir = Path(self.params.output_dir) / Path(mzml_path).stem
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save results
            self._save_results(output_dir, scan_info_df, spec_dict, ms1_xic_df)

        except Exception as e:
            logger.error(f"Error processing {mzml_path}: {str(e)}")

    def _save_results(self, output_dir: Path, scan_info_df: pd.DataFrame,
                      spec_dict: Dict, ms1_xic_df: pd.DataFrame) -> None:
        """Save processing results"""
        try:
            # Use ThreadPoolExecutor for I/O operations
            with ThreadPoolExecutor() as executor:
                # Save main DataFrames
                executor.submit(scan_info_df.to_parquet,
                                output_dir / "scan_info.parquet")
                executor.submit(ms1_xic_df.to_parquet,
                                output_dir / "ms1_xic.parquet")

                # Save spectrum dictionary entries
                futures = []
                for spec_idx, spec_df in spec_dict.items():
                    future = executor.submit(spec_df.to_parquet,
                                             output_dir / f"spec_{spec_idx}.parquet")
                    futures.append(future)

                # Wait for all saves to complete
                for future in futures:
                    future.result()

            logger.info(f"Results stored in: {output_dir}")

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    def _init_spectrum_reader(self, mzml_path: str) -> pymzml.run.Reader:
        """Initialize the pymzML reader"""
        try:
            return pymzml.run.Reader(mzml_path, MS1_Precision=5e-6,
                                     obo_version="4.1.33")
        except Exception as e:
            logger.error(f"Failed to initialize mzML reader: {str(e)}")
            raise

    def extract_spectra(self, mzml_path: str) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
        """Main method to extract spectra from mzML file"""
        if not Path(mzml_path).is_file():
            raise FileNotFoundError(f"File not found: {mzml_path}")

        logger.info(f"Extracting: {mzml_path}")
        spec_obj = self._init_spectrum_reader(mzml_path)

        # Extract polarity from filename
        polarity = self._extract_polarity(Path(mzml_path).name)
        return self._process_spectra(spec_obj, polarity)

    def _extract_polarity(self, filename: str) -> str:
        """Extract polarity from filename"""
        filename = filename.lower()
        if 'pos' in filename:
            return 'positive'
        elif 'neg' in filename:
            return 'negative'
        else:
            return 'unknown'

    def _process_spectra(self, spec_obj, polarity: str) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
        """Process spectra from mzML file"""
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
                    mz_array, int_array = spec.get_peaks()
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
                            mz_array, int_array = spec.get_peaks()
                            scan_time = spec.scan_time_in_minutes()

                            if len(mz_array) > 0:
                                scan_info["spec_index"].append(spec_idx)
                                scan_info["scan_time"].append(scan_time)
                                scan_info["dda_event_idx"].append(current_dda_event - 1)
                                scan_info["DDA_rank"].append(len([x for x in scan_info["dda_event_idx"]
                                                                  if x == current_dda_event - 1]))
                                scan_info["scan_number"].append(spec_idx)

                                spec_dict[spec_idx] = pd.DataFrame({
                                    'mz': mz_array,
                                    'intensity': int_array,
                                    'precursor_mz': precursor_mz,
                                    'precursor_intensity': precursor_intensity
                                })
            except Exception as e:
                logger.error(f"Error processing spectrum {spec_idx}: {str(e)}")
                continue

        scan_info["polarity"] = [polarity] * len(scan_info["spec_index"])
        scan_info_df = pd.DataFrame(scan_info)

        ms1_xic_df = pd.DataFrame()
        if ms1_data:
            max_length = max(len(data['mz_array']) for data in ms1_data)
            ms1_xic_df = pd.DataFrame({
                'scan_time': [data['scan_time'] for data in ms1_data],
                'mz_array': [np.pad(data['mz_array'], (0, max_length - len(data['mz_array'])))
                             for data in ms1_data],
                'int_array': [np.pad(data['int_array'], (0, max_length - len(data['int_array'])))
                              for data in ms1_data]
            })

        logger.info(f"Processed {len(scan_info_df)} spectra, {len(ms1_data)} MS1 scans")
        return scan_info_df, spec_dict, ms1_xic_df

    def _validate_spectrum(self, spec) -> bool:
        """Validate spectrum data"""
        if spec is None:
            return False
        if not hasattr(spec, 'ms_level') or spec.ms_level not in [1, 2]:
            return False
        try:
            peaks = spec.get_peaks()
            return len(peaks[0]) > 0
        except:
            return False


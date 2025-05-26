# MSImageProcessor.py
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import pandas as pd
import numpy as np
import pymzml
import logging
from dataclasses import dataclass
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import h5py
from tqdm import tqdm


@dataclass
class MSParameters:
    ms1_threshold: float = 1000.0
    ms2_threshold: float = 100.0
    mz_tolerance: float = 0.01
    rt_tolerance: float = 0.5
    min_intensity: float = 100.0
    output_dir: str = "output"
    n_workers: int = multiprocessing.cpu_count()


@dataclass
class ProcessedSpectrum:
    mz_array: np.ndarray
    intensity_array: np.ndarray
    metadata: Dict[str, Any]


class MSImageProcessor:
    def __init__(self, params: Optional[MSParameters] = None):
        self.params = params or MSParameters()
        self.supported_formats = {'.mzML'}
        self.mzml_reader = MZMLReader(self.params)

    def load_spectrum(self, path: Path) -> List[ProcessedSpectrum]:
        if path.suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        try:
            scan_info_df, spec_dict, ms1_xic_df = self.mzml_reader.extract_spectra(str(path))
            processed_spectra = []

            for spec_idx, spec_df in spec_dict.items():
                scan_info = scan_info_df[scan_info_df['spec_index'] == spec_idx].iloc[0]
                processed_spectra.append(ProcessedSpectrum(
                    mz_array=spec_df['mz'].values,
                    intensity_array=spec_df['intensity'].values,
                    metadata={
                        'scan_time': scan_info['scan_time'],
                        'dda_event_idx': scan_info['dda_event_idx'],
                        'DDA_rank': scan_info['DDA_rank'],
                        'scan_number': scan_info['scan_number'],
                        'polarity': scan_info['polarity']
                    }
                ))

            return processed_spectra
        except Exception as e:
            raise RuntimeError(f"Error reading file {path}: {str(e)}")

    def save_processed_spectra(self, processed_spectra: List[ProcessedSpectrum], output_path: Path) -> None:
        try:
            with h5py.File(output_path, 'w') as f:
                for i, spectrum in enumerate(processed_spectra):
                    grp = f.create_group(f'spectrum_{i}')
                    grp.create_dataset('mz_array', data=spectrum.mz_array)
                    grp.create_dataset('intensity_array', data=spectrum.intensity_array)
                    metadata_grp = grp.create_group('metadata')
                    for key, value in spectrum.metadata.items():
                        metadata_grp.attrs[key] = value
            logging.info(f"Saved processed spectra to {output_path}")
        except Exception as e:
            raise RuntimeError(f"Error saving processed spectra: {str(e)}")


class MZMLReader:
    def __init__(self, params: MSParameters):
        self.params = params

    def process_files_parallel(self, mzml_files: List[str]) -> None:
        logging.info(f"Processing {len(mzml_files)} files using {self.params.n_workers} workers")
        with ProcessPoolExecutor(max_workers=self.params.n_workers) as executor:
            futures = [executor.submit(self.process_single_file, mzml_file)
                       for mzml_file in mzml_files]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error in parallel processing: {str(e)}")

    def process_single_file(self, mzml_path: str) -> None:
        try:
            logging.info(f"Processing file: {mzml_path}")
            scan_info_df, spec_dict, ms1_xic_df = self.extract_spectra(mzml_path)
            output_dir = Path(self.params.output_dir) / Path(mzml_path).stem
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_results(output_dir, scan_info_df, spec_dict, ms1_xic_df)
        except Exception as e:
            logging.error(f"Error processing {mzml_path}: {str(e)}")

    def _save_results(self, output_dir: Path, scan_info_df: pd.DataFrame,
                      spec_dict: Dict, ms1_xic_df: pd.DataFrame) -> None:
        try:
            with ThreadPoolExecutor() as executor:
                executor.submit(scan_info_df.to_parquet, output_dir / "scan_info.parquet")
                executor.submit(ms1_xic_df.to_parquet, output_dir / "ms1_xic.parquet")
                futures = []
                for spec_idx, spec_df in spec_dict.items():
                    future = executor.submit(spec_df.to_parquet,
                                             output_dir / f"spec_{spec_idx}.parquet")
                    futures.append(future)
                for future in futures:
                    future.result()
            logging.info(f"Results stored in: {output_dir}")
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")

    def extract_spectra(self, mzml_path: str) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
        if not Path(mzml_path).is_file():
            raise FileNotFoundError(f"File not found: {mzml_path}")

        logging.info(f"Extracting: {mzml_path}")
        spec_obj = pymzml.run.Reader(mzml_path, MS1_Precision=5e-6, obo_version="4.1.33")
        polarity = self._extract_polarity(Path(mzml_path).name)
        return self._process_spectra(spec_obj, polarity)

    def _extract_polarity(self, filename: str) -> str:
        filename = filename.lower()
        if 'pos' in filename:
            return 'positive'
        elif 'neg' in filename:
            return 'negative'
        else:
            return 'unknown'

    def _process_spectra(self, spec_obj, polarity: str) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
        scan_info = {
            "spec_index": [], "scan_time": [], "dda_event_idx": [],
            "DDA_rank": [], "scan_number": [], "polarity": []
        }
        spec_dict = {}
        ms1_data = []
        current_dda_event = 0

        for spec in spec_obj:
            try:
                spec_idx = int(spec.ID)
                ms_level = spec.ms_level

                if ms_level == 1:
                    # Extract peaks using multiple methods
                    mz_array, int_array = [], []
                    try:
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
                    except Exception as e:
                        logging.error(f"Error extracting peaks from spectrum {spec_idx}: {str(e)}")
                        continue

                    # Convert to numpy arrays and ensure they're the same length
                    mz_array = np.array(mz_array)
                    int_array = np.array(int_array)

                    if len(mz_array) == 0 or len(int_array) == 0:
                        logging.warning(f"No peaks found in spectrum {spec_idx}")
                        continue

                    scan_time = spec.scan_time_in_minutes()

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
                            # Extract MS2 peaks using the same methods
                            mz_array, int_array = [], []
                            try:
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
                            except Exception as e:
                                logging.error(f"Error extracting peaks from MS2 spectrum {spec_idx}: {str(e)}")
                                continue

                            if len(mz_array) == 0 or len(int_array) == 0:
                                logging.warning(f"No peaks found in MS2 spectrum {spec_idx}")
                                continue

                            scan_time = spec.scan_time_in_minutes()

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
                logging.error(f"Error processing spectrum {spec_idx}: {str(e)}")
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

        return scan_info_df, spec_dict, ms1_xic_df

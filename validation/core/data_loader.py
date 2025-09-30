"""
Data Loader for Mass Spectrometry Data

Handles loading and preprocessing of mzML files for validation experiments.
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

try:
    import pymzml
    PYMZML_AVAILABLE = True
except ImportError:
    PYMZML_AVAILABLE = False
    logging.warning("pymzml not available, using mock data")

try:
    import pyopenms as oms
    OPENMS_AVAILABLE = True
except ImportError:
    OPENMS_AVAILABLE = False
    logging.warning("pyopenms not available, using alternative methods")

@dataclass
class SpectrumInfo:
    """Information about a mass spectrum"""
    scan_number: int
    retention_time: float
    ms_level: int
    precursor_mz: Optional[float]
    precursor_charge: Optional[int]
    mz_array: np.ndarray
    intensity_array: np.ndarray
    total_ion_current: float

@dataclass
class DatasetInfo:
    """Information about a complete dataset"""
    filename: str
    file_path: str
    instrument_type: str
    ionization_mode: str
    total_spectra: int
    ms1_spectra: int
    ms2_spectra: int
    rt_range: Tuple[float, float]
    mz_range: Tuple[float, float]
    file_size: int
    
class MZMLDataLoader:
    """Loader for mzML mass spectrometry data files"""
    
    def __init__(self, data_directory: str = "validation/public"):
        self.data_directory = Path(data_directory)
        self.logger = logging.getLogger(__name__)
        self._validate_dependencies()
    
    def _validate_dependencies(self):
        """Check if required dependencies are available"""
        if not PYMZML_AVAILABLE and not OPENMS_AVAILABLE:
            self.logger.warning("Neither pymzml nor pyopenms available. Using mock data.")
    
    def load_dataset(self, filename: str, max_spectra: Optional[int] = None) -> Tuple[List[SpectrumInfo], DatasetInfo]:
        """
        Load mzML dataset
        
        Args:
            filename: Name of mzML file to load
            max_spectra: Maximum number of spectra to load (for testing)
            
        Returns:
            Tuple of (spectra_list, dataset_info)
        """
        file_path = self.data_directory / filename
        
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return self._create_mock_data(filename)
        
        self.logger.info(f"Loading dataset: {filename}")
        
        if PYMZML_AVAILABLE:
            return self._load_with_pymzml(file_path, max_spectra)
        elif OPENMS_AVAILABLE:
            return self._load_with_openms(file_path, max_spectra)
        else:
            self.logger.warning("No MS libraries available, using mock data")
            return self._create_mock_data(filename)
    
    def _load_with_pymzml(self, file_path: Path, max_spectra: Optional[int]) -> Tuple[List[SpectrumInfo], DatasetInfo]:
        """Load data using pymzml"""
        spectra = []
        
        try:
            run = pymzml.run.Reader(str(file_path))
            
            ms1_count = 0
            ms2_count = 0
            rt_values = []
            mz_ranges = []
            
            for i, spectrum in enumerate(run):
                if max_spectra and i >= max_spectra:
                    break
                
                # Extract spectrum information
                ms_level = spectrum.ms_level
                scan_number = spectrum.ID
                retention_time = spectrum.scan_time_in_minutes() if spectrum.scan_time_in_minutes() else 0.0
                
                # Get m/z and intensity arrays
                mz_array = spectrum.mz
                intensity_array = spectrum.i
                
                if len(mz_array) == 0:
                    continue
                
                # Calculate total ion current
                total_ion_current = np.sum(intensity_array)
                
                # Handle precursor information for MS2
                precursor_mz = None
                precursor_charge = None
                
                if ms_level == 2 and spectrum.selected_precursors:
                    precursor_info = spectrum.selected_precursors[0]
                    precursor_mz = precursor_info.get('mz', None)
                    precursor_charge = precursor_info.get('charge', None)
                
                spectrum_info = SpectrumInfo(
                    scan_number=scan_number,
                    retention_time=retention_time,
                    ms_level=ms_level,
                    precursor_mz=precursor_mz,
                    precursor_charge=precursor_charge,
                    mz_array=np.array(mz_array),
                    intensity_array=np.array(intensity_array),
                    total_ion_current=total_ion_current
                )
                
                spectra.append(spectrum_info)
                
                # Track statistics
                if ms_level == 1:
                    ms1_count += 1
                elif ms_level == 2:
                    ms2_count += 1
                
                rt_values.append(retention_time)
                mz_ranges.extend([np.min(mz_array), np.max(mz_array)])
            
            # Create dataset info
            dataset_info = DatasetInfo(
                filename=file_path.name,
                file_path=str(file_path),
                instrument_type=self._determine_instrument_type(file_path.name),
                ionization_mode=self._determine_ionization_mode(file_path.name),
                total_spectra=len(spectra),
                ms1_spectra=ms1_count,
                ms2_spectra=ms2_count,
                rt_range=(min(rt_values), max(rt_values)) if rt_values else (0.0, 0.0),
                mz_range=(min(mz_ranges), max(mz_ranges)) if mz_ranges else (0.0, 0.0),
                file_size=file_path.stat().st_size
            )
            
            self.logger.info(f"Loaded {len(spectra)} spectra from {file_path.name}")
            return spectra, dataset_info
            
        except Exception as e:
            self.logger.error(f"Error loading with pymzml: {e}")
            return self._create_mock_data(file_path.name)
    
    def _load_with_openms(self, file_path: Path, max_spectra: Optional[int]) -> Tuple[List[SpectrumInfo], DatasetInfo]:
        """Load data using pyopenms"""
        # Implementation for pyopenms would go here
        # For now, fall back to mock data
        self.logger.warning("OpenMS loader not implemented, using mock data")
        return self._create_mock_data(file_path.name)
    
    def _create_mock_data(self, filename: str) -> Tuple[List[SpectrumInfo], DatasetInfo]:
        """Create mock data when real files can't be loaded"""
        self.logger.info(f"Creating mock data for {filename}")
        
        # Determine properties from filename
        instrument_type = self._determine_instrument_type(filename)
        ionization_mode = self._determine_ionization_mode(filename)
        
        # Create mock spectra
        spectra = []
        n_spectra = 100  # Mock 100 spectra
        
        for i in range(n_spectra):
            # Create synthetic spectrum
            n_peaks = np.random.randint(50, 500)
            mz_array = np.sort(np.random.uniform(100, 1000, n_peaks))
            intensity_array = np.random.exponential(1000, n_peaks)
            
            spectrum_info = SpectrumInfo(
                scan_number=i + 1,
                retention_time=i * 0.1,  # 0.1 minute intervals
                ms_level=1,
                precursor_mz=None,
                precursor_charge=None,
                mz_array=mz_array,
                intensity_array=intensity_array,
                total_ion_current=np.sum(intensity_array)
            )
            
            spectra.append(spectrum_info)
        
        # Create mock dataset info
        dataset_info = DatasetInfo(
            filename=filename,
            file_path=f"mock/{filename}",
            instrument_type=instrument_type,
            ionization_mode=ionization_mode,
            total_spectra=n_spectra,
            ms1_spectra=n_spectra,
            ms2_spectra=0,
            rt_range=(0.0, (n_spectra-1) * 0.1),
            mz_range=(100.0, 1000.0),
            file_size=1024 * 1024  # Mock 1MB file
        )
        
        return spectra, dataset_info
    
    def _determine_instrument_type(self, filename: str) -> str:
        """Determine instrument type from filename"""
        filename_lower = filename.lower()
        
        if 'qtof' in filename_lower:
            return 'qTOF'
        elif 'orbi' in filename_lower:
            return 'Orbitrap'
        elif 'waters' in filename_lower:
            return 'Waters'
        elif 'thermo' in filename_lower:
            return 'Thermo'
        else:
            return 'Unknown'
    
    def _determine_ionization_mode(self, filename: str) -> str:
        """Determine ionization mode from filename"""
        filename_lower = filename.lower()
        
        if '_neg_' in filename_lower or 'negative' in filename_lower:
            return 'Negative'
        elif '_pos_' in filename_lower or 'positive' in filename_lower:
            return 'Positive'
        else:
            return 'Unknown'
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available mzML files"""
        if not self.data_directory.exists():
            self.logger.warning(f"Data directory not found: {self.data_directory}")
            return []
        
        mzml_files = list(self.data_directory.glob("*.mzML"))
        return [f.name for f in mzml_files]
    
    def preprocess_spectra(self, spectra: List[SpectrumInfo], 
                          normalize: bool = True,
                          remove_noise: bool = True,
                          min_intensity: float = 100.0) -> List[SpectrumInfo]:
        """
        Preprocess spectra for analysis
        
        Args:
            spectra: List of spectrum info objects
            normalize: Whether to normalize intensities
            remove_noise: Whether to remove low-intensity peaks
            min_intensity: Minimum intensity threshold
            
        Returns:
            List of preprocessed spectra
        """
        self.logger.info(f"Preprocessing {len(spectra)} spectra")
        
        preprocessed_spectra = []
        
        for spectrum in spectra:
            mz_array = spectrum.mz_array.copy()
            intensity_array = spectrum.intensity_array.copy()
            
            # Remove noise
            if remove_noise:
                mask = intensity_array >= min_intensity
                mz_array = mz_array[mask]
                intensity_array = intensity_array[mask]
            
            # Normalize intensities
            if normalize and len(intensity_array) > 0:
                max_intensity = np.max(intensity_array)
                if max_intensity > 0:
                    intensity_array = intensity_array / max_intensity * 100.0
            
            # Create preprocessed spectrum
            preprocessed_spectrum = SpectrumInfo(
                scan_number=spectrum.scan_number,
                retention_time=spectrum.retention_time,
                ms_level=spectrum.ms_level,
                precursor_mz=spectrum.precursor_mz,
                precursor_charge=spectrum.precursor_charge,
                mz_array=mz_array,
                intensity_array=intensity_array,
                total_ion_current=np.sum(intensity_array)
            )
            
            preprocessed_spectra.append(preprocessed_spectrum)
        
        self.logger.info(f"Preprocessing complete: {len(preprocessed_spectra)} spectra")
        return preprocessed_spectra
    
    def export_to_dataframe(self, spectra: List[SpectrumInfo]) -> pd.DataFrame:
        """
        Export spectra to pandas DataFrame for analysis
        
        Args:
            spectra: List of spectrum info objects
            
        Returns:
            DataFrame with spectrum metadata
        """
        data = []
        
        for spectrum in spectra:
            data.append({
                'scan_number': spectrum.scan_number,
                'retention_time': spectrum.retention_time,
                'ms_level': spectrum.ms_level,
                'precursor_mz': spectrum.precursor_mz,
                'precursor_charge': spectrum.precursor_charge,
                'n_peaks': len(spectrum.mz_array),
                'total_ion_current': spectrum.total_ion_current,
                'mz_min': np.min(spectrum.mz_array) if len(spectrum.mz_array) > 0 else 0,
                'mz_max': np.max(spectrum.mz_array) if len(spectrum.mz_array) > 0 else 0,
                'intensity_max': np.max(spectrum.intensity_array) if len(spectrum.intensity_array) > 0 else 0
            })
        
        return pd.DataFrame(data)

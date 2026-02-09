"""
DataContainer for Mass Spectrometry Data Organization
=====================================================

Comprehensive data structure for organizing extracted mzML data with support for:
- Precursor-fragment linking
- DDA event organization
- Polarity and metadata extraction from filename
- S-Entropy coordinate system integration
- Phase-lock signature preparation
- Dual-modality (numerical + visual) processing support

Author: Kundai Chinyamakobvu
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict


@dataclass
class SpectrumMetadata:
    """Metadata for individual spectrum."""
    spec_index: int
    scan_number: int
    scan_time: float  # RT in minutes
    dda_event_idx: int
    dda_rank: int  # 0 for MS1, 1+ for MS2
    ms_level: int  # 1 or 2
    precursor_mz: float = 0.0  # For MS2 only
    precursor_intensity: float = 0.0  # From MS1
    precursor_charge: Optional[int] = None
    collision_energy: Optional[float] = None
    total_ion_current: float = 0.0
    base_peak_intensity: float = 0.0
    base_peak_mz: float = 0.0
    num_peaks: int = 0

    # S-Entropy related (to be computed)
    s_entropy_coords: Optional[np.ndarray] = None
    s_entropy_features: Optional[np.ndarray] = None

    # Phase-lock related (to be computed)
    phase_lock_signature: Optional[np.ndarray] = None
    categorical_state: Optional[int] = None


@dataclass
class PrecursorFragmentPair:
    """Links MS1 precursor with MS2 fragment spectrum."""
    precursor_spec_index: int
    fragment_spec_index: int
    precursor_mz: float
    precursor_rt: float
    fragment_rt: float
    dda_event_idx: int
    dda_rank: int

    # Precursor peak info
    precursor_intensity: float = 0.0
    precursor_ppm_error: float = 0.0

    # MS1 and MS2 DataFrames
    ms1_spectrum: Optional[pd.DataFrame] = None
    ms2_spectrum: Optional[pd.DataFrame] = None

    # Isotope pattern (from MS1)
    isotope_pattern: Optional[pd.DataFrame] = None

    # Complementarity measures (for validation)
    by_ion_complementarity: Optional[float] = None
    temporal_proximity: Optional[float] = None


@dataclass
class DDAEvent:
    """Represents a complete DDA cycle (1 MS1 + multiple MS2)."""
    dda_event_idx: int
    ms1_spec_index: int
    ms1_scan_time: float
    ms2_spec_indices: List[int] = field(default_factory=list)
    ms2_precursor_mzs: List[float] = field(default_factory=list)
    ms2_scan_times: List[float] = field(default_factory=list)

    # Full spectra
    ms1_spectrum: Optional[pd.DataFrame] = None
    ms2_spectra: Dict[int, pd.DataFrame] = field(default_factory=dict)

    # Precursor-fragment pairs for this event
    precursor_fragment_pairs: List[PrecursorFragmentPair] = field(default_factory=list)


class MSDataContainer:
    """
    Comprehensive container for mass spectrometry data.

    Organizes extracted mzML data with support for:
    - Precursor-fragment linking
    - DDA event navigation
    - Metadata extraction from filename
    - S-Entropy framework integration
    - Phase-lock signature computation
    - Dual-modality processing
    """

    def __init__(
        self,
        mzml_filepath: str,
        scan_info_df: pd.DataFrame,
        spectra_dict: Dict[int, pd.DataFrame],
        ms1_xic_df: pd.DataFrame,
        extraction_params: Optional[Dict] = None
    ):
        """
        Initialize DataContainer with extracted mzML data.

        Args:
            mzml_filepath: Path to original mzML file
            scan_info_df: DataFrame with columns [dda_event_idx, spec_index, scan_time,
                         DDA_rank, scan_number, MS2_PR_mz]
            spectra_dict: Dictionary mapping spec_index to DataFrame with columns [mz, i]
            ms1_xic_df: DataFrame with MS1 XIC data [mz, i, rt, spec_idx, dda_event_idx,
                       DDA_rank, scan_number]
            extraction_params: Parameters used for extraction (thresholds, precision, etc.)
        """
        # Core extracted data (preserved exactly as extracted)
        self.mzml_filepath = Path(mzml_filepath)
        self.scan_info_df = scan_info_df
        self.spectra_dict = spectra_dict
        self.ms1_xic_df = ms1_xic_df
        self.extraction_params = extraction_params or {}

        # File metadata extracted from filename
        self._extract_file_metadata()

        # Organized data structures
        self._spectrum_metadata: Dict[int, SpectrumMetadata] = {}
        self._dda_events: Dict[int, DDAEvent] = {}
        self._precursor_fragment_pairs: List[PrecursorFragmentPair] = []

        # Indices for fast lookup
        self._ms1_indices: List[int] = []
        self._ms2_indices: List[int] = []
        self._precursor_mz_index: Dict[float, List[int]] = defaultdict(list)
        self._rt_sorted_indices: List[int] = []

        # S-Entropy and phase-lock data (computed on demand)
        self._s_entropy_computed = False
        self._phase_lock_computed = False

        # Initialize data structures
        self._build_data_structures()

    def _extract_file_metadata(self):
        """
        Extract metadata from mzML filename.

        Common patterns:
        - Polarity: _pos, _neg, _POS, _NEG, positive, negative
        - Sample name: typically before polarity marker
        - Replicate: rep1, rep2, r1, r2, etc.
        - Batch: batch1, b1, etc.
        """
        filename = self.mzml_filepath.stem  # Filename without extension

        # Extract polarity
        polarity_patterns = [
            (r'[_\-\.]pos(?:itive)?[_\-\.]?', 'positive'),
            (r'[_\-\.]neg(?:ative)?[_\-\.]?', 'negative'),
            (r'[_\-\.]POS(?:ITIVE)?[_\-\.]?', 'positive'),
            (r'[_\-\.]NEG(?:ATIVE)?[_\-\.]?', 'negative'),
        ]

        self.polarity = None
        for pattern, pol in polarity_patterns:
            if re.search(pattern, filename):
                self.polarity = pol
                break

        if self.polarity is None:
            # Default to positive if not specified
            self.polarity = 'positive'
            print(f"[WARNING] Could not extract polarity from filename: {filename}. Defaulting to 'positive'.")

        # Extract sample name (before polarity marker)
        sample_match = re.match(r'^(.+?)(?:[_\-\.](?:pos|neg|POS|NEG))', filename)
        if sample_match:
            self.sample_name = sample_match.group(1)
        else:
            self.sample_name = filename

        # Extract replicate number
        rep_match = re.search(r'(?:rep|r|replicate)[_\-]?(\d+)', filename, re.IGNORECASE)
        self.replicate = int(rep_match.group(1)) if rep_match else None

        # Extract batch number
        batch_match = re.search(r'(?:batch|b)[_\-]?(\d+)', filename, re.IGNORECASE)
        self.batch = int(batch_match.group(1)) if batch_match else None

        # Store full metadata
        self.file_metadata = {
            'filename': filename,
            'filepath': str(self.mzml_filepath),
            'sample_name': self.sample_name,
            'polarity': self.polarity,
            'replicate': self.replicate,
            'batch': self.batch,
        }

        print("[INFO] File metadata extracted:")
        print(f"  Sample: {self.sample_name}")
        print(f"  Polarity: {self.polarity}")
        if self.replicate:
            print(f"  Replicate: {self.replicate}")
        if self.batch:
            print(f"  Batch: {self.batch}")

    def _build_data_structures(self):
        """Build organized data structures from extracted raw data."""
        print("[INFO] Building organized data structures...")

        # 1. Create spectrum metadata for all spectra
        for idx, row in self.scan_info_df.iterrows():
            spec_idx = int(row['spec_index'])

            # Get spectrum DataFrame
            spec_df = self.spectra_dict.get(spec_idx)
            if spec_df is None or spec_df.empty:
                continue

            # Compute basic statistics
            total_ion_current = float(spec_df['i'].sum())
            base_peak_idx = spec_df['i'].idxmax()
            base_peak_intensity = float(spec_df.loc[base_peak_idx, 'i'])
            base_peak_mz = float(spec_df.loc[base_peak_idx, 'mz'])
            num_peaks = len(spec_df)

            # Create metadata
            ms_level = 1 if row['DDA_rank'] == 0 else 2
            precursor_mz = float(row['MS2_PR_mz']) if ms_level == 2 else 0.0

            metadata = SpectrumMetadata(
                spec_index=spec_idx,
                scan_number=int(row['scan_number']),
                scan_time=float(row['scan_time']),
                dda_event_idx=int(row['dda_event_idx']),
                dda_rank=int(row['DDA_rank']),
                ms_level=ms_level,
                precursor_mz=precursor_mz,
                total_ion_current=total_ion_current,
                base_peak_intensity=base_peak_intensity,
                base_peak_mz=base_peak_mz,
                num_peaks=num_peaks,
            )

            self._spectrum_metadata[spec_idx] = metadata

            # Build indices
            if ms_level == 1:
                self._ms1_indices.append(spec_idx)
            else:
                self._ms2_indices.append(spec_idx)
                self._precursor_mz_index[round(precursor_mz, 4)].append(spec_idx)

        # Sort indices by RT
        self._rt_sorted_indices = sorted(
            self._spectrum_metadata.keys(),
            key=lambda idx: self._spectrum_metadata[idx].scan_time
        )

        # 2. Build DDA events
        self._build_dda_events()

        # 3. Build precursor-fragment pairs
        self._build_precursor_fragment_pairs()

        print(f"[INFO] Data structure built:")
        print(f"  Total spectra: {len(self._spectrum_metadata)}")
        print(f"  MS1 spectra: {len(self._ms1_indices)}")
        print(f"  MS2 spectra: {len(self._ms2_indices)}")
        print(f"  DDA events: {len(self._dda_events)}")
        print(f"  Precursor-fragment pairs: {len(self._precursor_fragment_pairs)}")

    def _build_dda_events(self):
        """Organize spectra into DDA events."""
        for dda_idx in self.scan_info_df['dda_event_idx'].unique():
            event_scans = self.scan_info_df[
                self.scan_info_df['dda_event_idx'] == dda_idx
            ]

            # Get MS1 (DDA_rank == 0)
            ms1_scans = event_scans[event_scans['DDA_rank'] == 0]
            if ms1_scans.empty:
                continue

            ms1_spec_idx = int(ms1_scans.iloc[0]['spec_index'])
            ms1_rt = float(ms1_scans.iloc[0]['scan_time'])

            # Get MS2 scans (DDA_rank > 0)
            ms2_scans = event_scans[event_scans['DDA_rank'] > 0]
            ms2_spec_indices = ms2_scans['spec_index'].astype(int).tolist()
            ms2_precursor_mzs = ms2_scans['MS2_PR_mz'].astype(float).tolist()
            ms2_scan_times = ms2_scans['scan_time'].astype(float).tolist()

            # Create DDA event
            dda_event = DDAEvent(
                dda_event_idx=int(dda_idx),
                ms1_spec_index=ms1_spec_idx,
                ms1_scan_time=ms1_rt,
                ms2_spec_indices=ms2_spec_indices,
                ms2_precursor_mzs=ms2_precursor_mzs,
                ms2_scan_times=ms2_scan_times,
                ms1_spectrum=self.spectra_dict.get(ms1_spec_idx),
                ms2_spectra={idx: self.spectra_dict.get(idx) for idx in ms2_spec_indices}
            )

            self._dda_events[int(dda_idx)] = dda_event

    def _build_precursor_fragment_pairs(self):
        """Build precursor-fragment pairs with linking information."""
        ms1_precision = self.extraction_params.get('ms1_precision', 50e-6)

        for dda_event in self._dda_events.values():
            ms1_spec = dda_event.ms1_spectrum
            if ms1_spec is None or ms1_spec.empty:
                continue

            for ms2_idx, precursor_mz in zip(
                dda_event.ms2_spec_indices,
                dda_event.ms2_precursor_mzs
            ):
                ms2_spec = self.spectra_dict.get(ms2_idx)
                if ms2_spec is None or ms2_spec.empty:
                    continue

                # Find precursor in MS1
                mz_tolerance = precursor_mz * ms1_precision
                precursor_peaks = ms1_spec[
                    (ms1_spec['mz'] >= precursor_mz - mz_tolerance) &
                    (ms1_spec['mz'] <= precursor_mz + mz_tolerance)
                ]

                precursor_intensity = 0.0
                precursor_ppm_error = 0.0
                if not precursor_peaks.empty:
                    # Take most intense peak in range
                    best_peak = precursor_peaks.loc[precursor_peaks['i'].idxmax()]
                    precursor_intensity = float(best_peak['i'])
                    precursor_ppm_error = 1e6 * (best_peak['mz'] - precursor_mz) / precursor_mz

                # Get fragment RT
                ms2_metadata = self._spectrum_metadata.get(ms2_idx)
                fragment_rt = ms2_metadata.scan_time if ms2_metadata else 0.0

                # Create pair
                pair = PrecursorFragmentPair(
                    precursor_spec_index=dda_event.ms1_spec_index,
                    fragment_spec_index=ms2_idx,
                    precursor_mz=precursor_mz,
                    precursor_rt=dda_event.ms1_scan_time,
                    fragment_rt=fragment_rt,
                    dda_event_idx=dda_event.dda_event_idx,
                    dda_rank=self._spectrum_metadata[ms2_idx].dda_rank,
                    precursor_intensity=precursor_intensity,
                    precursor_ppm_error=precursor_ppm_error,
                    ms1_spectrum=ms1_spec.copy(),
                    ms2_spectrum=ms2_spec.copy(),
                )

                self._precursor_fragment_pairs.append(pair)
                dda_event.precursor_fragment_pairs.append(pair)

    # ========================================================================
    # Public API: Data Access Methods
    # ========================================================================

    def get_spectrum(self, spec_index: int) -> Optional[pd.DataFrame]:
        """Get spectrum DataFrame by index."""
        return self.spectra_dict.get(spec_index)

    def get_spectrum_metadata(self, spec_index: int) -> Optional[SpectrumMetadata]:
        """Get spectrum metadata by index."""
        return self._spectrum_metadata.get(spec_index)

    def get_dda_event(self, dda_event_idx: int) -> Optional[DDAEvent]:
        """Get complete DDA event by index."""
        return self._dda_events.get(dda_event_idx)

    def get_ms1_spectra(self) -> Dict[int, pd.DataFrame]:
        """Get all MS1 spectra."""
        return {idx: self.spectra_dict[idx] for idx in self._ms1_indices}

    def get_ms2_spectra(self) -> Dict[int, pd.DataFrame]:
        """Get all MS2 spectra."""
        return {idx: self.spectra_dict[idx] for idx in self._ms2_indices}

    def get_spectra_in_rt_range(
        self,
        rt_min: float,
        rt_max: float,
        ms_level: Optional[int] = None
    ) -> Dict[int, pd.DataFrame]:
        """Get spectra within RT range."""
        spectra = {}
        for idx in self._rt_sorted_indices:
            metadata = self._spectrum_metadata[idx]
            if rt_min <= metadata.scan_time <= rt_max:
                if ms_level is None or metadata.ms_level == ms_level:
                    spectra[idx] = self.spectra_dict[idx]
        return spectra

    def get_spectra_by_precursor_mz(
        self,
        precursor_mz: float,
        tolerance_ppm: float = 10.0
    ) -> List[Tuple[int, pd.DataFrame]]:
        """Get MS2 spectra with precursor m/z within tolerance."""
        tolerance = precursor_mz * tolerance_ppm * 1e-6
        results = []

        for idx in self._ms2_indices:
            metadata = self._spectrum_metadata[idx]
            if abs(metadata.precursor_mz - precursor_mz) <= tolerance:
                results.append((idx, self.spectra_dict[idx]))

        return results

    def get_precursor_fragment_pairs(
        self,
        dda_event_idx: Optional[int] = None
    ) -> List[PrecursorFragmentPair]:
        """
        Get precursor-fragment pairs.

        Args:
            dda_event_idx: If specified, return only pairs from this DDA event
        """
        if dda_event_idx is not None:
            event = self._dda_events.get(dda_event_idx)
            return event.precursor_fragment_pairs if event else []
        return self._precursor_fragment_pairs

    def extract_xic(
        self,
        target_mz: float,
        tolerance_ppm: float = 10.0,
        rt_range: Optional[Tuple[float, float]] = None
    ) -> pd.DataFrame:
        """
        Extract XIC for target m/z.

        Args:
            target_mz: Target m/z value
            tolerance_ppm: Mass tolerance in ppm
            rt_range: Optional (rt_min, rt_max) tuple

        Returns:
            DataFrame with columns [rt, mz, i, ppm]
        """
        tolerance = target_mz * tolerance_ppm * 1e-6
        mz_min = target_mz - tolerance
        mz_max = target_mz + tolerance

        query = f"{mz_min} <= mz <= {mz_max}"
        xic_df = self.ms1_xic_df.query(query).copy()

        if rt_range is not None:
            rt_min, rt_max = rt_range
            xic_df = xic_df[(xic_df['rt'] >= rt_min) & (xic_df['rt'] <= rt_max)]

        # Calculate ppm error
        xic_df['ppm'] = 1e6 * (xic_df['mz'] - target_mz) / target_mz
        xic_df['ppm'] = xic_df['ppm'].abs()

        # Sort by RT
        xic_df = xic_df.sort_values('rt').reset_index(drop=True)

        return xic_df[['rt', 'mz', 'i', 'ppm']]

    # ========================================================================
    # S-Entropy Framework Integration
    # ========================================================================

    def compute_s_entropy_coordinates(
        self,
        spectrum_indices: Optional[List[int]] = None,
        force_recompute: bool = False
    ):
        """
        Compute S-Entropy coordinates for spectra.

        This will be integrated with your S-Entropy framework.
        Placeholder for now - to be implemented with actual S-Entropy calculation.

        Args:
            spectrum_indices: Specific spectra to compute, or None for all
            force_recompute: Recompute even if already computed
        """
        if self._s_entropy_computed and not force_recompute:
            return

        indices = spectrum_indices or list(self._spectrum_metadata.keys())

        print(f"[INFO] Computing S-Entropy coordinates for {len(indices)} spectra...")

        for idx in indices:
            spec_df = self.spectra_dict.get(idx)
            metadata = self._spectrum_metadata.get(idx)

            if spec_df is None or metadata is None:
                continue

            # Placeholder: actual S-Entropy computation will go here
            # This should call your S-Entropy coordinate transformer
            # For now, create dummy coordinates
            s_coords = np.array([
                metadata.base_peak_mz / 1000.0,  # S_knowledge
                metadata.scan_time / 60.0,        # S_time
                metadata.total_ion_current / 1e6  # S_entropy
            ])

            metadata.s_entropy_coords = s_coords

        self._s_entropy_computed = True
        print("[INFO] S-Entropy coordinates computed.")

    def compute_phase_lock_signatures(
        self,
        spectrum_indices: Optional[List[int]] = None,
        force_recompute: bool = False
    ):
        """
        Compute phase-lock signatures for spectra.

        This integrates with your phase-lock theory and categorical completion.
        Placeholder for now - to be implemented with actual phase-lock calculation.

        Args:
            spectrum_indices: Specific spectra to compute, or None for all
            force_recompute: Recompute even if already computed
        """
        if self._phase_lock_computed and not force_recompute:
            return

        indices = spectrum_indices or list(self._spectrum_metadata.keys())

        print(f"[INFO] Computing phase-lock signatures for {len(indices)} spectra...")

        for idx in indices:
            metadata = self._spectrum_metadata.get(idx)
            if metadata is None or metadata.s_entropy_coords is None:
                continue

            # Placeholder: actual phase-lock signature computation
            # This will involve your dual-modality analysis
            phase_lock_sig = np.random.rand(14)  # 14D features
            metadata.phase_lock_signature = phase_lock_sig

        self._phase_lock_computed = True
        print("[INFO] Phase-lock signatures computed.")

    # ========================================================================
    # Summary and Export Methods
    # ========================================================================

    def summary(self) -> Dict:
        """Get summary statistics of the data."""
        rt_range = (
            min(m.scan_time for m in self._spectrum_metadata.values()),
            max(m.scan_time for m in self._spectrum_metadata.values())
        )

        ms1_tics = [m.total_ion_current for m in self._spectrum_metadata.values()
                    if m.ms_level == 1]
        ms2_tics = [m.total_ion_current for m in self._spectrum_metadata.values()
                    if m.ms_level == 2]

        return {
            'file_info': self.file_metadata,
            'extraction_params': self.extraction_params,
            'statistics': {
                'total_spectra': len(self._spectrum_metadata),
                'ms1_spectra': len(self._ms1_indices),
                'ms2_spectra': len(self._ms2_indices),
                'dda_events': len(self._dda_events),
                'precursor_fragment_pairs': len(self._precursor_fragment_pairs),
                'rt_range_min': rt_range[0],
                'rt_range_max': rt_range[1],
                'ms1_median_tic': np.median(ms1_tics) if ms1_tics else 0,
                'ms2_median_tic': np.median(ms2_tics) if ms2_tics else 0,
                's_entropy_computed': self._s_entropy_computed,
                'phase_lock_computed': self._phase_lock_computed,
            }
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export all spectrum metadata to DataFrame for analysis.

        Returns:
            DataFrame with all spectrum metadata
        """
        records = []
        for idx, metadata in self._spectrum_metadata.items():
            record = {
                'spec_index': metadata.spec_index,
                'scan_number': metadata.scan_number,
                'scan_time': metadata.scan_time,
                'ms_level': metadata.ms_level,
                'dda_event_idx': metadata.dda_event_idx,
                'dda_rank': metadata.dda_rank,
                'precursor_mz': metadata.precursor_mz,
                'total_ion_current': metadata.total_ion_current,
                'base_peak_intensity': metadata.base_peak_intensity,
                'base_peak_mz': metadata.base_peak_mz,
                'num_peaks': metadata.num_peaks,
                'sample_name': self.sample_name,
                'polarity': self.polarity,
                'replicate': self.replicate,
                'batch': self.batch,
            }

            # Add S-Entropy coords if computed
            if metadata.s_entropy_coords is not None:
                record['s_knowledge'] = metadata.s_entropy_coords[0]
                record['s_time'] = metadata.s_entropy_coords[1]
                record['s_entropy'] = metadata.s_entropy_coords[2]

            records.append(record)

        return pd.DataFrame(records)

    def __repr__(self) -> str:
        """String representation."""
        summary = self.summary()
        stats = summary['statistics']
        return (
            f"MSDataContainer(\n"
            f"  Sample: {self.sample_name}\n"
            f"  Polarity: {self.polarity}\n"
            f"  Spectra: {stats['total_spectra']} (MS1: {stats['ms1_spectra']}, MS2: {stats['ms2_spectra']})\n"
            f"  DDA Events: {stats['dda_events']}\n"
            f"  RT Range: {stats['rt_range_min']:.2f} - {stats['rt_range_max']:.2f} min\n"
            f"  S-Entropy: {'computed' if stats['s_entropy_computed'] else 'not computed'}\n"
            f"  Phase-Lock: {'computed' if stats['phase_lock_computed'] else 'not computed'}\n"
            f")"
        )

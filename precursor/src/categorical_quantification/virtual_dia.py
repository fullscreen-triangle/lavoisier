#!/usr/bin/env python3
"""
VirtualDIA.py

Virtual Data-Independent Acquisition using categorical windows.
Solves the deconvolution problem by operating in S-entropy space.

Author: Kundai Farai Sachikonye (with AI assistance)
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from StStellasTandemMS import SEntropyCalculator, CategoricalState
from VirtualMS2Engine import VirtualMS2Engine


@dataclass
class CategoricalWindow:
    """Represents a window in S-entropy space."""
    window_id: int
    center: np.ndarray  # 14D S-entropy center
    width: float  # Window width in S-entropy units
    precursors: List[CategoricalState]

    def contains(self, state: CategoricalState) -> bool:
        """Check if state falls within window."""
        distance = np.linalg.norm(state.s_entropy - self.center)
        return distance < self.width


class VirtualDIAEngine:
    """
    Virtual DIA engine using categorical windows.
    """

    def __init__(self, window_width: float = 0.5, overlap: float = 0.1):
        """
        Args:
            window_width: Width of categorical windows in S-entropy units
            overlap: Overlap between adjacent windows (for edge cases)
        """
        self.window_width = window_width
        self.overlap = overlap

        self.sentropy_calc = SEntropyCalculator()
        self.virtual_ms2 = VirtualMS2Engine()

    def define_categorical_windows(self,
                                   ms1_spectrum: Dict,
                                   num_windows: int = 20) -> List[CategoricalWindow]:
        """
        Define categorical windows in S-entropy space.

        Unlike traditional DIA (fixed m/z windows), these windows adapt
        to the S-entropy distribution of the sample.
        """
        print(f"Defining {num_windows} categorical windows...")

        # Compute S-entropy for all precursors
        precursor_states = []

        for i, (mz, intensity) in enumerate(zip(ms1_spectrum['mz'], ms1_spectrum['intensity'])):
            rt_array = np.array([ms1_spectrum['rt']])

            s_vec = self.sentropy_calc.compute_sentropy_vector(
                np.array([mz]), np.array([intensity]), rt_array, mz
            )

            state = CategoricalState(
                mz=mz,
                intensity=intensity,
                rt=ms1_spectrum['rt'],
                precursor_mz=mz,
                s_entropy=s_vec,
                spectrum_id=i
            )
            precursor_states.append(state)

        # Cluster in S-entropy space
        from sklearn.cluster import KMeans

        sentropy_matrix = np.array([s.s_entropy for s in precursor_states])

        kmeans = KMeans(n_clusters=num_windows, random_state=42)
        kmeans.fit(sentropy_matrix)

        # Create categorical windows
        windows = []

        for i in range(num_windows):
            center = kmeans.cluster_centers_[i]

            # Find precursors in this cluster
            cluster_mask = kmeans.labels_ == i
            cluster_precursors = [s for s, mask in zip(precursor_states, cluster_mask) if mask]

            window = CategoricalWindow(
                window_id=i,
                center=center,
                width=self.window_width,
                precursors=cluster_precursors
            )
            windows.append(window)

        print(f"  ✓ Defined {len(windows)} windows")
        print(f"  ✓ Average precursors per window: {np.mean([len(w.precursors) for w in windows]):.1f}")

        return windows

    def acquire_virtual_dia(self,
                           ms1_spectrum: Dict,
                           collision_energy: float = 25.0) -> List[Dict]:
        """
        Perform virtual DIA acquisition.

        Returns:
            List of virtual MS² spectra, one per categorical window
        """
        print("\nPerforming virtual DIA acquisition...")

        # Define categorical windows
        windows = self.define_categorical_windows(ms1_spectrum)

        # Generate virtual MS² for each window
        virtual_dia_spectra = []

        for window in windows:
            print(f"  Processing window {window.window_id + 1}/{len(windows)}...", end='\r')

            # Generate virtual MS² for all precursors in window
            window_fragments = []

            for precursor in window.precursors:
                virtual_spec = self.virtual_ms2.generate_virtual_ms2(
                    precursor.mz,
                    precursor.intensity,
                    precursor.rt,
                    precursor.charge,
                    collision_energy
                )

                # Tag fragments with precursor info
                for mz, intensity in zip(virtual_spec['fragment_mz'],
                                        virtual_spec['fragment_intensity']):
                    window_fragments.append({
                        'mz': mz,
                        'intensity': intensity,
                        'precursor_mz': precursor.mz,
                        'window_id': window.window_id
                    })

            # Combine fragments from all precursors in window
            # (this is the "multiplexed" DIA spectrum)
            if window_fragments:
                window_df = pd.DataFrame(window_fragments)

                # Merge overlapping peaks (within 0.01 Da)
                merged_spectrum = self._merge_overlapping_peaks(window_df)

                virtual_dia_spectra.append({
                    'window_id': window.window_id,
                    'window_center': window.center,
                    'num_precursors': len(window.precursors),
                    'fragment_mz': merged_spectrum['mz'].values,
                    'fragment_intensity': merged_spectrum['intensity'].values,
                    'precursor_assignments': merged_spectrum['precursor_mz'].values
                })

        print(f"  ✓ Generated {len(virtual_dia_spectra)} virtual DIA spectra" + " " * 20)

        return virtual_dia_spectra

    def _merge_overlapping_peaks(self, fragments_df: pd.DataFrame,
                                 tolerance: float = 0.01) -> pd.DataFrame:
        """
        Merge overlapping peaks in DIA spectrum.
        """
        # Sort by m/z
        fragments_df = fragments_df.sort_values('mz')

        merged = []
        current_mz = fragments_df.iloc[0]['mz']
        current_intensity = fragments_df.iloc[0]['intensity']
        current_precursors = [fragments_df.iloc[0]['precursor_mz']]

        for _, row in fragments_df.iloc[1:].iterrows():
            if abs(row['mz'] - current_mz) < tolerance:
                # Merge with current peak
                current_intensity += row['intensity']
                current_precursors.append(row['precursor_mz'])
            else:
                # Save current peak and start new one
                merged.append({
                    'mz': current_mz,
                    'intensity': current_intensity,
                    'precursor_mz': current_precursors[0],  # Most abundant precursor
                    'num_precursors': len(set(current_precursors))
                })

                current_mz = row['mz']
                current_intensity = row['intensity']
                current_precursors = [row['precursor_mz']]

        # Add last peak
        merged.append({
            'mz': current_mz,
            'intensity': current_intensity,
            'precursor_mz': current_precursors[0],
            'num_precursors': len(set(current_precursors))
        })

        return pd.DataFrame(merged)

    def deconvolve_dia_spectrum(self,
                               dia_spectrum: Dict,
                               categorical_window: CategoricalWindow) -> List[Dict]:
        """
        Deconvolve DIA spectrum back to individual precursors.

        Key advantage: Deconvolution is TRIVIAL because we know which
        fragments came from which categorical equivalence class!
        """
        print(f"\nDeconvolving DIA spectrum for window {dia_spectrum['window_id']}...")

        # For each precursor in window, extract its fragments
        deconvolved_spectra = []

        for precursor in categorical_window.precursors:
            # Find fragments assigned to this precursor
            mask = dia_spectrum['precursor_assignments'] == precursor.mz

            if np.any(mask):
                precursor_spectrum = {
                    'precursor_mz': precursor.mz,
                    'precursor_intensity': precursor.intensity,
                    'fragment_mz': dia_spectrum['fragment_mz'][mask],
                    'fragment_intensity': dia_spectrum['fragment_intensity'][mask],
                    'num_fragments': np.sum(mask)
                }
                deconvolved_spectra.append(precursor_spectrum)

        print(f"  ✓ Deconvolved into {len(deconvolved_spectra)} precursor spectra")

        return deconvolved_spectra

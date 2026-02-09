"""
S-Entropy Core Functions for Proteomics Validation
Implements complete molecular encoding framework from the paper
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass
from scipy.stats import entropy

# ============================================================================
# AMINO ACID BASE COORDINATE MAPPING
# Following the physicochemical property approach from the paper
# ============================================================================

# Amino acid physicochemical properties (normalized to [0,1] or [-1,1])
AMINO_ACID_PROPERTIES = {
    # (hydrophobicity, polarity, size) - base coordinates
    'A': (0.62, 0.0, 0.09),  # Alanine
    'R': (-2.53, 1.0, 0.26),  # Arginine (positive)
    'N': (-0.78, 0.5, 0.17),  # Asparagine (polar)
    'D': (-0.90, -1.0, 0.16),  # Aspartate (negative)
    'C': (0.29, 0.5, 0.15),  # Cysteine
    'Q': (-0.85, 0.5, 0.21),  # Glutamine (polar)
    'E': (-0.74, -1.0, 0.20),  # Glutamate (negative)
    'G': (0.48, 0.0, 0.00),  # Glycine (smallest)
    'H': (-0.40, 1.0, 0.21),  # Histidine (positive)
    'I': (1.38, 0.0, 0.19),  # Isoleucine
    'L': (1.06, 0.0, 0.19),  # Leucine
    'K': (-1.50, 1.0, 0.21),  # Lysine (positive)
    'M': (0.64, 0.0, 0.21),  # Methionine
    'F': (1.19, 0.0, 0.23),  # Phenylalanine
    'P': (0.12, 0.0, 0.14),  # Proline
    'S': (-0.18, 0.5, 0.12),  # Serine (polar)
    'T': (-0.05, 0.5, 0.14),  # Threonine (polar)
    'W': (0.81, 0.0, 0.28),  # Tryptophan (largest)
    'Y': (0.26, 0.5, 0.25),  # Tyrosine (polar)
    'V': (1.08, 0.0, 0.16),  # Valine
}


@dataclass
class SEntropySpectrum:
    """S-Entropy representation of MS/MS spectrum"""
    spectrum_id: str
    precursor_mz: float
    precursor_charge: int
    peptide_sequence: Optional[str]

    # Raw data
    fragments_mz: np.ndarray
    fragments_intensity: np.ndarray
    retention_time: float

    # S-Entropy coordinates (3D per fragment)
    sentropy_coords_3d: np.ndarray  # Shape: (n_fragments, 3)

    # Aggregated 14D feature vector for clustering
    sentropy_features_14d: np.ndarray

    # Sequence encoding (if peptide sequence known)
    sequence_base_coords: Optional[np.ndarray]  # Base coordinates
    sequence_sentropy_coords: Optional[np.ndarray]  # S-Entropy weighted

    # Metrics
    s_value: float
    processing_time: float
    metadata: Dict


class SEntropyProteomicsEngine:
    """
    S-Entropy engine implementing complete molecular encoding framework
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.results = {
            'spectra': [],
            'features': [],
            'clustering': {},
            'benchmarks': {},
            'validation': {}
        }

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'sentropy': {
                'window_size': 5,  # Context window for weighting functions
                'use_adaptive_windows': True,
                'bayesian_weight': 0.968,
                'senn_weight': 0.031,
                'coord_weight': 0.001
            },
            'encoding': {
                'normalize_mz': True,
                'normalize_intensity': True,
                'use_retention_time': True,
                'use_sequence_context': True
            },
            'clustering': {
                'methods': ['hierarchical', 'kmeans', 'dbscan'],
                'n_clusters_range': [3, 5, 8, 10, 15, 20],
                'metrics': ['silhouette', 'davies_bouldin', 'calinski_harabasz']
            },
            'output': {
                'save_json': True,
                'save_csv': True,
                'save_figures': True,
                'figure_format': 'png',
                'figure_dpi': 300
            }
        }

    # ========================================================================
    # PEPTIDE SEQUENCE ENCODING (Following the paper exactly)
    # ========================================================================

    def encode_peptide_sequence_base(self, peptide_sequence: str) -> np.ndarray:
        """
        Step 1: Encode peptide to BASE coordinates ξ(a) = (h, p, s)

        Args:
            peptide_sequence: Amino acid sequence (e.g., "PEPTIDE")

        Returns:
            Base coordinate array of shape (len(sequence), 3)
            where columns are [hydrophobicity, polarity, size]
        """
        if not peptide_sequence:
            return np.array([])

        peptide_sequence = peptide_sequence.upper()

        base_coords = []
        for amino_acid in peptide_sequence:
            if amino_acid in AMINO_ACID_PROPERTIES:
                base_coords.append(AMINO_ACID_PROPERTIES[amino_acid])
            else:
                # Unknown amino acid - use neutral values
                base_coords.append((0.0, 0.0, 0.1))

        return np.array(base_coords)

    def encode_peptide_sequence_sentropy(
            self,
            peptide_sequence: str,
            base_coords: np.ndarray
    ) -> np.ndarray:
        """
        Step 2: Apply S-Entropy weighting to get Ξ(a,i,W_i)

        For amino acid at position i with context window W_i:
        Ξ(a,i,W_i) = (w_k * h, w_t * p, w_e * s)

        Args:
            peptide_sequence: Amino acid sequence
            base_coords: Base coordinates from step 1

        Returns:
            S-Entropy weighted coordinates of shape (len(sequence), 3)
        """
        n = len(peptide_sequence)
        window_size = self.config['sentropy']['window_size']

        sentropy_coords = np.zeros_like(base_coords)

        for i in range(n):
            # Define context window W_i
            window_start = max(0, i - window_size // 2)
            window_end = min(n, i + window_size // 2 + 1)
            window_indices = range(window_start, window_end)

            # Extract window context
            window_sequence = peptide_sequence[window_start:window_end]
            window_coords = base_coords[window_start:window_end]

            # Compute weighting functions
            w_k = self._compute_knowledge_weight_protein(window_sequence, i - window_start)
            w_t = self._compute_time_weight_protein(peptide_sequence, i)
            w_e = self._compute_entropy_weight_protein(window_coords)

            # Apply weights to base coordinates
            h, p, s = base_coords[i]
            sentropy_coords[i] = np.array([
                w_k * h,  # S_knowledge dimension
                w_t * p,  # S_time dimension
                w_e * s  # S_entropy dimension
            ])

        return sentropy_coords

    def _compute_knowledge_weight_protein(
            self,
            window_sequence: str,
            position_in_window: int
    ) -> float:
        """
        Knowledge weighting function: w_k(a,i,W_i) = -Σ p_j log₂(p_j)

        This is Shannon entropy of amino acid distribution in window
        """
        if len(window_sequence) == 0:
            return 1.0

        # Count amino acid frequencies in window
        aa_counts = {}
        for aa in window_sequence:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1

        # Compute probabilities
        total = len(window_sequence)
        probabilities = [count / total for count in aa_counts.values()]

        # Shannon entropy
        if len(probabilities) == 0:
            return 1.0

        shannon_entropy = entropy(probabilities, base=2)

        # Normalize to reasonable range [0.1, 2.0]
        # Max entropy for 20 amino acids is log₂(20) ≈ 4.32
        normalized = 0.1 + (shannon_entropy / 4.32) * 1.9

        return normalized

    def _compute_time_weight_protein(
            self,
            full_sequence: str,
            position: int
    ) -> float:
        """
        Time weighting function: w_t(a,i,W_i) = Σ ω_j · 1[a_j = a]

        With exponential decay: ω_j = exp(-(i-j)/τ)
        """
        if position == 0:
            return 0.1

        tau = 5.0  # Characteristic decay length
        current_aa = full_sequence[position]

        weight_sum = 0.0
        for j in range(position):
            if full_sequence[j] == current_aa:
                decay_weight = np.exp(-(position - j) / tau)
                weight_sum += decay_weight

        # Normalize to [0.1, 2.0]
        normalized = 0.1 + min(weight_sum / 3.0, 1.9)

        return normalized

    def _compute_entropy_weight_protein(self, window_coords: np.ndarray) -> float:
        """
        Entropy weighting function: w_e(a,i,W_i) = sqrt(variance of coords in window)

        This measures local disorder/organization
        """
        if len(window_coords) < 2:
            return 1.0

        # Compute mean coordinate in window
        mean_coord = np.mean(window_coords, axis=0)

        # Compute variance (sum of squared deviations)
        variance = np.mean(np.sum((window_coords - mean_coord) ** 2, axis=1))

        # Take square root and normalize
        std_dev = np.sqrt(variance)

        # Normalize to reasonable range [0.1, 2.0]
        normalized = 0.1 + min(std_dev / 1.0, 1.9)

        return normalized

    # ========================================================================
    # MS/MS FRAGMENT ENCODING
    # ========================================================================

    def encode_fragments_base(
            self,
            mz_array: np.ndarray,
            intensity_array: np.ndarray,
            precursor_mz: float
    ) -> np.ndarray:
        """
        Step 1: Encode MS/MS fragments to BASE coordinates

        Base coordinates for fragments:
        - Dimension 1: Normalized m/z (mass information)
        - Dimension 2: Normalized intensity (abundance information)
        - Dimension 3: m/z ratio to precursor (fragmentation pattern)

        Returns:
            Base coordinate array of shape (n_fragments, 3)
        """
        n_fragments = len(mz_array)

        # Normalize m/z
        if self.config['encoding']['normalize_mz']:
            mz_normalized = mz_array / precursor_mz
        else:
            mz_normalized = mz_array / np.max(mz_array)

        # Normalize intensity
        if self.config['encoding']['normalize_intensity']:
            intensity_normalized = intensity_array / np.sum(intensity_array)
        else:
            intensity_normalized = intensity_array / np.max(intensity_array)

        # m/z ratio (fragmentation pattern indicator)
        mz_ratio = mz_array / precursor_mz

        # Stack into base coordinates
        base_coords = np.column_stack([
            mz_normalized,
            intensity_normalized,
            mz_ratio
        ])

        return base_coords

    def encode_fragments_sentropy(
            self,
            base_coords: np.ndarray,
            intensity_array: np.ndarray,
            retention_time: float = 0.0
    ) -> np.ndarray:
        """
        Step 2: Apply S-Entropy weighting to fragment base coordinates

        For fragment at position i:
        Φ(f,i,W_i) = (w_k * coord_1, w_t * coord_2, w_e * coord_3)

        Returns:
            S-Entropy weighted coordinates of shape (n_fragments, 3)
        """
        n_fragments = len(base_coords)
        window_size = self.config['sentropy']['window_size']

        sentropy_coords = np.zeros_like(base_coords)

        for i in range(n_fragments):
            # Define context window
            window_start = max(0, i - window_size // 2)
            window_end = min(n_fragments, i + window_size // 2 + 1)

            window_coords = base_coords[window_start:window_end]
            window_intensities = intensity_array[window_start:window_end]

            # Compute weighting functions
            w_k = self._compute_knowledge_weight_fragments(window_intensities)
            w_t = self._compute_time_weight_fragments(i, n_fragments, retention_time)
            w_e = self._compute_entropy_weight_fragments(window_coords)

            # Apply weights
            sentropy_coords[i] = base_coords[i] * np.array([w_k, w_t, w_e])

        return sentropy_coords

    def _compute_knowledge_weight_fragments(self, window_intensities: np.ndarray) -> float:
        """
        Knowledge weight for fragments: Shannon entropy of intensity distribution
        """
        if len(window_intensities) == 0:
            return 1.0

        # Normalize to probabilities
        intensities_norm = window_intensities / np.sum(window_intensities)

        # Shannon entropy
        shannon_entropy = -np.sum(intensities_norm * np.log2(intensities_norm + 1e-10))

        # Normalize
        max_entropy = np.log2(len(window_intensities))
        if max_entropy > 0:
            normalized = 0.1 + (shannon_entropy / max_entropy) * 1.9
        else:
            normalized = 1.0

        return normalized

    def _compute_time_weight_fragments(
            self,
            position: int,
            total_fragments: int,
            retention_time: float
    ) -> float:
        """
        Time weight for fragments: position in spectrum + retention time
        """
        # Position-based component
        position_weight = position / total_fragments

        # Retention time component (if available)
        if self.config['encoding']['use_retention_time'] and retention_time > 0:
            rt_weight = retention_time / 100.0  # Normalize RT
            combined = 0.7 * position_weight + 0.3 * rt_weight
        else:
            combined = position_weight

        # Normalize to [0.1, 2.0]
        normalized = 0.1 + combined * 1.9

        return normalized

    def _compute_entropy_weight_fragments(self, window_coords: np.ndarray) -> float:
        """
        Entropy weight for fragments: local variance in coordinate space
        """
        if len(window_coords) < 2:
            return 1.0

        # Compute variance
        mean_coord = np.mean(window_coords, axis=0)
        variance = np.mean(np.sum((window_coords - mean_coord) ** 2, axis=1))
        std_dev = np.sqrt(variance)

        # Normalize
        normalized = 0.1 + min(std_dev / 0.5, 1.9)

        return normalized

    # ========================================================================
    # 14D FEATURE EXTRACTION (For clustering)
    # ========================================================================

    def extract_sentropy_features_14d(
            self,
            sentropy_coords: np.ndarray
    ) -> np.ndarray:
        """
        Extract 14D feature vector from S-Entropy coordinates

        Features:
        1-3: Mean of (S_knowledge, S_time, S_entropy)
        4-6: Std of (S_knowledge, S_time, S_entropy)
        7-9: Min of (S_knowledge, S_time, S_entropy)
        10-12: Max of (S_knowledge, S_time, S_entropy)
        13: S-value (total entropy)
        14: Coordinate density
        """
        features = np.zeros(14)

        if len(sentropy_coords) == 0:
            return features

        # Mean (1-3)
        features[0:3] = np.mean(sentropy_coords, axis=0)

        # Std (4-6)
        features[3:6] = np.std(sentropy_coords, axis=0)

        # Min (7-9)
        features[6:9] = np.min(sentropy_coords, axis=0)

        # Max (10-12)
        features[9:12] = np.max(sentropy_coords, axis=0)

        # S-value (13): sum of S_entropy dimension
        features[12] = np.sum(sentropy_coords[:, 2])

        # Coordinate density (14)
        features[13] = len(sentropy_coords) / (np.max(sentropy_coords[:, 0]) + 1e-10)

        return features

    # ========================================================================
    # MAIN PROCESSING FUNCTION
    # ========================================================================

    def process_spectrum(
            self,
            mz_array: np.ndarray,
            intensity_array: np.ndarray,
            precursor_mz: float,
            precursor_charge: int,
            spectrum_id: str,
            peptide_sequence: Optional[str] = None,
            retention_time: float = 0.0
    ) -> SEntropySpectrum:
        """
        Process single MS/MS spectrum with complete S-Entropy encoding

        Following the two-step process from the paper:
        1. Raw data → Base coordinates
        2. Base coordinates → S-Entropy weighted coordinates
        """
        start_time = time.time()

        # ===== PEPTIDE SEQUENCE ENCODING =====
        sequence_base_coords = None
        sequence_sentropy_coords = None

        if peptide_sequence and self.config['encoding']['use_sequence_context']:
            # Step 1: Base coordinates
            sequence_base_coords = self.encode_peptide_sequence_base(peptide_sequence)

            # Step 2: S-Entropy weighting
            sequence_sentropy_coords = self.encode_peptide_sequence_sentropy(
                peptide_sequence, sequence_base_coords
            )

        # ===== FRAGMENT ENCODING =====
        # Step 1: Base coordinates
        fragments_base_coords = self.encode_fragments_base(
            mz_array, intensity_array, precursor_mz
        )

        # Step 2: S-Entropy weighting
        fragments_sentropy_coords = self.encode_fragments_sentropy(
            fragments_base_coords, intensity_array, retention_time
        )

        # ===== FEATURE EXTRACTION =====
        sentropy_features_14d = self.extract_sentropy_features_14d(
            fragments_sentropy_coords
        )

        # ===== METRICS =====
        s_value = np.sum(fragments_sentropy_coords[:, 2])
        processing_time = time.time() - start_time

        # ===== METADATA =====
        metadata = {
            'n_fragments': len(mz_array),
            'total_intensity': np.sum(intensity_array),
            'base_peak_mz': mz_array[np.argmax(intensity_array)],
            'base_peak_intensity': np.max(intensity_array),
            'retention_time': retention_time,
            'has_sequence': peptide_sequence is not None
        }

        return SEntropySpectrum(
            spectrum_id=spectrum_id,
            precursor_mz=precursor_mz,
            precursor_charge=precursor_charge,
            peptide_sequence=peptide_sequence,
            fragments_mz=mz_array,
            fragments_intensity=intensity_array,
            retention_time=retention_time,
            sentropy_coords_3d=fragments_sentropy_coords,
            sentropy_features_14d=sentropy_features_14d,
            sequence_base_coords=sequence_base_coords,
            sequence_sentropy_coords=sequence_sentropy_coords,
            s_value=s_value,
            processing_time=processing_time,
            metadata=metadata
        )

    # ========================================================================
    # INTEGRATION WITH YOUR PIPELINE
    # ========================================================================

    def process_mzml_file(self, mzml_path: str) -> List[SEntropySpectrum]:
        """
        Process entire mzML file
        Will integrate with your MSnConvert.py and MSnProcessing.py
        """
        print(f"Processing {mzml_path}...")

        # TODO: Integrate with your existing code
        spectra = []

        return spectra

    def save_results(self, output_dir: str):
        """Save all results"""
        import os
        import json

        os.makedirs(output_dir, exist_ok=True)

        if self.config['output']['save_json']:
            self._save_json(output_dir)

        if self.config['output']['save_csv']:
            self._save_csv(output_dir)

    def _save_json(self, output_dir: str):
        """Save JSON results"""
        import json
        import os

        output_file = os.path.join(output_dir, 'sentropy_proteomics_results.json')

        results_serializable = {
            'n_spectra': len(self.results['spectra']),
            'total_processing_time': sum(s.processing_time for s in self.results['spectra']),
            'mean_s_value': float(np.mean([s.s_value for s in self.results['spectra']])) if self.results[
                'spectra'] else 0.0,
            'config': self.config
        }

        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"Saved: {output_file}")

    def _save_csv(self, output_dir: str):
        """Save CSV results"""
        import os

        if not self.results['spectra']:
            return

        feature_matrix = np.vstack([s.sentropy_features_14d for s in self.results['spectra']])
        spectrum_ids = [s.spectrum_id for s in self.results['spectra']]

        feature_names = [
            'mean_s_knowledge', 'mean_s_time', 'mean_s_entropy',
            'std_s_knowledge', 'std_s_time', 'std_s_entropy',
            'min_s_knowledge', 'min_s_time', 'min_s_entropy',
            'max_s_knowledge', 'max_s_time', 'max_s_entropy',
            's_value', 'coord_density'
        ]

        df = pd.DataFrame(feature_matrix, columns=feature_names, index=spectrum_ids)

        output_file = os.path.join(output_dir, 'sentropy_features_14d.csv')
        df.to_csv(output_file)

        print(f"Saved: {output_file}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def encode_peptide(peptide_sequence: str) -> Tuple[np.ndarray, np.ndarray]:
    """Quick peptide encoding - returns (base_coords, sentropy_coords)"""
    engine = SEntropyProteomicsEngine()
    base_coords = engine.encode_peptide_sequence_base(peptide_sequence)
    sentropy_coords = engine.encode_peptide_sequence_sentropy(peptide_sequence, base_coords)
    return base_coords, sentropy_coords


def process_single_spectrum(
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        precursor_mz: float,
        precursor_charge: int = 2,
        peptide_sequence: Optional[str] = None
) -> SEntropySpectrum:
    """Quick single spectrum processing"""
    engine = SEntropyProteomicsEngine()
    return engine.process_spectrum(
        mz_array, intensity_array, precursor_mz,
        precursor_charge, "spectrum_001", peptide_sequence
    )

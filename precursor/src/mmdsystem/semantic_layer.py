#!/usr/bin/env python3
"""
Semantic Navigation Layer
=========================

From st-stellas-spectrometry.tex Section 3.2:
"Non-sequential pattern recognition and dynamic synthesis
of molecular interpretations for cross-domain knowledge transfer."

This layer handles semantic understanding and knowledge synthesis.

Author: Kundai Sachikonye
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from core.EntropyTransformation import SEntropyCoordinates


class SemanticNavigator:
    """
    Semantic navigation through S-Entropy space.

    Provides:
    - Pattern recognition across sequences
    - Semantic similarity computation
    - Knowledge transfer from known to novel molecules
    """

    def __init__(self):
        """Initialize semantic navigator."""
        self.known_patterns: Dict[str, List[SEntropyCoordinates]] = {}

    def register_pattern(
        self,
        pattern_name: str,
        coords_path: List[SEntropyCoordinates]
    ):
        """
        Register a known S-Entropy pattern.

        Args:
            pattern_name: Pattern identifier (e.g., 'common_motif_1')
            coords_path: S-Entropy coordinate path
        """
        self.known_patterns[pattern_name] = coords_path

    def find_similar_patterns(
        self,
        query_path: List[SEntropyCoordinates],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find known patterns similar to query.

        Args:
            query_path: Query S-Entropy path
            top_k: Number of top matches to return

        Returns:
            List of (pattern_name, similarity) tuples
        """
        if len(self.known_patterns) == 0:
            return []

        similarities = []

        for pattern_name, pattern_path in self.known_patterns.items():
            sim = self._path_similarity(query_path, pattern_path)
            similarities.append((pattern_name, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def _path_similarity(
        self,
        path1: List[SEntropyCoordinates],
        path2: List[SEntropyCoordinates]
    ) -> float:
        """
        Calculate similarity between two S-Entropy paths.

        Uses dynamic time warping-inspired approach.

        Args:
            path1: First path
            path2: Second path

        Returns:
            Similarity score [0, 1]
        """
        if len(path1) == 0 or len(path2) == 0:
            return 0.0

        # Convert to arrays
        arr1 = np.array([c.to_array() for c in path1])
        arr2 = np.array([c.to_array() for c in path2])

        # Simple point-wise distance (placeholder for DTW)
        if len(arr1) == len(arr2):
            distances = np.linalg.norm(arr1 - arr2, axis=1)
            avg_distance = np.mean(distances)
        else:
            # Different lengths - use average distance to nearest neighbor
            avg_distance = 0.5

        # Convert distance to similarity
        similarity = np.exp(-avg_distance / 0.5)

        return float(similarity)


class CrossModalValidator:
    """
    Cross-modal validation across different measurement modalities.

    From st-stellas-spectrometry.tex Section 4:
    "Cross-modal pathway analysis validates molecular interpretations
    across MS, NMR, IR, and other modalities."

    For proteomics: validates across fragmentation patterns, retention time,
    isotopic distribution, etc.
    """

    def __init__(self):
        """Initialize cross-modal validator."""
        self.validation_modules = {
            'fragmentation': self._validate_fragmentation,
            'retention_time': self._validate_retention_time,
            'isotopic': self._validate_isotopic
        }

    def validate(
        self,
        sequence: str,
        observed_data: Dict[str, any],
        modalities: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Perform cross-modal validation.

        Args:
            sequence: Proposed sequence
            observed_data: Observed data from different modalities
            modalities: Which modalities to validate (None = all available)

        Returns:
            Dict mapping modality to validation score [0, 1]
        """
        if modalities is None:
            modalities = list(self.validation_modules.keys())

        scores = {}

        for modality in modalities:
            if modality in self.validation_modules:
                validator = self.validation_modules[modality]
                score = validator(sequence, observed_data.get(modality, {}))
                scores[modality] = score

        return scores

    def _validate_fragmentation(
        self,
        sequence: str,
        observed_fragments: dict
    ) -> float:
        """
        Validate against observed fragmentation pattern.

        Args:
            sequence: Proposed sequence
            observed_fragments: Observed fragment data

        Returns:
            Validation score [0, 1]
        """
        # Check if theoretical fragments match observed
        # (Placeholder - full implementation in mmd_orchestrator)
        return 0.8  # Default score

    def _validate_retention_time(
        self,
        sequence: str,
        observed_rt: dict
    ) -> float:
        """
        Validate against retention time.

        Uses hydrophobicity correlation.
        """
        if 'rt' not in observed_rt:
            return 1.0  # No data = no penalty

        from ..molecular_language.amino_acid_alphabet import STANDARD_AMINO_ACIDS

        # Calculate theoretical hydrophobicity
        hydrophobicity = 0.0
        for aa in sequence:
            if aa in STANDARD_AMINO_ACIDS:
                hydrophobicity += STANDARD_AMINO_ACIDS[aa].hydrophobicity

        # Expect correlation with RT
        # (Simplified - real validation would use prediction model)

        return 0.7  # Placeholder

    def _validate_isotopic(
        self,
        sequence: str,
        observed_isotopic: dict
    ) -> float:
        """
        Validate against isotopic distribution.
        """
        # Check if theoretical isotopic envelope matches observed
        # (Placeholder)
        return 0.9  # Placeholder

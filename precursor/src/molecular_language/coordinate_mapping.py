#!/usr/bin/env python3
"""
S-Entropy Coordinate Mapping for Molecular Sequences
====================================================

From st-stellas-molecular-language.tex Section 3:
Transforms molecular sequences into S-Entropy coordinate paths.

Author: Kundai Sachikonye
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from core.EntropyTransformation import SEntropyCoordinates

from .amino_acid_alphabet import AminoAcidAlphabet, STANDARD_AMINO_ACIDS


def amino_acid_to_sentropy(amino_acid_symbol: str) -> Optional[SEntropyCoordinates]:
    """
    Map single amino acid to S-Entropy coordinates.

    From st-stellas-molecular-language.tex Equation 3.2:
    φ_AA: AA → (S_k, S_t, S_e)

    Args:
        amino_acid_symbol: Single letter amino acid code

    Returns:
        S-Entropy coordinates for the amino acid
    """
    aa = STANDARD_AMINO_ACIDS.get(amino_acid_symbol)
    if aa:
        return aa.s_entropy_coords
    return None


def ptm_to_sentropy(
    base_symbol: str,
    ptm_name: str,
    alphabet: AminoAcidAlphabet
) -> Optional[SEntropyCoordinates]:
    """
    Map modified amino acid to S-Entropy coordinates.

    PTM transformation: S'= S_base + ΔS_ptm

    Args:
        base_symbol: Base amino acid symbol
        ptm_name: PTM name (e.g., 'Phosphorylation')
        alphabet: Amino acid alphabet with PTM definitions

    Returns:
        S-Entropy coordinates for modified amino acid
    """
    base_coords = alphabet.get_sentropy_coords(base_symbol)
    if not base_coords:
        return None

    return alphabet.apply_ptm(base_coords, ptm_name)


def sequence_to_sentropy_path(
    sequence: str,
    modifications: Optional[dict] = None,
    alphabet: Optional[AminoAcidAlphabet] = None
) -> List[SEntropyCoordinates]:
    """
    Convert peptide sequence to S-Entropy coordinate path.

    From st-stellas-molecular-language.tex Section 3.3:
    Sliding window analysis over sequence generates coordinate trajectory.

    Args:
        sequence: Peptide sequence (single letter codes)
        modifications: Dict of {position: ptm_name} for modified residues
        alphabet: Amino acid alphabet (uses default if None)

    Returns:
        List of S-Entropy coordinates (one per residue)
    """
    if alphabet is None:
        alphabet = AminoAcidAlphabet()

    if modifications is None:
        modifications = {}

    coords_path = []

    for i, aa_symbol in enumerate(sequence):
        if i in modifications:
            # Modified amino acid
            coords = ptm_to_sentropy(aa_symbol, modifications[i], alphabet)
        else:
            # Unmodified amino acid
            coords = amino_acid_to_sentropy(aa_symbol)

        if coords:
            coords_path.append(coords)
        else:
            # Unknown amino acid - use default coordinates
            coords_path.append(SEntropyCoordinates(
                s_knowledge=0.5,
                s_time=0.5,
                s_entropy=0.5
            ))

    return coords_path


def calculate_sequence_entropy(coords_path: List[SEntropyCoordinates]) -> float:
    """
    Calculate sequence S-Entropy (Shannon entropy of coordinate distribution).

    From st-stellas-sequence.tex Section 3:
    S_seq = -Σ p(s_i) log p(s_i)

    Args:
        coords_path: S-Entropy coordinate path for sequence

    Returns:
        Sequence entropy value
    """
    if len(coords_path) == 0:
        return 0.0

    # Convert to array
    coords_array = np.array([[c.s_knowledge, c.s_time, c.s_entropy] for c in coords_path])

    # Bin coordinates into discrete states (for Shannon entropy)
    n_bins = 10
    binned_coords = np.floor(coords_array * n_bins).astype(int)

    # Concatenate dimensions to get unique states
    states = [tuple(row) for row in binned_coords]

    # Count state frequencies
    unique_states, counts = np.unique(states, axis=0, return_counts=True)
    probabilities = counts / len(states)

    # Shannon entropy
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

    return float(entropy)


def calculate_sequence_complexity(sequence: str) -> float:
    """
    Calculate sequence complexity (Kolmogorov-inspired).

    From st-stellas-sequence.tex:
    Complexity based on:
    1. Shannon entropy
    2. Longest repeating substring
    3. Compressibility

    Args:
        sequence: Peptide sequence

    Returns:
        Complexity score [0, 1]
    """
    if len(sequence) == 0:
        return 0.0

    # 1. Shannon entropy of amino acid distribution
    aa_counts = {}
    for aa in sequence:
        aa_counts[aa] = aa_counts.get(aa, 0) + 1

    probs = np.array(list(aa_counts.values())) / len(sequence)
    shannon_entropy = -np.sum(probs * np.log2(probs + 1e-10))

    # 2. Repetition penalty
    # Find longest repeating substring
    max_repeat = 1
    for i in range(len(sequence)):
        for j in range(i + 1, len(sequence)):
            k = 0
            while (i + k < len(sequence) and j + k < len(sequence) and
                   sequence[i + k] == sequence[j + k]):
                k += 1
            max_repeat = max(max_repeat, k)

    repetition_penalty = 1.0 - (max_repeat / len(sequence))

    # Combined complexity
    complexity = (shannon_entropy / 4.32) * repetition_penalty  # Max Shannon ~4.32 for 20 AAs

    return float(np.clip(complexity, 0.0, 1.0))

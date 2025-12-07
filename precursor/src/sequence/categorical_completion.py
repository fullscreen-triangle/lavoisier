#!/usr/bin/env python3
"""
Categorical Completion for Gap Filling
======================================

From st-stellas-sequence.tex Algorithm 2 Step 4:
"Fill gaps via categorical completion (the 'miracle')"

When fragments don't cover the entire sequence, we infer missing
amino acids by minimizing total S-Entropy.

Author: Kundai Sachikonye
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from core.EntropyTransformation import SEntropyCoordinates
# Note: CategoricalState import removed - not needed for gap filling

from dictionary.sentropy_dictionary import SEntropyDictionary
from molecular_language.amino_acid_alphabet import AminoAcidAlphabet, STANDARD_AMINO_ACIDS
from molecular_language.coordinate_mapping import sequence_to_sentropy_path, calculate_sequence_entropy


@dataclass
class GapRegion:
    """
    Gap in fragment coverage.

    Attributes:
        start_fragment: Fragment before gap
        end_fragment: Fragment after gap
        mass_gap: Mass difference to fill
        sentropy_start: S-Entropy coords at gap start
        sentropy_end: S-Entropy coords at gap end
    """
    start_fragment: Optional[str]
    end_fragment: Optional[str]
    mass_gap: float
    sentropy_start: Optional[SEntropyCoordinates]
    sentropy_end: Optional[SEntropyCoordinates]


class CategoricalCompleter:
    """
    Categorical completion engine for gap filling.

    From st-stellas-sequence.tex Section 3.3:
    "Categorical completion fills gaps by finding amino acid sequences
    that minimize total S-Entropy while respecting mass constraints."

    This is the CORE innovation that enables database-free sequencing!
    """

    def __init__(
        self,
        dictionary: SEntropyDictionary,
        alphabet: Optional[AminoAcidAlphabet] = None,
        mass_tolerance: float = 0.5
    ):
        """
        Initialize categorical completer.

        Args:
            dictionary: S-Entropy dictionary for AA identification
            alphabet: Amino acid alphabet
            mass_tolerance: Mass tolerance for gap filling (Da)
        """
        self.dictionary = dictionary
        self.alphabet = alphabet if alphabet else AminoAcidAlphabet()
        self.mass_tolerance = mass_tolerance

    def fill_gap(
        self,
        gap: GapRegion,
        max_gap_size: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Fill a gap using categorical completion.

        From st-stellas-sequence.tex Algorithm 2 Step 4:

        1. Enumerate possible AA combinations that match mass
        2. Calculate S-Entropy for each combination
        3. Select combination with minimum total S-Entropy
        4. Validate using categorical state constraints

        Args:
            gap: Gap region to fill
            max_gap_size: Maximum number of AAs in gap

        Returns:
            List of (amino_acid, confidence) tuples
        """
        if gap.mass_gap < 50:  # Too small for even one AA
            return []

        # Estimate number of amino acids in gap
        avg_aa_mass = 110.0  # Average amino acid mass
        estimated_aa_count = int(round(gap.mass_gap / avg_aa_mass))
        estimated_aa_count = max(1, min(estimated_aa_count, max_gap_size))

        print(f"[Categorical] Filling gap of {gap.mass_gap:.1f} Da "
              f"(~{estimated_aa_count} residues)")

        # Try different gap sizes around estimate
        best_sequence = None
        best_entropy = float('inf')
        best_confidence = 0.0

        for n_aa in range(max(1, estimated_aa_count - 1),
                         min(max_gap_size + 1, estimated_aa_count + 2)):

            # Find combinations that match mass
            sequences = self._enumerate_sequences_by_mass(
                target_mass=gap.mass_gap,
                n_residues=n_aa,
                tolerance=self.mass_tolerance
            )

            for seq in sequences[:100]:  # Limit search
                # Calculate S-Entropy for this sequence
                coords_path = sequence_to_sentropy_path(seq, alphabet=self.alphabet)

                # Total entropy
                entropy = calculate_sequence_entropy(coords_path)

                # Validate with categorical state
                confidence = self._validate_with_categorical_state(
                    seq,
                    coords_path,
                    gap
                )

                # Penalize by entropy, reward by confidence
                score = entropy / (confidence + 0.1)

                if score < best_entropy:
                    best_entropy = score
                    best_sequence = seq
                    best_confidence = confidence

        if best_sequence:
            # Convert to (AA, confidence) list
            per_residue_conf = best_confidence / len(best_sequence)
            result = [(aa, per_residue_conf) for aa in best_sequence]

            print(f"[Categorical] Gap filled: {best_sequence} "
                  f"(entropy {best_entropy:.3f}, conf {best_confidence:.3f})")

            return result

        return []

    def _enumerate_sequences_by_mass(
        self,
        target_mass: float,
        n_residues: int,
        tolerance: float = 0.5,
        max_combinations: int = 1000
    ) -> List[str]:
        """
        Enumerate amino acid sequences that match target mass.

        Args:
            target_mass: Target mass to match
            n_residues: Number of residues in sequence
            tolerance: Mass tolerance (Da)
            max_combinations: Maximum combinations to return

        Returns:
            List of sequence strings
        """
        # Get all amino acid symbols and masses
        aa_symbols = list(STANDARD_AMINO_ACIDS.keys())
        aa_masses = [STANDARD_AMINO_ACIDS[aa].mass for aa in aa_symbols]

        # Recursive enumeration
        sequences = []

        def recurse(current_seq: str, current_mass: float, remaining: int):
            if len(sequences) >= max_combinations:
                return

            if remaining == 0:
                # Check if mass matches
                if abs(current_mass - target_mass) <= tolerance:
                    sequences.append(current_seq)
                return

            # Try each amino acid
            for aa, mass in zip(aa_symbols, aa_masses):
                new_mass = current_mass + mass

                # Pruning: check if we can still reach target
                min_possible = new_mass + (remaining - 1) * min(aa_masses)
                max_possible = new_mass + (remaining - 1) * max(aa_masses)

                if min_possible <= target_mass + tolerance and \
                   max_possible >= target_mass - tolerance:
                    recurse(current_seq + aa, new_mass, remaining - 1)

        recurse('', 0.0, n_residues)

        return sequences

    def _validate_with_categorical_state(
        self,
        sequence: str,
        coords_path: List[SEntropyCoordinates],
        gap: GapRegion
    ) -> float:
        """
        Validate sequence using categorical state constraints.

        From core.CategoricalState:
        Check if proposed sequence transitions smoothly between
        gap boundaries in categorical space.

        Args:
            sequence: Proposed sequence
            coords_path: S-Entropy coordinate path
            gap: Gap region

        Returns:
            Validation confidence [0, 1]
        """
        if len(coords_path) == 0:
            return 0.0

        # Check smoothness of S-Entropy trajectory
        # Start and end should align with gap boundaries
        confidence = 1.0

        # Check start alignment
        if gap.sentropy_start:
            start_coords = coords_path[0]
            start_dist = np.linalg.norm(
                start_coords.to_array() - gap.sentropy_start.to_array()
            )
            start_score = np.exp(-start_dist / 0.3)
            confidence *= start_score

        # Check end alignment
        if gap.sentropy_end:
            end_coords = coords_path[-1]
            end_dist = np.linalg.norm(
                end_coords.to_array() - gap.sentropy_end.to_array()
            )
            end_score = np.exp(-end_dist / 0.3)
            confidence *= end_score

        # Check smoothness of internal transitions
        for i in range(len(coords_path) - 1):
            c1 = coords_path[i].to_array()
            c2 = coords_path[i + 1].to_array()
            transition_dist = np.linalg.norm(c2 - c1)

            # Penalize large jumps (not smooth)
            if transition_dist > 0.5:
                confidence *= 0.8

        return float(confidence)


class GapFiller:
    """
    High-level gap filling orchestration.

    Combines categorical completion with validation.
    """

    def __init__(
        self,
        completer: CategoricalCompleter
    ):
        """
        Initialize gap filler.

        Args:
            completer: Categorical completer instance
        """
        self.completer = completer

    def identify_gaps(
        self,
        fragment_masses: List[float],
        precursor_mass: float,
        mass_tolerance: float = 0.5
    ) -> List[GapRegion]:
        """
        Identify gaps in fragment coverage.

        Args:
            fragment_masses: List of observed fragment masses
            precursor_mass: Precursor mass
            mass_tolerance: Tolerance for coverage check

        Returns:
            List of gap regions
        """
        if len(fragment_masses) == 0:
            # Entire sequence is a gap
            return [GapRegion(
                start_fragment=None,
                end_fragment=None,
                mass_gap=precursor_mass,
                sentropy_start=None,
                sentropy_end=None
            )]

        # Sort fragments by mass
        sorted_masses = sorted(fragment_masses)

        gaps = []

        # Check gaps between consecutive fragments
        for i in range(len(sorted_masses) - 1):
            mass_diff = sorted_masses[i + 1] - sorted_masses[i]

            # Gap if difference > single amino acid mass
            if mass_diff > (STANDARD_AMINO_ACIDS['G'].mass + mass_tolerance):
                gaps.append(GapRegion(
                    start_fragment=f"frag_{i}",
                    end_fragment=f"frag_{i+1}",
                    mass_gap=mass_diff,
                    sentropy_start=None,  # Will be filled later
                    sentropy_end=None
                ))

        return gaps

    def fill_all_gaps(
        self,
        gaps: List[GapRegion]
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Fill all identified gaps.

        Args:
            gaps: List of gap regions

        Returns:
            Dict mapping gap index to filled sequence
        """
        filled_gaps = {}

        for i, gap in enumerate(gaps):
            filled_seq = self.completer.fill_gap(gap)
            if len(filled_seq) > 0:
                filled_gaps[i] = filled_seq
                print(f"[GapFiller] Gap {i}: {''.join(aa for aa, _ in filled_seq)}")

        return filled_gaps

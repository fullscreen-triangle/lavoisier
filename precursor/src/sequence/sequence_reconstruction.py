#!/usr/bin/env python3
"""
Database-Free Peptide Sequence Reconstruction
=============================================

From st-stellas-sequence.tex Algorithm 2:
Complete peptide sequence reconstruction from MS/MS fragments
WITHOUT requiring a sequence database.

Author: Kundai Sachikonye
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from core.EntropyTransformation import SEntropyCoordinates

from sequence.fragment_graph import FragmentNode, FragmentGraph, build_fragment_graph_from_spectra
from sequence.categorical_completion import CategoricalCompleter, GapFiller, GapRegion
from dictionary.zero_shot_identification import ZeroShotIdentifier
from dictionary.sentropy_dictionary import SEntropyDictionary


@dataclass
class ReconstructionResult:
    """
    Result of sequence reconstruction.

    Attributes:
        sequence: Reconstructed peptide sequence
        confidence: Overall reconstruction confidence [0, 1]
        fragment_coverage: Fraction of sequence covered by fragments
        gap_filled_regions: Regions filled by categorical completion
        total_entropy: Total S-Entropy of reconstruction
        validation_scores: Dict of validation metrics
    """
    sequence: str
    confidence: float
    fragment_coverage: float
    gap_filled_regions: List[Tuple[int, int, str]]  # (start, end, sequence)
    total_entropy: float
    validation_scores: Dict[str, float]

    def __str__(self) -> str:
        return (f"Sequence: {self.sequence}\n"
                f"Confidence: {self.confidence:.3f}\n"
                f"Coverage: {self.fragment_coverage:.1%}\n"
                f"Gaps filled: {len(self.gap_filled_regions)}")


class SequenceReconstructor:
    """
    Complete database-free sequence reconstruction system.

    From st-stellas-sequence.tex Algorithm 2:
    Categorical Sequence Reconstruction

    Steps:
    1. Build fragment graph
    2. Extract S-Entropy manifold
    3. Find Hamiltonian path
    4. Fill gaps via categorical completion
    5. Identify fragments via zero-shot lookup
    6. Concatenate fragments
    7. Validate reconstruction
    """

    def __init__(
        self,
        dictionary: SEntropyDictionary,
        zero_shot_identifier: ZeroShotIdentifier,
        categorical_completer: CategoricalCompleter
    ):
        """
        Initialize sequence reconstructor.

        Args:
            dictionary: S-Entropy dictionary
            zero_shot_identifier: Zero-shot identification engine
            categorical_completer: Categorical completion engine
        """
        self.dictionary = dictionary
        self.zero_shot_identifier = zero_shot_identifier
        self.categorical_completer = categorical_completer
        self.gap_filler = GapFiller(categorical_completer)

    def reconstruct(
        self,
        fragments: List[FragmentNode],
        precursor_mass: Optional[float] = None,
        precursor_charge: int = 2
    ) -> ReconstructionResult:
        """
        Reconstruct peptide sequence from fragments.

        From st-stellas-sequence.tex Algorithm 2 (adapted for proteomics):

        Args:
            fragments: List of fragment nodes with S-Entropy coordinates
            precursor_mass: Precursor mass (optional, helps validation)
            precursor_charge: Precursor charge state

        Returns:
            ReconstructionResult with reconstructed sequence
        """
        print(f"\n[Reconstruction] Starting with {len(fragments)} fragments")

        # STEP 1: Build fragment graph
        print("[Reconstruction] Step 1: Building fragment graph...")
        fragment_graph = build_fragment_graph_from_spectra(fragments, precursor_mass)

        # STEP 2: Extract S-Entropy manifold (already in fragments)
        print("[Reconstruction] Step 2: S-Entropy manifold extracted")

        # STEP 3: Find Hamiltonian path through fragments
        print("[Reconstruction] Step 3: Finding Hamiltonian path...")
        path = fragment_graph.find_hamiltonian_path()

        if path is None or len(path) == 0:
            print("[Reconstruction] WARNING: No path found through fragments!")
            return ReconstructionResult(
                sequence="",
                confidence=0.0,
                fragment_coverage=0.0,
                gap_filled_regions=[],
                total_entropy=0.0,
                validation_scores={}
            )

        print(f"[Reconstruction] Found path through {len(path)} fragments")

        # STEP 4 & 5: Identify fragments and detect gaps
        print("[Reconstruction] Step 4-5: Identifying fragments and detecting gaps...")
        identified_fragments, gaps = self._identify_fragments_and_gaps(
            path,
            fragment_graph
        )

        # STEP 6: Fill gaps via categorical completion
        print(f"[Reconstruction] Step 6: Filling {len(gaps)} gaps...")
        filled_gaps = self.gap_filler.fill_all_gaps(gaps)

        # STEP 7: Concatenate fragments and gaps
        print("[Reconstruction] Step 7: Concatenating sequence...")
        final_sequence, gap_regions = self._concatenate_sequence(
            path,
            identified_fragments,
            filled_gaps,
            gaps
        )

        # STEP 8: Calculate metrics
        total_entropy = fragment_graph.calculate_path_entropy(path)

        # Fragment coverage
        fragment_aa_count = sum(len(seq) for seq, _ in identified_fragments.values())
        total_aa_count = len(final_sequence) if final_sequence else 1
        coverage = fragment_aa_count / total_aa_count

        # Overall confidence
        frag_confidences = [conf for _, conf in identified_fragments.values()]
        gap_confidences = [min(conf for _, conf in gap_seq) for gap_seq in filled_gaps.values() if gap_seq]

        all_confidences = frag_confidences + gap_confidences
        overall_confidence = np.mean(all_confidences) if all_confidences else 0.0

        # Validation scores
        validation_scores = {
            'path_entropy': total_entropy,
            'mean_fragment_conf': np.mean(frag_confidences) if frag_confidences else 0.0,
            'mean_gap_conf': np.mean(gap_confidences) if gap_confidences else 1.0,
            'n_fragments': len(identified_fragments),
            'n_gaps': len(gaps)
        }

        result = ReconstructionResult(
            sequence=final_sequence,
            confidence=float(overall_confidence),
            fragment_coverage=float(coverage),
            gap_filled_regions=gap_regions,
            total_entropy=total_entropy,
            validation_scores=validation_scores
        )

        print(f"\n[Reconstruction] COMPLETE!")
        print(result)

        return result

    def _identify_fragments_and_gaps(
        self,
        path: List[str],
        graph: FragmentGraph
    ) -> Tuple[Dict[str, Tuple[str, float]], List[GapRegion]]:
        """
        Identify fragment sequences and detect gaps.

        Args:
            path: Ordered fragment IDs
            graph: Fragment graph

        Returns:
            (identified_fragments, gaps)
        """
        identified = {}
        gaps = []

        for i, frag_id in enumerate(path):
            frag = graph.nodes[frag_id]

            # If fragment sequence already known, use it
            if frag.sequence:
                identified[frag_id] = (frag.sequence, frag.confidence)
                continue

            # Otherwise, identify via zero-shot
            result = self.zero_shot_identifier.identify(
                frag.s_entropy_coords,
                frag.mass
            )

            if result.top_match and result.confidence > 0.3:
                # Single amino acid identification
                identified[frag_id] = (result.top_match.symbol, result.confidence)
            else:
                # Could not identify - treat as gap
                identified[frag_id] = ("X", 0.1)  # Unknown residue

        # Detect gaps between consecutive fragments
        for i in range(len(path) - 1):
            frag_i = graph.nodes[path[i]]
            frag_j = graph.nodes[path[i + 1]]

            mass_diff = frag_j.mass - frag_i.mass

            # Expected mass difference
            seq_i = identified.get(path[i], ("", 0.0))[0]
            seq_j = identified.get(path[i + 1], ("", 0.0))[0]

            from molecular_language.amino_acid_alphabet import STANDARD_AMINO_ACIDS

            # Rough estimate of mass that should be covered
            if len(seq_i) > 0 and len(seq_j) > 0:
                min_aa_mass = min(aa.mass for aa in STANDARD_AMINO_ACIDS.values())

                # If mass diff is too large, there's a gap
                if mass_diff > 2 * min_aa_mass:
                    gap = GapRegion(
                        start_fragment=path[i],
                        end_fragment=path[i + 1],
                        mass_gap=mass_diff - min_aa_mass,  # Subtract expected AA
                        sentropy_start=frag_i.s_entropy_coords,
                        sentropy_end=frag_j.s_entropy_coords
                    )
                    gaps.append(gap)

        return identified, gaps

    def _concatenate_sequence(
        self,
        path: List[str],
        identified_fragments: Dict[str, Tuple[str, float]],
        filled_gaps: Dict[int, List[Tuple[str, float]]],
        original_gaps: List[GapRegion]
    ) -> Tuple[str, List[Tuple[int, int, str]]]:
        """
        Concatenate fragments and filled gaps into final sequence.

        Args:
            path: Ordered fragment IDs
            identified_fragments: Identified fragment sequences
            filled_gaps: Filled gap sequences (indexed by gap number)
            original_gaps: Original gap list

        Returns:
            (final_sequence, gap_regions)
        """
        sequence_parts = []
        gap_regions = []
        current_position = 0
        gap_index = 0

        for i, frag_id in enumerate(path):
            # Add fragment sequence
            if frag_id in identified_fragments:
                frag_seq, _ = identified_fragments[frag_id]
                sequence_parts.append(frag_seq)
                current_position += len(frag_seq)

            # Check if there's a gap after this fragment
            if i < len(path) - 1:
                # Check if gap_index corresponds to a gap after this fragment
                if gap_index < len(original_gaps):
                    gap = original_gaps[gap_index]

                    if gap.start_fragment == frag_id or \
                       (i < len(path) - 1 and gap.end_fragment == path[i + 1]):

                        # Add filled gap
                        if gap_index in filled_gaps:
                            gap_seq = ''.join(aa for aa, _ in filled_gaps[gap_index])
                            sequence_parts.append(gap_seq)

                            gap_start = current_position
                            gap_end = current_position + len(gap_seq)
                            gap_regions.append((gap_start, gap_end, gap_seq))
                            current_position = gap_end

                        gap_index += 1

        final_sequence = ''.join(sequence_parts)

        return final_sequence, gap_regions

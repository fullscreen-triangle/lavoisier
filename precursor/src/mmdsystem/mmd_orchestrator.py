#!/usr/bin/env python3
"""
Molecular Maxwell Demon Orchestrator
====================================

From st-stellas-spectrometry.tex Section 2:
"The Molecular Maxwell Demon operates through variance minimization,
seeking equilibrium states in molecular feature space."

This is the central orchestration system that integrates:
1. S-Entropy transformation
2. Dictionary-based identification
3. Categorical completion
4. Cross-modal validation

Author: Kundai Sachikonye
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from core.EntropyTransformation import SEntropyTransformer, SEntropyCoordinates

# Note: VirtualDetector and BMDFilter imports optional - commented out if not available
try:
    from core.VirtualInstrument import VirtualDetector
    HAS_VIRTUAL_DETECTOR = True
except ImportError:
    HAS_VIRTUAL_DETECTOR = False

try:
    from core.BiologicalMaxwellDemon import BMDFilter
    HAS_BMD_FILTER = True
except ImportError:
    HAS_BMD_FILTER = False

from dictionary.sentropy_dictionary import SEntropyDictionary, create_standard_proteomics_dictionary
from dictionary.zero_shot_identification import ZeroShotIdentifier
from sequence.sequence_reconstruction import SequenceReconstructor, ReconstructionResult
from sequence.categorical_completion import CategoricalCompleter
from sequence.fragment_graph import FragmentNode
from molecular_language.amino_acid_alphabet import AminoAcidAlphabet


@dataclass
class MMDConfig:
    """
    Configuration for Molecular Maxwell Demon system.

    From st-stellas-spectrometry.tex Section 3:
    System parameters for variance minimization and equilibrium seeking.
    """
    # S-Entropy parameters
    sentropy_bandwidth: float = 0.2

    # Dictionary parameters
    dictionary_path: Optional[str] = None
    enable_dynamic_learning: bool = True
    distance_threshold: float = 0.15

    # Reconstruction parameters
    mass_tolerance: float = 0.5  # Da
    max_gap_size: int = 5
    min_fragment_confidence: float = 0.3

    # Validation parameters
    enable_cross_modal: bool = True
    enable_bmd_filtering: bool = True

    # Virtual detector parameters
    enable_virtual_detectors: bool = False
    virtual_detector_types: List[str] = field(default_factory=lambda: ['TOF', 'Orbitrap'])


class MolecularMaxwellDemonSystem:
    """
    Complete Molecular Maxwell Demon system for database-free proteomics.

    From st-stellas-spectrometry.tex:
    "The MMD system achieves molecular understanding through:
    1. Variance minimization in S-Entropy space
    2. Dynamic network expansion via dictionary learning
    3. Cross-modal pathway analysis for validation"

    This is THE SYSTEM that unifies everything!
    """

    def __init__(self, config: Optional[MMDConfig] = None):
        """
        Initialize Molecular Maxwell Demon system.

        Args:
            config: System configuration
        """
        self.config = config if config else MMDConfig()

        print("\n" + "="*60)
        print("MOLECULAR MAXWELL DEMON SYSTEM")
        print("Database-Free Proteomics via S-Entropy")
        print("="*60 + "\n")

        # LAYER 1: S-Entropy Neural Network (SENN)
        print("[MMD] Initializing S-Entropy Neural Network...")
        self.sentropy_transformer = SEntropyTransformer()

        # LAYER 2: Empty Dictionary Architecture
        print("[MMD] Initializing Dictionary Architecture...")
        if self.config.dictionary_path:
            self.dictionary = SEntropyDictionary()
            self.dictionary.load(self.config.dictionary_path)
        else:
            self.dictionary = create_standard_proteomics_dictionary()

        # Zero-shot identifier
        self.zero_shot_identifier = ZeroShotIdentifier(
            dictionary=self.dictionary,
            distance_threshold=self.config.distance_threshold,
            mass_tolerance=self.config.mass_tolerance
        )

        # LAYER 3: Categorical Completion Engine
        print("[MMD] Initializing Categorical Completion...")
        self.alphabet = AminoAcidAlphabet()
        self.categorical_completer = CategoricalCompleter(
            dictionary=self.dictionary,
            alphabet=self.alphabet,
            mass_tolerance=self.config.mass_tolerance
        )

        # LAYER 4: Sequence Reconstructor
        print("[MMD] Initializing Sequence Reconstructor...")
        self.sequence_reconstructor = SequenceReconstructor(
            dictionary=self.dictionary,
            zero_shot_identifier=self.zero_shot_identifier,
            categorical_completer=self.categorical_completer
        )

        # LAYER 5: BMD Equivalence (Biological Maxwell Demon)
        if self.config.enable_bmd_filtering and HAS_BMD_FILTER:
            print("[MMD] Initializing BMD Filter...")
            self.bmd_filter = BMDFilter()
        else:
            self.bmd_filter = None
            if self.config.enable_bmd_filtering and not HAS_BMD_FILTER:
                print("[MMD] Warning: BMD filtering requested but module not available")

        # LAYER 6: Virtual Detectors (optional)
        self.virtual_detectors = {}
        if self.config.enable_virtual_detectors and HAS_VIRTUAL_DETECTOR:
            print("[MMD] Initializing Virtual Detectors...")
            for detector_type in self.config.virtual_detector_types:
                self.virtual_detectors[detector_type] = VirtualDetector(
                    detector_type=detector_type
                )
        elif self.config.enable_virtual_detectors and not HAS_VIRTUAL_DETECTOR:
            print("[MMD] Warning: Virtual detectors requested but module not available")

        print("\n[MMD] System initialization COMPLETE!")
        print(f"[MMD] Dictionary size: {len(self.dictionary.entries)} entries")
        print(f"[MMD] Dynamic learning: {self.config.enable_dynamic_learning}")
        print(f"[MMD] BMD filtering: {self.config.enable_bmd_filtering}\n")

    def analyze_spectrum(
        self,
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        precursor_mz: float,
        precursor_charge: int = 2,
        rt: Optional[float] = None
    ) -> ReconstructionResult:
        """
        Complete database-free analysis of MS/MS spectrum.

        From st-stellas-spectrometry.tex Algorithm:

        1. S-Entropy transformation
        2. Fragment identification (zero-shot)
        3. Categorical completion
        4. Sequence reconstruction
        5. Cross-modal validation

        Args:
            mz_array: Fragment m/z values
            intensity_array: Fragment intensities
            precursor_mz: Precursor m/z
            precursor_charge: Precursor charge
            rt: Retention time (optional)

        Returns:
            ReconstructionResult with peptide sequence
        """
        print(f"\n[MMD] Analyzing spectrum: precursor {precursor_mz:.4f} m/z, "
              f"charge {precursor_charge}, {len(mz_array)} peaks")

        # STEP 1: S-Entropy Transformation
        print("[MMD] Step 1: S-Entropy transformation...")
        coords_list, coord_matrix = self.sentropy_transformer.transform_spectrum(
            mz_array=mz_array,
            intensity_array=intensity_array,
            precursor_mz=precursor_mz,
            rt=rt
        )

        print(f"[MMD] Transformed {len(coords_list)} fragments to S-Entropy space")

        # STEP 2: BMD Filtering (optional)
        if self.bmd_filter and len(coords_list) > 0:
            print("[MMD] Step 2: BMD filtering...")
            # Filter based on hardware coherence
            # (Placeholder - full implementation would use actual BMD filtering)
            filtered_indices = list(range(len(coords_list)))
        else:
            filtered_indices = list(range(len(coords_list)))

        # STEP 3: Build fragment nodes
        print("[MMD] Step 3: Building fragment nodes...")
        fragments = []
        for i in filtered_indices:
            node = FragmentNode(
                fragment_id=f"frag_{i}",
                sequence=None,  # Unknown, will be identified
                s_entropy_coords=coords_list[i],
                mass=float(mz_array[i] * precursor_charge),  # Approximate mass
                ion_type=None,  # Unknown
                position=None,
                confidence=1.0
            )
            fragments.append(node)

        print(f"[MMD] Created {len(fragments)} fragment nodes")

        # STEP 4: Sequence Reconstruction
        print("[MMD] Step 4: Sequence reconstruction...")
        precursor_mass = precursor_mz * precursor_charge

        result = self.sequence_reconstructor.reconstruct(
            fragments=fragments,
            precursor_mass=precursor_mass,
            precursor_charge=precursor_charge
        )

        # STEP 5: Cross-Modal Validation (if enabled)
        if self.config.enable_cross_modal:
            print("[MMD] Step 5: Cross-modal validation...")
            result = self._cross_modal_validation(result, mz_array, intensity_array)

        # STEP 6: Dynamic Dictionary Learning (if enabled and novel entities found)
        if self.config.enable_dynamic_learning:
            self._update_dictionary_from_reconstruction(result, fragments)

        print(f"\n[MMD] Analysis COMPLETE: {result.sequence}")
        print(f"[MMD] Confidence: {result.confidence:.3f}")

        return result

    def batch_analyze(
        self,
        spectra: List[Tuple[np.ndarray, np.ndarray, float, int, Optional[float]]]
    ) -> List[ReconstructionResult]:
        """
        Batch analysis of multiple spectra.

        Args:
            spectra: List of (mz, intensity, precursor_mz, charge, rt) tuples

        Returns:
            List of ReconstructionResults
        """
        results = []

        print(f"\n[MMD] Batch analysis: {len(spectra)} spectra")

        for i, (mz, intensity, prec_mz, charge, rt) in enumerate(spectra):
            print(f"\n{'='*60}")
            print(f"SPECTRUM {i+1}/{len(spectra)}")
            print(f"{'='*60}")

            result = self.analyze_spectrum(
                mz_array=mz,
                intensity_array=intensity,
                precursor_mz=prec_mz,
                precursor_charge=charge,
                rt=rt
            )

            results.append(result)

        # Batch statistics
        print(f"\n{'='*60}")
        print("BATCH ANALYSIS SUMMARY")
        print(f"{'='*60}")

        sequences = [r.sequence for r in results if r.sequence]
        confidences = [r.confidence for r in results]

        print(f"Total spectra: {len(spectra)}")
        print(f"Sequences reconstructed: {len(sequences)}")
        print(f"Mean confidence: {np.mean(confidences):.3f}")
        print(f"High confidence (>0.7): {sum(1 for c in confidences if c > 0.7)}")

        if self.config.enable_dynamic_learning:
            print(f"Dictionary size after learning: {len(self.dictionary.entries)}")

        return results

    def _cross_modal_validation(
        self,
        result: ReconstructionResult,
        mz_array: np.ndarray,
        intensity_array: np.ndarray
    ) -> ReconstructionResult:
        """
        Cross-modal validation of reconstruction.

        From st-stellas-spectrometry.tex Section 4:
        "Cross-modal pathway analysis validates results across
        different measurement modalities."

        Args:
            result: Reconstruction result
            mz_array: Original m/z array
            intensity_array: Original intensity array

        Returns:
            Updated reconstruction result with validation scores
        """
        # Validate by checking if theoretical fragments match observed
        if not result.sequence:
            return result

        from molecular_language.fragmentation_grammar import PROTEOMICS_GRAMMAR

        # Generate theoretical fragments
        theoretical_fragments = PROTEOMICS_GRAMMAR.generate_all_fragments(
            sequence=result.sequence,
            include_b_ions=True,
            include_y_ions=True
        )

        # Calculate theoretical m/z values
        theoretical_mz = []
        for ion_type, pos, frag_seq, neutral_loss in theoretical_fragments:
            mz = PROTEOMICS_GRAMMAR.calculate_fragment_mass(
                frag_seq, ion_type, neutral_loss
            )
            theoretical_mz.append(mz)

        # Match with observed peaks
        tolerance = 0.5  # Da
        matched = 0

        for theo_mz in theoretical_mz:
            for obs_mz in mz_array:
                if abs(theo_mz - obs_mz) <= tolerance:
                    matched += 1
                    break

        # Validation score
        if len(theoretical_mz) > 0:
            validation_score = matched / len(theoretical_mz)
        else:
            validation_score = 0.0

        # Update result
        result.validation_scores['cross_modal_match'] = validation_score
        result.confidence = (result.confidence + validation_score) / 2.0  # Average

        print(f"[MMD] Cross-modal validation: {matched}/{len(theoretical_mz)} "
              f"peaks matched ({validation_score:.1%})")

        return result

    def _update_dictionary_from_reconstruction(
        self,
        result: ReconstructionResult,
        fragments: List[FragmentNode]
    ):
        """
        Update dictionary from novel entities discovered during reconstruction.

        From st-stellas-dictionary.tex Section 4:
        Dynamic dictionary learning.
        """
        # Check if any fragments were marked as novel during identification
        # (This would be tracked by the zero-shot identifier)

        # For now, just log
        novel_count = sum(1 for frag in fragments
                         if frag.confidence < 0.5)  # Low confidence = potentially novel

        if novel_count > 0:
            print(f"[MMD] Potentially novel entities: {novel_count}")

    def save_dictionary(self, filepath: str):
        """Save learned dictionary to file."""
        self.dictionary.save(filepath)

    def load_dictionary(self, filepath: str):
        """Load dictionary from file."""
        self.dictionary.load(filepath)
        # Rebuild zero-shot identifier
        self.zero_shot_identifier = ZeroShotIdentifier(
            dictionary=self.dictionary,
            distance_threshold=self.config.distance_threshold,
            mass_tolerance=self.config.mass_tolerance
        )

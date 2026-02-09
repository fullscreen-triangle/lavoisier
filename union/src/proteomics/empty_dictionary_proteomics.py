"""
Empty Dictionary Proteomics: Database-Free Sequence Reconstruction
===================================================================

Implements the Empty Dictionary architecture from St. Stella's Dictionary
for proteomics sequence reconstruction without database lookups.

Key Concepts:
- Amino acid → Codon transformation (protein to genomic representation)
- ATGC → Cardinal direction walks (N/S/E/W)
- Semantic gas molecular model (fragment ions = perturbations)
- Circular validation (correct sequence = path returns to origin)

Based on:
- docs/publication/st-stellas-dictionary.tex
- docs/publication/st-stellas-sequence.tex

The Empty Dictionary principle:
"Starting from nothing, traverse coordinates to synthesize meaning dynamically.
No static lookup. No database. Just navigation through S-Entropy space."

Author: Kundai Sachikonye
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import time


# ============================================================================
# CODON TABLE: AMINO ACID TO GENOMIC REPRESENTATION
# ============================================================================

# Standard genetic code: Amino acid → preferred codon(s)
# Using the most common codon for each amino acid
AMINO_ACID_TO_CODON = {
    'A': 'GCT',   # Alanine (GCT, GCC, GCA, GCG)
    'R': 'CGT',   # Arginine (CGT, CGC, CGA, CGG, AGA, AGG)
    'N': 'AAT',   # Asparagine (AAT, AAC)
    'D': 'GAT',   # Aspartate (GAT, GAC)
    'C': 'TGT',   # Cysteine (TGT, TGC)
    'E': 'GAA',   # Glutamate (GAA, GAG)
    'Q': 'CAA',   # Glutamine (CAA, CAG)
    'G': 'GGT',   # Glycine (GGT, GGC, GGA, GGG)
    'H': 'CAT',   # Histidine (CAT, CAC)
    'I': 'ATT',   # Isoleucine (ATT, ATC, ATA)
    'L': 'CTT',   # Leucine (CTT, CTC, CTA, CTG, TTA, TTG)
    'K': 'AAA',   # Lysine (AAA, AAG)
    'M': 'ATG',   # Methionine (ATG - START)
    'F': 'TTT',   # Phenylalanine (TTT, TTC)
    'P': 'CCT',   # Proline (CCT, CCC, CCA, CCG)
    'S': 'TCT',   # Serine (TCT, TCC, TCA, TCG, AGT, AGC)
    'T': 'ACT',   # Threonine (ACT, ACC, ACA, ACG)
    'W': 'TGG',   # Tryptophan (TGG)
    'Y': 'TAT',   # Tyrosine (TAT, TAC)
    'V': 'GTT',   # Valine (GTT, GTC, GTA, GTG)
    # Non-standard amino acids
    'O': 'TAG',   # Pyrrolysine (22nd amino acid, TAG recoded)
    'U': 'TGA',   # Selenocysteine (21st amino acid, TGA recoded)
}

# Reverse mapping: Codon → Amino acid
CODON_TO_AMINO_ACID = {
    # Alanine
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    # Arginine
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
    # Asparagine
    'AAT': 'N', 'AAC': 'N',
    # Aspartate
    'GAT': 'D', 'GAC': 'D',
    # Cysteine
    'TGT': 'C', 'TGC': 'C',
    # Glutamate
    'GAA': 'E', 'GAG': 'E',
    # Glutamine
    'CAA': 'Q', 'CAG': 'Q',
    # Glycine
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    # Histidine
    'CAT': 'H', 'CAC': 'H',
    # Isoleucine
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I',
    # Leucine
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L', 'TTA': 'L', 'TTG': 'L',
    # Lysine
    'AAA': 'K', 'AAG': 'K',
    # Methionine
    'ATG': 'M',
    # Phenylalanine
    'TTT': 'F', 'TTC': 'F',
    # Proline
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    # Serine
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',
    # Threonine
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    # Tryptophan
    'TGG': 'W',
    # Tyrosine
    'TAT': 'Y', 'TAC': 'Y',
    # Valine
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    # Non-standard amino acids (recoded stop codons)
    'TAG': 'O',  # Pyrrolysine (in some organisms)
    'TGA': 'U',  # Selenocysteine (in some organisms)
    # Standard stop codons (when not recoded)
    'TAA': '*',
}


# ============================================================================
# CARDINAL DIRECTION TRANSFORMATION (St. Stella's Framework)
# ============================================================================

# Cardinal direction mapping for nucleotides
# From St. Stella's Sequence: A→North, T→South, G→East, C→West
NUCLEOTIDE_CARDINAL = {
    'A': np.array([0, 1]),    # North (up)
    'T': np.array([0, -1]),   # South (down)
    'G': np.array([1, 0]),    # East (right)
    'C': np.array([-1, 0]),   # West (left)
}

# Direction names for readability
DIRECTION_NAMES = {
    'A': 'N',  # North
    'T': 'S',  # South
    'G': 'E',  # East
    'C': 'W',  # West
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CardinalWalk:
    """
    Cardinal direction walk for a sequence.

    Represents the trajectory through 2D space as nucleotides are traversed.
    """
    sequence: str  # Original sequence (nucleotides)
    positions: np.ndarray  # 2D positions at each step
    final_position: np.ndarray  # Final position
    path_length: float  # Total Euclidean path length
    displacement: float  # Distance from origin to end
    closure_distance: float  # Distance from end back to origin
    is_closed: bool  # True if path returns to origin (within tolerance)

    # Direction sequence (N/S/E/W)
    direction_sequence: str = ""

    # S-Entropy coordinates along path
    s_entropy_trajectory: np.ndarray = field(default=None)


@dataclass
class SemanticGasState:
    """
    Semantic gas state for fragment processing.

    In the semantic gas molecular model:
    - Molecules = Fragment ions
    - Temperature = Spectral intensity distribution
    - Pressure = Fragment density in m/z space
    - Volume = Mass range covered
    - Perturbation = New fragment ion
    - Equilibrium = Correct sequence found
    """
    n_fragments: int
    temperature: float  # Related to intensity distribution
    pressure: float  # Related to fragment density
    volume: float  # Mass range

    # S-Entropy coordinates
    s_knowledge: float  # Information content
    s_time: float  # Temporal ordering
    s_entropy: float  # Disorder measure

    # Equilibrium metrics
    perturbation_energy: float = 0.0
    equilibrium_distance: float = 0.0
    is_equilibrium: bool = False


@dataclass
class CircularValidation:
    """
    Result of circular validation for a sequence.

    Circular validation principle:
    A → B → C → A
    If the path closes (returns to origin), the sequence is valid.
    """
    sequence: str
    is_valid: bool
    closure_distance: float  # Should be close to 0 for valid sequence
    closure_score: float  # 0-1, higher = better closure

    # Walk details
    walk: CardinalWalk = None

    # Semantic gas state
    gas_state: SemanticGasState = None

    # Processing time
    processing_time_ms: float = 0.0


@dataclass
class EmptyDictionaryResult:
    """
    Result from Empty Dictionary sequence reconstruction.

    The Empty Dictionary principle:
    - No static database lookup
    - Meaning emerges through coordinate navigation
    - Sequence is discovered by minimizing S-distance to equilibrium
    """
    predicted_sequence: str
    confidence: float

    # Cardinal walk analysis
    walk: CardinalWalk = None

    # Circular validation
    circular_validation: CircularValidation = None

    # Candidate sequences explored
    candidates_explored: int = 0

    # Best closure score among candidates
    best_closure_score: float = 0.0

    # Semantic gas state at solution
    final_gas_state: SemanticGasState = None

    # Processing metrics
    processing_time_ms: float = 0.0
    n_fragments: int = 0


# ============================================================================
# EMPTY DICTIONARY TRANSFORMER
# ============================================================================

class EmptyDictionaryTransformer:
    """
    Empty Dictionary Transformer for Proteomics.

    Implements the Empty Dictionary architecture from St. Stella's Dictionary:
    - Transforms proteins to genomic representation (via codons)
    - Applies cardinal direction walks (ATGC → N/S/E/W)
    - Uses semantic gas model for fragment processing
    - Validates through circular path closure

    Key insight: The correct sequence closes the path.
    Wrong sequences leave the path open (displaced from origin).
    """

    def __init__(
        self,
        closure_tolerance: float = 0.5,
        mass_tolerance: float = 0.5,
        semantic_temperature: float = 300.0
    ):
        """
        Initialize Empty Dictionary transformer.

        Args:
            closure_tolerance: Distance tolerance for path closure validation
            mass_tolerance: Mass tolerance for amino acid matching (Da)
            semantic_temperature: Temperature parameter for semantic gas model
        """
        self.closure_tolerance = closure_tolerance
        self.mass_tolerance = mass_tolerance
        self.semantic_temperature = semantic_temperature

        # Build mass lookup for fragment matching
        self._build_mass_lookup()

    def _build_mass_lookup(self):
        """Build mass-to-amino acid lookup."""
        # Standard amino acid masses (monoisotopic)
        self.amino_acid_masses = {
            'A': 71.03711,   'R': 156.10111,  'N': 114.04293,
            'D': 115.02694,  'C': 103.00919,  'E': 129.04259,
            'Q': 128.05858,  'G': 57.02146,   'H': 137.05891,
            'I': 113.08406,  'L': 113.08406,  'K': 128.09496,
            'M': 131.04049,  'F': 147.06841,  'P': 97.05276,
            'S': 87.03203,   'T': 101.04768,  'W': 186.07931,
            'Y': 163.06333,  'V': 99.06841,
            # Non-standard amino acids
            'O': 255.15829,  # Pyrrolysine
            'U': 167.95681,  # Selenocysteine
        }

        # Reverse lookup
        self.mass_to_aa = {}
        for aa, mass in self.amino_acid_masses.items():
            mass_key = round(mass, 1)
            if mass_key not in self.mass_to_aa:
                self.mass_to_aa[mass_key] = []
            self.mass_to_aa[mass_key].append(aa)

    # ========================================================================
    # PROTEIN TO GENOMIC TRANSFORMATION
    # ========================================================================

    def protein_to_nucleotides(self, peptide_sequence: str) -> str:
        """
        Transform protein sequence to nucleotide sequence via codon mapping.

        This is the key transformation that enables the Empty Dictionary:
        Protein → Genome → Cardinal walks → S-Entropy space

        Args:
            peptide_sequence: Amino acid sequence (e.g., "PEPTIDE")

        Returns:
            Nucleotide sequence (e.g., "CCTGAAACTATCGATACAT...")
        """
        nucleotides = []

        for aa in peptide_sequence.upper():
            if aa in AMINO_ACID_TO_CODON:
                nucleotides.append(AMINO_ACID_TO_CODON[aa])
            else:
                # Unknown amino acid - use placeholder codon
                nucleotides.append('NNN')

        return ''.join(nucleotides)

    def nucleotides_to_protein(self, nucleotide_sequence: str) -> str:
        """
        Transform nucleotide sequence back to protein sequence.

        Args:
            nucleotide_sequence: Nucleotide sequence

        Returns:
            Amino acid sequence
        """
        protein = []

        for i in range(0, len(nucleotide_sequence) - 2, 3):
            codon = nucleotide_sequence[i:i+3].upper()
            if codon in CODON_TO_AMINO_ACID:
                aa = CODON_TO_AMINO_ACID[codon]
                if aa != '*':  # Skip stop codons
                    protein.append(aa)
            else:
                protein.append('X')  # Unknown

        return ''.join(protein)

    # ========================================================================
    # CARDINAL DIRECTION WALKS
    # ========================================================================

    def cardinal_walk(self, nucleotide_sequence: str) -> CardinalWalk:
        """
        Perform cardinal direction walk through nucleotide sequence.

        Maps each nucleotide to a direction:
        A → North (0, 1)
        T → South (0, -1)
        G → East (1, 0)
        C → West (-1, 0)

        Args:
            nucleotide_sequence: Nucleotide sequence (ATGC)

        Returns:
            CardinalWalk with trajectory information
        """
        sequence = nucleotide_sequence.upper()
        n = len(sequence)

        if n == 0:
            return CardinalWalk(
                sequence="",
                positions=np.array([[0, 0]]),
                final_position=np.array([0, 0]),
                path_length=0.0,
                displacement=0.0,
                closure_distance=0.0,
                is_closed=True,
                direction_sequence=""
            )

        # Track positions
        positions = [np.array([0.0, 0.0])]  # Start at origin
        current = np.array([0.0, 0.0])
        direction_chars = []

        # S-entropy trajectory
        s_trajectory = []

        for i, nucleotide in enumerate(sequence):
            if nucleotide in NUCLEOTIDE_CARDINAL:
                direction = NUCLEOTIDE_CARDINAL[nucleotide]
                current = current + direction
                positions.append(current.copy())
                direction_chars.append(DIRECTION_NAMES.get(nucleotide, '?'))

                # Compute S-entropy at this step
                s_k = np.log1p(np.abs(current[0]) + 0.01)  # Knowledge: x-extent
                s_t = (i + 1) / n  # Time: position in sequence
                s_e = np.linalg.norm(current) / (i + 1) if i > 0 else 0  # Entropy: displacement per step
                s_trajectory.append([s_k, s_t, s_e])

        positions = np.array(positions)
        final_position = positions[-1]

        # Compute path metrics
        path_length = 0.0
        for i in range(1, len(positions)):
            path_length += np.linalg.norm(positions[i] - positions[i-1])

        displacement = np.linalg.norm(final_position)
        closure_distance = displacement  # Distance back to origin
        is_closed = closure_distance < self.closure_tolerance

        return CardinalWalk(
            sequence=sequence,
            positions=positions,
            final_position=final_position,
            path_length=path_length,
            displacement=displacement,
            closure_distance=closure_distance,
            is_closed=is_closed,
            direction_sequence=''.join(direction_chars),
            s_entropy_trajectory=np.array(s_trajectory) if s_trajectory else None
        )

    def peptide_cardinal_walk(self, peptide_sequence: str) -> CardinalWalk:
        """
        Perform cardinal walk for a peptide sequence.

        First transforms to nucleotides, then performs walk.

        Args:
            peptide_sequence: Amino acid sequence

        Returns:
            CardinalWalk for the peptide
        """
        nucleotides = self.protein_to_nucleotides(peptide_sequence)
        return self.cardinal_walk(nucleotides)

    # ========================================================================
    # SEMANTIC GAS MODEL
    # ========================================================================

    def compute_semantic_gas_state(
        self,
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        precursor_mz: float
    ) -> SemanticGasState:
        """
        Compute semantic gas state from fragment spectrum.

        In the semantic gas molecular model:
        - Fragment ions = Gas molecules
        - Intensity = Kinetic energy
        - m/z distribution = Spatial distribution
        - Finding equilibrium = Finding correct sequence

        Args:
            mz_array: Fragment m/z values
            intensity_array: Fragment intensities
            precursor_mz: Precursor m/z

        Returns:
            SemanticGasState describing the fragment ensemble
        """
        n_fragments = len(mz_array)

        if n_fragments == 0:
            return SemanticGasState(
                n_fragments=0,
                temperature=0.0,
                pressure=0.0,
                volume=0.0,
                s_knowledge=0.0,
                s_time=0.0,
                s_entropy=0.0
            )

        # Normalize intensities
        total_intensity = np.sum(intensity_array) + 1e-10
        p_i = intensity_array / total_intensity

        # Semantic temperature: variance of intensity distribution
        # High variance = high temperature = more disorder
        temperature = np.var(intensity_array) / (np.mean(intensity_array) + 1e-10)
        temperature = temperature * self.semantic_temperature / 1000.0  # Scale

        # Semantic pressure: fragment density in m/z space
        mass_range = np.max(mz_array) - np.min(mz_array) + 1.0
        pressure = n_fragments / mass_range

        # Semantic volume: mass range covered
        volume = mass_range

        # S-Entropy coordinates
        # S_knowledge: information from intensity-weighted mass
        s_knowledge = -np.sum(p_i * np.log(p_i + 1e-10))

        # S_time: ordering by m/z (temporal analog)
        sorted_idx = np.argsort(mz_array)
        mz_sorted = mz_array[sorted_idx]
        mz_normalized = (mz_sorted - mz_sorted.min()) / (mz_sorted.max() - mz_sorted.min() + 1e-10)
        s_time = np.mean(mz_normalized)

        # S_entropy: fragmentation entropy
        s_entropy = s_knowledge  # Use Shannon entropy

        # Perturbation energy: deviation from uniform distribution
        uniform_p = 1.0 / n_fragments
        perturbation_energy = np.sum((p_i - uniform_p) ** 2)

        # Equilibrium distance: how far from "equilibrium" state
        # Equilibrium = fragments evenly spaced in m/z, similar intensities
        ideal_spacing = (precursor_mz - np.min(mz_array)) / n_fragments
        actual_spacing = np.diff(np.sort(mz_array))
        spacing_variance = np.var(actual_spacing) if len(actual_spacing) > 0 else 0
        equilibrium_distance = np.sqrt(spacing_variance + perturbation_energy)

        is_equilibrium = equilibrium_distance < 0.5

        return SemanticGasState(
            n_fragments=n_fragments,
            temperature=temperature,
            pressure=pressure,
            volume=volume,
            s_knowledge=s_knowledge,
            s_time=s_time,
            s_entropy=s_entropy,
            perturbation_energy=perturbation_energy,
            equilibrium_distance=equilibrium_distance,
            is_equilibrium=is_equilibrium
        )

    # ========================================================================
    # CIRCULAR VALIDATION
    # ========================================================================

    def circular_validate(
        self,
        peptide_sequence: str,
        mz_array: Optional[np.ndarray] = None,
        intensity_array: Optional[np.ndarray] = None,
        precursor_mz: Optional[float] = None
    ) -> CircularValidation:
        """
        Perform circular validation for a peptide sequence.

        Circular validation principle:
        1. Transform peptide to nucleotides
        2. Perform cardinal walk
        3. Check if path closes (returns to origin)
        4. Closed path = valid sequence

        This is the "Empty Dictionary" insight:
        The correct sequence creates a closed trajectory.
        Wrong sequences leave the path open.

        Args:
            peptide_sequence: Candidate peptide sequence
            mz_array: Optional fragment m/z values (for gas state)
            intensity_array: Optional fragment intensities
            precursor_mz: Optional precursor m/z

        Returns:
            CircularValidation result
        """
        start_time = time.time()

        # Perform cardinal walk
        walk = self.peptide_cardinal_walk(peptide_sequence)

        # Compute closure score
        # Score = 1 / (1 + closure_distance)
        # Higher is better, max = 1 for perfect closure
        closure_score = 1.0 / (1.0 + walk.closure_distance)

        # Check if valid (path closes within tolerance)
        is_valid = walk.is_closed

        # Compute semantic gas state if spectrum provided
        gas_state = None
        if mz_array is not None and intensity_array is not None:
            gas_state = self.compute_semantic_gas_state(
                mz_array, intensity_array, precursor_mz or 0.0
            )

        processing_time = (time.time() - start_time) * 1000

        return CircularValidation(
            sequence=peptide_sequence,
            is_valid=is_valid,
            closure_distance=walk.closure_distance,
            closure_score=closure_score,
            walk=walk,
            gas_state=gas_state,
            processing_time_ms=processing_time
        )

    # ========================================================================
    # SEQUENCE RECONSTRUCTION (EMPTY DICTIONARY APPROACH)
    # ========================================================================

    def reconstruct_from_fragments(
        self,
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        precursor_mz: float,
        precursor_charge: int = 2,
        max_candidates: int = 100
    ) -> EmptyDictionaryResult:
        """
        Reconstruct peptide sequence using Empty Dictionary approach.

        The Empty Dictionary principle:
        1. Start from nothing (empty sequence)
        2. Navigate through coordinate space
        3. Meaning emerges through trajectory
        4. Correct sequence = closed path = equilibrium

        Args:
            mz_array: Fragment m/z values
            intensity_array: Fragment intensities
            precursor_mz: Precursor m/z
            precursor_charge: Precursor charge
            max_candidates: Maximum candidates to evaluate

        Returns:
            EmptyDictionaryResult with reconstructed sequence
        """
        start_time = time.time()

        # Compute semantic gas state
        gas_state = self.compute_semantic_gas_state(
            mz_array, intensity_array, precursor_mz
        )

        # Infer mass differences (potential amino acids)
        sorted_idx = np.argsort(mz_array)
        mz_sorted = mz_array[sorted_idx]

        mass_diffs = np.diff(mz_sorted)

        # Match mass differences to amino acids
        inferred_aas = []
        for diff in mass_diffs:
            aa = self._match_mass_to_aa(diff)
            if aa:
                inferred_aas.append(aa)

        # Generate candidate sequences
        candidates = self._generate_candidates(
            inferred_aas, precursor_mz, precursor_charge, max_candidates
        )

        # Evaluate candidates using circular validation
        best_candidate = None
        best_score = 0.0

        for candidate in candidates:
            validation = self.circular_validate(
                candidate, mz_array, intensity_array, precursor_mz
            )

            if validation.closure_score > best_score:
                best_score = validation.closure_score
                best_candidate = candidate
                best_walk = validation.walk

        # If no good candidates, return best guess from inferred AAs
        if best_candidate is None:
            best_candidate = ''.join(inferred_aas[:10])  # First 10 inferred
            best_walk = self.peptide_cardinal_walk(best_candidate)
            best_score = 0.0

        # Final circular validation
        final_validation = self.circular_validate(
            best_candidate, mz_array, intensity_array, precursor_mz
        )

        processing_time = (time.time() - start_time) * 1000

        return EmptyDictionaryResult(
            predicted_sequence=best_candidate,
            confidence=best_score,
            walk=best_walk,
            circular_validation=final_validation,
            candidates_explored=len(candidates),
            best_closure_score=best_score,
            final_gas_state=gas_state,
            processing_time_ms=processing_time,
            n_fragments=len(mz_array)
        )

    def _match_mass_to_aa(self, mass_diff: float) -> Optional[str]:
        """Match mass difference to amino acid."""
        for aa, mass in self.amino_acid_masses.items():
            if abs(mass_diff - mass) < self.mass_tolerance:
                return aa
        return None

    def _generate_candidates(
        self,
        inferred_aas: List[str],
        precursor_mz: float,
        precursor_charge: int,
        max_candidates: int
    ) -> List[str]:
        """
        Generate candidate sequences from inferred amino acids.

        Uses permutations and combinations to explore sequence space.
        """
        if not inferred_aas:
            return []

        # Target mass
        target_mass = precursor_mz * precursor_charge - precursor_charge * 1.007276

        candidates = []

        # Start with direct sequence
        direct_seq = ''.join(inferred_aas)
        if direct_seq:
            candidates.append(direct_seq)

        # Reversed sequence
        reverse_seq = direct_seq[::-1]
        if reverse_seq and reverse_seq != direct_seq:
            candidates.append(reverse_seq)

        # Generate permutations (limited)
        from itertools import permutations

        # Only permute if small enough
        if len(inferred_aas) <= 8:
            for perm in permutations(inferred_aas):
                seq = ''.join(perm)
                if seq not in candidates:
                    candidates.append(seq)
                if len(candidates) >= max_candidates:
                    break

        # Add subsequences of different lengths
        for length in range(len(inferred_aas) - 1, max(1, len(inferred_aas) - 3), -1):
            subseq = ''.join(inferred_aas[:length])
            if subseq and subseq not in candidates:
                candidates.append(subseq)

        return candidates[:max_candidates]

    # ========================================================================
    # S-DISTANCE NAVIGATION (EMPTY DICTIONARY CORE)
    # ========================================================================

    def s_distance(self, walk1: CardinalWalk, walk2: CardinalWalk) -> float:
        """
        Compute S-distance between two cardinal walks.

        S-distance measures similarity in S-Entropy space:
        - Similar trajectories → small S-distance
        - Different trajectories → large S-distance

        Args:
            walk1: First cardinal walk
            walk2: Second cardinal walk

        Returns:
            S-distance (Euclidean distance in S-Entropy space)
        """
        if walk1.s_entropy_trajectory is None or walk2.s_entropy_trajectory is None:
            # Fallback: use final position distance
            return np.linalg.norm(walk1.final_position - walk2.final_position)

        # Interpolate to same length
        len1 = len(walk1.s_entropy_trajectory)
        len2 = len(walk2.s_entropy_trajectory)
        max_len = max(len1, len2)

        if len1 < max_len:
            s1 = np.zeros((max_len, 3))
            for i in range(3):
                s1[:, i] = np.interp(
                    np.linspace(0, 1, max_len),
                    np.linspace(0, 1, len1),
                    walk1.s_entropy_trajectory[:, i]
                )
        else:
            s1 = walk1.s_entropy_trajectory

        if len2 < max_len:
            s2 = np.zeros((max_len, 3))
            for i in range(3):
                s2[:, i] = np.interp(
                    np.linspace(0, 1, max_len),
                    np.linspace(0, 1, len2),
                    walk2.s_entropy_trajectory[:, i]
                )
        else:
            s2 = walk2.s_entropy_trajectory

        # Compute mean distance along trajectories
        distances = np.linalg.norm(s1 - s2, axis=1)

        return float(np.mean(distances))

    def navigate_to_equilibrium(
        self,
        gas_state: SemanticGasState,
        candidates: List[str],
        target_closure: float = 0.95
    ) -> Tuple[str, float]:
        """
        Navigate through S-Entropy space to find equilibrium.

        This is the core of the Empty Dictionary:
        - Start from perturbation (fragment ions)
        - Navigate through candidate sequences
        - Find equilibrium (closed path)

        Args:
            gas_state: Current semantic gas state
            candidates: Candidate sequences to evaluate
            target_closure: Target closure score for equilibrium

        Returns:
            Tuple of (best_sequence, closure_score)
        """
        best_seq = ""
        best_score = 0.0

        for candidate in candidates:
            validation = self.circular_validate(candidate)

            if validation.closure_score > best_score:
                best_score = validation.closure_score
                best_seq = candidate

                if best_score >= target_closure:
                    # Reached equilibrium
                    break

        return best_seq, best_score


# ============================================================================
# INTEGRATION WITH PIPELINE
# ============================================================================

def empty_dictionary_reconstruct(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    precursor_mz: float,
    precursor_charge: int = 2,
    closure_tolerance: float = 0.5
) -> EmptyDictionaryResult:
    """
    Convenience function for Empty Dictionary reconstruction.

    Args:
        mz_array: Fragment m/z values
        intensity_array: Fragment intensities
        precursor_mz: Precursor m/z
        precursor_charge: Precursor charge
        closure_tolerance: Tolerance for path closure

    Returns:
        EmptyDictionaryResult with reconstruction
    """
    transformer = EmptyDictionaryTransformer(
        closure_tolerance=closure_tolerance
    )

    return transformer.reconstruct_from_fragments(
        mz_array, intensity_array, precursor_mz, precursor_charge
    )


def validate_sequence_circular(
    peptide_sequence: str,
    mz_array: Optional[np.ndarray] = None,
    intensity_array: Optional[np.ndarray] = None,
    precursor_mz: Optional[float] = None
) -> CircularValidation:
    """
    Convenience function for circular validation.

    Args:
        peptide_sequence: Sequence to validate
        mz_array: Optional fragment m/z
        intensity_array: Optional intensities
        precursor_mz: Optional precursor m/z

    Returns:
        CircularValidation result
    """
    transformer = EmptyDictionaryTransformer()

    return transformer.circular_validate(
        peptide_sequence, mz_array, intensity_array, precursor_mz
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Empty Dictionary Proteomics - Validation")
    print("=" * 70)

    # Initialize transformer
    transformer = EmptyDictionaryTransformer()

    # Test 1: Protein to nucleotide transformation
    print("\n1. PROTEIN TO NUCLEOTIDE TRANSFORMATION")
    print("-" * 40)

    test_peptide = "PEPTIDE"
    nucleotides = transformer.protein_to_nucleotides(test_peptide)
    recovered = transformer.nucleotides_to_protein(nucleotides)

    print(f"Original peptide: {test_peptide}")
    print(f"Nucleotides: {nucleotides}")
    print(f"Recovered: {recovered}")
    print(f"Match: {test_peptide == recovered}")

    # Test 2: Cardinal direction walk
    print("\n2. CARDINAL DIRECTION WALK")
    print("-" * 40)

    walk = transformer.peptide_cardinal_walk(test_peptide)

    print(f"Sequence: {walk.sequence}")
    print(f"Direction sequence: {walk.direction_sequence}")
    print(f"Path length: {walk.path_length:.2f}")
    print(f"Displacement: {walk.displacement:.2f}")
    print(f"Closure distance: {walk.closure_distance:.2f}")
    print(f"Is closed: {walk.is_closed}")
    print(f"Final position: ({walk.final_position[0]:.1f}, {walk.final_position[1]:.1f})")

    # Test 3: Circular validation
    print("\n3. CIRCULAR VALIDATION")
    print("-" * 40)

    # Test different sequences
    test_sequences = ["PEPTIDE", "PEPTIED", "MAAAA", "GGGG", "ATAT"]

    for seq in test_sequences:
        validation = transformer.circular_validate(seq)
        print(f"{seq}: closure={validation.closure_distance:.2f}, "
              f"score={validation.closure_score:.3f}, "
              f"valid={validation.is_valid}")

    # Test 4: Semantic gas state
    print("\n4. SEMANTIC GAS STATE")
    print("-" * 40)

    # Simulated spectrum
    mz_array = np.array([100.0, 171.0, 284.0, 397.0, 510.0, 657.0])
    intensity_array = np.array([1000, 5000, 8000, 12000, 6000, 3000])
    precursor_mz = 800.0

    gas_state = transformer.compute_semantic_gas_state(
        mz_array, intensity_array, precursor_mz
    )

    print(f"N fragments: {gas_state.n_fragments}")
    print(f"Temperature: {gas_state.temperature:.2f}")
    print(f"Pressure: {gas_state.pressure:.4f}")
    print(f"Volume: {gas_state.volume:.1f}")
    print(f"S_knowledge: {gas_state.s_knowledge:.4f}")
    print(f"S_time: {gas_state.s_time:.4f}")
    print(f"S_entropy: {gas_state.s_entropy:.4f}")
    print(f"Equilibrium distance: {gas_state.equilibrium_distance:.4f}")
    print(f"Is equilibrium: {gas_state.is_equilibrium}")

    # Test 5: Full reconstruction
    print("\n5. EMPTY DICTIONARY RECONSTRUCTION")
    print("-" * 40)

    result = transformer.reconstruct_from_fragments(
        mz_array, intensity_array, precursor_mz, precursor_charge=1
    )

    print(f"Predicted sequence: {result.predicted_sequence}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Candidates explored: {result.candidates_explored}")
    print(f"Best closure score: {result.best_closure_score:.3f}")
    print(f"Processing time: {result.processing_time_ms:.2f} ms")

    if result.circular_validation:
        print(f"Circular validation: valid={result.circular_validation.is_valid}")

    print("\n" + "=" * 70)
    print("Empty Dictionary Validation Complete")
    print("=" * 70)

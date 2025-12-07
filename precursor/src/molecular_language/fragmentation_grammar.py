#!/usr/bin/env python3
"""
Fragmentation Grammar for Proteomics
====================================

From st-stellas-molecular-language.tex Section 4.2:
Molecular grammar defines production rules for fragmentation.

MS/MS fragmentation is interpreted as string parsing:
- Peptide sequence S is parsed into fragments F₁, F₂, ..., Fₙ
- Production rules P_frag define allowed fragmentations
- b/y ions are complementary substrings

Author: Kundai Sachikonye
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
from enum import Enum


class IonType(Enum):
    """Fragment ion types in MS/MS."""
    B_ION = 'b'  # N-terminal fragments
    Y_ION = 'y'  # C-terminal fragments
    A_ION = 'a'  # b - CO
    C_ION = 'c'  # b + NH3
    X_ION = 'x'  # y + CO
    Z_ION = 'z'  # y - NH3
    PRECURSOR = 'M'  # Intact peptide


@dataclass
class FragmentationRule:
    """
    Production rule for peptide fragmentation.

    From st-stellas-molecular-language.tex Equation 4.4:
    P_frag: S → F₁ ⊕ F₂

    Where:
    - S is the peptide sequence
    - F₁, F₂ are complementary fragments
    - ⊕ is the fragmentation operator

    Attributes:
        bond_position: Cleavage position (0-indexed from N-terminus)
        ion_type: Type of fragment ion produced
        neutral_loss: Neutral loss (H2O, NH3, etc.)
        charge_retention: Which fragment retains charge
    """
    bond_position: int
    ion_type: IonType
    neutral_loss: Optional[str] = None
    charge_retention: str = 'both'  # 'N-term', 'C-term', or 'both'

    def apply(self, sequence: str, charge: int = 1) -> Tuple[str, str]:
        """
        Apply fragmentation rule to sequence.

        Returns:
            (N-terminal fragment, C-terminal fragment)
        """
        if self.bond_position < 0 or self.bond_position >= len(sequence):
            return ('', '')

        n_term = sequence[:self.bond_position + 1]
        c_term = sequence[self.bond_position + 1:]

        return (n_term, c_term)


class MolecularGrammar:
    """
    Complete molecular grammar for proteomics fragmentation.

    Implements:
    1. Production rules P_frag
    2. Complementarity constraints (b_i + y_{n-i} = M)
    3. Sequential relationships (b_i, b_{i+1} differ by one AA)
    4. Frequency coupling (all fragments from same collision event)
    """

    def __init__(self, enable_neutral_losses: bool = True):
        """
        Initialize grammar.

        Args:
            enable_neutral_losses: Include -H2O, -NH3 rules
        """
        self.enable_neutral_losses = enable_neutral_losses

        # Common neutral losses
        self.neutral_losses = {
            'H2O': 18.011,  # Water
            'NH3': 17.027,  # Ammonia
            'CO': 27.995,   # Carbon monoxide
            'H3PO4': 97.977  # Phosphoric acid (for phosphopeptides)
        }

    def generate_all_fragments(
        self,
        sequence: str,
        charge: int = 2,
        include_y_ions: bool = True,
        include_b_ions: bool = True
    ) -> List[Tuple[IonType, int, str, Optional[str]]]:
        """
        Generate all theoretical fragments for a peptide sequence.

        This is the complete P_frag production rule set.

        Args:
            sequence: Peptide sequence (single letter codes)
            charge: Precursor charge state
            include_y_ions: Generate y-ion series
            include_b_ions: Generate b-ion series

        Returns:
            List of (ion_type, position, fragment_sequence, neutral_loss)
        """
        fragments = []
        n = len(sequence)

        # B-ion series (N-terminal fragments)
        if include_b_ions:
            for i in range(1, n):  # b1 to b_{n-1}
                frag_seq = sequence[:i]
                fragments.append((IonType.B_ION, i, frag_seq, None))

                # Neutral losses
                if self.enable_neutral_losses:
                    # -H2O (common for S, T, E, D containing peptides)
                    if any(aa in frag_seq for aa in ['S', 'T', 'E', 'D']):
                        fragments.append((IonType.B_ION, i, frag_seq, 'H2O'))

                    # -NH3 (common for R, K, N, Q containing peptides)
                    if any(aa in frag_seq for aa in ['R', 'K', 'N', 'Q']):
                        fragments.append((IonType.B_ION, i, frag_seq, 'NH3'))

        # Y-ion series (C-terminal fragments)
        if include_y_ions:
            for i in range(1, n):  # y1 to y_{n-1}
                frag_seq = sequence[n-i:]
                fragments.append((IonType.Y_ION, i, frag_seq, None))

                # Neutral losses
                if self.enable_neutral_losses:
                    if any(aa in frag_seq for aa in ['S', 'T', 'E', 'D']):
                        fragments.append((IonType.Y_ION, i, frag_seq, 'H2O'))

                    if any(aa in frag_seq for aa in ['R', 'K', 'N', 'Q']):
                        fragments.append((IonType.Y_ION, i, frag_seq, 'NH3'))

        return fragments

    def calculate_fragment_mass(
        self,
        fragment_sequence: str,
        ion_type: IonType,
        neutral_loss: Optional[str] = None,
        charge: int = 1
    ) -> float:
        """
        Calculate theoretical m/z for a fragment.

        Args:
            fragment_sequence: Fragment sequence
            ion_type: Ion type (b or y)
            neutral_loss: Neutral loss (if any)
            charge: Charge state

        Returns:
            Theoretical m/z value
        """
        from .amino_acid_alphabet import STANDARD_AMINO_ACIDS

        # Sum amino acid masses
        mass = sum(STANDARD_AMINO_ACIDS[aa].mass for aa in fragment_sequence if aa in STANDARD_AMINO_ACIDS)

        # Add terminus modifications
        if ion_type == IonType.B_ION:
            # b-ion: +H (N-terminus intact)
            mass += 1.008
        elif ion_type == IonType.Y_ION:
            # y-ion: +H + OH (C-terminus intact)
            mass += 19.018

        # Subtract neutral loss
        if neutral_loss and neutral_loss in self.neutral_losses:
            mass -= self.neutral_losses[neutral_loss]

        # Calculate m/z
        mz = (mass + charge * 1.008) / charge

        return mz

    def validate_complementarity(
        self,
        b_fragment: str,
        y_fragment: str,
        original_sequence: str
    ) -> bool:
        """
        Validate b/y ion complementarity.

        From theory: b_i + y_{n-i} = M (precursor)

        Args:
            b_fragment: b-ion sequence
            y_fragment: y-ion sequence
            original_sequence: Original peptide sequence

        Returns:
            True if fragments are complementary
        """
        # Check if concatenation gives original sequence
        reconstructed = b_fragment + y_fragment
        return reconstructed == original_sequence

    def identify_sequential_series(
        self,
        fragments: List[Tuple[IonType, int, str, Optional[str]]]
    ) -> List[List[int]]:
        """
        Identify sequential ion series (b₁, b₂, b₃, ... or y₁, y₂, y₃, ...).

        Sequential ions differ by exactly one amino acid mass.
        Strong frequency coupling expected between sequential ions.

        Returns:
            List of sequential series (indices into fragments list)
        """
        series = []

        # Group by ion type
        b_ions = [(i, f) for i, f in enumerate(fragments) if f[0] == IonType.B_ION and f[3] is None]
        y_ions = [(i, f) for i, f in enumerate(fragments) if f[0] == IonType.Y_ION and f[3] is None]

        # Sort by position
        b_ions.sort(key=lambda x: x[1][1])
        y_ions.sort(key=lambda x: x[1][1])

        # Build series
        if len(b_ions) > 1:
            b_series = [idx for idx, _ in b_ions]
            series.append(b_series)

        if len(y_ions) > 1:
            y_series = [idx for idx, _ in y_ions]
            series.append(y_series)

        return series


# Global proteomics grammar instance
PROTEOMICS_GRAMMAR = MolecularGrammar(enable_neutral_losses=True)

#!/usr/bin/env python3
"""
Amino Acid Alphabet with S-Entropy Coordinates
==============================================

From st-stellas-molecular-language.tex Section 3.2:
Amino acids mapped to S-Entropy space based on physicochemical properties.

Each amino acid is characterized by:
- Hydrophobicity (S_knowledge dimension)
- Size/Mass (S_time dimension)
- Charge/Polarity (S_entropy dimension)

Author: Kundai Sachikonye
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from pathlib import Path
import sys

# Import S-Entropy core
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.EntropyTransformation import SEntropyCoordinates


@dataclass
class AminoAcid:
    """
    Single amino acid with physicochemical properties and S-Entropy coordinates.

    Attributes:
        symbol: Single letter code (e.g., 'A' for Alanine)
        name: Full name (e.g., 'Alanine')
        mass: Monoisotopic mass (Da)
        hydrophobicity: Kyte-Doolittle scale [-4.5, 4.5]
        volume: Van der Waals volume (Å³)
        charge: Charge state at pH 7 {-1, 0, +1}
        polarity: Polar (True) or nonpolar (False)
        s_entropy_coords: S-Entropy coordinates (s_k, s_t, s_e)
    """
    symbol: str
    name: str
    mass: float
    hydrophobicity: float  # Kyte-Doolittle
    volume: float  # Van der Waals volume
    charge: int  # At pH 7
    polarity: bool
    s_entropy_coords: Optional[SEntropyCoordinates] = None

    def __post_init__(self):
        """Compute S-Entropy coordinates from physicochemical properties."""
        if self.s_entropy_coords is None:
            # Map physicochemical properties to S-Entropy coordinates
            # S_knowledge: Information content ~ hydrophobicity (normalized to [0,1])
            s_k = (self.hydrophobicity + 4.5) / 9.0  # Kyte-Doolittle range [-4.5, 4.5]

            # S_time: Temporal/sequential ~ mass/volume (normalized)
            s_t = self.volume / 250.0  # Max volume ~228 Å³ for Trp

            # S_entropy: Disorder/complexity ~ charge magnitude + polarity
            s_e = (abs(self.charge) + int(self.polarity)) / 2.0  # [0, 1]

            self.s_entropy_coords = SEntropyCoordinates(
                s_knowledge=s_k,
                s_time=s_t,
                s_entropy=s_e
            )


@dataclass
class PostTranslationalModification:
    """
    Post-translational modification (PTM) with S-Entropy shift.

    PTMs are represented as Δ shifts in S-Entropy space.
    """
    name: str
    mass_shift: float  # Da
    target_residues: Tuple[str, ...]  # Which amino acids can be modified
    s_entropy_shift: Optional[np.ndarray] = None  # (Δs_k, Δs_t, Δs_e)

    def __post_init__(self):
        """Compute S-Entropy shift from mass and chemical change."""
        if self.s_entropy_shift is None:
            # S_knowledge shift ~ chemical information change
            delta_s_k = np.tanh(self.mass_shift / 100.0) * 0.2

            # S_time shift ~ mass change
            delta_s_t = self.mass_shift / 200.0  # Normalized shift

            # S_entropy shift ~ structural disorder introduced
            delta_s_e = 0.1 if self.mass_shift > 0 else -0.1

            self.s_entropy_shift = np.array([delta_s_k, delta_s_t, delta_s_e])


# Standard 20 amino acids with physicochemical properties
# Data from: Kyte-Doolittle hydrophobicity, Zamyatnin volumes
STANDARD_AMINO_ACIDS = {
    'A': AminoAcid('A', 'Alanine', 71.037, 1.8, 67, 0, False),
    'R': AminoAcid('R', 'Arginine', 156.101, -4.5, 148, 1, True),
    'N': AminoAcid('N', 'Asparagine', 114.043, -3.5, 96, 0, True),
    'D': AminoAcid('D', 'Aspartic acid', 115.027, -3.5, 91, -1, True),
    'C': AminoAcid('C', 'Cysteine', 103.009, 2.5, 86, 0, True),
    'Q': AminoAcid('Q', 'Glutamine', 128.059, -3.5, 114, 0, True),
    'E': AminoAcid('E', 'Glutamic acid', 129.043, -3.5, 109, -1, True),
    'G': AminoAcid('G', 'Glycine', 57.021, -0.4, 48, 0, False),
    'H': AminoAcid('H', 'Histidine', 137.059, -3.2, 118, 0, True),
    'I': AminoAcid('I', 'Isoleucine', 113.084, 4.5, 124, 0, False),
    'L': AminoAcid('L', 'Leucine', 113.084, 3.8, 124, 0, False),
    'K': AminoAcid('K', 'Lysine', 128.095, -3.9, 135, 1, True),
    'M': AminoAcid('M', 'Methionine', 131.040, 1.9, 124, 0, False),
    'F': AminoAcid('F', 'Phenylalanine', 147.068, 2.8, 135, 0, False),
    'P': AminoAcid('P', 'Proline', 97.053, -1.6, 90, 0, False),
    'S': AminoAcid('S', 'Serine', 87.032, -0.8, 73, 0, True),
    'T': AminoAcid('T', 'Threonine', 101.048, -0.7, 93, 0, True),
    'W': AminoAcid('W', 'Tryptophan', 186.079, -0.9, 163, 0, False),
    'Y': AminoAcid('Y', 'Tyrosine', 163.063, -1.3, 141, 0, True),
    'V': AminoAcid('V', 'Valine', 99.068, 4.2, 105, 0, False),
}

# Common PTMs
COMMON_PTMS = {
    'Oxidation': PostTranslationalModification(
        'Oxidation', 15.995, ('M', 'W', 'P')
    ),
    'Phosphorylation': PostTranslationalModification(
        'Phosphorylation', 79.966, ('S', 'T', 'Y')
    ),
    'Acetylation': PostTranslationalModification(
        'Acetylation', 42.011, ('K', 'S', 'T', 'Y')
    ),
    'Methylation': PostTranslationalModification(
        'Methylation', 14.016, ('K', 'R')
    ),
    'Carbamidomethyl': PostTranslationalModification(
        'Carbamidomethyl', 57.021, ('C',)
    ),
    'Deamidation': PostTranslationalModification(
        'Deamidation', 0.984, ('N', 'Q')
    ),
}


class AminoAcidAlphabet:
    """
    Complete amino acid alphabet with S-Entropy coordinate system.

    Provides:
    - Standard 20 amino acids
    - Common PTMs
    - Novel amino acid learning
    - S-Entropy coordinate lookup
    """

    def __init__(self):
        """Initialize with standard amino acids and PTMs."""
        self.amino_acids: Dict[str, AminoAcid] = STANDARD_AMINO_ACIDS.copy()
        self.ptms: Dict[str, PostTranslationalModification] = COMMON_PTMS.copy()

        # Novel amino acids discovered during analysis
        self.novel_amino_acids: Dict[str, AminoAcid] = {}

    def get_sentropy_coords(self, symbol: str) -> Optional[SEntropyCoordinates]:
        """Get S-Entropy coordinates for amino acid."""
        aa = self.amino_acids.get(symbol)
        if aa:
            return aa.s_entropy_coords

        # Check novel amino acids
        aa = self.novel_amino_acids.get(symbol)
        if aa:
            return aa.s_entropy_coords

        return None

    def add_novel_amino_acid(
        self,
        symbol: str,
        name: str,
        mass: float,
        s_entropy_coords: SEntropyCoordinates
    ):
        """
        Add a novel amino acid discovered via zero-shot identification.

        This is the "dynamic dictionary learning" from st-stellas-dictionary.tex.
        """
        # Create amino acid with provided S-Entropy coordinates
        aa = AminoAcid(
            symbol=symbol,
            name=name,
            mass=mass,
            hydrophobicity=0.0,  # Unknown, will be inferred
            volume=100.0,  # Default
            charge=0,
            polarity=False,
            s_entropy_coords=s_entropy_coords
        )

        self.novel_amino_acids[symbol] = aa
        print(f"[Alphabet] Learned novel amino acid: {symbol} ({name}) at S-Entropy {s_entropy_coords}")

    def get_mass(self, symbol: str) -> Optional[float]:
        """Get mass for amino acid."""
        aa = self.amino_acids.get(symbol) or self.novel_amino_acids.get(symbol)
        return aa.mass if aa else None

    def apply_ptm(
        self,
        base_coords: SEntropyCoordinates,
        ptm_name: str
    ) -> SEntropyCoordinates:
        """
        Apply PTM shift to base S-Entropy coordinates.

        PTM transforms: S' = S + ΔS_ptm
        """
        ptm = self.ptms.get(ptm_name)
        if not ptm:
            return base_coords

        base_array = base_coords.to_array()
        shifted = base_array + ptm.s_entropy_shift

        return SEntropyCoordinates(
            s_knowledge=float(shifted[0]),
            s_time=float(shifted[1]),
            s_entropy=float(shifted[2])
        )

    def get_all_symbols(self) -> list:
        """Get all amino acid symbols (standard + novel)."""
        return list(self.amino_acids.keys()) + list(self.novel_amino_acids.keys())

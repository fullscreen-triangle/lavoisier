#!/usr/bin/env python3
"""
St. Stella's Molecular Language
================================

Universal molecular notation and grammar for proteomics.
Extends the genomic coordinate system to amino acids and PTMs.

Author: Kundai Sachikonye
"""

from .amino_acid_alphabet import (
    AminoAcid,
    AminoAcidAlphabet,
    STANDARD_AMINO_ACIDS,
    COMMON_PTMS
)

from .fragmentation_grammar import (
    FragmentationRule,
    MolecularGrammar,
    PROTEOMICS_GRAMMAR
)

from .coordinate_mapping import (
    amino_acid_to_sentropy,
    ptm_to_sentropy,
    sequence_to_sentropy_path,
    calculate_sequence_entropy,
    calculate_sequence_complexity
)

__all__ = [
    'AminoAcid',
    'AminoAcidAlphabet',
    'STANDARD_AMINO_ACIDS',
    'COMMON_PTMS',
    'FragmentationRule',
    'MolecularGrammar',
    'PROTEOMICS_GRAMMAR',
    'amino_acid_to_sentropy',
    'ptm_to_sentropy',
    'sequence_to_sentropy_path',
    'calculate_sequence_entropy',
    'calculate_sequence_complexity'
]

#!/usr/bin/env python3
"""
St. Stella's S-Entropy Dictionary
==================================

Machine-readable dictionary for molecular entities defined by S-Entropy coordinates.
Enables zero-shot identification via nearest-neighbor lookup.

From st-stellas-dictionary.tex

Author: Kundai Sachikonye
"""

from .dictionary_entry import (
    DictionaryEntry,
    EquivalenceClass
)

from .sentropy_dictionary import (
    SEntropyDictionary,
    create_standard_proteomics_dictionary
)

from .zero_shot_identification import (
    ZeroShotIdentifier,
    IdentificationResult
)

__all__ = [
    'DictionaryEntry',
    'EquivalenceClass',
    'SEntropyDictionary',
    'create_standard_proteomics_dictionary',
    'ZeroShotIdentifier',
    'IdentificationResult'
]

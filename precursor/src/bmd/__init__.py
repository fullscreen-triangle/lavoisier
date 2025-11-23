"""
Biological Maxwell Demon (BMD) Module for Precursor

Implements hardware-constrained categorical completion through BMD operations.
BMDs are information catalysts that perform dual filtering:
  - Input filter: Select signal from noise based on phase-lock coherence
  - Output filter: Target physically-grounded interpretations

References:
- Mizraji, E. (2021). The biological Maxwell's demons: exploring ideas about
  the information processing in biological systems. Theory in Biosciences.
- Hardware-Constrained Categorical Completion (Sachikonye, 2025)
- Phase-Locked Molecular Ensembles as Information-Encoding Structures (Sachikonye, 2025)
"""

from .categorical_state import CategoricalState, CategoricalStateSpace
from .bmd_state import BMDState, OscillatoryHole, PhaseStructure
from .bmd_algebra import (
    compare_bmd_with_region,
    generate_bmd_from_comparison,
    compute_ambiguity,
    compute_stream_divergence,
    integrate_hierarchical
)
from .bmd_reference import BiologicalMaxwellDemonReference, HardwareBMDStream
from .sentropy_integration import (
    sentropy_to_categorical_state,
    categorical_state_to_bmd,
    spectrum_to_categorical_space,
    build_spectrum_bmd_network,
    compute_spectrum_ambiguity
)

__all__ = [
    # Categorical states
    'CategoricalState',
    'CategoricalStateSpace',

    # BMD states
    'BMDState',
    'OscillatoryHole',
    'PhaseStructure',

    # BMD algebra operations
    'compare_bmd_with_region',
    'generate_bmd_from_comparison',
    'compute_ambiguity',
    'compute_stream_divergence',
    'integrate_hierarchical',

    # Hardware BMD reference
    'BiologicalMaxwellDemonReference',
    'HardwareBMDStream',

    # S-Entropy integration
    'sentropy_to_categorical_state',
    'categorical_state_to_bmd',
    'spectrum_to_categorical_space',
    'build_spectrum_bmd_network',
    'compute_spectrum_ambiguity',
]

__version__ = '1.0.0'

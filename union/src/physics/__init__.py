# Physics module
"""
First-principles physics derivations for mass spectrometry.

Key concepts:
- Ionization as partition coordinate assignment
- CID as partition cascade with selection rules
- Spectroscopy derived from capacity formula C(n) = 2n^2

Selection rules for CID:
- Delta_l = +/- 1 (angular momentum change)
- Delta_m = 0, +/- 1 (orientation change)
- Delta_s = 0 (chirality preserved)

The periodic table structure emerges from C(n) = 2n^2:
- n=1: 2 elements (H, He)
- n=2: 8 elements (Li-Ne)
- n=3: 18 elements (Na-Ar + first transition metals)
- n=4: 32 elements (includes lanthanides)
"""

from .ionization_physics import (
    ESIModel,
    MALDIModel,
    EIModel,
    IonizationEngine
)

from .collision_induced_dissociation import (
    CIDEngine,
    PeptideCIDEngine,
    PartitionCascade,
    FragmentIon,
    CIDValidator
)

from .spectroscopy_derivation import (
    ChromatographyDerivation,
    MS1PeakDerivation,
    FragmentPeakDerivation,
    ElementDerivation,
    SpectroscopyValidator
)

__all__ = [
    # Ionization
    'ESIModel',
    'MALDIModel',
    'EIModel',
    'IonizationEngine',
    # CID
    'CIDEngine',
    'PeptideCIDEngine',
    'PartitionCascade',
    'FragmentIon',
    'CIDValidator',
    # Spectroscopy
    'ChromatographyDerivation',
    'MS1PeakDerivation',
    'FragmentPeakDerivation',
    'ElementDerivation',
    'SpectroscopyValidator'
]

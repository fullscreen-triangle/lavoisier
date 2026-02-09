# Entropy module
"""
S-entropy coordinates and thermodynamic transformations.

Key concepts:
- S-entropy coordinates: (S_k, S_t, S_e)
  - S_k: Knowledge entropy (information about state)
  - S_t: Temporal entropy (time evolution uncertainty)
  - S_e: Evolution entropy (trajectory complexity)

- Phase-lock networks for coherent computation
- Oscillatory computation paradigm
"""

from .EntropyTransformation import *
from .PhaseLockNetworks import *

# OscillatoryComputation depends on hardware module - optional import
try:
    from .OscillatoryComputation import *
    _OSCILLATORY_AVAILABLE = True
except ImportError:
    _OSCILLATORY_AVAILABLE = False

__all__ = [
    'EntropyTransformation',
    'PhaseLockNetworks',
]

if _OSCILLATORY_AVAILABLE:
    __all__.append('OscillatoryComputation')

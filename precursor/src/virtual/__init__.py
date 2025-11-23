"""
Virtual Mass Spectrometry Framework
====================================

Implements Molecular Maxwell Demon (MMD) based virtual instruments
for mass spectrometry. Based on St-Stellas categorical framework.

Core Modules:
- molecular_demon_state_architecture: MMD core classes and categorical states
- frequency_hierarchy: 8-scale hardware oscillation hierarchy
- finite_observers: Finite and transcendent observers for phase-lock detection
- mass_spec_ensemble: Virtual instrument ensemble orchestrator
- virtual_detector: Legacy virtual detector interface

Key Concepts:
1. MMD = Information catalyst filtering potential → actual states
2. Dual filtering: Input (noise) + Output (unphysical)
3. Categorical state = Compressed from ~10^6 equivalent molecular configurations
4. S-coordinates: (S_k, S_t, S_e) = Sufficient statistics
5. Recursive self-similarity: Each S-coordinate is itself an MMD
6. No simulation of intermediate stages (unknowable ion trajectories)
7. Read categorical states at convergence nodes (predetermined endpoints)

Author: Kundai Farai Sachikonye
Date: 2025
"""

# Core MMD architecture
from .molecular_demon_state_architecture import (
    MolecularMaxwellDemon,
    CategoricalState,
    OscillatoryHole,
    PhaseStructure,
    InstrumentProjection
)

# Frequency hierarchy
from .frequency_hierarchy import (
    FrequencyHierarchyTree,
    FrequencyHierarchyNode,
    HardwareScale
)

# Finite observers
from .finite_observers import (
    FiniteObserver,
    TranscendentObserver,
    PhaseLockSignature,
    ObserverType
)

# Mass spec ensemble
from .mass_spec_ensemble import (
    VirtualMassSpecEnsemble,
    VirtualInstrumentResult,
    MassSpecEnsembleResult
)

# Legacy virtual detector (compatible with existing code)
try:
    from .virtual_detector import (
        VirtualMassSpectrometer,
        VirtualIonDetector,
        VirtualPhotodetector,
        VirtualDetectorFactory
    )
    LEGACY_DETECTORS_AVAILABLE = True
except ImportError:
    LEGACY_DETECTORS_AVAILABLE = False

__all__ = [
    # MMD architecture
    'MolecularMaxwellDemon',
    'CategoricalState',
    'OscillatoryHole',
    'PhaseStructure',
    'InstrumentProjection',

    # Frequency hierarchy
    'FrequencyHierarchyTree',
    'FrequencyHierarchyNode',
    'HardwareScale',

    # Finite observers
    'FiniteObserver',
    'TranscendentObserver',
    'PhaseLockSignature',
    'ObserverType',

    # Ensemble
    'VirtualMassSpecEnsemble',
    'VirtualInstrumentResult',
    'MassSpecEnsembleResult',
]

# Add legacy detectors if available
if LEGACY_DETECTORS_AVAILABLE:
    __all__.extend([
        'VirtualMassSpectrometer',
        'VirtualIonDetector',
        'VirtualPhotodetector',
        'VirtualDetectorFactory'
    ])

__version__ = '0.1.0'
__author__ = 'Kundai Farai Sachikonye'

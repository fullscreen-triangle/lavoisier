"""
Instruments Module - Hardware-Based Virtual Instruments for Categorical Measurement

This module implements the virtual instrument suite based on the partition coordinate
framework. All instruments derive measurements from real hardware oscillator timing,
creating categorical states rather than simulating them.

Theoretical Foundation:
- Oscillation ≡ Category ≡ Partition (Fundamental Equivalence)
- S = k_B * M * ln(n) (Unified Entropy Formula)
- Every partition operation takes positive time τ_p > 0
- Categorical distance ≠ f(physical distance)

Instruments:
-----------
Partition Coordinate Instruments:
    - ShellResonator: Measures partition depth n
    - AngularAnalyser: Measures complexity l
    - OrientationMapper: Measures orientation m
    - ChiralityDiscriminator: Measures chirality s

Thermodynamic Instruments:
    - PartitionLagDetector: Measures undetermined residue entropy
    - HeatEntropyDecoupler: Demonstrates heat-entropy independence
    - CrossInstrumentConvergenceValidator: Validates equivalence theorem

Network Instruments:
    - PhaseLockNetworkMapper: Visualizes phase-lock topology
    - VibrationAnalyzer: Characterizes oscillatory coupling

Categorical Navigation Instruments:
    - CategoricalDistanceMeter: Measures S-space vs physical distance
    - NullGeodesicDetector: Identifies partition-free traversals
    - NonActualisationShellScanner: Maps non-actualisation geometry

Field Instruments:
    - NegationFieldMapper: Visualizes electric-like negation fields

Metabolomics Instruments:
    - FragmentationTopologyMapper: Maps MS2 as categorical completion
    - SEntropyMassSpectrometer: Virtual MS in S-coordinates
"""

from .base import (
    VirtualInstrument,
    HardwareOscillator,
    CategoricalState,
    SEntropyCoordinate,
    BOLTZMANN_CONSTANT
)

from .partition_coordinates import (
    ShellResonator,
    AngularAnalyser,
    OrientationMapper,
    ChiralityDiscriminator,
    PartitionCoordinateMeasurer
)

from .thermodynamic import (
    PartitionLagDetector,
    HeatEntropyDecoupler,
    CrossInstrumentConvergenceValidator
)

from .network import (
    PhaseLockNetworkMapper,
    VibrationAnalyzer
)

from .categorical_navigation import (
    CategoricalDistanceMeter,
    NullGeodesicDetector,
    NonActualisationShellScanner
)

from .field import (
    NegationFieldMapper
)

from .metabolomics import (
    FragmentationTopologyMapper,
    SEntropyMassSpectrometer
)

__all__ = [
    # Base
    'VirtualInstrument',
    'HardwareOscillator', 
    'CategoricalState',
    'SEntropyCoordinate',
    'BOLTZMANN_CONSTANT',
    # Partition Coordinates
    'ShellResonator',
    'AngularAnalyser',
    'OrientationMapper',
    'ChiralityDiscriminator',
    'PartitionCoordinateMeasurer',
    # Thermodynamic
    'PartitionLagDetector',
    'HeatEntropyDecoupler',
    'CrossInstrumentConvergenceValidator',
    # Network
    'PhaseLockNetworkMapper',
    'VibrationAnalyzer',
    # Categorical Navigation
    'CategoricalDistanceMeter',
    'NullGeodesicDetector',
    'NonActualisationShellScanner',
    # Field
    'NegationFieldMapper',
    # Metabolomics
    'FragmentationTopologyMapper',
    'SEntropyMassSpectrometer',
]


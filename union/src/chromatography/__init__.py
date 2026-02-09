# Chromatography module
"""
Implements chromatography as electric trap array computation.

Key concepts:
- Column = Electric trap array with partition wells
- Retention time = Partition lag tau_p
- Plate height = Partition spacing
- Resolution = Partition distinguishability

The Triple Equivalence Theorem shows:
Classical mechanics (diffusion) = Quantum mechanics (tunneling) = Partition coordinates
"""

from .transport_phenomena import (
    SEntropyCoordinate,
    PartitionCoordinates,
    ElectricTrap,
    ChromatographicTrapArray,
    PartitionLagCalculator,
    ChromatographicQuantumComputer,
    PlateTheoryValidator,
    compute_chromatographic_trajectory
)

__all__ = [
    'SEntropyCoordinate',
    'PartitionCoordinates',
    'ElectricTrap',
    'ChromatographicTrapArray',
    'PartitionLagCalculator',
    'ChromatographicQuantumComputer',
    'PlateTheoryValidator',
    'compute_chromatographic_trajectory'
]

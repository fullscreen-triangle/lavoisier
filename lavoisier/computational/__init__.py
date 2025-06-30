"""
Lavoisier Computational Module

Advanced computational methods for virtual molecular simulation,
hardware-assisted validation, and resonance-based detection.
"""

from .hardware_integration import HardwareHarvester, SystemOscillationProfiler
from .simulation import VirtualMolecularSimulator, MolecularResonanceEngine
from .resonance import ResonanceSpectrometer, HardwareMolecularValidator
from .noise_modeling import DynamicNoiseCharacterizer, OscillationPatternAnalyzer
from .optimization import TrajectoryOptimizer, ConvergenceAccelerator
from .prediction import PredictiveAnalytics, OptimalPathwayPredictor

__all__ = [
    'HardwareHarvester',
    'SystemOscillationProfiler',
    'VirtualMolecularSimulator',
    'MolecularResonanceEngine',
    'ResonanceSpectrometer',
    'HardwareMolecularValidator',
    'DynamicNoiseCharacterizer',
    'OscillationPatternAnalyzer',
    'TrajectoryOptimizer',
    'ConvergenceAccelerator',
    'PredictiveAnalytics',
    'OptimalPathwayPredictor'
]

"""
Precursor - Lavoisier Mass Spectrometry Analysis Framework
===========================================================

Hardware-Constrained Categorical Completion for Platform-Independent
Metabolomics and Proteomics.

Modules:
    - pipeline: Theatre-Stage-Process hierarchical observers
    - core: S-Entropy transformation, spectral readers, phase-lock networks
    - bmd: Biological Maxwell Demon components
    - hardware: Hardware oscillation harvesters
    - metabolomics: Metabolite identification and annotation
    - proteomics: Peptide/protein sequencing
    - analysis: Validation and quality control
    - utils: Utility functions

Author: Kundai Farai Sachikonye
"""

__version__ = "1.0.0"
__author__ = "Kundai Farai Sachikonye"

# Make key components easily accessible
try:
    from .pipeline.theatre import Theatre
    from .pipeline.stages import StageObserver, ProcessObserver
    from .bmd import (
        BiologicalMaxwellDemonReference,
        HardwareBMDStream,
        BMDState,
        CategoricalState
    )
    BMD_AVAILABLE = True
except ImportError:
    BMD_AVAILABLE = False

__all__ = [
    'Theatre',
    'StageObserver',
    'ProcessObserver',
    'BMD_AVAILABLE'
]

if BMD_AVAILABLE:
    __all__.extend([
        'BiologicalMaxwellDemonReference',
        'HardwareBMDStream',
        'BMDState',
        'CategoricalState'
    ])

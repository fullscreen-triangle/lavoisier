# Union src module
"""
Source modules for the Union of Two Crowns framework.

Submodules:
- chromatography: Transport phenomena and partition lag calculation
- physics: Ionization, CID, and spectroscopy from first principles
- virtual: Virtual detectors with multi-modal detection
- entropy: S-entropy transformations
- numerical: Numerical methods and data structures
- visual: Visualization and image processing
"""

from .pipeline_runner import PipelineRunner, run_pipeline

__all__ = ['PipelineRunner', 'run_pipeline']

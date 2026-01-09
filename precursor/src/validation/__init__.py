"""
Validation Module for Union of Two Crowns
==========================================

This module provides complete validation infrastructure for the
theoretical framework presented in the Union of Two Crowns paper.

Main Components:
- pipeline_3d_objects: Generate 3D objects at each pipeline stage
- batch_generate_3d_objects: Batch processing for multiple experiments
- visualize_3d_pipeline: Visualization tools for 3D objects
- run_validation: Main validation script

Usage:
    python -m precursor.src.validation.run_validation
"""

from .pipeline_3d_objects import (
    Pipeline3DObjectGenerator,
    Object3D,
    SEntropyCoordinate,
    ThermodynamicProperties,
    generate_pipeline_objects_for_experiment
)

from .batch_generate_3d_objects import Batch3DObjectGenerator

from .visualize_3d_pipeline import Pipeline3DVisualizer, visualize_experiment

__all__ = [
    'Pipeline3DObjectGenerator',
    'Object3D',
    'SEntropyCoordinate',
    'ThermodynamicProperties',
    'generate_pipeline_objects_for_experiment',
    'Batch3DObjectGenerator',
    'Pipeline3DVisualizer',
    'visualize_experiment',
]

__version__ = '1.0.0'


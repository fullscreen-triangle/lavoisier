#!/usr/bin/env python3
"""
Visualization module for Lavoisier Validation Framework
Integrates oscillatory.py and panel.py visualizations with actual validation results
"""

from .oscillatory import LavoisierVisualizationSuite
from .panel import generate_all_panels, print_instructions
from .validation_visualizer import ValidationVisualizationIntegrator

__all__ = [
    'LavoisierVisualizationSuite',
    'generate_all_panels', 
    'print_instructions',
    'ValidationVisualizationIntegrator'
]
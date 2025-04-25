"""
Lavoisier - High-performance mass spectrometry analysis with metacognitive orchestration
"""

__version__ = "0.1.0"
__author__ = "The Lavoisier Team"
__license__ = "MIT"

from lavoisier.core.config import LavoisierConfig, CONFIG
from lavoisier.core.metacognition import (
    Orchestrator, 
    PipelineType, 
    AnalysisStatus, 
    create_orchestrator
)

# Expose key components at package level
__all__ = [
    'LavoisierConfig',
    'CONFIG',
    'Orchestrator',
    'PipelineType',
    'AnalysisStatus',
    'create_orchestrator'
]

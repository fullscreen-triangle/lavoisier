"""
Lavoisier - High-performance mass spectrometry analysis with metacognitive orchestration
"""

__version__ = "0.1.0"
__author__ = "The Lavoisier Team"
__license__ = "MIT"

# Import from the correct modules to avoid circular imports
from lavoisier.core.config.config import GlobalConfig, CONFIG

# Expose key components at package level
__all__ = [
    'GlobalConfig',
    'CONFIG'
]

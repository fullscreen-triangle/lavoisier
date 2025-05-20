"""
Configuration utilities for Lavoisier
"""

# Import and re-export GlobalConfig and CONFIG from config.py
from lavoisier.core.config.config import GlobalConfig, CONFIG

# Make these available when importing from lavoisier.core.config
__all__ = ["GlobalConfig", "CONFIG"]

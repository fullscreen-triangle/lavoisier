"""
Derivation Module - First-Principles Ion Journey Validation
============================================================

Validates every theorem in the partition framework by tracing
individual ions through each stage of the mass spectrometry pipeline.
"""

from .ion_journey_validator import (
    IonJourneyValidator,
    IonInput,
    StageResult,
    JourneyResult,
)

__all__ = [
    "IonJourneyValidator",
    "IonInput",
    "StageResult",
    "JourneyResult",
]

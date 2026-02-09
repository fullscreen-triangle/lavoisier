# Virtual instruments module
"""
Virtual mass spectrometry instruments with physics-based detection.

Key concepts:
- DDA Linkage: MS1-MS2 linking via dda_event_idx (not retention time!)
- Multi-modal detection: 15 detection modes providing ~180 bits
- Categorical detector: Zero back-action measurement
- Differential image current: Perfect background subtraction

The critical insight is that dda_event_idx is the only reliable
way to link MS1 precursors to their MS2 fragmentation spectra.
"""

from .dda_linkage import (
    ScanMetadata,
    DDAEvent,
    MS1MS2Linkage,
    DDALinkageManager,
    CategoricalLinkageValidator
)

from .multimodal_detector import (
    ReferenceIon,
    DetectionResult,
    CompleteCharacterization,
    MultiModalDetector
)

from .detector_physics import (
    CategoricalDetector,
    DifferentialImageCurrentDetector,
    QuantumNonDemolitionDetector
)

__all__ = [
    # DDA Linkage
    'ScanMetadata',
    'DDAEvent',
    'MS1MS2Linkage',
    'DDALinkageManager',
    'CategoricalLinkageValidator',
    # Multi-modal detection
    'ReferenceIon',
    'DetectionResult',
    'CompleteCharacterization',
    'MultiModalDetector',
    # Detector physics
    'CategoricalDetector',
    'DifferentialImageCurrentDetector',
    'QuantumNonDemolitionDetector'
]

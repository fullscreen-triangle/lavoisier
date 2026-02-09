# Visual module
"""
Visualization and image processing for mass spectrometry data.

Key component: IonToDropletConverter
- Bijective transformation from ions to thermodynamic droplets
- Zero information loss - spectra can be fully recovered
- Encodes S-Entropy coordinates in droplet parameters
- Physics validation ensures plausible transformations
"""

# Core components - always available
from .PhysicsValidator import *

__all__ = ['PhysicsValidator']

# IonToDropletConverter - requires cv2 (optional)
try:
    from .IonToDropletConverter import (
        IonToDropletConverter,
        SEntropyCalculator,
        DropletMapper,
        ThermodynamicWaveGenerator,
        IonDroplet,
        SEntropyCoordinates,
        DropletParameters
    )
    __all__.extend([
        'IonToDropletConverter',
        'SEntropyCalculator',
        'DropletMapper',
        'ThermodynamicWaveGenerator',
        'IonDroplet',
        'SEntropyCoordinates',
        'DropletParameters'
    ])
    _ION_DROPLET_AVAILABLE = True
except ImportError as e:
    _ION_DROPLET_AVAILABLE = False
    import warnings
    warnings.warn(f"IonToDropletConverter not available: {e}")

# MSImageProcessor - requires h5py (optional)
try:
    from .MSImageProcessor import *
    from .MSImageDatabase_Enhanced import *
    __all__.extend(['MSImageProcessor', 'MSImageDatabase_Enhanced'])
    _MS_IMAGE_AVAILABLE = True
except ImportError as e:
    _MS_IMAGE_AVAILABLE = False

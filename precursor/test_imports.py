#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import sys
from pathlib import Path

# Add precursor root to path
precursor_root = Path(__file__).parent
sys.path.insert(0, str(precursor_root))

print("Testing imports...")
print("=" * 60)

# Test 1: Core imports
print("\n[1/6] Testing core imports...")
try:
    from src.core.SpectraReader import extract_mzml
    from src.core.EntropyTransformation import SEntropyTransformer
    from src.core.PhaseLockNetworks import PhaseLockNetwork
    print("  ✓ Core imports successful")
except ImportError as e:
    print(f"  ✗ Core import failed: {e}")
    sys.exit(1)

# Test 2: Pipeline imports
print("\n[2/6] Testing pipeline imports...")
try:
    from src.pipeline.theatre import Theatre, TheatreResult
    from src.pipeline.stages import StageObserver, ProcessObserver
    print("  ✓ Pipeline imports successful")
except ImportError as e:
    print(f"  ✗ Pipeline import failed: {e}")
    sys.exit(1)

# Test 3: BMD imports (optional)
print("\n[3/6] Testing BMD imports...")
try:
    from src.bmd import (
        BiologicalMaxwellDemonReference,
        HardwareBMDStream,
        BMDState,
        CategoricalState
    )
    print("  ✓ BMD imports successful")
    BMD_AVAILABLE = True
except ImportError as e:
    print(f"  ⚠ BMD imports failed (optional): {e}")
    BMD_AVAILABLE = False

# Test 4: Metabolomics pipeline import
print("\n[4/6] Testing metabolomics pipeline import...")
try:
    from src.pipeline.metabolomics import (
        MetabolomicsTheatre,
        run_metabolomics_analysis
    )
    print("  ✓ Metabolomics pipeline imports successful")
except ImportError as e:
    print(f"  ✗ Metabolomics pipeline import failed: {e}")
    sys.exit(1)

# Test 5: Check experimental files
print("\n[5/6] Checking experimental files...")
data_dir = precursor_root / "public" / "metabolomics"
files = [
    data_dir / "PL_Neg_Waters_qTOF.mzML",
    data_dir / "TG_Pos_Thermo_Orbi.mzML"
]

all_exist = True
for f in files:
    if f.exists():
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  ✓ {f.name} ({size_mb:.2f} MB)")
    else:
        print(f"  ✗ Missing: {f.name}")
        all_exist = False

if not all_exist:
    print("\n  ERROR: Some experimental files are missing!")
    sys.exit(1)

# Test 6: Check dependencies
print("\n[6/6] Checking key dependencies...")
dependencies = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scipy': 'scipy',
    'pymzml': 'pymzml',
    'sklearn': 'scikit-learn',
    'networkx': 'networkx'
}

missing = []
for module, package in dependencies.items():
    try:
        __import__(module)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} - MISSING")
        missing.append(package)

if missing:
    print(f"\n  ERROR: Missing dependencies: {', '.join(missing)}")
    print(f"  Install with: pip install {' '.join(missing)}")
    sys.exit(1)

# All tests passed
print("\n" + "=" * 60)
print("✓ All import tests passed!")
print("=" * 60)
print(f"\nBMD components available: {BMD_AVAILABLE}")
print("\nYou can now run:")
print("  python run_metabolomics_analysis.py")
print()

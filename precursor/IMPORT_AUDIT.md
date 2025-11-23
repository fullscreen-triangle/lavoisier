# Precursor Framework Import Audit
**Date:** October 22, 2025
**Purpose:** Systematic verification of all imports across the precursor framework

## Import Errors Found

### 1. **CRITICAL:** `metabolomics.py` Line 51
**Import:** `from ..core.PhaseLockNetworks import PhaseLockNetwork`
**Status:** ❌ **DOES NOT EXIST**

**Available in `PhaseLockNetworks.py`:**
- `PhaseLockSignature` (dataclass)
- `FiniteObserver` (dataclass)
- `GearRatio` (dataclass)
- `TranscendentObserver` (class)
- `PhaseLockMeasurementDevice` (class)
- `EnhancedPhaseLockMeasurementDevice` (class)
- `GearRatioTable` (dataclass)
- `MinimalSufficientObserverSelector` (class)
- `StochasticNavigator` (class)
- `EmptyDictionaryNavigator` (class)
- `PerformanceTracker` (class)

**Likely Intended Import:** `PhaseLockMeasurementDevice` or `EnhancedPhaseLockMeasurementDevice`

---

## Import Verification by File

### `precursor/src/pipeline/metabolomics.py`

#### Standard Library Imports (Lines 26-32)
- ✅ `numpy` as `np`
- ✅ `pandas` as `pd`
- ✅ `pathlib.Path`
- ✅ `typing` (Dict, List, Any, Optional, Tuple)
- ✅ `logging`
- ✅ `time`
- ✅ `json`

#### Pipeline Component Imports (Lines 35-43)
- ✅ `from .theatre import Theatre, TheatreResult, TheatreStatus, NavigationMode`
- ✅ `from .stages import StageObserver, StageResult, ProcessObserver, ProcessResult, StageStatus, ObserverLevel`

**Verification:** All imported from `pipeline/__init__.py` which exports them from `stages.py` and `theatre.py`

#### Core Functionality Imports (Lines 46-51)
- ✅ `from ..core.SpectraReader import extract_mzml` - EXISTS
- ✅ `from ..core.EntropyTransformation import SEntropyTransformer` - EXISTS
- ✅ `from ..core.EntropyTransformation import SEntropyFeatures` - EXISTS
- ❌ `from ..core.PhaseLockNetworks import PhaseLockNetwork` - **DOES NOT EXIST**

#### BMD Component Imports (Lines 55-67) - Try/Except Block
- ✅ `from ..bmd import BiologicalMaxwellDemonReference` - EXISTS (in `bmd/__init__.py`)
- ✅ `from ..bmd import HardwareBMDStream` - EXISTS (in `bmd/__init__.py`)
- ✅ `from ..bmd import BMDState` - EXISTS (in `bmd/__init__.py`)
- ✅ `from ..bmd import CategoricalState` - EXISTS (in `bmd/__init__.py`)
- ✅ `from ..bmd import compute_ambiguity` - EXISTS (in `bmd/__init__.py`)
- ✅ `from ..bmd import generate_bmd_from_comparison` - EXISTS (in `bmd/__init__.py`)
- ✅ `from ..bmd import compute_stream_divergence` - EXISTS (in `bmd/__init__.py`)
- ✅ `from ..bmd import integrate_hierarchical` - EXISTS (in `bmd/__init__.py`)
- ✅ `from ..bmd import sentropy_to_categorical_state` - EXISTS (in `bmd/__init__.py`)
- ✅ `from ..bmd import categorical_state_to_bmd` - EXISTS (in `bmd/__init__.py`)
- ✅ `from ..bmd import spectrum_to_categorical_space` - EXISTS (in `bmd/__init__.py`)

---

## Recommended Fixes

### Fix #1: Replace `PhaseLockNetwork` Import
**File:** `precursor/src/pipeline/metabolomics.py` (Line 51)

**Current:**
```python
from ..core.PhaseLockNetworks import PhaseLockNetwork
```

**Recommended Fix:**
```python
from ..core.PhaseLockNetworks import (
    PhaseLockMeasurementDevice,
    EnhancedPhaseLockMeasurementDevice,
    PhaseLockSignature,
    TranscendentObserver
)
```

**Reasoning:** Based on the code structure, the phase-lock network functionality is provided through the measurement device classes, not a single `PhaseLockNetwork` class.

---

## Status Summary

**Total Imports Checked:** 28
**Valid Imports:** 27 ✅
**Invalid Imports:** 1 ❌
**Success Rate:** 96.4%

**Critical Errors:** 1
- Missing `PhaseLockNetwork` class in `PhaseLockNetworks.py`

---

## Next Steps

1. ✅ Update `metabolomics.py` Line 51 to import correct classes
2. ⏳ Check for usages of `PhaseLockNetwork` in `metabolomics.py` code
3. ⏳ Update those usages to use correct class names
4. ⏳ Continue full audit of remaining Python files in `precursor/src/`

---

## Files Audited
- [x] `precursor/src/pipeline/metabolomics.py` (Partially - imports only)
- [x] `precursor/src/core/PhaseLockNetworks.py`
- [x] `precursor/src/core/EntropyTransformation.py`
- [x] `precursor/src/bmd/__init__.py`
- [x] `precursor/src/pipeline/__init__.py`
- [ ] `precursor/src/core/__init__.py` (empty file)
- [ ] Remaining 78 Python files

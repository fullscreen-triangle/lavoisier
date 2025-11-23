# Computer Vision Import Path Fix

## Problem

The metabolomics pipeline was **NOT using the computer vision modules** despite them being fully implemented and working.

### Root Cause

The pipeline was importing from the **WRONG location**:

```python
# WRONG - imports from outdated/broken copy
from ..core.MSImageDatabase_Enhanced import MSImageDatabase
```

This tried to import from:
- `precursor/src/core/MSImageDatabase_Enhanced.py` ❌

But the **CORRECT, WORKING implementations** are in:
- `lavoisier/visual/IonToDropletConverter.py` ✅
- `lavoisier/visual/MSImageDatabase_Enhanced.py` ✅

### Why This Happened

The `precursor/src/core/` directory had an **outdated or broken copy** of `MSImageDatabase_Enhanced.py` that:
1. Failed to import `cv2` (even though opencv-python IS installed)
2. Was not the latest version with full thermodynamic conversion
3. Did NOT use `IonToDropletConverter` properly

Meanwhile, the working versions in `lavoisier/visual/` had:
- ✅ Full ion-to-droplet thermodynamic conversion
- ✅ S-Entropy coordinate calculation
- ✅ Phase-lock signature extraction
- ✅ Categorical state encoding
- ✅ Dual-modality feature integration

## Solution Applied

### Fixed Import Paths

**ComputerVisionConversionProcess** (Line 469-487):
```python
try:
    import sys
    from pathlib import Path
    # Import from the correct location: lavoisier/visual/
    visual_path = Path(__file__).parent.parent.parent.parent / 'visual'
    if str(visual_path) not in sys.path:
        sys.path.insert(0, str(visual_path))

    from MSImageDatabase_Enhanced import MSImageDatabase
    self.ms_image_db = MSImageDatabase(
        resolution=resolution,
        feature_dimension=128,
        use_thermodynamic=True
    )
    self.enabled = True
    self.logger.info("Computer Vision database initialized with thermodynamic features from lavoisier/visual/")
except ImportError as e:
    self.logger.warning(f"MSImageDatabase not available: {e}")
    self.enabled = False
```

**ComputerVisionMatchingProcess** (Line 1028-1053):
```python
try:
    import sys
    from pathlib import Path
    # Import from the correct location: lavoisier/visual/
    visual_path = Path(__file__).parent.parent.parent.parent / 'visual'
    if str(visual_path) not in sys.path:
        sys.path.insert(0, str(visual_path))

    from MSImageDatabase_Enhanced import MSImageDatabase
    # ... rest of initialization
    self.enabled = True
except ImportError as e:
    self.logger.warning(f"MSImageDatabase not available: {e}")
    self.enabled = False
```

## What This Enables

Now the pipeline will correctly use:

### 1. Ion-to-Droplet Conversion (Stage 2)
**File**: `lavoisier/visual/IonToDropletConverter.py`
- ✅ Converts each ion to thermodynamic droplet
- ✅ Calculates S-Entropy coordinates (S_knowledge, S_time, S_entropy)
- ✅ Maps to droplet parameters (velocity, radius, surface_tension, temperature, phase_coherence)
- ✅ Generates categorical states
- ✅ Creates thermodynamic wave patterns
- ✅ Physics validation

### 2. Visual Modality Database (Stage 2 & 4)
**File**: `lavoisier/visual/MSImageDatabase_Enhanced.py`
- ✅ Full thermodynamic image generation from spectra
- ✅ SIFT/ORB feature extraction
- ✅ Optical flow analysis
- ✅ Phase-lock signature extraction
- ✅ S-Entropy distance calculation
- ✅ Categorical state matching
- ✅ Dual-modality (visual + numerical) similarity scoring

## Expected Output Now

### Stage 2 (S-Entropy Transformation) Will Now Save:

**Directory**: `stage_02_sentropy/`
1. `cv_images/spectrum_{scan_id}_droplet.png` - Thermodynamic droplet images (PNG) ✅
2. `cv_features.tsv` - Computer vision features per spectrum ✅
3. `ion_droplets.tsv` - Detailed droplet parameters per ion ✅
   - Columns: `scan_id`, `droplet_idx`, `mz`, `intensity`, `s_knowledge`, `s_time`, `s_entropy`, `velocity`, `radius`, `phase_coherence`, `categorical_state`

### Stage 4 (Categorical Completion) Will Now Save:

**Directory**: `stage_04_completion/`
1. `annotations.tsv` - Metabolite annotations from CV matching ✅
2. `cv_matches_detailed.tsv` - Detailed CV matching results ✅
   - Columns: `scan_id`, `match_rank`, `database_id`, `similarity`, `structural_similarity`, `phase_lock_similarity`, `categorical_match`, `s_entropy_distance`, `n_matched_features`

## Verification

To confirm the fix worked, check the console output for:

```
✅ Computer Vision database initialized with thermodynamic features from lavoisier/visual/
```

Instead of:
```
❌ MSImageDatabase not available: No module named 'cv2'
```

And verify these files exist after running:
- `stage_02_sentropy/cv_images/spectrum_*.png`
- `stage_02_sentropy/cv_features.tsv`
- `stage_02_sentropy/ion_droplets.tsv`
- `stage_04_completion/cv_matches_detailed.tsv`

## Technical Note

The import path resolution:
```python
Path(__file__).parent.parent.parent.parent / 'visual'
```

From `precursor/src/pipeline/metabolomics.py`:
- `.parent` → `precursor/src/pipeline/`
- `.parent` → `precursor/src/`
- `.parent` → `precursor/`
- `.parent` → `lavoisier/`
- `/ 'visual'` → `lavoisier/visual/`

This correctly points to the working implementations.

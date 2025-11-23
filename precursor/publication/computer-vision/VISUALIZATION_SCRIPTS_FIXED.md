# Visualization Scripts - GridSpec SubplotSpec Fix

## Summary
Fixed `TypeError: 'SubplotSpec' object is not subscriptable` errors across all visualization scripts and optimized memory usage for `droplet_analysis.py` and `feature_extraction.py`.

## Issue
When passing a GridSpec row slice like `gs[0, :]` to plotting methods, the methods tried to subscript it again (e.g., `gs[0]`), causing a TypeError because SubplotSpec objects cannot be subscripted directly.

## Solution
Modified all plotting methods that receive a row/column slice from GridSpec to create a sub-grid using `.subgridspec()` before accessing individual panels.

### Pattern Applied:
```python
# OLD (broken):
def _plot_method(self, fig, gs):
    ax1 = fig.add_subplot(gs[0])  # Error: SubplotSpec not subscriptable
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

# NEW (working):
def _plot_method(self, fig, gs):
    # Create a 1x3 sub-grid within the passed SubplotSpec
    gs_sub = gs.subgridspec(1, 3, wspace=0.3)

    ax1 = fig.add_subplot(gs_sub[0])  # Works!
    ax2 = fig.add_subplot(gs_sub[1])
    ax3 = fig.add_subplot(gs_sub[2])
```

## Critical Fixes Applied

### Pickle Error in `feature_extraction.py` ✅
**Issue:** `TypeError: cannot pickle 'cv2.KeyPoint' object`

**Root Cause:** OpenCV KeyPoint objects (from SIFT, ORB, AKAZE) cannot be directly pickled.

**Solution:** Convert KeyPoint objects to serializable dictionaries before saving:
```python
# Convert KeyPoint to dict with all attributes
{
    'pt': kp.pt,           # (x, y) position
    'size': kp.size,       # diameter
    'angle': kp.angle,     # orientation
    'response': kp.response,  # strength
    'octave': kp.octave,   # pyramid level
    'class_id': kp.class_id  # cluster ID
}
```

This preserves all KeyPoint information in a format that can be pickled and later reconstructed if needed.

---

## Files Modified

### 1. `complementarity.py`
**Fixed Methods:**
- `_plot_confidence_comparison()` - Row 1 (3 panels: A, B, C)
- `_plot_scenario_performance()` - Row 2 (3 panels: D, E, F)
- `_plot_feature_space()` - Row 3 (3 panels: G, H, I)
- `_plot_complementarity_metrics()` - Row 4 (3 panels: J, K, L)

**Total:** 12 subplot fixes

### 2. `complexity_scaling.py`
**Fixed Methods:**
- `_plot_count_distributions()` - Row 1 (3 panels: A, B, C)
- `_plot_complexity_metrics()` - Row 5 (3 panels: M, N, O)
- `_plot_scaling_relationships()` - Row 6 (3 panels: P, Q, R)

**Note:** Rows 2-4 (`_plot_intensity_distributions`, `_plot_cv_sentropy_distributions`, `_plot_cv_physical_distributions`) were already correct as they iterate over individual grid cells.

**Total:** 9 subplot fixes

### 3. `individual_spectrum.py`
**Status:** Already correct - uses individual grid cells (e.g., `gs[0, 0]`, `gs[0, 1]`) rather than row/column slices.

**No changes needed**

### 4. `droplet_analysis.py`
**Memory Optimization:**
- Modified `_detect_first_spectrum()` to prefer smaller spectra (100-104)
- Explicitly avoids spectrum 105 (65,870 droplets causing memory crashes)
- Changed default fallback from 105 to 100

**Memory Impact:**
- Old default: Spectrum 105 (65,870 droplets) → ~Memory crash
- New default: Spectrum 100 (862 droplets) → ~76x less data

### 5. `feature_extraction.py`
**Memory Optimization:**
- Modified `_detect_first_spectrum()` to prefer smaller spectra (100-104)
- Explicitly avoids spectrum 105
- Changed default fallback from 105 to 100

**Pickle Fix:**
- Converts cv2.KeyPoint objects to serializable dictionaries before saving
- Preserves all KeyPoint attributes (pt, size, angle, response, octave, class_id)
- Fixes `TypeError: cannot pickle 'cv2.KeyPoint' object`

**Memory Impact:** Same as `droplet_analysis.py`

## Testing Results

### Before Fix:
```
TypeError: 'SubplotSpec' object is not subscriptable
  File "complementarity.py", line 284, in _plot_confidence_comparison
    ax1 = fig.add_subplot(gs[0])
```

### After Fix:
✅ All scripts should now run without SubplotSpec errors
✅ Memory-intensive scripts default to smaller, manageable spectra

## Data Organization
All scripts correctly reference the nested data structure:
```
precursor/publication/computer-vision/data/
├── numerical/               # Raw spectra (spectrum_*.tsv)
├── entropy/                # S-Entropy data
└── vision/
    ├── droplets/           # Droplet parameters (spectrum_*_droplets.tsv)
    └── images/             # Droplet images (spectrum_*_droplet.png)
```

## Matplotlib Backend
All scripts use the non-interactive 'Agg' backend for figure generation:
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
```

This avoids TclError issues on systems without proper Tcl/Tk installation.

## Running the Scripts

All scripts can now be run without arguments:
```bash
cd precursor/publication/computer-vision/visualisations

python complementarity.py
python complexity_scaling.py
python individual_spectrum.py
python droplet_analysis.py
python feature_extraction.py
```

They will:
1. Auto-detect available data in `../data/`
2. Process all relevant spectra (or prefer smaller ones for memory-intensive scripts)
3. Generate publication-quality PNG and PDF figures
4. Save outputs in the current directory

## Output Files Generated

- `complementarity_analysis.png` / `.pdf`
- `complexity_scaling_comprehensive.png` / `.pdf`
- `spectrum_<ID>_deepdive.png` / `.pdf`
- `spectrum_<ID>_transformation_pipeline.png` / `.pdf`
- `spectrum_<ID>_cv_features.png` / `.pdf`
- `spectrum_<ID>_features.pkl` (feature extraction cache)

---

**Date:** 2025-10-27
**Status:** All visualization scripts operational
**Memory:** Optimized for systems with limited RAM

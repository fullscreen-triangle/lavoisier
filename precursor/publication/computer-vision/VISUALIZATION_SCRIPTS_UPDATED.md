# Visualization Scripts Update Summary

## Changes Made

All five visualization scripts have been updated to remove `argparse` dependency and directly reference the data folder, similar to how `run_cv_only.py` works.

**IMPORTANT FIXES (2025-10-27):**
1. Updated all scripts to use the correct subdirectory structure:
   - `data/numerical/` for raw spectra
   - `data/vision/droplets/` for droplet TSV files
   - `data/vision/images/` for droplet PNG images

   The scripts were initially looking for files directly in `data/`, but the actual folder structure uses organized subdirectories.

2. **Fixed matplotlib backend issue**: Added `matplotlib.use('Agg')` before importing pyplot in all scripts to use non-interactive backend. This prevents TclError on systems without Tcl/Tk and allows scripts to save figures without trying to display them interactively.

## Updated Scripts

### 1. `complementarity.py` ✅
**Changes:**
- Added `from pathlib import Path` import
- Updated `ComplementarityAnalyzer.__init__()`:
  - Removed `spectra_ids` parameter
  - Added `self.data_dir = Path(__file__).parent.parent / 'data'`
  - Added `_detect_spectra()` method to auto-detect available spectra from data folder
- Updated `load_data()`:
  - Changed file paths to use `self.data_dir / filename`
  - Updated error messages to be more descriptive

**Usage:**
```python
# Before (with args):
analyzer = ComplementarityAnalyzer(spectra_ids=[100, 101, 102])

# After (auto-detect):
analyzer = ComplementarityAnalyzer()  # Auto-detects from data folder
```

---

### 2. `complexity_scaling.py` ✅
**Changes:**
- Added `from pathlib import Path` import
- Updated `ComplexityScalingAnalyzer.__init__()`:
  - Removed `spectra_ids` parameter
  - Added `self.data_dir = Path(__file__).parent.parent / 'data'`
  - Added `_detect_spectra()` method to auto-detect available spectra
- Updated `load_data()`:
  - Changed file paths to use `self.data_dir / filename`
  - Updated error messages

**Usage:**
```python
# Before (with args):
analyzer = ComplexityScalingAnalyzer(spectra_ids=[100, 101, 102])

# After (auto-detect):
analyzer = ComplexityScalingAnalyzer()  # Auto-detects from data folder
```

---

### 3. `individual_spectrum.py` ✅
**Changes:**
- Added `from pathlib import Path` import
- Updated `SpectrumDeepDive.__init__()`:
  - Added `self.data_dir = Path(__file__).parent.parent / 'data'`
- Updated `load_data()`:
  - Changed file paths to use `self.data_dir / filename`
  - Updated error messages to include data directory path
- Updated `main()`:
  - Removed hardcoded `spectra_ids` list
  - Added auto-detection logic to find all available spectra in data folder
  - Added informative print statements showing data directory and found spectra

**Usage:**
```python
# Before (manual list):
spectra_ids = [100, 101, 102, 103, 104, 105]

# After (auto-detect):
# Automatically finds all spectrum_*_droplets.tsv files in data folder
```

---

### 4. `droplet_analysis.py` ✅
**Changes:**
- Added `from pathlib import Path` import
- Updated `CVMethodVisualizer.__init__()`:
  - Changed `spectrum_id` parameter to optional (default `None`)
  - Added `self.data_dir = Path(__file__).parent.parent / 'data'`
  - Added `_detect_first_spectrum()` method to auto-detect first available spectrum
  - Added informative print statements
- Updated `load_data()`:
  - Changed all file paths to use `self.data_dir / filename`
  - Updated error messages to include data directory path
- Updated `main()`:
  - Removed `argparse` import and argument parsing
  - Simplified to just create visualizer and run

**Usage:**
```python
# Before (with argparse):
python droplet_analysis.py --spectrum 105

# After (auto-detect):
python droplet_analysis.py  # Uses first available spectrum in data folder
```

---

### 5. `feature_extraction.py` ✅
**Changes:**
- Added `from pathlib import Path` import (matplotlib was already imported)
- Updated `CVFeatureExtractor.__init__()`:
  - Changed `spectrum_id` parameter to optional (default `None`)
  - Added `self.data_dir = Path(__file__).parent.parent / 'data'`
  - Added `_detect_first_spectrum()` method to auto-detect first available spectrum
  - Added informative print statements
- Updated `load_data()`:
  - Changed all file paths to use `self.data_dir / filename`
  - Updated error messages to include data directory path
- Updated `main()`:
  - Removed `argparse` import and all argument parsing
  - Simplified to just create extractor and run
  - Always saves features to pickle file (removed `--save-features` flag)

**Usage:**
```python
# Before (with argparse):
python feature_extraction.py --spectrum 105 --save-features

# After (auto-detect, always saves):
python feature_extraction.py  # Uses first available spectrum, always saves features
```

---

## Data Folder Structure

All scripts now expect data files in the following organized structure:
```
precursor/publication/computer-vision/data/
├── entropy/
│   ├── spectrum_100.tsv
│   ├── spectrum_101.tsv
│   └── ...
├── numerical/
│   ├── spectrum_100.tsv
│   ├── spectrum_101.tsv
│   └── ...
└── vision/
    ├── droplets/
    │   ├── spectrum_100_droplets.tsv
    │   ├── spectrum_101_droplets.tsv
    │   └── ...
    └── images/
        ├── spectrum_100_droplet.png
        ├── spectrum_101_droplet.png
        └── ...
```

**Key directories:**
- `data/numerical/` - Raw numerical spectra (for complementarity, complexity_scaling, individual_spectrum, droplet_analysis)
- `data/vision/droplets/` - Droplet TSV files (used by all scripts for auto-detection)
- `data/vision/images/` - Droplet PNG images (for droplet_analysis and feature_extraction)
- `data/entropy/` - S-Entropy spectra (currently not used by visualization scripts)

## Auto-Detection Logic

All scripts now include auto-detection logic that searches in the correct subdirectories:

```python
def _detect_spectra(self):
    """Auto-detect available spectrum IDs from data directory"""
    spectra_ids = set()
    droplets_dir = self.data_dir / 'vision' / 'droplets'
    if droplets_dir.exists():
        for file in droplets_dir.glob('spectrum_*_droplets.tsv'):
            # Extract spectrum ID from filename
            spec_id = int(file.stem.split('_')[1])
            spectra_ids.add(spec_id)
    return sorted(list(spectra_ids))
```

This searches for all files matching the pattern `spectrum_*_droplets.tsv` in the `data/vision/droplets/` folder and extracts the spectrum IDs.

## Running the Scripts

All scripts can now be run without any command-line arguments:

```bash
cd precursor/publication/computer-vision/visualisations

# Run complementarity analysis
python complementarity.py

# Run complexity scaling analysis
python complexity_scaling.py

# Run individual spectrum deep dive (all spectra)
python individual_spectrum.py

# Run droplet analysis (first spectrum)
python droplet_analysis.py

# Run feature extraction (first spectrum, saves features)
python feature_extraction.py
```

## Benefits

1. **No manual configuration**: Scripts automatically find and process all available data
2. **Consistent with project style**: Matches the pattern used in `run_cv_only.py`
3. **More robust**: Less prone to user error from incorrect command-line arguments
4. **Better error messages**: Shows data directory path when files are not found
5. **Cleaner code**: Removed argparse boilerplate

## Notes

- All scripts print the data directory path and detected spectra on startup
- If no spectra are found, scripts will gracefully handle the empty case
- For `droplet_analysis.py` and `feature_extraction.py`, you can still manually specify a spectrum ID:
  ```python
  visualizer = CVMethodVisualizer(spectrum_id=105)
  extractor = CVFeatureExtractor(spectrum_id=105)
  ```
- `feature_extraction.py` now always saves features to a pickle file (no flag needed)

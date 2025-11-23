# Critical Bugs Fixed - Lavoisier Pipeline

## Date: 2025-10-24

This document lists the critical bugs identified from the console output and the fixes applied.

## Bug 1: `'ms_level'` KeyError in SpectralAcquisitionProcess
**Location**: `precursor/src/pipeline/metabolomics.py` lines 134-136
**Error**: `KeyError: 'ms_level'`
**Root Cause**: The code tried to access `scan_info_df['ms_level']` but that column doesn't exist in the DataFrame returned by `extract_mzml()`. The actual column structure is: `["dda_event_idx", "spec_index", "scan_time", "DDA_rank", "scan_number", "MS2_PR_mz"]`

**Fix Applied**:
```python
# BEFORE:
n_ms1 = len(scan_info_df[scan_info_df['ms_level'] == 1])
n_ms2 = len(scan_info_df[scan_info_df['ms_level'] == 2])

# AFTER:
# MS1 scans have DDA_rank == 0, MS2 scans have DDA_rank > 0
n_ms1 = len(scan_info_df[scan_info_df['DDA_rank'] == 0])
n_ms2 = len(scan_info_df[scan_info_df['DDA_rank'] > 0])
```

## Bug 2: `'WindowsPath' object is not subscriptable` in SEntropyTransformProcess
**Location**: `precursor/src/pipeline/metabolomics.py` line 397 (now 401)
**Error**: `TypeError: 'WindowsPath' object is not subscriptable`
**Root Cause**: When Stage 1 fails, it returns `output_data=None`. The Theatre then passes the original input (the mzML file path, a WindowsPath object) to Stage 2. Stage 2's `SEntropyTransformProcess` tries to access `input_data['filtered_spectra']`, which fails because `input_data` is a Path, not a dict.

**Fix Applied**:
```python
# Added input validation at line 397-399:
# Validate input
if not isinstance(input_data, dict):
    raise ValueError(f"Expected dict input, got {type(input_data)}")
```

## Bug 3: `argument of type 'WindowsPath' is not iterable` in `_save_ms_data`
**Location**: `precursor/src/pipeline/stages.py` line 622-659
**Error**: `TypeError: argument of type 'WindowsPath' is not iterable`
**Root Cause**: When stages fail, they might have a WindowsPath in `output_data` instead of a dict. The `_save_ms_data` method tried to check `'spectra' in data` without first validating that `data` is a dictionary.

**Fix Applied**:
```python
# Added type check at line 631-635:
data = self.result.output_data

# Safety check: data must be a dictionary
if not isinstance(data, dict):
    return
```

## Impact of Fixes

### Stage 1 (Spectral Preprocessing)
**Before**: Failed with `KeyError: 'ms_level'` after successfully reading spectra
**After**: Should complete successfully, correctly counting MS1/MS2 scans using `DDA_rank`

### Stage 2 (S-Entropy Transformation)
**Before**: Failed immediately with `'WindowsPath' object is not subscriptable'` when Stage 1 failed
**After**: Will fail gracefully with a clear error message if it receives invalid input type

### Stage Saving
**Before**: Crashed with `argument of type 'WindowsPath' is not iterable` when trying to save failed stage results
**After**: Safely checks data type and skips MS data saving if not a dictionary

## Next Steps

1. **Run the pipeline** to verify Stage 1 now completes successfully
2. **Install opencv-python** to enable computer vision components: `pip install opencv-python`
3. **Verify data saving** - check that actual spectral data, CV images, and annotations are saved
4. **Check remaining process observers** for similar input validation issues in:
   - `CategoricalStateMappingProcess` (line 571)
   - `StreamCoherenceCheckProcess` (line 740)
   - `OscillatoryHoleIdentificationProcess` (line 854)
   - `DatabaseSearchProcess` (line 925)

## Technical Notes

### Data Flow in Theatre
When a stage fails:
1. The stage returns a `StageResult` with `output_data=None` or `output_data=original_input`
2. Theatre's `_execute_dependency` checks `if result.output_data is not None:`
3. If None, `current_data` remains as the ORIGINAL input (the mzML Path)
4. This Path gets passed to the next stage, causing subscript errors

### Proper Solution
All `ProcessObserver.observe()` methods should:
1. Validate `input_data` type before accessing keys
2. Return a meaningful error with clear message
3. Allow the Theatre to continue (log the error but don't crash the entire pipeline)

This enables better debugging and allows later stages to run even if earlier ones fail (where appropriate).

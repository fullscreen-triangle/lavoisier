# RUNTIME BUGS - PRECURSOR METABOLOMICS PIPELINE

## Bug #1: DataFrame.append() Deprecated
**File:** `precursor/src/core/SpectraReader.py`
**Line:** 187, 374, 387, 477
**Error:** `'DataFrame' object has no attribute 'append'`
**Cause:** Pandas 2.0+ removed `df.append()` method
**Fix:** Use `pd.concat([ms1_xic_df, _tmp_spec_df])` instead

---

## Bug #2: WindowsPath Not Subscriptable
**File:** `precursor/src/pipeline/metabolomics.py`
**Lines:** 335, 406, 689
**Error:** `'WindowsPath' object is not subscriptable`
**Cause:** Input is Path object but code tries `input_data['filtered_spectra']`
**Root Cause:** First process returns data dict, but theatre passes original Path to all stages
**Fix:** Check if input is Path, if so load from first stage result instead

---

## Bug #3: HardwareBMDStream.coherence Doesn't Exist
**File:** `precursor/src/pipeline/metabolomics.py`
**Lines:** 522, 535
**Error:** `'HardwareBMDStream' object has no attribute 'coherence'`
**Actual Attribute:** `phase_lock_quality`
**Fix:** Change `hardware_stream.coherence` to `hardware_stream.phase_lock_quality`

---

## Bug #4: Result Dict vs Object Access
**File:** `precursor/run_metabolomics_analysis.py`
**Line:** 165
**Error:** `'dict' object has no attribute 'metrics'`
**Cause:** `result.stage_results` returns dict, need `.get()` not attribute access
**Fix:** Use `.get('bmd_grounding', {}).get('metrics', {})` instead

---

## FIXING ORDER:

1. Fix SpectraReader DataFrame.append (critical - blocks stage 1)
2. Fix WindowsPath subscripting (blocks stages 2, 4)
3. Fix HardwareBMDStream.coherence (blocks stage 3)
4. Fix result access in run script

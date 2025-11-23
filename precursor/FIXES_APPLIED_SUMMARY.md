# RUNTIME FIXES APPLIED

## ✅ Fixed Bugs:

### 1. **DataFrame.append() → pd.concat()**
**File:** `precursor/src/core/SpectraReader.py` Line 187
**Fixed:** Changed deprecated `.append()` to `pd.concat()`
```python
# Before:
ms1_xic_df = ms1_xic_df.append(_tmp_spec_df)

# After:
ms1_xic_df = pd.concat([ms1_xic_df, _tmp_spec_df], ignore_index=True)
```

### 2. **HardwareBMDStream.coherence → phase_lock_quality**
**File:** `precursor/src/pipeline/metabolomics.py` Lines 522, 535
**Fixed:** Correct attribute name
```python
# Before:
hardware_stream.coherence

# After:
hardware_stream.phase_lock_quality
```

### 3. **device_bmds not component_bmds**
**File:** `precursor/src/pipeline/metabolomics.py` Line 523
**Fixed:** Correct dict name
```python
# Before:
len(hardware_stream.component_bmds)

# After:
len(hardware_stream.device_bmds)
```

---

## ⚠️ Remaining Issue: Data Flow

**Problem:** Stage 1 still failing would cause Stage 2 to receive Path object instead of dict.

**Root Cause:** If Stage 1's first process fails, it returns `output_data=None`, so Theatre passes original input to Stage 2.

**Verification Needed:**
1. Ensure SpectraReader fix is loaded
2. Check if Stage 1 completes successfully
3. If Stage 1 succeeds, data will flow correctly to Stage 2

---

## 📋 To Verify Fixes Work:

```bash
cd precursor
python run_metabolomics_analysis.py
```

**Expected Behavior:**
- ✅ Stage 1: Should complete (no DataFrame.append error)
- ✅ Stage 2: Should receive dict with 'filtered_spectra' key
- ✅ Stage 3: Should not have coherence AttributeError
- ✅ All stages save .json, .tab files

---

## 🔍 If Still Failing:

Check these files were saved:
1. `precursor/src/core/SpectraReader.py` (line 187 has `pd.concat`)
2. `precursor/src/pipeline/metabolomics.py` (lines 522, 535, 523 have correct attributes)

Reload Python:
```bash
# Exit and restart Python to reload modules
```

# Virtual Spectrometry Validation - Quick Start Guide

## What Does This Do?

The virtual spectrometry validation demonstrates **Molecular Maxwell Demon (MMD)** theory applied to mass spectrometry:

- ✅ **Multiple virtual instruments** (TOF, Orbitrap, FT-ICR, IMS, etc.) measure the **SAME categorical state** simultaneously
- ✅ **Zero simulation** of intermediate stages (no TOF tube simulation, no quadrupole simulation)
- ✅ **Platform independent**: Same molecular state regardless of hardware
- ✅ **Zero marginal cost**: Each additional instrument is "free" (exists only during measurement)

---

## Prerequisites

1. **Python environment** (already set up if you've run other scripts)
2. **Real data files** (optional, but recommended):
   - `public/metabolomics/PL_Neg_Waters_qTOF.mzML`
   - `public/metabolomics/TG_Pos_Thermo_Orbi.mzML`

---

## How to Run (3 Simple Commands)

### Option 1: Quick Test (No Real Data Needed)

```bash
cd precursor
python test_virtual_mass_spec_ensemble.py
```

**What happens:**
- **Test 1**: Creates a simple synthetic spectrum and measures it with multiple virtual instruments
- Shows how TOF, Orbitrap, FT-ICR, IMS, etc. all read the SAME categorical state
- Takes ~5-10 seconds

**You'll see:**
```
TEST 1: SINGLE SPECTRUM - MULTIPLE VIRTUAL INSTRUMENTS
========================================================

Input spectrum:
  Peaks: 4
  m/z range: 100.05 - 400.20

RESULTS
--------
Virtual instruments materialized: 6
Convergence nodes found: 4
Total phase-locks detected: 32
Total time: 0.123 s

Virtual instrument measurements:

  TOF:
    m/z: [100.05, 200.10, 300.15, 400.20]
    Arrival time: 0.000042 s
    Categorical state: (S_k=0.523, S_t=0.412, S_e=0.334)

  ORBITRAP:
    m/z: [100.05, 200.10, 300.15, 400.20]
    Exact frequency: 1.52e+06 Hz
    Categorical state: (S_k=0.523, S_t=0.412, S_e=0.334)

  ... (more instruments)
```

---

### Option 2: Full Test Suite (With Real Data)

**If you have the real data files:**

```bash
cd precursor
python test_virtual_mass_spec_ensemble.py
```

**What happens:**
- **Test 1**: Synthetic spectrum (simple case)
- **Test 2**: Real Waters qTOF data (demonstrates hardware oscillation harvesting)
- **Test 3**: Platform independence (Waters vs Thermo comparison)

**Takes:** ~30-60 seconds (depends on data size)

**Results saved to:** `precursor/results/virtual_ensemble_tests/`

---

### Option 3: Custom Script (Your Own Data)

Create a Python script:

```python
from src.virtual import VirtualMassSpecEnsemble
import numpy as np

# Your data
mz = np.array([100.05, 200.10, 300.15])
intensity = np.array([1000, 800, 600])

# Create ensemble (all instruments enabled)
ensemble = VirtualMassSpecEnsemble(
    enable_all_instruments=True,
    enable_hardware_grounding=True,
    coherence_threshold=0.3
)

# Measure with virtual ensemble
result = ensemble.measure_spectrum(
    mz=mz,
    intensity=intensity,
    rt=15.5,
    metadata={'sample': 'my_test'}
)

# Print results
print(f"Virtual instruments: {result.n_instruments}")
print(f"Convergence nodes: {result.convergence_nodes_count}")
print(f"Phase-locks: {result.total_phase_locks}")

# Access individual instruments
for vi in result.virtual_instruments:
    print(f"\n{vi.instrument_type}:")
    print(f"  m/z: {vi.measurement['mz']}")
    print(f"  S-coordinates: (S_k={vi.categorical_state.S_k:.3f}, "
          f"S_t={vi.categorical_state.S_t:.3f}, "
          f"S_e={vi.categorical_state.S_e:.3f})")

# Save results
ensemble.save_results(result, "results/my_test")
```

Run it:
```bash
python my_custom_script.py
```

---

## Understanding the Output

### Key Metrics

1. **Virtual instruments materialized**: How many instrument types (TOF, Orbitrap, etc.)
2. **Convergence nodes**: Where MMDs materialized (high phase-lock density sites)
3. **Phase-locks detected**: Hardware-molecular oscillation couplings found
4. **S-coordinates**: `(S_k, S_t, S_e)` = categorical state compression
   - **S_k** (Knowledge): Which equivalence class? Information deficit
   - **S_t** (Time): When in categorical sequence? Temporal position
   - **S_e** (Entropy): Constraint density, thermodynamic accessibility

### What Each Test Shows

**Test 1 - Single Spectrum:**
- Multiple instruments reading SAME categorical state
- No simulation needed (categorical access)
- All instruments materialize at same convergence node
- Zero marginal cost per instrument

**Test 2 - Real Data:**
- Hardware oscillation harvesting from actual computer
- Frequency hierarchy (8 scales: CPU → Memory → Disk → Network → ...)
- Finite observers detecting phase-locks
- Categorical state extraction from real spectrum

**Test 3 - Platform Independence:**
- Waters qTOF data → S-coordinates
- Thermo Orbitrap data → S-coordinates
- **SAME categorical state** despite different hardware!
- Platform-independent molecular representation

---

## Troubleshooting

### "Hardware harvesters not available"

This is just a **warning**, not an error. The script will use simulated oscillations instead.

**To enable hardware harvesting** (optional):
- Hardware harvesting only works on the actual machine (not in simulation)
- Uses real CPU clock, memory bus, disk I/O, etc. oscillations
- Not required for basic functionality

### "Real data file not found"

Tests 2 and 3 will be **skipped** (not errors). Test 1 will still run with synthetic data.

**To enable real data tests:**
- Download or copy the `.mzML` files to `precursor/public/metabolomics/`
- Files: `PL_Neg_Waters_qTOF.mzML`, `TG_Pos_Thermo_Orbi.mzML`

### "ModuleNotFoundError: No module named 'src.virtual'"

Make sure you're in the `precursor` directory:
```bash
cd precursor
python test_virtual_mass_spec_ensemble.py
```

---

## What You Should See (Summary)

✅ **Test 1 PASS**: Multiple virtual instruments measuring simple spectrum
✅ **Test 2 PASS** (if real data): Waters qTOF data processed
✅ **Test 3 PASS** (if real data): Platform independence demonstrated

**Conclusion:**
```
Virtual mass spectrometers successfully demonstrate:
  • Multiple instrument types from SAME categorical state
  • Zero simulation of unknowable intermediate stages
  • Platform-independent categorical state extraction
  • Zero marginal cost (instruments exist only during measurement)
  • Molecular Maxwell Demon (MMD) based information filtering

This is measurement without measurement - accessing what IS,
not forcing it into a particular eigenstate through classical
simulation of inherently unknowable ion trajectories.
```

---

## Next Steps

1. **Explore results**: Check `precursor/results/virtual_ensemble_tests/`
   - JSON files with full ensemble results
   - Includes all S-coordinates, measurements, validation

2. **Read the README**: `precursor/src/virtual/README.md`
   - Detailed theory (St-Stellas categories, MMD formalism)
   - Architecture explanation
   - Why simulation doesn't work (unknowable trajectories)

3. **Try your own data**: Use the custom script template above

4. **Understand the theory**: Read the paper sections in `precursor/virtual-instruments/`
   - `sections/molecular-maxwell-demon.tex`: MMD theory
   - `sections/virtual-detector.tex`: Virtual detector architecture
   - `sections/charged-ion-ensembles.tex`: Ensemble construction

---

## One-Line Summary

**Just run this:**
```bash
cd precursor && python test_virtual_mass_spec_ensemble.py
```

**It will validate that multiple virtual instruments can measure the same categorical state simultaneously with zero simulation!**

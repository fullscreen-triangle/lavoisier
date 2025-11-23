# Quick Start Guide

## Step 1: Activate Virtual Environment

```powershell
# PowerShell
.\.venv\Scripts\Activate.ps1
```

Or if that doesn't work:

```cmd
# Command Prompt
.venv\Scripts\activate.bat
```

## Step 2: Test Imports (Optional but Recommended)

```bash
python test_imports.py
```

This will verify:
- ✓ All modules import correctly
- ✓ Experimental files exist
- ✓ Dependencies are installed

## Step 3: Run Metabolomics Analysis

```bash
python run_metabolomics_analysis.py
```

This will process:
1. **PL_Neg_Waters_qTOF.mzML** (Phospholipids, Waters)
2. **TG_Pos_Thermo_Orbi.mzML** (Triglycerides, Thermo)

Through 4 stages:
- Stage 1: Spectral Preprocessing (BMD Input Filter)
- Stage 2: S-Entropy Transformation (Platform-Independent)
- Stage 3: Hardware BMD Grounding (Reality Check)
- Stage 4: Categorical Completion (Temporal Navigation)

## Expected Output

```
================================================================================
Processing: PL_Neg_Waters_qTOF.mzML
================================================================================

  Stage: spectral_preprocessing
    Status: completed
    Time: 2.45s
    MS2 spectra: 1234
    Peaks filtered: 45678

  Stage: sentropy_transformation
    Status: completed
    Time: 0.54s
    Throughput: 2287 spec/s
    Unique categorical states: 892

  Stage: hardware_bmd_grounding
    Status: completed
    Time: 0.12s
    Mean divergence: 0.089
    Warnings: 0

  Stage: categorical_completion
    Status: completed
    Time: 1.23s
    Annotations: 1198 (avg confidence: 0.867)

Completed: PL_Neg_Waters_qTOF.mzML
Status: completed
Total execution time: 4.34s
```

## Results Location

```
results/metabolomics_analysis/
├── PL_Neg_Waters_qTOF/
│   ├── theatre_result.json
│   ├── stage_01_preprocessing/
│   ├── stage_02_sentropy/
│   ├── stage_03_bmd/
│   └── stage_04_completion/
└── TG_Pos_Thermo_Orbi/
    └── [same structure]
```

## Troubleshooting

### Import Error
```bash
# Make sure you're in precursor directory
cd precursor

# Test imports first
python test_imports.py
```

### Missing Dependencies
```bash
pip install numpy pandas scipy scikit-learn pymzml
# Note: ursgal is already installed in your environment
```

### Virtual Environment Not Activated
Check your prompt - should show `(.venv)` at the start:
```
(.venv) PS C:\...\precursor>
```

## What Each File Does

- `run_metabolomics_analysis.py` - Main analysis script
- `test_imports.py` - Verify setup is correct
- `src/pipeline/metabolomics.py` - Pipeline implementation
- `src/bmd/` - Biological Maxwell Demon components
- `src/core/` - S-Entropy transformation and spectral processing

## Key Metrics to Watch

| Metric | Good Value | Warning |
|--------|-----------|---------|
| Stream Divergence | < 0.15 | > 0.3 |
| S-Entropy Throughput | > 2000 spec/s | < 500 spec/s |
| Annotation Confidence | > 0.8 | < 0.6 |
| Platform CV | < 1% | > 5% |

## For More Details

See `METABOLOMICS_PIPELINE_README.md` for:
- Complete API documentation
- Theoretical framework
- Configuration options
- Advanced usage examples

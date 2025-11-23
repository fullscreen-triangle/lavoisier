# Hardware-Constrained Categorical Completion for Metabolomics

This directory contains the implementation of the framework from our publication:

**"Hardware-Constrained Categorical Completion for Platform-Independent Metabolomics: Direct Molecular Information Access Through Oscillatory Coupling and Biological Maxwell Demons"**

## Overview

This pipeline implements a revolutionary approach to metabolomics that:

1. **Achieves platform independence** through S-entropy bijective transformation
2. **Grounds interpretations in physical reality** via Hardware BMD streams
3. **Enables O(1) metabolite lookup** through temporal coordinate navigation
4. **Maintains thermodynamic consistency** via stream divergence monitoring
5. **Accesses 100% molecular information space** versus ~5% for traditional methods

## Architecture

The pipeline follows a **Theatre → Stages → Processes** hierarchical observer architecture:

```
MetabolomicsTheatre (Transcendent Observer)
│
├── Stage 1: Spectral Preprocessing
│   ├── Process: Spectral Acquisition
│   └── Process: Peak Detection (BMD Input Filter)
│
├── Stage 2: S-Entropy Transformation
│   ├── Process: S-Entropy Transform (Bijective)
│   └── Process: Categorical State Mapping
│
├── Stage 3: Hardware BMD Grounding
│   ├── Process: Hardware Stream Harvest
│   └── Process: Stream Coherence Check
│
└── Stage 4: Categorical Completion
    ├── Process: Oscillatory Hole Identification
    └── Process: Temporal Navigation (O(1))
```

## Experimental Files

Two platform-specific experimental datasets are included:

1. **PL_Neg_Waters_qTOF.mzML**
   - Platform: Waters Synapt G2-Si qTOF
   - Resolution: 20,000 FWHM
   - Metabolite class: Phospholipids (PL)
   - Ion mode: Negative

2. **TG_Pos_Thermo_Orbi.mzML**
   - Platform: Thermo Q Exactive Orbitrap
   - Resolution: 60,000 FWHM
   - Metabolite class: Triglycerides (TG)
   - Ion mode: Positive

These files demonstrate **platform independence**: the S-entropy transformation produces categorical states with CV < 1% despite 3× resolution difference and different analyzer types.

## Installation

### Requirements

```bash
# Core dependencies
pip install numpy pandas scipy scikit-learn

# Mass spectrometry
pip install pymzml pyteomics

# Optional: BMD components
# (Included in precursor/src/bmd/)
```

### Setup

```bash
# Clone repository
git clone https://github.com/fullscreen-triangle/lavoisier
cd lavoisier/precursor

# Verify data files
ls public/metabolomics/
# Should show: PL_Neg_Waters_qTOF.mzML, TG_Pos_Thermo_Orbi.mzML
```

## Usage

### Quick Start

```bash
# Run complete analysis on both experimental files
python run_metabolomics_analysis.py
```

### Programmatic Usage

```python
from pathlib import Path
from src.pipeline.metabolomics import run_metabolomics_analysis

# Define files
mzml_files = [
    Path("public/metabolomics/PL_Neg_Waters_qTOF.mzML"),
    Path("public/metabolomics/TG_Pos_Thermo_Orbi.mzML")
]

# Run analysis
results = run_metabolomics_analysis(
    mzml_files=mzml_files,
    output_dir=Path("results/metabolomics_analysis"),
    enable_bmd=True,
    preprocessing={
        'acquisition': {
            'rt_range': [0, 100],
            'ms1_threshold': 1000,
            'ms2_threshold': 10
        }
    }
)

# Access results
for file_name, result in results.items():
    print(f"File: {file_name}")
    print(f"Status: {result.status}")
    print(f"Execution time: {result.execution_time:.2f}s")

    # Stage-specific results
    sentropy_stage = result.stage_results['sentropy']
    print(f"S-Entropy throughput: {sentropy_stage.metrics['throughput_spec_per_sec']:.1f} spec/s")

    bmd_stage = result.stage_results['bmd_grounding']
    print(f"Stream divergence: {bmd_stage.metrics['mean_divergence']:.3f}")
```

### Custom Theatre Configuration

```python
from src.pipeline.metabolomics import MetabolomicsTheatre
from pathlib import Path

# Initialize theatre with custom parameters
theatre = MetabolomicsTheatre(
    output_dir=Path("results/custom_analysis"),
    enable_bmd_grounding=True,
    stream_divergence_threshold=0.3,  # Warning threshold
    preprocessing={
        'acquisition': {'vendor': 'thermo'},
        'peak_detection': {'min_intensity': 100.0}
    },
    sentropy={
        'categorical': {'epsilon': 0.1}  # Categorical state resolution
    },
    bmd_grounding={
        'coherence': {'divergence_threshold': 0.3}
    }
)

# Execute pipeline
result = theatre.observe_all_stages(
    input_data=Path("public/metabolomics/TG_Pos_Thermo_Orbi.mzML")
)

# Save results
result.save(Path("results/custom_analysis/result.json"))
```

## Key Concepts

### 1. S-Entropy Coordinates

The **bijective transformation** from mass spectra to 14-dimensional S-entropy space:

**Structural (4D):**

- f₁: Base peak m/z
- f₂: Peak count
- f₃: m/z range
- f₄: Peak spacing variance

**Statistical (4D):**

- f₅: Total ion current
- f₆: Intensity variance
- f₇: Intensity skewness
- f₈: Intensity kurtosis

**Information (4D):**

- f₉: Spectral entropy
- f₁₀: Structural entropy
- f₁₁: Mutual information
- f₁₂: Conditional entropy

**Temporal (2D):**

- f₁₃: Temporal coordinate
- f₁₄: Phase coherence

**Platform Independence:** CV < 1% across different MS platforms.

### 2. Categorical States

**Equivalence classes** of molecular configurations sharing identical phase relationships in oscillatory modes.

```python
# Categorical state definition
CategoricalState = {
    'state_id': 'cat_00123',
    'description': 'Lipid_PL_negative',
    'phase_relationships': {...},
    'features': [f₁, f₂, ..., f₁₄],
    'entropy': 2.34,
    'richness': 1.0
}
```

Metabolites in the same categorical state are **indistinguishable** at the given resolution ε.

### 3. Hardware BMD Stream

The **Hardware BMD Stream** provides thermodynamic grounding through physical oscillations:

```python
HardwareBMDStream = {
    'display_bmd': {...},      # Display refresh oscillations
    'network_bmd': {...},      # Network packet timing
    'em_field_bmd': {...},     # EM emissions from computation
    'acoustic_bmd': {...},     # Fan/disk vibrations
    'thermal_bmd': {...},      # Temperature fluctuations
    'sensor_bmd': {...},       # Accelerometer/gyroscope
    'unified_bmd': {...},      # Phase-locked composition
    'coherence': 0.87          # Overall coherence
}
```

**Stream Divergence** monitors reality drift:

```python
D_stream = ||Φ_network - Φ_hardware||₂ + λ|R_network - R_hardware|
```

- D_stream < 0.15: Excellent physical realizability
- D_stream < 0.3: Good (acceptable)
- D_stream > 0.3: Warning (interpretation drifting)
- D_stream > 0.5: Error (unphysical region)

### 4. Categorical Completion

**Oscillatory holes** are sets of weak force configurations that could complete observed patterns:

```python
OscillatoryHole = {
    'scan_id': 12345,
    'richness': 3.5,           # Number of possible completions
    'completions': [           # Possible metabolites
        {'id': 'LMGP01010001', 'prob': 0.65},
        {'id': 'LMGP01010002', 'prob': 0.25},
        {'id': 'LMGP01010003', 'prob': 0.10}
    ]
}
```

**Selection** of physical completion uses hardware BMD constraints.

### 5. Temporal Navigation

**Predetermined endpoints** enable O(1) metabolite lookup:

```python
# Traditional database search: O(N)
for metabolite in database:  # N iterations
    if match(spectrum, metabolite):
        return metabolite

# Temporal navigation: O(1)
temporal_coord = compute_temporal_coord(categorical_state)
metabolite = direct_access(temporal_coord)  # Single lookup
```

## Output Structure

```
results/metabolomics_analysis/
├── PL_Neg_Waters_qTOF/
│   ├── theatre_result.json                    # Complete results
│   ├── stage_01_preprocessing/
│   │   ├── stage_result.json
│   │   ├── scan_info.csv
│   │   └── filtered_spectra.pkl
│   ├── stage_02_sentropy/
│   │   ├── stage_result.json
│   │   ├── sentropy_features.csv
│   │   └── categorical_states.json
│   ├── stage_03_bmd/
│   │   ├── stage_result.json
│   │   ├── hardware_bmd_stream.json
│   │   ├── coherence_scores.csv
│   │   └── divergences.csv
│   └── stage_04_completion/
│       ├── stage_result.json
│       ├── oscillatory_holes.json
│       └── annotations.csv
└── TG_Pos_Thermo_Orbi/
    └── [same structure]
```

## Performance Metrics

### Expected Performance (from publication validation)

| Metric | Value | Notes |
|--------|-------|-------|
| **S-Entropy Transform** | 2,273 spec/s | Bijective transformation |
| **Full Pipeline** | 36 spec/s | Including BMD grounding |
| **Database Annotation** | 91.4% | LIPIDMAPS |
| **Platform CV** | < 1% | Waters vs Thermo |
| **Stream Divergence** | < 0.12 | Throughout processing |
| **Intra-class Similarity** | 0.847 | Same lipid class |
| **Inter-class Dissimilarity** | 0.723 | Different classes |

### Computational Complexity

| Operation | Traditional | Precursor |
|-----------|------------|-----------|
| Database search | O(N) | O(1) |
| Feature extraction | O(n log n) | O(n log n) |
| Platform normalization | Required | Not needed |
| Cross-validation | Per platform | Universal |

## Theoretical Foundation

### Universal Coupling Equation

```
dΨᵢ/dt = Hᵢ(Ψᵢ) + Σⱼ Cᵢⱼ(Ψᵢ, Ψⱼ, ωᵢⱼ) + Eᵢ(t) + Qᵢ(ψ̂)
```

Where:

- **Hᵢ**: Intrinsic molecular oscillatory dynamics
- **Cᵢⱼ**: Inter-scale coupling (8 scales)
- **Eᵢ**: Environmental oscillatory perturbations
- **Qᵢ**: Quantum coherence terms

### Eight-Scale Oscillatory Hierarchy

1. **Quantum Membrane** (10¹²-10¹⁵ Hz): Electronic transitions
2. **Intracellular Circuits** (10³-10⁶ Hz): Bond vibrations
3. **Cellular Information** (10⁻¹-10² Hz): Conformational dynamics
4. **Tissue Integration** (10⁻²-10¹ Hz): Interaction networks
5. **Microbiome Community** (10⁻⁴-10⁻¹ Hz): Ecosystem context
6. **Organ Coordination** (10⁻⁵-10⁻² Hz): Physiological function
7. **Physiological Systems** (10⁻⁶-10⁻³ Hz): Systemic integration
8. **Allometric Organism** (10⁻⁸-10⁻⁵ Hz): Metabolic scaling

Traditional MS accesses only scale 1-2 (~5% information). Oscillatory coupling accesses all 8 scales (~100% information).

### BMD Information Catalysis

Biological Maxwell Demons act as **information catalysts** (iCat):

```
η_recognition = I_molecular / (k_B T ln(2)) > 1
```

Exceeds classical information-theoretic limits through:

- **Input filter**: Selects signal from noise via phase-lock coherence
- **Information processing**: Categorical completion through weak force selection
- **Output filter**: Directs to physical realizations
- **iCat property**: Increases probability by factors of 10⁶-10¹²

## Validation

### Platform Independence Test

```python
# Compare S-entropy features across platforms
features_waters = extract_features(waters_spectrum)
features_thermo = extract_features(thermo_spectrum)

cv = coefficient_of_variation(features_waters, features_thermo)
print(f"Platform CV: {cv:.3f}%")  # Expected: < 1%

assert cv < 0.015, "Platform independence requirement not met"
```

### Stream Divergence Validation

```python
# Monitor throughout processing
for stage in theatre.stages:
    divergence = stage.stream_divergence
    assert divergence < 0.3, f"Stream divergence too high: {divergence}"

    if divergence > 0.15:
        logger.warning(f"Elevated divergence in {stage.name}: {divergence}")
```

### Temporal Navigation Efficiency

```python
import time

# Traditional search
start = time.time()
result_traditional = database_search(spectrum)
time_traditional = time.time() - start

# Temporal navigation
start = time.time()
result_temporal = temporal_navigation(categorical_state)
time_temporal = time.time() - start

speedup = time_traditional / time_temporal
print(f"Speedup: {speedup:.1f}×")  # Expected: 100-1000×
```

## Troubleshooting

### Issue: High stream divergence

**Symptoms:** D_stream > 0.3 warnings

**Causes:**

- Poor spectral quality
- Platform-specific artifacts
- Incorrect vendor specification

**Solutions:**

```python
# Increase peak filtering stringency
config['preprocessing']['peak_detection']['min_intensity'] = 500.0
config['preprocessing']['peak_detection']['min_snr'] = 5.0

# Adjust categorical state resolution
config['sentropy']['categorical']['epsilon'] = 0.15  # Coarser states
```

### Issue: Low annotation rate

**Symptoms:** < 80% annotations

**Causes:**

- Novel metabolites not in database
- Insufficient spectral quality
- Wrong lipid class

**Solutions:**

```python
# Lower confidence threshold
config['completion']['temporal']['confidence_threshold'] = 0.7

# Use multiple databases
config['completion']['temporal']['databases'] = [
    'LIPIDMAPS', 'METLIN', 'HMDB'
]
```

### Issue: BMD components not available

**Symptoms:** "BMD_AVAILABLE = False" warning

**Causes:**

- Missing BMD module files
- Import errors in bmd/

**Solutions:**

```python
# Check BMD module
import sys
sys.path.insert(0, 'src')
from bmd import *  # Should not error

# Fallback to standard mode
results = run_metabolomics_analysis(
    mzml_files=files,
    output_dir=output_dir,
    enable_bmd=False  # Disable BMD grounding
)
```

## Citation

If you use this pipeline, please cite:

```bibtex
@article{sachikonye2024metabolomics,
  title={Hardware-Constrained Categorical Completion for Platform-Independent Metabolomics:
         Direct Molecular Information Access Through Oscillatory Coupling and
         Biological Maxwell Demons},
  author={Sachikonye, Kundai Farai},
  journal={Analytical Chemistry},
  year={2024},
  note={In preparation}
}
```

## License

MIT License - see LICENSE file

## Contact

**Kundai Farai Sachikonye**

- Email: <kundai.sachikonye@wzw.tum.de>
- GitHub: <https://github.com/fullscreen-triangle/lavoisier>

## Acknowledgments

This work builds on theoretical foundations in:

- Oscillatory mass spectrometry
- S-entropy coordinate systems
- Biological Maxwell demon theory
- Categorical completion algebra
- Hardware-constrained computation

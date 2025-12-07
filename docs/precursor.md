---
layout: default
title: Precursor Framework
nav_order: 2
---

# Precursor Framework

**Hardware-Grounded Mass Spectrometry via S-Entropy Coordinates**

The Precursor framework implements a revolutionary approach to mass spectrometry analysis through categorical coordinate systems and virtual instruments.

## Overview

Traditional mass spectrometry analysis treats spectra as platform-specific data that requires instrument-specific processing. Precursor transforms this paradigm by mapping all spectral data to a universal **S-Entropy Coordinate System**.

```
Traditional:  m/z + intensity → platform-specific features → analysis

Precursor:    m/z + intensity → (S_k, S_t, S_e) coordinates
                              → categorical state assignment
                              → platform-independent analysis
```

## Core Components

### 1. S-Entropy Coordinates

The S-Entropy coordinate system maps each ion to a 3D categorical space:

| Coordinate | Symbol | Description | Range |
|------------|--------|-------------|-------|
| S_knowledge | S_k | Structural information content | [0, 1] |
| S_time | S_t | Temporal/kinetic positioning | [0, 1] |
| S_entropy | S_e | Thermodynamic entropy state | [0, 1] |

**Properties:**
- Bijective mapping (reversible transformation)
- Platform-independent representation
- Preserves thermodynamic information
- Enables categorical state assignment

```python
from precursor.src.core.EntropyTransformation import SEntropyTransformer

transformer = SEntropyTransformer()
coords_list, coord_matrix = transformer.transform_spectrum(mz_array, intensity_array)

for coord in coords_list:
    print(f"S_k={coord.s_knowledge:.3f}, S_t={coord.s_time:.3f}, S_e={coord.s_entropy:.3f}")
```

### 2. Ion-to-Droplet Computer Vision

Bijective transformation of mass spectra into thermodynamic droplet images:

- **Physics Validation**: Navier-Stokes constrained parameters
- **Surface Tension**: Realistic droplet dynamics
- **Phase Coherence**: Ensemble behavior modeling

```python
from precursor.src.core.IonToDropletConverter import IonToDropletConverter

converter = IonToDropletConverter(
    resolution=(512, 512),
    enable_physics_validation=True,
    validation_threshold=0.3
)

image, ion_droplets = converter.convert_spectrum_to_image(
    mzs=mz_array,
    intensities=intensity_array,
    rt=retention_time
)

# Each droplet contains:
# - S-Entropy coordinates
# - Droplet parameters (velocity, radius, phase_coherence)
# - Physics quality score
# - Categorical state assignment
```

### 3. Virtual Instrument Ensemble

Hardware-grounded virtual mass spectrometers for cross-platform analysis:

| Instrument Type | Characteristics |
|----------------|-----------------|
| qTOF | High resolution, accurate mass |
| Orbitrap | Ultra-high resolution |
| Triple Quad | High sensitivity, MRM |
| Ion Trap | MSn capability |
| Linear Ion Trap | Fast scanning |

```python
from precursor.src.virtual import VirtualMassSpecEnsemble

ensemble = VirtualMassSpecEnsemble(
    enable_all_instruments=True,
    enable_hardware_grounding=True,
    coherence_threshold=0.3
)

result = ensemble.measure_spectrum(mz, intensity, rt)

# Result contains:
# - n_instruments: Number of virtual instruments
# - total_phase_locks: Detected molecular ensembles
# - convergence_nodes_count: Cross-platform agreement points
```

### 4. Molecular Maxwell Demon

Information-theoretic fragmentation analysis using categorical states:

- **Fragmentation Network**: Graph-based fragment relationships
- **Phase-Lock Detection**: Molecular ensemble identification
- **Categorical Completion**: Gap-filling via trajectory analysis

```python
from precursor.src.mmdsystem import MolecularMaxwellDemonSystem

mmd = MolecularMaxwellDemonSystem()
analysis = mmd.analyze_fragmentation(precursor_mz, fragment_mzs, intensities)
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Spectral Acquisition                                  │
│  - mzML extraction (any vendor: Waters, Thermo, Agilent, etc.)  │
│  - Peak detection and quality control                           │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: S-Entropy Transformation                              │
│  - Bijective mapping to (S_k, S_t, S_e) coordinates             │
│  - Categorical state assignment                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2B: Computer Vision Modality                             │
│  - Ion-to-droplet thermodynamic conversion                      │
│  - Physics validation (Navier-Stokes, surface tension)          │
│  - Droplet image generation                                     │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2.5: Fragmentation Network                               │
│  - Phase-lock network construction                               │
│  - Precursor-fragment relationship mapping                      │
│  - Molecular Maxwell Demon analysis                              │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: BMD Hardware Grounding                                │
│  - Categorical state coherence validation                       │
│  - Hardware oscillation harvesting                              │
│  - Reality grounding metrics                                    │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4: Categorical Completion                                │
│  - S-entropy trajectory analysis                                │
│  - Gap-filling via categorical navigation                       │
│  - Annotation confidence scoring                                │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 5: Virtual Instrument Ensemble                           │
│  - Multi-instrument materialization (qTOF, Orbitrap, etc.)      │
│  - Phase-lock network consensus                                 │
│  - Cross-platform validation                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Running the Pipeline

### Complete Analysis

```bash
cd precursor
python run_ucdavis_complete_analysis.py
```

### Resume from Stage 2B (Faster)

If you've already completed Stages 1 and 2, resume with limited CV processing:

```bash
python run_ucdavis_resume.py
```

This processes only 50 spectra per file for droplet conversion, significantly reducing execution time.

## Results Structure

```
results/ucdavis_complete_analysis/
├── {file_name}/
│   ├── stage_01_preprocessing/
│   │   ├── scan_info.csv           # Scan metadata
│   │   ├── scan_info.tsv
│   │   ├── spectra/                # Individual spectra
│   │   │   └── spectrum_*.tsv
│   │   ├── spectra_summary.csv
│   │   └── stage_01_metrics.json
│   │
│   ├── stage_02_sentropy/
│   │   ├── sentropy_features.csv   # S-Entropy coordinates
│   │   ├── matrices/               # Coordinate matrices
│   │   │   └── sentropy_*.tsv
│   │   └── stage_02_metrics.json
│   │
│   ├── stage_02_cv/
│   │   ├── images/                 # Droplet images
│   │   │   └── droplet_*.png
│   │   ├── droplets/               # Droplet parameters
│   │   │   └── droplets_*.tsv
│   │   ├── cv_summary.csv
│   │   └── stage_02_cv_metrics.json
│   │
│   ├── stage_02_5_fragmentation/
│   │   └── stage_02_5_metrics.json
│   │
│   ├── stage_03_bmd/
│   │   ├── coherence_results.csv
│   │   └── stage_03_metrics.json
│   │
│   ├── stage_04_completion/
│   │   ├── completion_results.csv
│   │   └── stage_04_metrics.json
│   │
│   ├── stage_05_virtual/
│   │   ├── virtual_results.csv
│   │   └── stage_05_metrics.json
│   │
│   └── pipeline_results.json       # Complete pipeline summary
│
├── analysis_summary.csv            # Cross-file summary
└── master_results.json             # Master results file
```

## Performance Metrics

| Stage | Throughput | Notes |
|-------|------------|-------|
| Spectral Acquisition | ~1000 spectra/s | Depends on file I/O |
| S-Entropy Transform | 800-900 spectra/s | Fully vectorized |
| Computer Vision | 50-100 spectra/s | With physics validation |
| BMD Grounding | 500+ spectra/s | Coherence calculation |
| Virtual Ensemble | 100+ spectra/s | Multi-instrument |

## Theoretical Foundation

### S-Entropy Mathematics

The S-Entropy transformation is defined as:

$$
\mathbf{S}(m/z, I) = \begin{pmatrix} S_k \\ S_t \\ S_e \end{pmatrix}
$$

Where:
- $S_k = f_k(m/z, I)$ encodes structural information
- $S_t = f_t(m/z, I)$ encodes temporal positioning
- $S_e = f_e(m/z, I)$ encodes thermodynamic state

### Categorical States

Each point in S-Entropy space maps to a categorical state:

$$
\text{State}(\mathbf{S}) = \text{Hash}(\lfloor S_k \cdot N \rfloor, \lfloor S_t \cdot N \rfloor, \lfloor S_e \cdot N \rfloor)
$$

Where $N$ is the resolution parameter.

### Phase-Lock Networks

Molecular ensembles are detected through coherence analysis:

$$
\text{Coherence}(i, j) = \frac{|\langle \mathbf{S}_i, \mathbf{S}_j \rangle|}{\|\mathbf{S}_i\| \|\mathbf{S}_j\|}
$$

## Publications

See `precursor/publication/` for detailed LaTeX documentation:

- `entropy-coordinates/` - S-Entropy coordinate system theory
- `computer-vision/` - Ion-to-droplet conversion methods
- `virtual-instruments/` - Virtual mass spectrometer ensemble
- `molecular-language/` - Categorical amino acid alphabet
- `fragmentation/` - Molecular Maxwell Demon analysis

---

*Next: [Buhera Scripting Language](README_BUHERA.md)*

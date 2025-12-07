# Lavoisier Precursor

**Hardware-Grounded Mass Spectrometry via S-Entropy Coordinates and Virtual Instruments**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-validated-brightgreen.svg)]()

## Overview

Precursor is a novel mass spectrometry analysis framework implementing:

- **S-Entropy Coordinates**: 3D categorical coordinate system (S_knowledge, S_time, S_entropy) for platform-independent spectral representation
- **Ion-to-Droplet Computer Vision**: Bijective transformation of mass spectra into thermodynamic droplet images with physics validation
- **Virtual Instrument Ensemble**: Hardware-grounded virtual mass spectrometers (qTOF, Orbitrap, etc.) with phase-lock networks
- **Molecular Maxwell Demon**: Information-theoretic fragmentation analysis using categorical states
- **Categorical Completion**: Gap-filling and annotation through S-entropy trajectory analysis

## Key Innovation

**S-Entropy Categorical Coordinates** provide a unified representation across all mass spectrometry platforms:

```
Traditional:  m/z + intensity → platform-specific features

Precursor:    m/z + intensity → (S_knowledge, S_time, S_entropy)
                              → categorical state assignment
                              → platform-independent analysis
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

## Installation

```bash
# Clone repository
git clone https://github.com/lavoisier-project/precursor.git
cd precursor

# Install dependencies
pip install -e .

# Or with pnpm for Node components
pnpm install
```

### Requirements

- Python ≥ 3.8
- NumPy, Pandas, SciPy
- OpenCV (cv2)
- NetworkX
- scikit-learn

See `requirements.txt` for complete list.

## Quick Start

### 1. Run Complete Analysis Pipeline

```python
from pathlib import Path
from src.core.SpectraReader import extract_mzml
from src.core.EntropyTransformation import SEntropyTransformer
from src.core.IonToDropletConverter import IonToDropletConverter

# Load data
scan_info, spectra, xic = extract_mzml("your_data.mzML")

# Transform to S-Entropy coordinates
transformer = SEntropyTransformer()
for scan_id, spectrum in spectra.items():
    coords, matrix = transformer.transform_spectrum(
        spectrum['mz'].values,
        spectrum['intensity'].values
    )

# Generate droplet images
converter = IonToDropletConverter(resolution=(512, 512))
image, droplets = converter.convert_spectrum_to_image(
    mzs=spectrum['mz'].values,
    intensities=spectrum['intensity'].values
)
```

### 2. Run UC Davis Metabolomics Analysis

```bash
# Full pipeline (may take a while for large datasets)
python run_ucdavis_complete_analysis.py

# Resume from Stage 2B with limited spectra (faster)
python run_ucdavis_resume.py
```

### 3. Virtual Instrument Ensemble

```python
from src.virtual import VirtualMassSpecEnsemble

ensemble = VirtualMassSpecEnsemble(
    enable_all_instruments=True,
    enable_hardware_grounding=True
)

result = ensemble.measure_spectrum(
    mz=mz_array,
    intensity=intensity_array,
    rt=retention_time
)

print(f"Phase-locks detected: {result.total_phase_locks}")
print(f"Convergence nodes: {result.convergence_nodes_count}")
```

## Core Modules

### S-Entropy Coordinates

The S-Entropy coordinate system maps each ion to a 3D categorical space:

- **S_knowledge** (S_k): Structural information content [0, 1]
- **S_time** (S_t): Temporal/kinetic positioning [0, 1]
- **S_entropy** (S_e): Thermodynamic entropy state [0, 1]

```python
from src.core.EntropyTransformation import SEntropyTransformer, SEntropyCoordinates

transformer = SEntropyTransformer()
coords_list, coord_matrix = transformer.transform_spectrum(mz, intensity)

# Each coordinate is an SEntropyCoordinates object
for coord in coords_list:
    print(f"S_k={coord.s_knowledge:.3f}, S_t={coord.s_time:.3f}, S_e={coord.s_entropy:.3f}")
```

### Ion-to-Droplet Converter

Bijective transformation of mass spectra into thermodynamic droplet images:

```python
from src.core.IonToDropletConverter import IonToDropletConverter

converter = IonToDropletConverter(
    resolution=(512, 512),
    enable_physics_validation=True
)

image, ion_droplets = converter.convert_spectrum_to_image(
    mzs=mz_array,
    intensities=intensity_array
)

# Each droplet contains physics-validated parameters
for droplet in ion_droplets:
    print(f"m/z: {droplet.mz:.4f}")
    print(f"S-Entropy: ({droplet.s_entropy_coords.s_knowledge:.3f}, "
          f"{droplet.s_entropy_coords.s_time:.3f}, {droplet.s_entropy_coords.s_entropy:.3f})")
    print(f"Physics quality: {droplet.physics_quality:.3f}")
```

### Virtual Mass Spec Ensemble

Hardware-grounded virtual instruments for cross-platform analysis:

```python
from src.virtual import VirtualMassSpecEnsemble

ensemble = VirtualMassSpecEnsemble(
    enable_all_instruments=True,  # qTOF, Orbitrap, TOF, etc.
    coherence_threshold=0.3
)

result = ensemble.measure_spectrum(mz, intensity, rt)
```

### Molecular Maxwell Demon

Information-theoretic fragmentation analysis:

```python
from src.mmdsystem import MolecularMaxwellDemonSystem

mmd = MolecularMaxwellDemonSystem()
analysis = mmd.analyze_fragmentation(precursor_mz, fragment_mzs, intensities)
```

## Results Structure

After running the analysis, results are saved in:

```
results/
├── ucdavis_complete_analysis/
│   ├── A_M3_negPFP_03/
│   │   ├── stage_01_preprocessing/
│   │   │   ├── scan_info.csv
│   │   │   ├── spectra/
│   │   │   └── stage_01_metrics.json
│   │   ├── stage_02_sentropy/
│   │   │   ├── sentropy_features.csv
│   │   │   ├── matrices/
│   │   │   └── stage_02_metrics.json
│   │   ├── stage_02_cv/
│   │   │   ├── images/
│   │   │   ├── droplets/
│   │   │   └── cv_summary.csv
│   │   ├── stage_02_5_fragmentation/
│   │   ├── stage_03_bmd/
│   │   ├── stage_04_completion/
│   │   ├── stage_05_virtual/
│   │   └── pipeline_results.json
│   └── analysis_summary.csv
└── visualizations/
    ├── entropy_space/
    ├── molecular_language/
    ├── phase_lock/
    └── trajectories/
```

## Publications

This framework is documented in several publications:

- **S-Entropy Coordinates**: Categorical coordinate system for mass spectrometry
- **Ion-to-Droplet Computer Vision**: Bijective thermodynamic image generation
- **Virtual Instruments**: Hardware-grounded virtual mass spectrometers
- **Molecular Language**: Categorical amino acid alphabet and fragmentation grammar

See the `publication/` directory for LaTeX sources.

## Testing

```bash
# Test all modules
python run_all_tests.py

# Test specific module
python test_molecular_language.py
python test_dictionary.py
python test_mmd_system_complete.py

# Generate visualizations
python run_all_visualizations.py
```

## Performance

- **S-Entropy transformation**: 800-900 spectra/second
- **Ion-to-droplet conversion**: 50-100 spectra/second (with physics validation)
- **Virtual ensemble**: 100+ spectra/second
- **Complete pipeline**: ~1000 spectra/minute per file

## Theoretical Foundation

1. **S-Entropy Coordinates**: Bijective mapping from (m/z, intensity) to categorical states preserving thermodynamic information
2. **Categorical Completion**: Gap-filling via trajectory analysis in S-entropy space
3. **Phase-Lock Networks**: Molecular ensemble detection through coherence analysis
4. **Hardware Grounding**: Reality validation through oscillation harvesting
5. **Virtual Instruments**: Cross-platform consensus through categorical state comparison

## License

MIT License - See [LICENSE](LICENSE) file for details

## Authors

Lavoisier Project Team

## Citation

```bibtex
@software{lavoisier_precursor,
  title = {Lavoisier Precursor: Hardware-Grounded Mass Spectrometry via S-Entropy Coordinates},
  author = {Lavoisier Project Team},
  year = {2025},
  url = {https://github.com/lavoisier-project/precursor},
  version = {2.0.0}
}
```

---

**Status: ✅ Validated on UC Davis Metabolomics Dataset**

Precursor v2.0.0 - December 2025

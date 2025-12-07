---
layout: default
title: Home
nav_order: 1
---

# Lavoisier Documentation

Welcome to the comprehensive documentation for the Lavoisier mass spectrometry analysis framework. Lavoisier is a high-performance computing framework that combines numerical and visual processing methods with integrated artificial intelligence modules for automated compound identification and structural elucidation.

## 🎯 NEW: Precursor Framework

The **Precursor** module introduces a revolutionary approach to mass spectrometry analysis through **S-Entropy Coordinates** and **Virtual Instruments**.

### Key Innovations

- **S-Entropy Coordinates**: 3D categorical coordinate system (S_knowledge, S_time, S_entropy) for platform-independent spectral representation
- **Ion-to-Droplet Computer Vision**: Bijective transformation of mass spectra into thermodynamic droplet images
- **Virtual Instrument Ensemble**: Hardware-grounded virtual mass spectrometers with phase-lock networks
- **Molecular Maxwell Demon**: Information-theoretic fragmentation analysis using categorical states
- **Categorical Completion**: Gap-filling and annotation through S-entropy trajectory analysis

### Precursor Pipeline

```
Spectral Acquisition → S-Entropy Transform → Computer Vision
                                              ↓
Virtual Instruments ← Categorical Completion ← BMD Grounding
```

### Validated on UC Davis Metabolomics Dataset

The Precursor framework has been validated on the UC Davis metabolomics dataset (10 mzML files, ~16,000 spectra total), demonstrating:

- **S-Entropy transformation**: 800+ spectra/second
- **Physics-validated droplet conversion**: 50-100 spectra/second
- **Cross-platform categorical consistency**: Coherence > 0.85

## 🎯 Buhera Scripting Language

Lavoisier includes **Buhera**, a domain-specific scripting language that transforms mass spectrometry analysis by encoding the scientific method as executable scripts.

### Buhera Documentation

- **[📋 Buhera Overview](README_BUHERA.md)** - Complete introduction to the Buhera scripting language
- **[📖 Language Reference](buhera-language-reference.md)** - Comprehensive syntax and semantics reference
- **[🔧 Integration Guide](buhera-integration.md)** - Detailed guide to Buhera-Lavoisier integration
- **[📚 Tutorials](buhera-tutorials.md)** - Step-by-step tutorials from beginner to advanced
- **[💼 Script Examples](buhera-examples.md)** - Practical examples for various applications

### Key Buhera Features

- 🎯 **Objective-First Analysis**: Scripts declare explicit scientific goals before execution
- ✅ **Pre-flight Validation**: Catch experimental flaws before wasting time and resources
- 🧠 **Goal-Directed AI**: Bayesian evidence networks optimized for specific objectives
- 🔬 **Scientific Rigor**: Enforced statistical requirements and biological coherence

## Core Framework

### System Architecture & Installation

- **[🏗️ Architecture Overview](architecture.md)** - System design and component relationships
- **[⚙️ Installation Guide](installation.md)** - Setup instructions and requirements
- **[🚀 Performance Benchmarks](performance.md)** - System performance characteristics

### AI Modules & Intelligence

- **[🤖 AI Modules Overview](ai-modules.md)** - Comprehensive guide to all AI modules
- **[🧠 Specialized Intelligence](specialised.md)** - Domain-specific AI capabilities
- **[🔗 HuggingFace Integration](huggingface-models.md)** - Machine learning model integration
- **[📊 Embodied Understanding](embodied-understanding.md)** - 3D molecular reconstruction validation

### Analysis Pipelines

- **[🔢 Numerical Analysis](algorithms.md)** - Mathematical foundations and algorithms
- **[👁️ Visual Processing](visualization.md)** - Computer vision and image analysis
- **[📈 Results & Validation](results.md)** - Analysis outputs and validation metrics

### Development & Integration

- **[🔧 Implementation Roadmap](implementation-roadmap.md)** - Development planning and milestones
- **[🦀 Rust Integration](rust-integration.md)** - High-performance Rust components
- **[🐍 Python Integration](module-summary.md)** - Python module organization
- **[🚗 Autobahn Integration](autobahn-integration.md)** - Probabilistic reasoning integration

## Quick Start Guide

### 1. Precursor Analysis (NEW!)

```python
from precursor.src.core.SpectraReader import extract_mzml
from precursor.src.core.EntropyTransformation import SEntropyTransformer
from precursor.src.core.IonToDropletConverter import IonToDropletConverter

# Load data
scan_info, spectra, xic = extract_mzml("your_data.mzML")

# Transform to S-Entropy coordinates
transformer = SEntropyTransformer()
coords, matrix = transformer.transform_spectrum(mz, intensity)

# Generate droplet images with physics validation
converter = IonToDropletConverter(resolution=(512, 512))
image, droplets = converter.convert_spectrum_to_image(mz, intensity)

# Access categorical coordinates
for droplet in droplets:
    s_k = droplet.s_entropy_coords.s_knowledge  # Structural knowledge
    s_t = droplet.s_entropy_coords.s_time       # Temporal position
    s_e = droplet.s_entropy_coords.s_entropy    # Thermodynamic entropy
```

### 2. Virtual Instrument Ensemble

```python
from precursor.src.virtual import VirtualMassSpecEnsemble

# Create ensemble with all instruments
ensemble = VirtualMassSpecEnsemble(
    enable_all_instruments=True,
    enable_hardware_grounding=True
)

# Measure with cross-platform consensus
result = ensemble.measure_spectrum(mz, intensity, rt)
print(f"Phase-locks: {result.total_phase_locks}")
print(f"Convergence: {result.convergence_nodes_count}")
```

### 3. Complete Pipeline

```bash
cd precursor

# Run complete analysis on UC Davis dataset
python run_ucdavis_complete_analysis.py

# Or resume from Stage 2B (faster)
python run_ucdavis_resume.py
```

### 4. Buhera Script Analysis

```bash
# Build Buhera language
cd lavoisier-buhera && cargo build --release

# Create and execute a script
buhera validate biomarker_discovery.bh
buhera execute biomarker_discovery.bh
```

## Pipeline Results Structure

After running Precursor analysis:

```
results/
├── ucdavis_complete_analysis/
│   ├── {file_name}/
│   │   ├── stage_01_preprocessing/
│   │   │   ├── scan_info.csv
│   │   │   └── spectra/
│   │   ├── stage_02_sentropy/
│   │   │   ├── sentropy_features.csv
│   │   │   └── matrices/
│   │   ├── stage_02_cv/
│   │   │   ├── images/        # Droplet images
│   │   │   └── droplets/      # Physics-validated data
│   │   ├── stage_02_5_fragmentation/
│   │   ├── stage_03_bmd/
│   │   ├── stage_04_completion/
│   │   └── stage_05_virtual/
│   └── analysis_summary.csv
└── visualizations/
    ├── entropy_space/
    ├── molecular_language/
    └── phase_lock/
```

## Use Cases

### 🔬 Scientific Research
- **Metabolomics**: S-Entropy coordinate analysis for metabolite identification
- **Proteomics**: Fragmentation pattern analysis with categorical completion
- **Biomarker Discovery**: Virtual instrument consensus for robust markers
- **Cross-Platform Studies**: Platform-independent categorical representation

### 🤖 Computer Vision
- **Ion-to-Droplet Conversion**: Thermodynamic image generation
- **Physics Validation**: Navier-Stokes constrained droplet parameters
- **Multi-Modal Analysis**: Spectral + visual feature fusion

### 🔗 Virtual Instruments
- **Ensemble Consensus**: Multi-instrument agreement scoring
- **Hardware Grounding**: Reality validation through oscillation harvesting
- **Phase-Lock Networks**: Molecular ensemble detection

## Publications

The framework is documented in several publications under `precursor/publication/`:

- **S-Entropy Coordinates**: Categorical coordinate system for mass spectrometry
- **Ion-to-Droplet Computer Vision**: Bijective thermodynamic image generation
- **Virtual Instruments**: Hardware-grounded virtual mass spectrometers
- **Molecular Language**: Categorical amino acid alphabet and fragmentation grammar

## Contributing

We welcome contributions to:

1. **Precursor Framework**: S-Entropy, virtual instruments, computer vision
2. **Buhera Language**: Rust-based language implementation
3. **Documentation**: Tutorials, examples, and best practices
4. **Validation**: Test cases and benchmarking datasets

See our [implementation roadmap](implementation-roadmap.md) for current development priorities.

## Community

- **GitHub**: [lavoisier](https://github.com/fullscreen-triangle/lavoisier)
- **Issues**: Report bugs and request features
- **Discussions**: Share use cases and get help

## License

Lavoisier is released under the MIT License. See LICENSE file for details.

---

*"Only the extraordinary can beget the extraordinary"* - Antoine Lavoisier

Transform your mass spectrometry analysis with S-Entropy coordinates and virtual instruments.

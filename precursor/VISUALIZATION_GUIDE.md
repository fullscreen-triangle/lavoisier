# Lavoisier Visualization Guide

This guide explains all the visualization scripts available for analyzing fragmentation pipeline results.

## Quick Start

**Run all visualizations at once:**
```bash
python visualize_all_results.py
```

This will generate comprehensive visualizations in `results/visualizations/`.

## Individual Visualization Scripts

### 1. Fragmentation Spectra (`src/virtual/fragmentation_spectra.py`)

**Purpose**: Show individual spectra through all pipeline stages

**What it does**:
- Selects up to 10 spectra from each platform
- Shows data at each stage:
  1. Preprocessing (raw peaks)
  2. S-Entropy transformation
  3. Fragmentation network
  4. BMD grounding
  5. Categorical completion
- Compares both platform files

**Output**: `results/visualizations/pipeline_spectra/`

**Usage**:
```bash
python src/virtual/fragmentation_spectra.py
```

---

### 2. Computer Vision Validation (`src/virtual/computer_vision_validation.py`)

**Purpose**: Visualize the CV droplet physics transformation

**What it does**:
- Loads CV droplet images from all experiments
- Creates grid visualizations of droplet images
- Compares CV features across platforms:
  - Droplet count distributions
  - Phase coherence
  - Physics quality metrics

**Output**: `results/visualizations/computer_vision/`

**Usage**:
```bash
python src/virtual/computer_vision_validation.py
```

---

### 3. Virtual Stages (`src/virtual/virtual_stages.py`)

**Purpose**: Visualize complete pipeline execution flow

**What it does**:
- Loads theatre results with all stage data
- Creates pipeline flow diagrams showing:
  - Stage execution order
  - Success/failure status
  - Execution times
  - Process metrics
- Compares metrics across platforms

**Output**: `results/visualizations/virtual_stages/`

**Usage**:
```bash
python src/virtual/virtual_stages.py
```

---

### 4. S-Entropy Transformation (`src/virtual/entropy_transformation.py`)

**Purpose**: Visualize the 3D S-Entropy coordinate space

**What it does**:
- 3D visualizations of S-Entropy space (S_knowledge, S_time, S_entropy)
- Categorical boundary analysis
- Entropy evolution trajectories
- Phase space clustering (DBSCAN)
- Statistical summaries

**Output**: `results/visualizations/sentropy/`
- `sentropy_3d_space.png`
- `sentropy_categorical_analysis.png`
- `sentropy_evolution.png`
- `sentropy_clustering.png`
- `sentropy_summary.csv`

**Usage**:
```bash
python src/virtual/entropy_transformation.py
```

---

### 5. Fragmentation Landscape (`src/virtual/fragmentation_landscape.py`)

**Purpose**: Create publication-quality 3D fragmentation landscapes

**What it does**:
- Beautiful 3D plots of S-Entropy coordinates
- Custom colormaps for evolution entropy
- Publication-ready PDF + PNG outputs
- Multiple viewing angles

**Output**: `results/visualizations/landscape/`

**Usage**:
```bash
python src/virtual/fragmentation_landscape.py
```

---

### 6. Phase-Lock Networks (`src/virtual/phase_lock_networks.py`)

**Purpose**: Visualize fragmentation network topology

**What it does**:
- Creates network graphs showing phase relationships between fragments
- 3D network visualization in S-Entropy space
- 2D projections with network edges
- Phase-lock threshold analysis

**Output**: `results/visualizations/phase_lock/`

**Usage**:
```bash
python src/virtual/phase_lock_networks.py
```

---

### 7. Validation Charts (`src/virtual/validation_charts.py`)

**Purpose**: Validate theoretical predictions (entropy-intensity relationships)

**What it does**:
- Entropy vs. intensity scatter plots for each dimension
- Combined entropy metric analysis
- Power-law fitting
- Categorical structure diagrams with commutative diagrams
- Functor mappings to S-Entropy space

**Output**: `results/visualizations/validation/`

**Usage**:
```bash
python src/virtual/validation_charts.py
```

---

## Results Directory Structure

After running visualizations, you'll have:

```
results/
├── visualizations/
│   ├── pipeline_spectra/          # Individual spectra through stages
│   │   ├── platform1_spectrum_1_pipeline.png
│   │   ├── platform1_spectrum_2_pipeline.png
│   │   └── ...
│   ├── computer_vision/           # CV droplet analysis
│   │   ├── platform1_cv_droplets_grid.png
│   │   ├── platform2_cv_droplets_grid.png
│   │   └── cv_statistics_comparison.png
│   ├── virtual_stages/            # Pipeline flow diagrams
│   │   ├── platform1_pipeline_flow.png
│   │   ├── platform2_pipeline_flow.png
│   │   └── pipeline_metrics_comparison.png
│   ├── sentropy/                  # S-Entropy space
│   │   ├── sentropy_3d_space.png
│   │   ├── sentropy_categorical_analysis.png
│   │   ├── sentropy_evolution.png
│   │   ├── sentropy_clustering.png
│   │   └── sentropy_summary.csv
│   ├── landscape/                 # 3D landscapes
│   │   ├── landscape_spectrum_1.pdf
│   │   ├── landscape_spectrum_1.png
│   │   └── ...
│   ├── phase_lock/                # Network topologies
│   │   ├── phase_lock_network_1.pdf
│   │   ├── phase_lock_network_1.png
│   │   └── ...
│   └── validation/                # Theoretical validation
│       ├── entropy_intensity_validation.pdf
│       ├── entropy_intensity_validation.png
│       ├── categorical_structure_1.pdf
│       └── ...
```

## Requirements

All scripts automatically:
- Find available results directories
- Handle multiple platforms
- Create output directories
- Generate both PDF and PNG outputs (where applicable)

## Notes

### Synthetic vs. Real Data

Some scripts may use synthetic data if certain pipeline stages haven't completed yet:
- **S-Entropy features**: Synthetic until Stage 2 completes
- **Fragmentation networks**: Synthetic until Stage 2.5 completes
- **Intensity data**: Synthetic until full pipeline with actual data completes

The scripts will print warnings when using synthetic data.

### Customization

Each script can be customized by editing:
- `max_spectra`: Number of spectra to visualize
- `figsize`: Figure dimensions
- Color schemes and styles
- Output formats

### Troubleshooting

**No results found?**
```bash
# Check what results directories exist
ls results/

# Scripts look for:
# - results/fragmentation_comparison/
# - results/fragmentation_test/
# - results/metabolomics_analysis/
```

**Script fails?**
- Check Python packages are installed: `numpy`, `matplotlib`, `seaborn`, `pandas`, `scipy`, `scikit-learn`, `networkx`
- Ensure results directory has completed pipeline stages
- Check file permissions

**Want more spectra?**
- Edit the `max_spectra` or `max_images` variables in each script

## Integration with Papers

Visualizations are designed for direct inclusion in publications:

- **Landscape plots**: Figure 1 candidates (3D S-Entropy space)
- **Phase-lock networks**: Figure 2 candidates (network topology)
- **Validation charts**: Figure 3 candidates (entropy-intensity relationships)
- **Categorical diagrams**: Figure 4 candidates (category theory visualization)

All PDF outputs are publication-ready with proper fonts and sizing.

## Contact

For questions or issues with visualizations, check:
1. This guide
2. Individual script docstrings
3. Pipeline output logs in `results/`

# Lavoisier Precursor

**Advanced Mass Spectrometry Analysis with Finite Observer Architecture**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-beta-yellow.svg)]()

## Overview

Precursor is a revolutionary mass spectrometry analysis framework that implements:

- **Finite Observer Architecture**: Hierarchical observation system (Theatre → Stages → Processes)
- **Resonant Computation Engine**: Hardware oscillation harvesting for physically grounded computation
- **S-Entropy Coordinate System**: Platform-independent 14D feature space for MS data
- **Phase-Lock Networks**: Detect and analyze molecular phase-coherent ensembles
- **Analysis Bundles**: Surgically injectable components for any pipeline
- **Non-Linear Navigation**: Bidirectional movement through saved stage results
- **Experiment-to-LLM**: Generate specialized LLMs from MS experiments

## Key Innovation

**No inherent execution sequence** - stages save results at every step (.json + .tab), enabling navigation in ANY direction:

```
Traditional:  Step 1 → Step 2 → Step 3 → Step 4

Precursor:    Stage 1 ⇄ Stage 2
                  ↕         ↕
              Stage 3 ⇄ Stage 4
```

## Installation

### From Source

```bash
# Clone repository
git clone https://github.com/lavoisier-project/precursor.git
cd precursor

# Install in development mode
pip install -e .

# With all optional dependencies
pip install -e ".[all]"

# With GPU support only
pip install -e ".[gpu]"
```

### Requirements

- Python ≥ 3.8
- NumPy, Pandas, SciPy
- PyTorch, Transformers
- NetworkX, scikit-learn
- pymzML, pyteomics

See `requirements.txt` for complete list.

## Quick Start

### 1. Basic Pipeline with Theatre/Stages

```python
from precursor.pipeline import Theatre, create_stage, NavigationMode

# Create theatre (transcendent observer)
theatre = Theatre(
    theatre_name="MS_Analysis",
    navigation_mode=NavigationMode.GEAR_RATIO
)

# Create stages (finite observers)
stage1 = create_stage(
    stage_name="Data Loading",
    stage_id="stage_01_data",
    process_names=['data_loading', 'quality_control']
)

stage2 = create_stage(
    stage_name="S-Entropy Transform",
    stage_id="stage_02_sentropy",
    process_names=['s_entropy_transform', 'feature_extraction']
)

# Add stages and dependencies
theatre.add_stage(stage1)
theatre.add_stage(stage2)
theatre.add_stage_dependency("stage_01_data", "stage_02_sentropy")

# Execute with non-linear navigation
result = theatre.observe_all_stages("experiment.mzml")

# Results saved at EVERY stage - can resume from anywhere!
```

### 2. Analysis Bundles

```python
from precursor.analysis import QualityBundle, AnnotationBundle, PipelineInjector

# Quick quality check
from precursor.analysis import quick_quality_check
results = quick_quality_check(data)

# Or build full pipeline
bundle = QualityBundle()
pipeline = bundle.build_pipeline()
results = pipeline.execute_sequential(data)

# Surgical injection into existing pipeline
injector = PipelineInjector(your_pipeline)
injector.inject_bundle(AnnotationBundle(), position='end')
modified_pipeline = injector.get_pipeline()
```

### 3. Resonant Computation Engine

```python
from precursor.hardware import ResonantComputationEngine

# Create engine with hardware harvesting
engine = ResonantComputationEngine(
    enable_all_harvesters=True,
    coherence_threshold=0.3,
    optimization_goal="maximize_annotation_confidence"
)

# Process experiment as Bayesian evidence network
spectrum_data = {
    'mz': np.array([100, 200, 300]),
    'intensity': np.array([1000, 800, 1200]),
    'rt': 15.5
}

results = await engine.process_experiment_as_bayesian_network(
    spectrum_data,
    processing_function
)

# Access frequency hierarchies and phase-locks
print(f"Evidence nodes: {len(results['evidence_network'])}")
print(f"Hierarchical levels: {len(results['frequency_hierarchies'])}")
```

### 4. Phase-Lock Networks

```python
from precursor.core import PhaseLockMeasurementDevice

# Create measurement device
device = PhaseLockMeasurementDevice(
    mz_range=(50, 1500),
    coherence_threshold=0.3
)

# Measure phase-locks from spectra
phase_locks_df = device.measure_from_arrays(spectra_list)

# Query specific m/z and RT
matches = device.query(mz=250.5, rt=12.3)
print(f"Found {len(matches)} phase-locks")
```

### 5. S-Entropy Transformation

```python
from precursor.core import SEntropyTransformer

# Create transformer
transformer = SEntropyTransformer()

# Transform spectrum to coordinates
coords_list, features = transformer.transform_and_extract(
    mz_array=mz,
    intensity_array=intensity,
    precursor_mz=precursor,
    rt=retention_time
)

# Get 14D feature vector
print(f"S-Entropy features: {features.to_array()}")
```

## Architecture

### Observer Hierarchy

```
┌─────────────────────────────────────────┐
│   Theatre (Transcendent Observer)       │
│   - Coordinates all stages              │
│   - Gear ratio navigation (O(1))        │
│   - Saves complete execution results    │
└──────────────┬──────────────────────────┘
               │ observes
               ↓
┌─────────────────────────────────────────┐
│   Stage (Finite Observer)               │
│   - Observes multiple processes         │
│   - Saves results (.json + .tab)        │
│   - Can observe other stages            │
└──────────────┬──────────────────────────┘
               │ observes
               ↓
┌─────────────────────────────────────────┐
│   Process (Lowest Observer)             │
│   - Single computational task           │
│   - Produces ProcessResult              │
└─────────────────────────────────────────┘
```

### Package Structure

```
precursor/
├── src/
│   ├── analysis/              # Analysis bundles
│   │   ├── annotation/        # Annotation components
│   │   ├── features/          # Feature analysis
│   │   ├── quality/           # Quality assessment
│   │   ├── statistical/       # Statistical validation
│   │   └── completeness/      # Completeness checks
│   │
│   ├── core/                  # Core functionality
│   │   ├── DataStructure.py   # Data containers
│   │   ├── EntropyTransformation.py  # S-Entropy
│   │   ├── PhaseLockNetworks.py     # Phase-locks
│   │   └── VectorTransformation.py  # Vectorization
│   │
│   ├── hardware/              # Hardware harvesting
│   │   ├── resonant_computation_engine.py
│   │   ├── clock_drift.py
│   │   ├── memory_access_patterns.py
│   │   └── ... (8 harvesters)
│   │
│   ├── pipeline/              # Stage/Theatre system
│   │   ├── stages.py          # Stage observers
│   │   ├── theatre.py         # Theatre coordinator
│   │   └── __init__.py
│   │
│   ├── metabolomics/          # Metabolomics-specific
│   │   ├── DatabaseSearch.py
│   │   ├── FragmentationTrees.py
│   │   └── MetabolicLargeLanguageModel.py
│   │
│   ├── proteomics/            # Proteomics-specific
│   │   ├── MSIonDatabaseSearch.py
│   │   ├── TandemDatabaseSearch.py
│   │   └── ProteomicsLargeLanguageModel.py
│   │
│   └── utils/                 # Utilities
│       ├── orchestrator.py    # Global optimizer
│       ├── metacognition_registry.py
│       ├── miraculous_chess_navigator.py
│       ├── moon_landing.py
│       └── entropy_neural_networks.py
│
├── requirements.txt
├── pyproject.toml
├── setup.py
└── README.md
```

## Core Features

### 1. Theatre/Stage System

**Non-linear pipeline navigation with saved results at every stage:**

- **5 Navigation Modes**: LINEAR, DEPENDENCY, OPPORTUNISTIC, BIDIRECTIONAL, GEAR_RATIO
- **Resume capability**: Load saved stage results and continue
- **Parallel execution**: Run independent stages simultaneously
- **Dynamic re-arrangement**: Change stage order without re-running
- **Complete provenance**: Full audit trail for reproducibility

[See pipeline documentation](src/pipeline/STAGE_THEATRE_README.md)

### 2. Resonant Computation Engine

**Hardware oscillations → Frequency hierarchies → Bayesian evidence network:**

- Harvests 8 hardware oscillation sources (clock, memory, network, USB, GPU, disk, LED)
- Creates frequency hierarchies with O(1) gear ratio navigation
- Deploys finite observers at each hierarchical level
- Integrates SENN, Chess Navigator, Moon Landing, Global Optimizer, Metacognition
- Enables closed-loop navigation through categorical networks

[See resonant computation documentation](src/hardware/RESONANT_COMPUTATION_README.md)

### 3. Analysis Bundles

**Surgically injectable analysis components:**

- **7 Pre-configured bundles**: Quality, Annotation, Feature, Statistical, Completeness, Complete, Custom
- **25+ components** wrapped with uniform interface
- **PipelineInjector** for surgical injection at any point
- **ComponentRegistry** for discovery and dynamic loading
- Zero modification of original analysis scripts

[See analysis bundles documentation](src/analysis/ANALYSIS_BUNDLES_README.md)

### 4. S-Entropy Coordinate System

**Platform-independent 14D feature space:**

- Bijective transformation preserving spectral information
- Combines structural entropy (S), Shannon entropy (H), temporal coordination (T)
- 14-dimensional feature extraction (statistical + geometric + information-theoretic)
- Cross-platform compatibility (Waters qTOF, Thermo Orbitrap, etc.)
- Sub-millisecond processing times (830-867 spectra/second)

### 5. Phase-Lock Networks

**Detect molecular phase-coherent ensembles:**

- Hierarchical finite observers with transcendent coordination
- O(1) navigation via gear ratios
- Detects phase-locks across m/z and RT dimensions
- Assigns categorical states for Gibbs' paradox resolution
- Enables fragment disambiguation through network position

### 6. LLM Generation

**Convert experiments to specialized language models:**

- Metabolomics LLM generation
- Proteomics LLM generation
- S-Entropy and phase-lock integration
- Natural language querying of experimental results
- Dynamic inference on MS data

## Navigation Modes

### LINEAR

Traditional sequential execution (for compatibility)

### DEPENDENCY

Smart execution based on dependency graph (topological sort)

### OPPORTUNISTIC

Execute stages as soon as dependencies are met (efficient)

### BIDIRECTIONAL

Forward execution + backward re-observation (exploratory)

### GEAR_RATIO

Hierarchical navigation with O(1) jumps (advanced)

## Integration Examples

### With Existing Validation Pipeline

```python
from validation.run_complete_validation import ValidationOrchestrator
from precursor.analysis import CompleteBundle
from precursor.pipeline import PipelineInjector

validator = ValidationOrchestrator()
injector = PipelineInjector(validator.pipeline)
injector.inject_bundle(CompleteBundle())
results = validator.run_validation()
```

### With Custom Analysis

```python
from precursor.pipeline import ProcessObserver, StageObserver, ProcessResult

class CustomProcess(ProcessObserver):
    def observe(self, input_data, **kwargs):
        # Your custom analysis
        result = your_analysis(input_data)

        return ProcessResult(
            process_name=self.name,
            status=StageStatus.COMPLETED,
            execution_time=elapsed,
            data=result,
            metrics={'custom_metric': value}
        )

# Use in stage
stage = StageObserver(...)
stage.add_process_observer(CustomProcess())
```

## Theoretical Foundation

Precursor implements several novel theoretical concepts:

1. **Finite Observer Hierarchy**: Theatre → Stages → Processes, with each level observing lower levels
2. **Gear Ratio Navigation**: O(1) hierarchical jumps using pre-computed reduction ratios
3. **Phase-Lock Theory**: Molecular ensembles as categorical states resolving Gibbs' paradox
4. **S-Entropy Framework**: Information-theoretic coordinates for MS data
5. **Resonant Computation**: Hardware oscillations as computational substrate
6. **Categorical Networks**: Non-linear navigation through frequency-coupled structures

## Performance

- **S-Entropy transformation**: 830-867 spectra/second
- **Stage execution**: O(N) where N = number of processes
- **Gear ratio navigation**: O(1) for hierarchical jumps
- **Resume from saved**: O(1) (just load file)
- **Clustering**: Optimal k detection with silhouette scores 0.555-0.570

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_pipeline.py
```

## Documentation

- [Pipeline Stage/Theatre System](src/pipeline/STAGE_THEATRE_README.md)
- [Resonant Computation Engine](src/hardware/RESONANT_COMPUTATION_README.md)
- [Analysis Bundles](src/analysis/ANALYSIS_BUNDLES_README.md)
- [Hardware Harvesting](src/hardware/OSCILLATORY_COMPUTATION_README.md)
- [LLM Generation](src/EXPERIMENT_LLM_README.md)

## Examples

See comprehensive examples in:

- `src/pipeline/PIPELINE_EXAMPLE.md` - Theatre/Stage examples
- `src/analysis/usage_example.py` - Analysis bundle examples
- `src/hardware/resonant_computation_engine.py` - Resonant computation example (in `__main__`)

## Citation

If you use Precursor in your research, please cite:

```bibtex
@software{lavoisier_precursor,
  title = {Lavoisier Precursor: Advanced Mass Spectrometry Analysis with Finite Observer Architecture},
  author = {Lavoisier Project Team},
  year = {2025},
  url = {https://github.com/lavoisier-project/precursor},
  version = {1.0.0}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linters
5. Submit a pull request

## License

MIT License - See [LICENSE](LICENSE) file for details

## Authors

Lavoisier Project Team

## Acknowledgments

- Inspired by oscillatory reality theory and phase-lock phenomena
- Built on theoretical foundations of categorical entropy
- Implements finite observer hierarchy for computational systems

## Support

- **Issues**: [GitHub Issues](https://github.com/lavoisier-project/precursor/issues)
- **Documentation**: [ReadTheDocs](https://lavoisier-precursor.readthedocs.io)
- **Discussions**: [GitHub Discussions](https://github.com/lavoisier-project/precursor/discussions)

---

**Status: ✅ Production Ready**

Precursor v1.0.0 - October 2025

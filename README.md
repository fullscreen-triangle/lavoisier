<h1 align="center">Lavoisier</h1>
<p align="center"><em> Only the extraordinary can beget the extraordinary</em></p>

<p align="center">
  <img src="Antoine_lavoisier-copy.jpg" alt="Spectacular Logo" width="300"/>
</p>

[![Python Version](https://img.shields.io/pypi/pyversions/science-platform.svg)](https://pypi.org/project/science-platform/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Lavoisier is a high-performance computing solution for mass spectrometry-based metabolomics data analysis pipelines. It combines traditional numerical methods with advanced visualization and AI-driven analytics to provide comprehensive insights from high-volume MS data.

## Core Architecture

Lavoisier features a metacognitive orchestration layer that coordinates two main pipelines:

1. **Numerical Analysis Pipeline**: Uses established computational methods for ion spectra extraction, annotates ion peaks through database search, fragmentation rules, and natural language processing.

2. **Visual Analysis Pipeline**: Converts spectra into video format and applies computer vision methods for annotation.

The orchestration layer manages workflow execution, resource allocation, and integrates LLM-powered intelligence for analysis and decision-making.

```
┌────────────────────────────────────────────────────────────────┐
│                   Metacognitive Orchestration                   │
│                                                                │
│  ┌──────────────────────┐          ┌───────────────────────┐   │
│  │                      │          │                       │   │
│  │  Numerical Pipeline  │◄────────►│  Visual Pipeline      │   │
│  │                      │          │                       │   │
│  └──────────────────────┘          └───────────────────────┘   │
│                 ▲                              ▲                │
│                 │                              │                │
│                 ▼                              ▼                │
│  ┌──────────────────────┐          ┌───────────────────────┐   │
│  │                      │          │                       │   │
│  │  Model Repository    │◄────────►│  LLM Integration      │   │
│  │                      │          │                       │   │
│  └──────────────────────┘          └───────────────────────┘   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Command Line Interface

Lavoisier provides a high-performance CLI interface for seamless interaction with all system components:

- Built with modern CLI frameworks for visually pleasing, intuitive interaction
- Color-coded outputs, progress indicators, and interactive components
- Command completions and contextual help
- Workflow management and pipeline orchestration
- Integrated with LLM assistants for natural language interaction
- Configuration management and parameter customization
- Results visualization and reporting

## Numerical Processing Pipeline

The numerical pipeline processes raw mass spectrometry data through a distributed computing architecture, specifically designed for handling large-scale MS datasets:

### Raw Data Processing
- Extracts MS1 and MS2 spectra from mzML files
- Performs intensity thresholding (MS1: 1000.0, MS2: 100.0 by default)
- Applies m/z tolerance filtering (0.01 Da default)
- Handles retention time alignment (0.5 min tolerance)

### Comprehensive MS2 Annotation
- Multi-database annotation system integrating multiple complementary resources
- Spectral matching against libraries (MassBank, METLIN, MzCloud, in-house)
- Accurate mass search across HMDB, LipidMaps, KEGG, and PubChem
- Fragmentation tree generation for structural elucidation
- Pathway integration with KEGG and HumanCyc databases
- Multi-component confidence scoring system for reliable identifications
- Deep learning models for MS/MS prediction and interpretation

### Enhanced MS2 Analysis
- Deep learning models for spectral interpretation
- Transfer learning from large-scale metabolomics datasets
- Model serialization for all analytical outputs
- Automated hyperparameter optimization

### Distributed Computing
- Utilizes Ray for parallel processing
- Implements Dask for large dataset handling
- Automatic resource management based on system capabilities
- Dynamic workload distribution across available cores

### Data Management
- Efficient data storage using Zarr format
- Compressed data storage with LZ4 compression
- Parallel I/O operations for improved performance
- Hierarchical data organization

### Processing Features
- Automatic chunk size optimization
- Memory-efficient processing
- Progress tracking and reporting
- Comprehensive error handling and logging

## Visual Analysis Pipeline

The visualization pipeline transforms processed MS data into interpretable visual formats:

### Spectrum Analysis
- MS image database creation and management
- Feature extraction from spectral data
- Resolution-specific image generation (default 1024x1024)
- Feature dimension handling (128-dimensional by default)

### Visualization Generation
- Creates time-series visualizations of MS data
- Generates analysis videos showing spectral changes
- Supports multiple visualization formats
- Custom color mapping and scaling

### Data Integration
- Combines multiple spectra into cohesive visualizations
- Temporal alignment of spectral data
- Metadata integration into visualizations
- Batch processing capabilities

### Output Formats
- High-resolution image generation
- Video compilation of spectral changes
- Interactive visualization options
- Multiple export formats support

## LLM Integration & Continuous Learning

Lavoisier integrates commercial and open-source LLMs to enhance analytical capabilities and enable continuous learning:

### Assistive Intelligence
- Natural language interface through CLI
- Context-aware analytical assistance
- Automated report generation
- Expert knowledge integration

### Solver Architecture
- Integration with Claude, GPT, and other commercial LLMs
- Local models via Ollama for offline processing
- Numerical model API endpoints for LLM queries
- Pipeline result interpretation

### Continuous Learning System
- Feedback loop capturing new analytical results
- Incremental model updates via train-evaluate cycles
- Knowledge distillation from commercial LLMs to local models
- Versioned model repository with performance tracking

### Metacognitive Query Generation
- Auto-generated queries of increasing complexity
- Integration of numerical model outputs with LLM knowledge
- Comparative analysis between numeric and visual pipelines
- Knowledge extraction and synthesis

## Specialized Models Integration

Lavoisier incorporates domain-specific models for advanced analysis tasks:

### Biomedical Language Models
- BioMedLM integration for biomedical text analysis and generation
- Context-aware analysis of mass spectrometry data
- Biological pathway interpretation and metabolite identification
- Custom prompting templates for different analytical tasks

### Scientific Text Encoders
- SciBERT model for scientific literature processing and embedding
- Multiple pooling strategies for optimal text representation
- Similarity-based search across scientific documents
- Batch processing of large text collections

### Chemical Named Entity Recognition
- PubMedBERT-NER-Chemical for extracting chemical compounds from text
- Identification and normalization of chemical nomenclature
- Entity replacement for text preprocessing
- High-precision extraction with confidence scoring

### Proteomics Analysis
- InstaNovo model for de novo peptide sequencing
- Integration of proteomics and metabolomics data
- Cross-modal analysis for comprehensive biomolecule profiling
- Advanced protein identification workflows

## Key Capabilities

### Performance
- Processing speeds: Up to 1000 spectra/second (hardware dependent)
- Memory efficiency: Streaming processing for large datasets
- Scalability: Automatic adjustment to available resources
- Parallel processing: Multi-core utilization

### Data Handling
- Input formats: mzML (primary), with extensible format support
- Output formats: Zarr, HDF5, video (MP4), images (PNG/JPEG)
- Data volumes: Capable of handling datasets >100GB
- Batch processing: Multiple file handling

### Annotation Capabilities
- Multi-tiered annotation combining spectral matching and accurate mass search
- Integrated pathway analysis for biological context
- Confidence scoring system weighing multiple evidence sources
- Parallelized database searches for rapid compound identification
- Isotope pattern matching and fragmentation prediction
- RT prediction for additional identification confidence

### Quality Control
- Automated validation checks
- Signal-to-noise ratio monitoring
- Quality metrics reporting
- Error detection and handling

### Analysis Features
- Peak detection and quantification
- Retention time alignment
- Mass accuracy verification
- Intensity normalization

## Use Cases

### Proteomics Research
- Protein identification workflows
- Peptide quantification
- Post-translational modification analysis
- Comparative proteomics studies
- De novo peptide sequencing with InstaNovo integration
- Cross-analysis of proteomics and metabolomics datasets
- Protein-metabolite interaction mapping

### Metabolomics Studies
- Metabolite profiling
- Pathway analysis
- Biomarker discovery
- Time-series metabolomics

### Quality Control
- Instrument performance monitoring
- Method validation
- Batch effect detection
- System suitability testing

### Data Visualization
- Scientific presentation
- Publication-quality figures
- Time-course analysis
- Comparative analysis visualization

## Results & Validation

Our comprehensive validation demonstrates the effectiveness of Lavoisier's dual-pipeline approach through rigorous statistical analysis and performance metrics:

### Core Performance Metrics
- **Feature Extraction Accuracy**: 0.989 similarity score between pipelines, with complementarity index of 0.961
- **Vision Pipeline Robustness**: 0.954 stability score against noise/perturbations
- **Annotation Performance**: Numerical pipeline achieves perfect accuracy (1.0) for known compounds
- **Temporal Consistency**: 0.936 consistency score for time-series analysis
- **Anomaly Detection**: Low anomaly score of 0.02, indicating reliable performance

### Example Analysis Results

#### Mass Spectrometry Analysis
![Full MS Scan](analytical_visualizations/20250527_094000/mass_spectra/full_scan.png)
*Full scan mass spectrum showing the comprehensive metabolite profile with high mass accuracy and resolution*

![MS/MS Analysis](analytical_visualizations/20250527_094000/mass_spectra/glucose_msms.png)
*MS/MS fragmentation pattern analysis for glucose, demonstrating detailed structural elucidation*

![Feature Comparison](analytical_visualizations/20250527_094000/feature_analysis/feature_comparison.png)
*Comparison of feature extraction between numerical and visual pipelines, showing high concordance and complementarity*

#### Visual Pipeline Output
https://github.com/username/lavoisier/raw/main/public/output/visual/videos/analysis_video.mp4

*Computer vision pipeline output showing real-time spectral analysis and feature tracking. The video demonstrates the system's ability to:*
- Track metabolite features across time
- Visualize intensity changes dynamically
- Highlight structural similarities
- Detect pattern changes in real-time

#### Analysis Outputs
The system generates comprehensive analytical outputs organized in:

1. **Time Series Analysis** (`time_series/`)
   - Chromatographic peak tracking
   - Retention time alignment
   - Intensity variation monitoring

2. **Feature Analysis** (`feature_analysis/`)
   - Principal component analysis
   - Feature clustering
   - Pattern recognition results

3. **Interactive Dashboards** (`interactive_dashboards/`)
   - Real-time data exploration
   - Dynamic filtering capabilities
   - Interactive peak annotation

4. **Publication Quality Figures** (`publication_figures/`)
   - High-resolution spectral plots
   - Statistical analysis visualizations
   - Comparative analysis figures

### Pipeline Complementarity
The dual-pipeline approach shows strong synergistic effects:
- **Feature Comparison**: Multiple validation scores [1.0, 0.999, 0.999, 0.999, 0.932, 1.0] across different aspects
- **Vision Analysis**: Robust performance in both noise resistance (0.914) and temporal analysis (0.936)
- **Annotation Synergy**: While numerical pipeline excels in accuracy, visual pipeline provides complementary insights

### Validation Methodology
For detailed information about our validation approach and complete results, please refer to:
- [Visualization Documentation](docs/visualization.md) - Comprehensive analysis framework
- `validation_results/` - Raw validation data and metrics
- `validation_visualizations/` - Interactive visualizations and temporal analysis
- `analytical_visualizations/` - Detailed analytical outputs

## Project Structure

```
lavoisier/
├── pyproject.toml            # Project metadata and dependencies
├── LICENSE                   # Project license
├── README.md                 # This file
├── docs/                     # Documentation
│   ├── user_guide.md         # User documentation
│   └── developer_guide.md    # Developer documentation
├── lavoisier/                # Main package
│   ├── __init__.py           # Package initialization
│   ├── cli/                  # Command-line interface
│   │   ├── __init__.py
│   │   ├── app.py            # CLI application entry point
│   │   ├── commands/         # CLI command implementations
│   │   └── ui/               # Terminal UI components
│   ├── core/                 # Core functionality
│   │   ├── __init__.py
│   │   ├── metacognition.py  # Orchestration layer
│   │   ├── config.py         # Configuration management
│   │   ├── logging.py        # Logging utilities
│   │   └── ml/               # Machine learning components
│   │       ├── __init__.py
│   │       ├── models.py     # ML model implementations
│   │       └── MSAnnotator.py # MS2 annotation engine
│   ├── numerical/            # Numerical pipeline
│   │   ├── __init__.py
│   │   ├── processing.py     # Data processing functions
│   │   ├── pipeline.py       # Main pipeline implementation
│   │   ├── ms1.py            # MS1 spectra analysis
│   │   ├── ms2.py            # MS2 spectra analysis
│   │   ├── ml/               # Machine learning components
│   │   │   ├── __init__.py
│   │   │   ├── models.py     # ML model definitions
│   │   │   └── training.py   # Training utilities
│   │   ├── distributed/      # Distributed computing
│   │   │   ├── __init__.py
│   │   │   ├── ray_utils.py  # Ray integration
│   │   │   └── dask_utils.py # Dask integration
│   │   └── io/               # Input/output operations
│   │       ├── __init__.py
│   │       ├── readers.py    # File format readers
│   │       └── writers.py    # File format writers
│   ├── visual/               # Visual pipeline
│   │   ├── __init__.py
│   │   ├── conversion.py     # Spectra to visual conversion
│   │   ├── processing.py     # Visual processing
│   │   ├── video.py          # Video generation
│   │   └── analysis.py       # Visual analysis
│   ├── llm/                  # LLM integration
│   │   ├── __init__.py
│   │   ├── api.py            # API for LLM communication
│   │   ├── ollama.py         # Ollama integration
│   │   ├── commercial.py     # Commercial LLM integrations
│   │   └── query_gen.py      # Query generation
│   ├── models/               # Model repository
│   │   ├── __init__.py
│   │   ├── repository.py     # Model management
│   │   ├── distillation.py   # Knowledge distillation
│   │   └── versioning.py     # Model versioning
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── helpers.py        # General helpers
│       └── validation.py     # Validation utilities
├── tests/                    # Tests
│   ├── __init__.py
│   ├── test_numerical.py
│   ├── test_visual.py
│   ├── test_llm.py
│   └── test_cli.py
└── examples/                 # Example workflows
    ├── basic_analysis.py
    ├── distributed_processing.py
    ├── llm_assisted_analysis.py
    └── visual_analysis.py
```

## Installation & Usage

### Installation

```bash
pip install lavoisier
```

For development installation:

```bash
git clone https://github.com/username/lavoisier.git
cd lavoisier
pip install -e ".[dev]"
```

### Basic Usage

Process a single MS file:

```bash
lavoisier process --input sample.mzML --output results/
```

Run with LLM assistance:

```bash
lavoisier analyze --input sample.mzML --llm-assist
```
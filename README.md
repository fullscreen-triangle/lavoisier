# Lavoisier

Lavoisier is a high performance computing solution for high data volume mass spectrometry based metabolomics data analysis 
pipelines. Lavoisier contains two main pipelines : 
1. A numerical analysis pipeline that uses established computational methods for ion spectra extraction, annotates ion peaks through a combination of database search, application of fragmentation rules and natural language processing methods.
2. An in-house visual method that converts spectra into a video format and applies computer vision methods for annotation.


### Numerical Processing Pipeline
The numerical pipeline processes raw mass spectrometry data through a distributed computing architecture, specifically designed for handling large-scale MS datasets. It provides:

1. **Raw Data Processing**
   - Extracts MS1 and MS2 spectra from mzML files
   - Performs intensity thresholding (MS1: 1000.0, MS2: 100.0 by default)
   - Applies m/z tolerance filtering (0.01 Da default)
   - Handles retention time alignment (0.5 min tolerance)

2. **Distributed Computing**
   - Utilizes Ray for parallel processing
   - Implements Dask for large dataset handling
   - Automatic resource management based on system capabilities
   - Dynamic workload distribution across available cores

3. **Data Management**
   - Efficient data storage using Zarr format
   - Compressed data storage with LZ4 compression
   - Parallel I/O operations for improved performance
   - Hierarchical data organization

4. **Processing Features**
   - Automatic chunk size optimization
   - Memory-efficient processing
   - Progress tracking and reporting
   - Comprehensive error handling and logging

### Visual Analysis Pipeline
The visualization pipeline transforms processed MS data into interpretable visual formats, focusing on:

1. **Spectrum Analysis**
   - MS image database creation and management
   - Feature extraction from spectral data
   - Resolution-specific image generation (default 1024x1024)
   - Feature dimension handling (128-dimensional by default)

2. **Visualization Generation**
   - Creates time-series visualizations of MS data
   - Generates analysis videos showing spectral changes
   - Supports multiple visualization formats
   - Custom color mapping and scaling

3. **Data Integration**
   - Combines multiple spectra into cohesive visualizations
   - Temporal alignment of spectral data
   - Metadata integration into visualizations
   - Batch processing capabilities

4. **Output Formats**
   - High-resolution image generation
   - Video compilation of spectral changes
   - Interactive visualization options
   - Multiple export formats support

### Key Capabilities

1. **Performance**
   - Processing speeds: Up to 1000 spectra/second (hardware dependent)
   - Memory efficiency: Streaming processing for large datasets
   - Scalability: Automatic adjustment to available resources
   - Parallel processing: Multi-core utilization

2. **Data Handling**
   - Input formats: mzML (primary), with extensible format support
   - Output formats: Zarr, HDF5, video (MP4), images (PNG/JPEG)
   - Data volumes: Capable of handling datasets >100GB
   - Batch processing: Multiple file handling

3. **Quality Control**
   - Automated validation checks
   - Signal-to-noise ratio monitoring
   - Quality metrics reporting
   - Error detection and handling

4. **Analysis Features**
   - Peak detection and quantification
   - Retention time alignment
   - Mass accuracy verification
   - Intensity normalization

### Use Cases

1. **Proteomics Research**
   - Protein identification workflows
   - Peptide quantification
   - Post-translational modification analysis
   - Comparative proteomics studies

2. **Metabolomics Studies**
   - Metabolite profiling
   - Pathway analysis
   - Biomarker discovery
   - Time-series metabolomics

3. **Quality Control**
   - Instrument performance monitoring
   - Method validation
   - Batch effect detection
   - System suitability testing

4. **Data Visualization**
   - Scientific presentation
   - Publication-quality figures
   - Time-course analysis
   - Comparative analysis visualization

## References 
7. [Sachikonye, K. (2025). Lavoisier: A High-Performance Computing Solution for MassSpectrometry-Based Metabolomics with Novel Video AnalysisPipeline](./lavoisier.pdf)

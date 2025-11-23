# Publication Validation Charts

## Overview

This directory contains 5 publication-ready visualization scripts for the Lavoisier metabolomics framework validation results. Each script generates a 4-panel figure (20 charts total) with embedded data, requiring no external file dependencies.

## Generated Figures

### 1. **Quality Control Validation** (`quality_control.py`)
**4 Panels:**
- A. Quality Score Distributions
- B. Threshold Filtering Effectiveness
- C. Assessment Performance Metrics
- D. Dataset Comparison Summary

**Data Source:** `step_02_quality_control_results.json`

**Key Findings:**
- Mean quality scores: 0.677 (PL_Neg) and 0.661 (TG_Pos)
- High-quality spectra: 85% and 79%
- Assessment rate: ~4,200 spectra/second
- Optimal quality threshold: 0.5

---

### 2. **Database Search Performance** (`database_search.py`)
**4 Panels:**
- A. Database Search Performance
- B. Database Hit Rate Matrix
- C. Search Efficiency by Category
- D. Search Performance Summary

**Data Source:** `step_03_database_search_results.json`

**Key Findings:**
- 8 databases queried (LIPIDMAPS, MSLIPIDS, PUBCHEM, METLIN, MASSBANK, MZCLOUD, KEGG, HUMANCYC)
- Search rate: ~6,000 spectra/second
- Current annotation rate: 0% (indicates need for improved database coverage)
- Fastest database: PUBCHEM (20,000 spec/s)

---

### 3. **Spectrum Embedding Analysis** (`spectrum_embedding.py`)
**4 Panels:**
- A. Embedding Performance Comparison
- B. Similarity Search Performance
- C. Dimensionality vs Performance (with bubble plot)
- D. Embedding Quality Summary

**Data Source:** `step_04_spectrum_embedding_results.json`

**Key Findings:**
- 3 embedding methods tested: spec2vec, stellas, fingerprint
- Best for speed: fingerprint (6,171 spec/s)
- Best for similarity: spec2vec (avg score 0.781-0.802)
- 100% success rate across all methods
- 180 total embeddings created

---

### 4. **Feature Clustering Results** (`feature_clustering_numerical.py`)
**4 Panels:**
- A. Clustering Quality vs Configuration
- B. Cluster Balance Analysis
- C. Clustering Performance
- D. Clustering Summary

**Data Source:** `step_05_feature_clustering_results.json`

**Key Findings:**
- 14-dimensional feature space
- 4 cluster configurations tested (k=3, 5, 8, 10)
- Best configuration: k=3 (quality score 0.867-0.909)
- Feature diversity: 0.555-0.570
- Clustering rate: 700-2,400 spectra/second

---

### 5. **Comparative Analysis: Numerical vs Visual** (`comparative_analysis.py`)
**4 Panels:**
- A. Processing Efficiency Comparison
- B. Annotation Performance Analysis
- C. Method Capabilities Profile (radar chart)
- D. Comprehensive Method Comparison

**Data Sources:**
- `numerical_validation_results.json`
- `visual_validation_results.json`

**Key Findings:**
- **Numerical Pipeline:** Faster (93 spec/s), better annotation (1.04%), multi-database
- **Visual Pipeline:** Complete ion extraction (167 ions/spec), perfect CV conversion (100%)
- **Strengths:** Numerical excels at speed and annotation; Visual excels at comprehensive ion information
- **Recommendation:** Numerical for production, Visual for research and validation

---

## Usage

### Generate All Charts
```bash
cd precursor/publication/entropy-coordinates/charts
python generate_all_charts.py
```

### Generate Individual Charts
```bash
python quality_control.py
python database_search.py
python spectrum_embedding.py
python feature_clustering_numerical.py
python comparative_analysis.py
```

## Output Files

All figures are saved as high-resolution PNG files (300 DPI) in the same directory:
- `quality_control_validation.png`
- `database_search_validation.png`
- `spectrum_embedding_validation.png`
- `feature_clustering_validation.png`
- `comparative_analysis.png`

## Technical Details

**Plotting Configuration:**
- Backend: Agg (non-interactive, for server/headless environments)
- Style: seaborn-v0_8-paper
- DPI: 300 (publication quality)
- Figure size: 16×12 inches per figure
- Font: Sans-serif, 11pt base

**Dependencies:**
- numpy
- matplotlib
- seaborn
- pathlib (standard library)

**Data Embedding:**
All JSON data is embedded directly in each script, making them self-contained and portable.

## Publication-Ready Features

✓ High-resolution (300 DPI) output
✓ Professional color schemes
✓ Clear axis labels and titles
✓ Comprehensive legends
✓ Statistical annotations
✓ Multi-panel layouts
✓ Consistent styling across all figures
✓ Self-contained (no external data dependencies)

## Modifications

To modify visualizations:
1. Edit the `RESULTS` dictionary in each script with updated data
2. Adjust plot parameters (colors, sizes, etc.) as needed
3. Rerun the script to regenerate the figure

## Author

Lavoisier Metabolomics Team
Date: October 27, 2025

## License

Part of the Lavoisier metabolomics framework.

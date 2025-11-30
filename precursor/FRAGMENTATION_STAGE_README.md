# Fragmentation Network Stage - Integration Documentation

## Overview

The **Fragmentation Network Stage (Stage 2.5)** implements the categorical fragmentation theory from the paper:

> **"Categorical Fragmentation: Phase-Lock Networks and the Resolution of Gibbs' Paradox in Mass Spectrometry"**

This stage integrates between S-Entropy transformation and BMD grounding to provide comprehensive fragmentation analysis for metabolomics.

---

## Theoretical Foundation

### Key Concepts

1. **Gibbs' Paradox Resolution**
   - Traditional fragmentation trees treat fragments as indistinguishable
   - S-Entropy coordinates place fragments in metric space
   - Network topology distinguishes fragments by their position in phase-lock networks

2. **Phase-Lock Networks**
   - Fragments are connected by harmonic relationships
   - Transforms hierarchical trees → dense scale-free networks
   - Small-world property: logarithmic path lengths

3. **Intensity-Entropy Relationship**
   ```
   I ∝ α(fragment) = exp(-|E_fragment|/⟨E⟩)
   ```
   - Fragment intensity proportional to oscillation termination probability
   - Network entropy (edge density) determines fragmentation patterns

4. **Platform Independence**
   - Categorical states (not instrument details) determine fragmentation
   - Network topology is invariant across platforms
   - Validation via coefficient of variation (CV) across instruments

---

## Pipeline Integration

### Stage Execution Order

```
Stage 1: Spectral Preprocessing
    ↓
Stage 2: S-Entropy Transformation
    ↓
Stage 2.5: Fragmentation Network Analysis  ← NEW
    ↓
Stage 3: Hardware BMD Grounding
    ↓
Stage 4: Categorical Completion
```

### Dependencies

- **Inputs**:
  - Raw MS1/MS2 spectra from Stage 1 (preprocessing)
  - S-Entropy features from Stage 2 (transformation)

- **Outputs**:
  - Fragmentation networks (precursor-fragment relationships)
  - Phase-lock network statistics
  - Intensity-entropy correlations
  - Platform independence metrics

---

## Stage Architecture

### Process 1: Fragmentation Network Build

**Purpose**: Build S-Entropy fragmentation networks from MS2 spectra

**Implementation**: `FragmentationNetworkBuildProcess`

**Operations**:
1. Group spectra by precursor (MS1 → MS2 relationships)
2. Create `PrecursorIon` and `FragmentIon` objects
3. Compute S-Entropy features for all ions
4. Build network edges based on semantic distance
5. Create directed acyclic graph (DAG) structure

**Outputs**:
- `fragmentation_network`: `SEntropyFragmentationNetwork` object
- Metrics: `n_precursors`, `n_fragments`, `n_edges`, `network_density`, `avg_degree`

---

### Process 2: Phase-Lock Network Analysis

**Purpose**: Detect harmonic coupling between fragments

**Implementation**: `PhaseLockNetworkAnalysisProcess`

**Operations**:
1. Identify edges representing phase-locks (semantic distance < tolerance)
2. Compute network topology metrics (degree distribution, clustering)
3. Detect scale-free property (power-law degree distribution)
4. Validate small-world property (clustering + short paths)

**Outputs**:
- `phase_locks`: List of phase-lock relationships
- Metrics: `n_phase_locks`, `clustering_coefficient`, `avg_path_length`, `max_degree`

---

### Process 3: Intensity-Entropy Analysis

**Purpose**: Validate theoretical intensity-entropy relationship

**Implementation**: `IntensityEntropyAnalysisProcess`

**Operations**:
1. For each fragment, compute local network entropy (edge density)
2. Calculate termination probability: α = exp(-|E|/⟨E⟩)
3. Correlate fragment intensity with termination probability
4. Validate theoretical prediction (positive correlation expected)

**Outputs**:
- `fragment_entropy_data`: Per-fragment entropy and intensity data
- Metrics: `n_fragments_analyzed`, `intensity_entropy_correlation`, `avg_termination_prob`

---

### Process 4: Platform Independence Validation

**Purpose**: Verify that categorical states are platform-independent

**Implementation**: `PlatformIndependenceValidationProcess`

**Operations**:
1. Compute network topology metrics (degree distribution, S-Entropy features)
2. Calculate coefficient of variation (CV) for key metrics
3. Compute platform independence score (lower CV = higher independence)
4. Validate that network structure is invariant across instruments

**Outputs**:
- Metrics: `degree_cv`, `sentropy_cv`, `edge_weight_cv`, `platform_independence_score`

---

## Configuration

### Parameters

```python
fragmentation={
    'similarity_threshold': 0.5,    # Semantic distance for edge creation (τ)
    'sigma': 0.2,                   # Scale parameter for edge weights
    'harmonic_tolerance': 0.01      # Phase-lock detection tolerance
}
```

### Example Usage

```python
from pathlib import Path
from src.pipeline.metabolomics import MetabolomicsTheatre

# Initialize theatre with fragmentation stage
theatre = MetabolomicsTheatre(
    output_dir=Path("./results/fragmentation_analysis"),
    enable_bmd_grounding=True,
    fragmentation={
        'similarity_threshold': 0.5,
        'sigma': 0.2,
        'harmonic_tolerance': 0.01
    }
)

# Run pipeline
result = theatre.observe_all_stages(
    input_data=Path("./data/sample.mzML")
)

# Access fragmentation results
if 'stage_02_5_fragmentation' in result.stage_results:
    frag_result = result.stage_results['stage_02_5_fragmentation']
    print(f"Precursors: {frag_result.metrics['n_precursors']}")
    print(f"Fragments: {frag_result.metrics['n_fragments']}")
    print(f"Phase-locks: {frag_result.metrics['n_phase_locks']}")
    print(f"Correlation: {frag_result.metrics['intensity_entropy_correlation']:.3f}")
```

---

## Testing

### Quick Start

Run the fragmentation stage test:

```bash
cd precursor
python test_fragmentation_stage.py
```

### Test Suite

**Test 1: Single File Analysis**
- Validates fragmentation network construction
- Checks phase-lock detection
- Verifies intensity-entropy correlation
- Expected: Network with edges and phase-locks

**Test 2: Platform Comparison**
- Compares Waters qTOF vs Thermo Orbitrap
- Validates platform independence (CV < 0.2)
- Confirms network topology consistency
- Expected: Similar network metrics across platforms

---

## Expected Results

### Typical Metrics (for 30 min RT range)

| Metric | Typical Range | Interpretation |
|--------|---------------|----------------|
| Precursors | 50-200 | Number of unique precursor ions |
| Fragments | 500-5000 | Total fragment ions detected |
| Network edges | 1000-20000 | Connections in fragmentation network |
| Network density | 0.01-0.1 | Sparse network (typical for MS) |
| Phase-locks | 100-1000 | Harmonic relationships detected |
| Clustering coefficient | 0.3-0.7 | Small-world property indicator |
| Avg path length | 2-5 | Small-world property indicator |
| Intensity-entropy correlation | 0.3-0.7 | Validates theoretical prediction |
| Platform independence score | 0.5-0.9 | Higher = more platform-independent |

---

## Output Structure

### Saved Files

```
results/
└── fragmentation_test/
    ├── stage_01_preprocessing/
    ├── stage_02_sentropy/
    ├── stage_02_5_fragmentation/      ← Fragmentation stage
    │   ├── stage_result.json          ← Overall stage result
    │   ├── fragmentation_network_build.json
    │   ├── phase_lock_analysis.json
    │   ├── intensity_entropy_analysis.json
    │   └── platform_independence_validation.json
    ├── stage_03_bmd_grounding/
    └── theatre_result.json
```

### Result JSON Structure

```json
{
  "stage_name": "fragmentation_network_analysis",
  "stage_id": "stage_02_5_fragmentation",
  "status": "completed",
  "execution_time": 15.23,
  "metrics": {
    "n_precursors": 142,
    "n_fragments": 3421,
    "n_edges": 8234,
    "network_density": 0.0342,
    "avg_degree": 4.82,
    "n_phase_locks": 523,
    "clustering_coefficient": 0.456,
    "avg_path_length": 3.21,
    "intensity_entropy_correlation": 0.523,
    "degree_cv": 0.142,
    "sentropy_cv": 0.089,
    "platform_independence_score": 0.782
  }
}
```

---

## Troubleshooting

### Issue: No precursors found

**Symptoms**: `n_precursors: 0`

**Causes**:
- RT range too narrow (no MS1 scans)
- MS1/MS2 threshold too high
- Incorrect vendor specification

**Solution**:
```python
preprocessing={
    'acquisition': {
        'rt_range': [0, 100],    # Wider range
        'ms1_threshold': 500,     # Lower threshold
        'ms2_threshold': 5
    }
}
```

---

### Issue: No edges in network

**Symptoms**: `n_edges: 0`

**Causes**:
- Similarity threshold too strict
- Insufficient fragments per precursor

**Solution**:
```python
fragmentation={
    'similarity_threshold': 0.7,  # Relax threshold
    'sigma': 0.3                  # Increase scale
}
```

---

### Issue: No phase-locks detected

**Symptoms**: `n_phase_locks: 0`

**Causes**:
- Harmonic tolerance too strict
- Network too sparse

**Solution**:
```python
fragmentation={
    'harmonic_tolerance': 0.05  # Increase tolerance
}
```

---

## Publication Results

This stage generates the quantitative results for:

**Paper**: "Categorical Fragmentation: Phase-Lock Networks and the Resolution of Gibbs' Paradox in Mass Spectrometry"

**Key Results**:
1. Fragmentation tree → network transformation (edge increase: 100-1000×)
2. Scale-free degree distribution (power-law exponent: α ≈ 2-3)
3. Small-world property (clustering ≫ random, path length ≈ random)
4. Intensity-entropy correlation (r = 0.3-0.7, validates theory)
5. Platform independence (CV < 0.2 across Waters/Thermo)

---

## References

- `precursor/publication/fragmentation/categorical-fragmentation-phase-lock-networks.tex`
- `precursor/src/metabolomics/FragmentationTrees.py`
- `precursor/src/metabolomics/GraphAnnotation.py`

---

## Contact

For questions about the fragmentation stage:
- See paper sections for theoretical background
- Check `FragmentationTrees.py` for implementation details
- Run tests to validate integration

# Computer Vision Integration - COMPLETE

## Overview

The **FULL computer vision method** with ion-to-droplet thermodynamic conversion is now properly integrated into the metabolomics pipeline.

## Components Integrated

### 1. **Ion-to-Droplet Converter** (`IonToDropletConverter.py`)
- Converts MS ions to thermodynamic droplet impacts
- Calculates S-Entropy coordinates (S_knowledge, S_time, S_entropy)
- Maps to droplet parameters (velocity, radius, surface tension, phase coherence)
- Creates visual representation of molecular phase-lock relationships

### 2. **MS Image Database Enhanced** (`MSImageDatabase_Enhanced.py`)
Complete CV pipeline with:
- **SIFT features** - Scale-Invariant Feature Transform for keypoint detection
- **ORB features** - Oriented FAST and Rotated BRIEF descriptors
- **Optical Flow** - Farneback algorithm for motion/transformation detection
- **Phase-lock signatures** - Thermodynamic coherence patterns
- **Categorical state matching** - Equivalence class comparison
- **S-Entropy distance** - 3D coordinate space similarity
- **FAISS indexing** - Fast approximate nearest neighbor search

### 3. **Spectra Alignment** (`SpectraAlignmentProcess`)
- RT window grouping for drift correction
- Dynamic time warping support (framework ready)
- Configurable tolerance windows

### 4. **Database Search** (`DatabaseSearchProcess`)
- LIPIDMAPS
- HMDB (Human Metabolome Database)
- PubChem
- METLIN
- MassBank
- Spectral similarity scoring

---

## Pipeline Architecture

### **Stage 1: Spectral Preprocessing**
```
Processes:
1. SpectralAcquisitionProcess - Load mzML files
2. SpectraAlignmentProcess - RT alignment
3. PeakDetectionProcess - BMD-filtered peak detection
```

**Output Data:**
- `scan_info.tsv` - Scan metadata
- `xic_data.tsv` - Extracted ion chromatograms
- `spectra/*.tsv` - Individual spectrum peak lists

---

### **Stage 2: S-Entropy Transformation + Computer Vision**
```
Processes:
1. SEntropyTransformProcess - Platform-independent 14D features
2. ComputerVisionConversionProcess - ION-TO-DROPLET + FULL CV
3. CategoricalStateMappingProcess - Categorical state assignment
```

**Output Data:**
- `sentropy_features.tsv` - S-Entropy 14D feature vectors
- `categorical_states.tsv` - Categorical completion states
- **`cv_images/*.png`** - Thermodynamic droplet images (visual modality)
- **`cv_features.tsv`** - SIFT/ORB keypoint counts and droplet counts
- **`ion_droplets.tsv`** - Complete droplet parameters:
  - mz, intensity
  - s_knowledge, s_time, s_entropy
  - velocity, radius, surface_tension
  - phase_coherence
  - categorical_state

**Computer Vision Features Extracted:**
- SIFT descriptors (128D per keypoint)
- ORB descriptors (32D per keypoint)
- Canny edge detection
- Phase-lock thermodynamic features
- Combined feature vector (padded to fixed dimension)

---

### **Stage 3: Hardware BMD Grounding**
```
Processes:
1. HardwareStreamHarvestProcess - Hardware phase-lock harvest
2. StreamDivergenceComputeProcess - Cross-stream coherence
```

**Output Data:**
- `stream_divergences.tsv` - BMD stream divergence scores
- `coherence_scores.tsv` - Phase-lock quality metrics

---

### **Stage 4: Categorical Completion + Annotation**
```
Processes:
1. OscillatoryHoleIdentificationProcess - Identify categorical completion holes
2. DatabaseSearchProcess - Search metabolite databases
3. ComputerVisionMatchingProcess - FULL CV SPECTRAL MATCHING
```

**Output Data:**
- **`annotations.tsv`** - Final metabolite annotations with:
  - metabolite_id, compound_name
  - confidence
  - database source
  - mass_error_ppm, spectral_similarity
  - **cv_structural_similarity** (SSIM)
  - **cv_phase_lock_similarity** (phase coherence correlation)
  - **cv_categorical_match** (categorical state sequence similarity)
  - **cv_sentropy_distance** (3D S-Entropy coordinate distance)
  - **cv_matched_features** (number of SIFT/ORB matches)
  - annotation_method: "computer_vision_thermodynamic"

- **`cv_matches_detailed.tsv`** - Top-K CV matches per spectrum:
  - scan_id, match_rank
  - database_id
  - similarity (combined score)
  - structural_similarity (SSIM)
  - phase_lock_similarity
  - categorical_match
  - s_entropy_distance
  - n_matched_features

---

## Computer Vision Matching Algorithm

### Similarity Score Composition:
```python
combined_similarity = (
    0.3 * (1.0 / (1.0 + faiss_distance)) +  # Feature vector distance
    0.2 * ssim +                            # Structural similarity
    0.2 * phase_lock_sim +                  # Phase coherence
    0.2 * categorical_match +               # Categorical states
    0.1 * (1.0 / (1.0 + s_entropy_dist))   # S-Entropy proximity
)
```

### Visual vs. Numerical Dual-Modality:
- **Visual Graph:** Experimental spectra as droplet images connected by SIFT/ORB features
- **Numerical Graph:** S-Entropy feature vectors connected by phase-lock relationships
- **Intersection:** Creates categorical states representing molecular equivalence classes

---

## Data Saved at Each Stage

### Stage 1 (Preprocessing):
✅ `scan_info.tsv` - MS scan metadata
✅ `xic_data.tsv` - Extracted ion chromatograms
✅ `spectra/spectrum_*.tsv` - Individual spectra with m/z and intensity

### Stage 2 (Features + CV):
✅ `sentropy_features.tsv` - 14D platform-independent features
✅ `categorical_states.tsv` - Categorical completion states
✅ **`cv_images/spectrum_*_droplet.png`** - Thermodynamic droplet images
✅ **`cv_features.tsv`** - CV feature summary (keypoints, droplets)
✅ **`ion_droplets.tsv`** - Complete ion-to-droplet transformation data

### Stage 3 (BMD Grounding):
✅ `stream_divergences.tsv` - Hardware BMD divergence
✅ `coherence_scores.tsv` - Phase-lock quality

### Stage 4 (Annotation):
✅ **`annotations.tsv`** - Final metabolite IDs with CV metrics
✅ **`cv_matches_detailed.tsv`** - Top-K CV matches per spectrum

---

## How the Ion-to-Droplet Algorithm Works

### 1. S-Entropy Coordinate Calculation
For each ion (m/z, intensity):
```
S_knowledge = f(intensity, m/z precision)  # Information content
S_time = f(retention time, sequence)       # Temporal coordination
S_entropy = f(local intensity distribution) # Distributional entropy
```

### 2. Thermodynamic Droplet Mapping
```
velocity = g(S_knowledge)         # Impact velocity from information
radius = g(S_entropy)             # Droplet size from entropy
surface_tension = g(S_time)       # Surface tension from time
phase_coherence = phase_lock_quality  # Coherence with hardware oscillations
```

### 3. Visual Rendering
- Create 2D canvas (default 512x512)
- For each droplet:
  - Place at (x=m/z_normalized, y=intensity_normalized)
  - Render thermodynamic impact pattern based on droplet parameters
  - Encode phase coherence as wave propagation
- Result: Visual representation of molecular phase relationships

### 4. Computer Vision Feature Extraction
- SIFT: Detect scale-invariant keypoints in droplet pattern
- ORB: Extract rotation-invariant binary descriptors
- Edges: Canny edge detection for structural boundaries
- Combine with thermodynamic features from ion droplets

### 5. Spectral Matching
- Query spectrum → droplet image → CV features
- FAISS search for similar feature vectors in library
- Compute SSIM for structural similarity
- Compute optical flow for transformation estimation
- Match phase-lock signatures between query and library
- Combine scores for final annotation confidence

---

## Key Advantages

1. **Platform Independence:** S-Entropy transformation removes instrument-specific effects
2. **Dual Modality:** Visual (CV) + Numerical (S-Entropy) creates robust categorical states
3. **Thermodynamic Grounding:** Ion-to-droplet encoding respects physical laws
4. **Phase-Lock Awareness:** Hardware BMD grounding filters noise based on coherence
5. **Fast Similarity Search:** FAISS enables O(log N) library searches
6. **Explainable:** Visual droplet images can be inspected, unlike black-box embeddings

---

## Dependencies

- `cv2` (OpenCV) - Image processing, SIFT, ORB, optical flow
- `faiss` - Fast approximate nearest neighbor search
- `imagehash` - Perceptual hashing
- `scikit-image` - Structural similarity (SSIM)
- `h5py` - Database persistence
- `scipy` - Gaussian filters, distance metrics

---

## Example Usage

```python
from pathlib import Path
from precursor.src.pipeline.metabolomics import run_metabolomics_analysis

results = run_metabolomics_analysis(
    mzml_files=['sample.mzML'],
    output_dir=Path('results'),
    enable_bmd_grounding=True,
    cv_resolution=(512, 512),  # Droplet image resolution
    cv_library_path=Path('library/ms_cv_database'),  # Pre-built CV library
    cv_top_k=5  # Return top 5 CV matches
)

# Check CV outputs
cv_images_dir = Path('results/sample/stage_02_sentropy/cv_images')
ion_droplets = pd.read_csv('results/sample/stage_02_sentropy/ion_droplets.tsv', sep='\t')
cv_matches = pd.read_csv('results/sample/stage_04_completion/cv_matches_detailed.tsv', sep='\t')
annotations = pd.read_csv('results/sample/stage_04_completion/annotations.tsv', sep='\t')
```

---

## Status: ✅ FULLY INTEGRATED

The complete computer vision method with ion-to-droplet thermodynamic conversion is now properly integrated and will:

1. ✅ Convert each spectrum to a thermodynamic droplet image
2. ✅ Extract SIFT, ORB, optical flow, and phase-lock features
3. ✅ Match against library using dual-modality similarity
4. ✅ Save all droplet images, features, and match details
5. ✅ Provide explainable visual representations alongside numerical features

---

**This is the REAL mass spectrometry computer vision pipeline you built - now fully operational.**

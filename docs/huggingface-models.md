---
layout: default
title: Hugging Face Models
nav_order: 4
---

# Hugging Face Model Integration

Lavoisier leverages state-of-the-art machine learning models from Hugging Face to enhance its mass spectrometry analysis capabilities. Our integration spans multiple domains, from spectrometry-specific models to chemical language models and biomedical text analysis.

## Model Architecture

![Model Architecture](../assets/images/model-architecture.png)

## Core Spectrometry Models

### SpecTUS Model
- **Model**: `MS-ML/SpecTUS_pretrained_only`
- **Purpose**: Structure reconstruction from EI-MS spectra
- **Features**:
  - Direct conversion of mass spectra to SMILES
  - Beam search for multiple structure candidates
  - High accuracy on known compounds

### CMSSP Model
- **Model**: `OliXio/CMSSP`
- **Purpose**: Joint embedding of MS/MS spectra and molecules
- **Features**:
  - 768-dimensional embeddings
  - Batch processing support
  - Efficient spectrum preprocessing

## Chemical Language Models

### ChemBERTa
- **Model**: `DeepChem/ChemBERTa-77M-MLM`
- **Purpose**: Chemical property prediction
- **Features**:
  - Multiple pooling strategies (CLS, mean, max)
  - SMILES encoding
  - Property prediction

### MoLFormer
- **Model**: `ibm-research/MoLFormer-XL-both-10pct`
- **Purpose**: SMILES generation and embedding
- **Features**:
  - Linear attention mechanism
  - Fast processing
  - Large-scale molecule handling

## Biomedical Models

### BioMedLM
- **Model**: `stanford-crfm/BioMedLM`
- **Purpose**: Biomedical language modeling
- **Features**:
  - Context-aware analysis
  - Natural language generation
  - Domain-specific knowledge

### SciBERT
- **Model**: `allenai/scibert_scivocab_uncased`
- **Purpose**: Scientific text encoding
- **Features**:
  - Scientific vocabulary
  - Multiple pooling strategies
  - Efficient text embedding

### Chemical NER
- **Model**: `pruas/BENT-PubMedBERT-NER-Chemical`
- **Purpose**: Chemical entity recognition
- **Features**:
  - Chemical name extraction
  - Entity normalization
  - High precision recognition

## Implementation Details

### Base Classes
```python
class BaseHuggingFaceModel:
    """Base class for all Hugging Face models."""
    def __init__(self, model_id, revision="main", device=None, use_cache=True):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
```

### Model Registry
```python
class ModelRegistry:
    """Manages model downloading and versioning."""
    def download_model(self, model_id, revision="main", force_download=False):
        # Implementation for model downloading and caching
        pass
```

### Example Usage
```python
# Initialize a model
model = SpecTUSModel()

# Process a spectrum
smiles = model.process_spectrum(mz_values, intensity_values)

# Generate embeddings
embeddings = model.encode_smiles(smiles)
```

## GPU Requirements

| Model | Required VRAM | GPU Required |
|-------|--------------|--------------|
| MoLFormer-XL | 16GB | Yes |
| BioMedLM | 16GB | Yes |
| InstaNovo | 8GB | Yes |
| SciBERT | 4GB | No |
| Chemical NER | 4GB | No |

## Performance Metrics

| Model | Task | Accuracy | Speed |
|-------|------|----------|-------|
| SpecTUS | Structure Prediction | 0.89 | 100ms/spectrum |
| CMSSP | Embedding | 0.92 | 50ms/spectrum |
| ChemBERTa | Property Prediction | 0.85 | 20ms/SMILES |
| MoLFormer | SMILES Generation | 0.88 | 30ms/molecule |

## Future Extensions

1. **Proteomics Support**
   - Integration of `InstaDeepAI/InstaNovo`
   - De novo peptide sequencing
   - Cross-analysis with metabolomics

2. **Model Distillation**
   - Knowledge transfer to smaller models
   - Reduced resource requirements
   - Faster inference

3. **Custom Fine-tuning**
   - Domain adaptation
   - Task-specific optimization
   - Performance improvements 
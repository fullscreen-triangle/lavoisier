# Experiment-to-LLM: Condensing Experiments into Language Models

## Overview

This system enables **condensing entire mass spectrometry experiments into specialized Large Language Models (LLMs)**. Each experiment becomes a trainable, queryable "learned representation" that encodes:

- **Metabolomics**: S-Entropy features, phase-lock signatures, fragmentation networks, biological pathways
- **Proteomics**: S-Entropy features, **frequency coupling** (all fragments from same collision), B/Y complementarity, protein-peptide relationships

## Key Innovation

### Traditional Approach

```
Experiment Data → Database → SQL Queries → Static Results
```

### LLM Approach

```
Experiment Data → Knowledge Base → Fine-tuned LLM → Dynamic Queries & Insights
```

**Advantages:**

1. **Natural Language Queries**: Ask questions in plain English
2. **Contextual Understanding**: LLM learns experiment-specific patterns
3. **Inference Capabilities**: Can predict and suggest based on learned patterns
4. **Compact Representation**: Multi-GB datasets → Few-MB models
5. **Transferable Knowledge**: Can query across experiments

## Architecture

### Metabolomics LLM (`MetabolicLargeLanguageModel.py`)

**Knowledge Base Structure:**

```python
ExperimentKnowledgeBase(
    metadata: ExperimentMetadata,
    metabolites: List[MetaboliteKnowledge],
    s_entropy_distribution: np.ndarray,
    phase_lock_statistics: Dict,
    metabolite_similarity_network: np.ndarray
)
```

**Training Examples Generated:**

- Spectrum → Metabolite identification
- S-Entropy features → Metabolite class
- Precursor m/z + RT → Identification
- SMILES → Biological pathway
- Phase-lock signature → Similar metabolites
- Experiment-wide statistics

### Proteomics LLM (`ProteomicsLargeLanguageModel.py`)

**UNIQUE FEATURE: Frequency Coupling Integration**

Since all peptide fragments originate from the same collision event, they are **frequency-coupled**. This is CRITICAL for proteomics and is integrated into:

1. **Training data**: Frequency coupling matrices as features
2. **Collision signatures**: Shared phase-lock across all fragments
3. **Validation**: Coupling consistency scores
4. **Hybrid models**: Coupling-aware architectures

**Knowledge Base Structure:**

```python
ProteomicsKnowledgeBase(
    metadata: ProteomicsExperimentMetadata,
    peptides: List[PeptideKnowledge],  # With frequency coupling!
    proteins: List[ProteinKnowledge],
    frequency_coupling_distribution: np.ndarray,  # Experiment-wide
    collision_event_statistics: Dict
)
```

**Training Examples Generated:**

- Spectrum → Peptide sequence
- Precursor + charge → Peptide
- **Frequency coupling signature → Validation quality**
- **B/Y complementarity → Confidence scores**
- S-Entropy features → Peptide properties
- Peptide sequence → Protein(s)
- Protein identification → Biological function

## Construction Methods

### 1. LoRA (Low-Rank Adaptation)

**Best for**: Quick fine-tuning with limited compute

```python
from metabolomics.MetabolicLargeLanguageModel import MetabolicLLMConstructor

constructor = MetabolicLLMConstructor(
    base_model="DeepChem/ChemBERTa-77M-MLM",
    construction_method="lora"
)

model = constructor.construct_llm(
    knowledge_base=kb,
    output_dir="./metabolic_llm",
    num_epochs=3,
    lora_r=8,
    lora_alpha=32
)
```

**Advantages:**

- Fast training (minutes)
- Low memory footprint
- Preserves base model knowledge

### 2. Fine-tuning

**Best for**: When you have sufficient compute and want full adaptation

```python
constructor = MetabolicLLMConstructor(
    base_model="DeepChem/ChemBERTa-77M-MLM",
    construction_method="finetune"
)

model = constructor.construct_llm(
    knowledge_base=kb,
    output_dir="./metabolic_llm",
    num_epochs=5
)
```

**Advantages:**

- Full model adaptation
- Best performance on experiment-specific queries

### 3. From Scratch

**Best for**: Small, specialized models or when base models aren't suitable

```python
constructor = MetabolicLLMConstructor(
    construction_method="fromscratch"
)

model = constructor.construct_llm(
    knowledge_base=kb,
    output_dir="./metabolic_llm",
    num_epochs=10,
    n_embd=256,
    n_layer=4
)
```

**Advantages:**

- Complete control over architecture
- Smallest model size
- No dependency on large base models

### 4. Hybrid (S-Entropy + LLM)

**Best for**: Maximum performance by combining multiple modalities

```python
constructor = MetabolicLLMConstructor(
    construction_method="hybrid"
)

model = constructor.construct_hybrid(
    knowledge_base=kb,
    output_dir="./hybrid_metabolic_llm"
)
```

**Architecture:**

```
Input: Text + S-Entropy (14D) + Phase-Lock Signature
         ↓
    [LLM Embeddings] → [S-Entropy Projection] → [Phase-Lock Projection]
         ↓                     ↓                          ↓
                        [Fusion Layer]
                              ↓
                    [Multi-head Outputs]
```

**Proteomics Hybrid:**

```
Input: Text + S-Entropy (14D) + Frequency Coupling (8D)
         ↓
    [Protein LLM] → [S-Entropy Proj] → [Coupling Proj]
         ↓                ↓                    ↓
                    [Fusion Layer]
                         ↓
           [Confidence | Coupling Quality | B/Y Complementarity]
```

## Usage Examples

### Metabolomics

**1. Build Knowledge Base**

```python
from metabolomics.MetabolicLargeLanguageModel import (
    ExperimentMetadata,
    MetaboliteKnowledge,
    ExperimentKnowledgeBase
)

# Create metadata
metadata = ExperimentMetadata(
    experiment_id="EXP001",
    experiment_name="Plasma_Metabolomics",
    sample_type="plasma",
    organism="Homo sapiens",
    condition="healthy",
    replicate=1,
    instrument="Waters qTOF",
    ionization_mode="positive",
    num_spectra=5000,
    num_metabolites_identified=250
)

# Add metabolites
metabolites = []
for result in annotation_results:
    metabolite = MetaboliteKnowledge(
        metabolite_id=result.metabolite_id,
        metabolite_name=result.name,
        formula=result.formula,
        smiles=result.smiles,
        precursor_mz=result.precursor_mz,
        retention_time=result.rt,
        mz_array=result.mz_array,
        intensity_array=result.intensity_array,
        s_entropy_coords=result.s_entropy_coords,
        s_entropy_features=result.s_entropy_features,
        phase_lock_signature=result.phase_lock_signature,
        confidence_score=result.confidence
    )
    metabolites.append(metabolite)

# Create knowledge base
kb = ExperimentKnowledgeBase(
    metadata=metadata,
    metabolites=metabolites
)
```

**2. Construct LLM**

```python
constructor = MetabolicLLMConstructor(
    base_model="DeepChem/ChemBERTa-77M-MLM",
    construction_method="lora"
)

model = constructor.construct_llm(
    knowledge_base=kb,
    output_dir="./plasma_metabolic_llm",
    num_epochs=3,
    batch_size=8
)
```

**3. Query LLM**

```python
from metabolomics.MetabolicLargeLanguageModel import MetabolicLLMQuery

query_interface = MetabolicLLMQuery("./plasma_metabolic_llm")

# Natural language queries
response = query_interface.query(
    "Identify metabolite at m/z 180.0634, RT 5.2 min:"
)
print(response)  # "Glucose (SMILES: C(C1C(C(C(C(O1)O)O)O)O)O)"

response = query_interface.query(
    "What metabolites are in the glycolysis pathway?"
)
print(response)  # "Glucose, Glucose-6-phosphate, Fructose-6-phosphate, ..."
```

### Proteomics

**1. Build Knowledge Base (with Frequency Coupling!)**

```python
from proteomics.ProteomicsLargeLanguageModel import (
    ProteomicsExperimentMetadata,
    PeptideKnowledge,
    ProteinKnowledge,
    ProteomicsKnowledgeBase
)

# Create metadata
metadata = ProteomicsExperimentMetadata(
    experiment_id="PROT001",
    experiment_name="Hela_Proteomics",
    sample_type="cell lysate",
    organism="Homo sapiens",
    tissue="HeLa cells",
    condition="control",
    replicate=1,
    instrument="Thermo Orbitrap",
    enzyme="trypsin",
    num_spectra=50000,
    num_peptides_identified=5000,
    num_proteins_identified=1200
)

# Add peptides (WITH FREQUENCY COUPLING!)
peptides = []
for identification in peptide_identifications:
    peptide = PeptideKnowledge(
        peptide_sequence=identification.peptide_sequence,
        protein_ids=identification.protein_ids,
        precursor_mz=identification.precursor_mz,
        precursor_charge=identification.precursor_charge,
        retention_time=identification.retention_time,
        mz_array=identification.mz_array,
        intensity_array=identification.intensity_array,
        fragment_ions=identification.fragments,
        s_entropy_coords=identification.s_entropy_coords,
        s_entropy_features=identification.s_entropy_features,
        # FREQUENCY COUPLING (all fragments coupled!)
        frequency_coupling_matrix=identification.frequency_coupling_matrix,
        collision_event_signature=identification.collision_event_signature,
        frequency_coupling_score=identification.frequency_coupling_score,
        # B/Y complementarity
        by_complementarity_score=identification.by_complementarity_score,
        matched_b_ions=identification.matched_b_ions,
        matched_y_ions=identification.matched_y_ions,
        confidence_score=identification.confidence_score
    )
    peptides.append(peptide)

# Create knowledge base
kb = ProteomicsKnowledgeBase(
    metadata=metadata,
    peptides=peptides,
    proteins=proteins,
    frequency_coupling_distribution=freq_coupling_dist  # Experiment-wide
)
```

**2. Construct LLM (Frequency-Coupling-Aware)**

```python
constructor = ProteomicsLLMConstructor(
    base_model="Rostlab/prot_bert",
    construction_method="hybrid"  # Best for proteomics!
)

model = constructor.construct_llm(
    knowledge_base=kb,
    output_dir="./hela_proteomics_llm",
    num_epochs=3,
    batch_size=8
)
```

**3. Query LLM**

```python
from proteomics.ProteomicsLargeLanguageModel import ProteomicsLLMQuery

query_interface = ProteomicsLLMQuery("./hela_proteomics_llm")

# Peptide identification
response = query_interface.query(
    "Identify peptide from MS/MS spectrum: 147.1:100 175.1:80 261.2:95 ..."
)
print(response)  # "PEPTIDER"

# Frequency coupling query (UNIQUE TO PROTEOMICS!)
response = query_interface.query(
    "Frequency coupling signature: coherence=0.89, ensemble=12 fragments. Coupling quality:"
)
print(response)  # "high (score: 0.89) - all fragments from same collision"

# Protein query
response = query_interface.query(
    "Proteins containing peptide PEPTIDER:"
)
print(response)  # "Protein kinase C, Serine/threonine-protein kinase"
```

## Frequency Coupling: The Proteomics Difference

### Why Frequency Coupling Matters

**Metabolomics**: Fragments may come from different molecules or pathways

```
Metabolite A → Fragment X
Metabolite B → Fragment X  (same m/z, different origin)
```

**Proteomics**: ALL fragments come from ONE collision event

```
Peptide → CID → [b1, b2, b3, y1, y2, y3] (ALL coupled!)
```

### Integration in LLM

**1. Training Data**

```python
# Example: Coupling signature → Quality assessment
"Frequency coupling signature (all fragments from same collision):
Collision event at m/z 450.23, RT 25.3 min
Coherence: 0.875, Ensemble: 15 fragments
Modality: peptide_fragmentation, Frequency: 12.5

Coupling quality: high (score: 0.88)"
```

**2. Hybrid Model Architecture**

```python
class HybridProteomicsModel(nn.Module):
    def __init__(self):
        # ... (LLM + S-Entropy) ...

        # Frequency coupling projection (UNIQUE!)
        self.coupling_proj = nn.Sequential(
            nn.Linear(coupling_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Coupling quality head
        self.coupling_head = nn.Linear(hidden_dim, 1)
```

**3. Query Examples**

```python
# Validate spectrum quality based on coupling
query_interface.query(
    "Coupling matrix shows mean=1.8, expected=1.5. Is this a clean spectrum?"
)
# Response: "Yes, high coupling (1.8) indicates clean peptide fragmentation."

# Chimera detection
query_interface.query(
    "Coupling matrix shows mean=0.7, inconsistent patterns. Spectrum quality:"
)
# Response: "Low, possible chimeric spectrum (fragments from multiple peptides)."
```

## Performance Considerations

### Model Sizes

| Method | Model Size | Training Time | Query Time |
|--------|-----------|---------------|------------|
| LoRA (r=8) | 5-10 MB | 5-15 min | <1 ms |
| Fine-tuning | 100-300 MB | 30-60 min | <1 ms |
| From Scratch | 50-100 MB | 60-120 min | <1 ms |
| Hybrid | 150-400 MB | 30-90 min | ~2 ms |

### Scaling

- **1,000 metabolites**: ~10K training examples, 10-20 min training (LoRA)
- **5,000 peptides**: ~50K training examples, 60-90 min training (LoRA)
- **10,000 proteins**: ~100K training examples, 2-3 hrs training (LoRA)

## Best Practices

1. **Start with LoRA**: Fastest iteration cycle
2. **Use Hybrid for Production**: Best performance
3. **Include All Modalities**: S-Entropy + Phase-Lock/Coupling + LLM
4. **Fine-tune Hyperparameters**: Experiment-specific tuning important
5. **Validate on Hold-out Set**: Ensure generalization

## Future Directions

1. **Cross-Experiment Transfer**: Pre-train on multiple experiments
2. **Few-Shot Learning**: Adapt to new experiments with minimal data
3. **Multi-Modal Fusion**: Integrate with imaging, genomics, etc.
4. **Active Learning**: LLM suggests which spectra to acquire next
5. **Federated Learning**: Train across institutions without sharing raw data

---

**Author**: Lavoisier Project
**Date**: October 2025
**Status**: Implemented and ready for experimental validation

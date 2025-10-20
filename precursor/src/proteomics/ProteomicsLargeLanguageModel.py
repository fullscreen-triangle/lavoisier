#!/usr/bin/env python3
"""
Proteomics Large Language Model Generation
===========================================

Generate specialized LLMs from proteomics experiments, where each experiment
is condensed into a trainable language model that encodes:
- S-Entropy coordinates for peptide spectra
- **Frequency coupling signatures (all fragments from same collision)**
- B/Y ion complementarity patterns
- Peptide sequences and modifications
- Protein inference knowledge
- Biological context (proteins, pathways, functions)

Key Innovation:
---------------
Each proteomics experiment becomes a "learned representation" that captures:
1. **Frequency-coupled fragment patterns (unique to proteomics)**
2. Peptide-protein relationships
3. Post-translational modifications
4. Collision event signatures
5. Sample-specific expression patterns

Construction Methods:
--------------------
1. Fine-tuning: Adapt protein language models (ProtBERT, ESM)
2. LoRA: Efficient fine-tuning with low-rank adaptation
3. From Scratch: Train small transformer on experiment data
4. Distillation: Compress knowledge from larger models
5. Hybrid: Combine S-Entropy + frequency coupling + LLM

Author: Lavoisier Project
Date: October 2025
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings
from collections import defaultdict

# Transformers imports
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType

# Import our frameworks
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.EntropyTransformation import SEntropyTransformer, SEntropyFeatures, SEntropyCoordinates
from core.PhaseLockNetworks import PhaseLockSignature, EnhancedPhaseLockMeasurementDevice
from proteomics.TandemDatabaseSearch import (
    PeptideSpectrum,
    PeptideFragment,
    ProteomicsAnnotationResult
)
from proteomics.MSIonDatabaseSearch import (
    PeptideIdentification,
    ProteinIdentification,
    ProteinEntry
)


@dataclass
class ProteomicsExperimentMetadata:
    """Metadata for a proteomics experiment."""
    experiment_id: str
    experiment_name: str
    sample_type: str
    organism: str
    tissue: Optional[str]
    condition: str
    replicate: int
    acquisition_date: str
    instrument: str
    enzyme: str
    num_spectra: int
    num_peptides_identified: int
    num_proteins_identified: int
    metadata: Dict = field(default_factory=dict)


@dataclass
class PeptideKnowledge:
    """
    Knowledge representation of a peptide identification.

    FREQUENCY COUPLING INTEGRATION:
    - Includes collision event signature (shared by all fragments)
    - Frequency coupling matrix for validation
    - B/Y ion complementarity scores
    """
    peptide_sequence: str
    protein_ids: List[str]
    precursor_mz: float
    precursor_charge: int
    retention_time: float

    # MS/MS spectrum
    mz_array: np.ndarray
    intensity_array: np.ndarray
    fragment_ions: List[PeptideFragment]

    # S-Entropy representation
    s_entropy_coords: List[SEntropyCoordinates]  # Per-fragment coords
    s_entropy_features: SEntropyFeatures  # Spectrum-level features

    # FREQUENCY COUPLING (UNIQUE TO PROTEOMICS!)
    frequency_coupling_matrix: Optional[np.ndarray]  # All fragments coupled
    collision_event_signature: Optional[PhaseLockSignature]  # Shared signature
    frequency_coupling_score: float  # Consistency score

    # B/Y ion complementarity
    by_complementarity_score: float
    matched_b_ions: List[float]
    matched_y_ions: List[float]

    # Annotation confidence
    confidence_score: float
    annotation_source: str

    # Modifications
    modifications: Optional[Dict[int, str]] = None

    # Validation flags
    is_validated: bool = False
    validation_flags: List[str] = field(default_factory=list)


@dataclass
class ProteinKnowledge:
    """Knowledge representation of a protein identification."""
    protein_id: str
    protein_name: str
    gene_name: str
    organism: str
    sequence: str

    # Supporting peptides
    unique_peptides: List[PeptideKnowledge] = field(default_factory=list)
    shared_peptides: List[PeptideKnowledge] = field(default_factory=list)

    # Protein-level metrics
    sequence_coverage: float
    protein_score: float
    num_unique_peptides: int

    # Biological context
    go_terms: List[str] = field(default_factory=list)
    pathways: List[str] = field(default_factory=list)
    function: Optional[str] = None

    # Validation
    is_validated: bool = False


@dataclass
class ProteomicsKnowledgeBase:
    """
    Complete knowledge base for a proteomics experiment.

    FREQUENCY COUPLING EMPHASIS:
    - Aggregates frequency coupling patterns across experiment
    - Tracks collision event signatures
    - Encodes peptide-specific coupling consistency
    """
    metadata: ProteomicsExperimentMetadata
    peptides: List[PeptideKnowledge] = field(default_factory=list)
    proteins: List[ProteinKnowledge] = field(default_factory=list)

    # Experiment-wide statistics
    s_entropy_distribution: Optional[np.ndarray] = None
    frequency_coupling_distribution: Optional[np.ndarray] = None  # Experiment-wide coupling
    collision_event_statistics: Dict[str, float] = field(default_factory=dict)

    # Network topology
    peptide_similarity_network: Optional[np.ndarray] = None
    protein_interaction_network: Optional[Dict] = None

    # Summary statistics
    summary_statistics: Dict[str, Any] = field(default_factory=dict)


class ProteomicsExperimentDataset(Dataset):
    """
    PyTorch Dataset for proteomics experiment data.

    FREQUENCY COUPLING INTEGRATION:
    - Training examples include frequency coupling information
    - Collision event signatures as features
    - B/Y complementarity as validation
    """

    def __init__(
        self,
        knowledge_base: ProteomicsKnowledgeBase,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        include_sentropy: bool = True,
        include_frequency_coupling: bool = True,
        include_by_complementarity: bool = True
    ):
        """
        Initialize dataset.

        Args:
            knowledge_base: Proteomics knowledge base
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
            include_sentropy: Include S-Entropy features
            include_frequency_coupling: Include frequency coupling (CRITICAL for proteomics!)
            include_by_complementarity: Include B/Y complementarity
        """
        self.knowledge_base = knowledge_base
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_sentropy = include_sentropy
        self.include_frequency_coupling = include_frequency_coupling
        self.include_by_complementarity = include_by_complementarity

        # Generate training examples
        self.examples = self._generate_examples()

    def _generate_examples(self) -> List[Dict]:
        """
        Generate training examples from knowledge base.

        FREQUENCY COUPLING EXAMPLES:
        - Coupling matrix → Peptide validation
        - Collision signature → Peptide identification
        - Coupling consistency → Spectrum quality
        """
        examples = []

        for peptide in self.knowledge_base.peptides:
            # Example 1: Spectrum description → Peptide sequence
            spectrum_desc = self._format_spectrum(peptide.mz_array, peptide.intensity_array)
            prompt = f"Identify peptide from MS/MS spectrum:\n{spectrum_desc}\n\nPeptide:"
            completion = f" {peptide.peptide_sequence}"
            examples.append({
                'prompt': prompt,
                'completion': completion,
                'peptide_seq': peptide.peptide_sequence
            })

            # Example 2: Precursor + charge → Peptide
            prompt = f"Peptide at m/z {peptide.precursor_mz:.4f}, charge {peptide.precursor_charge}+, RT {peptide.retention_time:.2f} min:"
            completion = f" {peptide.peptide_sequence}"
            examples.append({
                'prompt': prompt,
                'completion': completion,
                'peptide_seq': peptide.peptide_sequence
            })

            # Example 3: FREQUENCY COUPLING → Validation
            if self.include_frequency_coupling and peptide.collision_event_signature:
                coupling_desc = self._format_frequency_coupling(peptide.collision_event_signature)
                prompt = f"Frequency coupling signature (all fragments from same collision):\n{coupling_desc}\n\nCoupling quality:"
                quality = "high" if peptide.frequency_coupling_score > 0.7 else "medium" if peptide.frequency_coupling_score > 0.5 else "low"
                completion = f" {quality} (score: {peptide.frequency_coupling_score:.2f})"
                examples.append({
                    'prompt': prompt,
                    'completion': completion,
                    'peptide_seq': peptide.peptide_sequence
                })

            # Example 4: B/Y ion complementarity
            if self.include_by_complementarity:
                by_desc = f"B-ions: {len(peptide.matched_b_ions)}, Y-ions: {len(peptide.matched_y_ions)}"
                prompt = f"Peptide fragment complementarity:\n{by_desc}\n\nComplementarity score:"
                completion = f" {peptide.by_complementarity_score:.2f}"
                examples.append({
                    'prompt': prompt,
                    'completion': completion,
                    'peptide_seq': peptide.peptide_sequence
                })

            # Example 5: S-Entropy features → Peptide properties
            if self.include_sentropy and peptide.s_entropy_features:
                sentropy_desc = self._format_sentropy(peptide.s_entropy_features)
                prompt = f"S-Entropy features:\n{sentropy_desc}\n\nPeptide length:"
                completion = f" {len(peptide.peptide_sequence)} amino acids"
                examples.append({
                    'prompt': prompt,
                    'completion': completion,
                    'peptide_seq': peptide.peptide_sequence
                })

            # Example 6: Peptide sequence → Protein(s)
            protein_names = []
            for protein_id in peptide.protein_ids[:3]:  # Top 3
                for protein in self.knowledge_base.proteins:
                    if protein.protein_id == protein_id:
                        protein_names.append(protein.protein_name)
                        break

            if protein_names:
                prompt = f"Proteins containing peptide {peptide.peptide_sequence}:"
                completion = f" {', '.join(protein_names)}"
                examples.append({
                    'prompt': prompt,
                    'completion': completion,
                    'peptide_seq': peptide.peptide_sequence
                })

        # Example 7: Protein-level queries
        for protein in self.knowledge_base.proteins:
            # Protein identification
            prompt = f"Protein {protein.gene_name} ({protein.organism}):\nUnique peptides: {protein.num_unique_peptides}\nCoverage: {protein.sequence_coverage:.1%}\n\nProtein ID:"
            completion = f" {protein.protein_id}"
            examples.append({
                'prompt': prompt,
                'completion': completion,
                'peptide_seq': None
            })

            # Biological function
            if protein.function:
                prompt = f"Function of {protein.protein_name}:"
                completion = f" {protein.function}"
                examples.append({
                    'prompt': prompt,
                    'completion': completion,
                    'peptide_seq': None
                })

        # Example 8: Experiment-wide queries
        prompt = f"Experiment: {self.knowledge_base.metadata.experiment_name}\nSample: {self.knowledge_base.metadata.sample_type}\nEnzyme: {self.knowledge_base.metadata.enzyme}\n\nProteins identified:"
        completion = f" {self.knowledge_base.metadata.num_proteins_identified}"
        examples.append({
            'prompt': prompt,
            'completion': completion,
            'peptide_seq': None
        })

        # Example 9: FREQUENCY COUPLING experiment-wide
        if self.include_frequency_coupling and self.knowledge_base.frequency_coupling_distribution is not None:
            mean_coupling = np.mean(self.knowledge_base.frequency_coupling_distribution)
            prompt = f"Experiment frequency coupling quality (all peptides):"
            completion = f" Mean coupling: {mean_coupling:.2f} (all fragments temporally coupled)"
            examples.append({
                'prompt': prompt,
                'completion': completion,
                'peptide_seq': None
            })

        return examples

    def _format_spectrum(self, mz_array: np.ndarray, intensity_array: np.ndarray, top_k: int = 15) -> str:
        """Format spectrum as text."""
        if len(mz_array) > top_k:
            top_indices = np.argsort(intensity_array)[-top_k:]
            mz_array = mz_array[top_indices]
            intensity_array = intensity_array[top_indices]

        sorted_indices = np.argsort(mz_array)
        mz_array = mz_array[sorted_indices]
        intensity_array = intensity_array[sorted_indices]

        intensity_array = intensity_array / np.max(intensity_array) * 100

        peaks = [f"{mz:.1f}:{int:.0f}" for mz, int in zip(mz_array, intensity_array)]
        return " ".join(peaks)

    def _format_frequency_coupling(self, signature: PhaseLockSignature) -> str:
        """Format frequency coupling (collision event) signature."""
        return (f"Collision event at m/z {signature.mz_center:.2f}, RT {signature.rt_center:.2f} min\n"
                f"Coherence: {signature.coherence_strength:.3f}, Ensemble: {signature.ensemble_size} fragments\n"
                f"Modality: {signature.coupling_modality}, Frequency: {signature.oscillation_frequency:.3f}")

    def _format_sentropy(self, features: SEntropyFeatures) -> str:
        """Format S-Entropy features."""
        return (f"Mean magnitude: {features.mean_magnitude:.3f}, "
                f"Entropy: {features.coordinate_entropy:.3f}, "
                f"Knowledge: {features.mean_knowledge:.3f}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]

        # Tokenize
        full_text = example['prompt'] + example['completion']
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': tokenized['input_ids'].squeeze(),
            'peptide_seq': example['peptide_seq']
        }


class ProteomicsLLMConstructor:
    """
    Constructor for experiment-specific proteomics LLMs.

    FREQUENCY COUPLING INTEGRATION:
    - Training incorporates frequency coupling as a key feature
    - Models learn that all peptide fragments are temporally coupled
    - Collision event signatures used for validation
    """

    def __init__(
        self,
        base_model: str = "Rostlab/prot_bert",
        construction_method: str = "lora",
        device: Optional[str] = None
    ):
        """
        Initialize LLM constructor.

        Args:
            base_model: Base protein language model (ProtBERT, ESM, etc.)
            construction_method: Construction method (lora, finetune, fromscratch, hybrid)
            device: Device for training
        """
        self.base_model = base_model
        self.construction_method = construction_method
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"[ProteomicsLLM] Initializing constructor")
        print(f"  Base model: {self.base_model}")
        print(f"  Method: {self.construction_method}")
        print(f"  Device: {self.device}")

    def construct_llm(
        self,
        knowledge_base: ProteomicsKnowledgeBase,
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        **kwargs
    ) -> AutoModel:
        """
        Construct experiment-specific proteomics LLM.

        Args:
            knowledge_base: Proteomics knowledge base
            output_dir: Directory to save model
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            **kwargs: Additional training arguments

        Returns:
            Trained model
        """
        print(f"\n[ProteomicsLLM] Constructing LLM for experiment: {knowledge_base.metadata.experiment_name}")
        print(f"  Peptides: {len(knowledge_base.peptides)}")
        print(f"  Proteins: {len(knowledge_base.proteins)}")
        print(f"  Method: {self.construction_method}")
        print(f"  Frequency coupling: ENABLED (peptide-specific)")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create dataset
        dataset = ProteomicsExperimentDataset(
            knowledge_base,
            self.tokenizer,
            include_sentropy=True,
            include_frequency_coupling=True,  # CRITICAL for proteomics!
            include_by_complementarity=True
        )

        print(f"  Training examples: {len(dataset)}")

        # Construct based on method
        if self.construction_method == 'lora':
            model = self._construct_with_lora(dataset, output_dir, num_epochs, batch_size, learning_rate, **kwargs)
        elif self.construction_method == 'finetune':
            model = self._construct_with_finetuning(dataset, output_dir, num_epochs, batch_size, learning_rate, **kwargs)
        elif self.construction_method == 'fromscratch':
            model = self._construct_from_scratch(dataset, output_dir, num_epochs, batch_size, learning_rate, **kwargs)
        elif self.construction_method == 'hybrid':
            model = self._construct_hybrid(knowledge_base, output_dir, **kwargs)
        else:
            raise ValueError(f"Unknown construction method: {self.construction_method}")

        print(f"\n[ProteomicsLLM] Construction complete!")
        print(f"  Model saved to: {output_dir}")

        return model

    def _construct_with_lora(
        self,
        dataset: ProteomicsExperimentDataset,
        output_dir: str,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        **kwargs
    ) -> AutoModel:
        """Construct LLM with LoRA."""
        print("\n[LoRA] Loading base protein language model...")

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        model.to(self.device)

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=kwargs.get('lora_r', 16),  # Higher rank for proteomics complexity
            lora_alpha=kwargs.get('lora_alpha', 32),
            lora_dropout=kwargs.get('lora_dropout', 0.1),
            target_modules=kwargs.get('target_modules', ['q_proj', 'v_proj', 'k_proj'])
        )

        print(f"[LoRA] Applying LoRA with rank={lora_config.r}")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
            **{k: v for k, v in kwargs.items() if k in TrainingArguments.__dataclass_fields__}
        )

        # Data collator
        data_collator = DataCollatorWithPadding(self.tokenizer)

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )

        # Train
        print(f"\n[LoRA] Training for {num_epochs} epochs...")
        trainer.train()

        # Save
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return model

    def _construct_with_finetuning(
        self,
        dataset: ProteomicsExperimentDataset,
        output_dir: str,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        **kwargs
    ) -> AutoModel:
        """Construct LLM with full fine-tuning."""
        print("\n[Fine-tuning] Loading base protein language model...")

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )
        model.to(self.device)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
            **{k: v for k, v in kwargs.items() if k in TrainingArguments.__dataclass_fields__}
        )

        data_collator = DataCollatorWithPadding(self.tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )

        print(f"\n[Fine-tuning] Training for {num_epochs} epochs...")
        trainer.train()

        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return model

    def _construct_from_scratch(
        self,
        dataset: ProteomicsExperimentDataset,
        output_dir: str,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        **kwargs
    ) -> nn.Module:
        """Construct small transformer from scratch."""
        print("\n[From Scratch] Creating small transformer for proteomics...")

        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            vocab_size=len(self.tokenizer),
            n_positions=512,
            n_embd=384,  # Larger than metabolomics
            n_layer=6,  # More layers for protein complexity
            n_head=6,
            **kwargs
        )

        model = GPT2LMHeadModel(config)
        model.to(self.device)

        print(f"[From Scratch] Model parameters: {model.num_parameters():,}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False
        )

        data_collator = DataCollatorWithPadding(self.tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )

        print(f"\n[From Scratch] Training for {num_epochs} epochs...")
        trainer.train()

        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return model

    def _construct_hybrid(
        self,
        knowledge_base: ProteomicsKnowledgeBase,
        output_dir: str,
        **kwargs
    ) -> nn.Module:
        """
        Construct hybrid model combining S-Entropy + Frequency Coupling + LLM.

        This is the most powerful approach for proteomics!
        """
        print("\n[Hybrid] Creating hybrid S-Entropy + Frequency Coupling + LLM model...")

        class HybridProteomicsModel(nn.Module):
            """
            Hybrid model combining:
            - S-Entropy features (14D)
            - Frequency coupling signatures (collision event)
            - LLM embeddings (protein language model)
            """

            def __init__(self, base_model_name, sentropy_dim=14, coupling_dim=8, hidden_dim=512):
                super().__init__()

                # Load base protein LLM
                self.llm = AutoModel.from_pretrained(base_model_name)
                self.llm_dim = self.llm.config.hidden_size

                # S-Entropy projection
                self.sentropy_proj = nn.Sequential(
                    nn.Linear(sentropy_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim)
                )

                # Frequency coupling projection (UNIQUE TO PROTEOMICS!)
                self.coupling_proj = nn.Sequential(
                    nn.Linear(coupling_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim)
                )

                # Fusion layer (combines all modalities)
                self.fusion = nn.Sequential(
                    nn.Linear(self.llm_dim + 2 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim)
                )

                # Output heads
                self.confidence_head = nn.Linear(hidden_dim, 1)  # Identification confidence
                self.coupling_head = nn.Linear(hidden_dim, 1)  # Coupling quality
                self.by_head = nn.Linear(hidden_dim, 1)  # B/Y complementarity

            def forward(self, input_ids, attention_mask, sentropy_features, coupling_features):
                # LLM embeddings (peptide sequence)
                llm_output = self.llm(input_ids=input_ids, attention_mask=attention_mask)
                llm_emb = llm_output.last_hidden_state[:, 0, :]  # [CLS] token

                # S-Entropy projection
                sentropy_emb = self.sentropy_proj(sentropy_features)

                # Frequency coupling projection (ALL FRAGMENTS COUPLED!)
                coupling_emb = self.coupling_proj(coupling_features)

                # Fuse all modalities
                fused = torch.cat([llm_emb, sentropy_emb, coupling_emb], dim=1)
                fused = self.fusion(fused)

                # Multi-head outputs
                confidence = self.confidence_head(fused)
                coupling_quality = self.coupling_head(fused)
                by_complementarity = self.by_head(fused)

                return {
                    'confidence': confidence,
                    'coupling_quality': coupling_quality,
                    'by_complementarity': by_complementarity
                }

        model = HybridProteomicsModel(self.base_model, sentropy_dim=14, coupling_dim=8, hidden_dim=512)
        model.to(self.device)

        print(f"[Hybrid] Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"[Hybrid] FREQUENCY COUPLING: Integrated as core feature")
        print(f"[Hybrid] Saving to {output_dir}")

        # Save
        torch.save(model.state_dict(), Path(output_dir) / "hybrid_proteomics_model.pt")

        return model


class ProteomicsLLMQuery:
    """Query interface for experiment-specific proteomics LLMs."""

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize query interface.

        Args:
            model_path: Path to trained model
            device: Device for inference
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"[ProteomicsLLM Query] Loaded model from {model_path}")

    def query(
        self,
        prompt: str,
        max_length: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> Union[str, List[str]]:
        """
        Query the experiment-specific proteomics LLM.

        Args:
            prompt: Query prompt
            max_length: Maximum length of generated response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of responses to generate

        Returns:
            Generated response(s)
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Remove prompt
        prompt_length = len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
        responses = [resp[prompt_length:].strip() for resp in responses]

        if num_return_sequences == 1:
            return responses[0]
        return responses


if __name__ == "__main__":
    print("="*70)
    print("Proteomics Large Language Model Generation - Example")
    print("="*70)
    print("\nKey Feature: FREQUENCY COUPLING INTEGRATION")
    print("  - All peptide fragments are temporally coupled (same collision)")
    print("  - Collision event signatures encoded in LLM")
    print("  - B/Y complementarity validation")
    print("  - Phase-lock-aware peptide identification")
    print("="*70)

    print("\nSee documentation for usage examples.")

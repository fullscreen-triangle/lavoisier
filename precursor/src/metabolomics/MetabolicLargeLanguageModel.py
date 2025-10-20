#!/usr/bin/env python3
"""
Metabolic Large Language Model Generation
==========================================

Generate specialized LLMs from metabolomics experiments, where each experiment
is condensed into a trainable language model that encodes:
- S-Entropy coordinates and feature distributions
- Metabolite identifications and confidence scores
- Fragmentation patterns and phase-lock signatures
- Biological context and sample metadata

Key Innovation:
---------------
Each experiment becomes a "learned representation" - a small, specialized LLM
that can answer queries, predict metabolites, and provide biological insights
specific to that experimental dataset.

Construction Methods:
--------------------
1. Fine-tuning: Adapt pre-trained chemical LLMs (ChemBERTa, MoLFormer)
2. LoRA: Efficient fine-tuning with low-rank adaptation
3. From Scratch: Train small transformer on experiment data
4. Distillation: Compress knowledge from larger models
5. Hybrid: Combine S-Entropy features with LLM embeddings

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
    AutoModelForSequenceClassification,
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
from metabolomics.FragmentationTrees import MetaboliteNode, FragmentationNetwork
from metabolomics.MSIonDatabaseSearch import IonAnnotation, AnnotationResult


@dataclass
class ExperimentMetadata:
    """Metadata for a metabolomics experiment."""
    experiment_id: str
    experiment_name: str
    sample_type: str
    organism: str
    tissue: Optional[str]
    condition: str
    replicate: int
    acquisition_date: str
    instrument: str
    ionization_mode: str
    num_spectra: int
    num_metabolites_identified: int
    metadata: Dict = field(default_factory=dict)


@dataclass
class MetaboliteKnowledge:
    """
    Knowledge representation of a metabolite identification.

    This structures experiment results into LLM-trainable format.
    """
    metabolite_id: str
    metabolite_name: str
    formula: str
    smiles: str
    inchi: Optional[str]

    # MS/MS information
    precursor_mz: float
    retention_time: float
    mz_array: np.ndarray
    intensity_array: np.ndarray

    # S-Entropy representation
    s_entropy_coords: SEntropyCoordinates
    s_entropy_features: SEntropyFeatures

    # Phase-lock signature
    phase_lock_signature: Optional[PhaseLockSignature]

    # Annotation confidence
    confidence_score: float
    annotation_source: str

    # Fragmentation network
    fragmentation_network: Optional[Dict] = None

    # Biological context
    pathway: Optional[str] = None
    biological_function: Optional[str] = None


@dataclass
class ExperimentKnowledgeBase:
    """
    Complete knowledge base for an experiment.

    This aggregates all experimental information for LLM training.
    """
    metadata: ExperimentMetadata
    metabolites: List[MetaboliteKnowledge] = field(default_factory=list)

    # Experiment-wide distributions
    s_entropy_distribution: Optional[np.ndarray] = None
    phase_lock_statistics: Dict[str, float] = field(default_factory=dict)

    # Network topology
    metabolite_similarity_network: Optional[np.ndarray] = None

    # Summary statistics
    summary_statistics: Dict[str, Any] = field(default_factory=dict)


class MetabolicExperimentDataset(Dataset):
    """
    PyTorch Dataset for metabolic experiment data.

    Converts experiment knowledge into trainable examples.
    """

    def __init__(
        self,
        knowledge_base: ExperimentKnowledgeBase,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        include_sentropy: bool = True,
        include_phaselock: bool = True
    ):
        """
        Initialize dataset.

        Args:
            knowledge_base: Experiment knowledge base
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
            include_sentropy: Include S-Entropy features
            include_phaselock: Include phase-lock signatures
        """
        self.knowledge_base = knowledge_base
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_sentropy = include_sentropy
        self.include_phaselock = include_phaselock

        # Generate training examples
        self.examples = self._generate_examples()

    def _generate_examples(self) -> List[Dict]:
        """
        Generate training examples from knowledge base.

        Creates diverse example types:
        - Metabolite identification queries
        - Spectrum-to-structure predictions
        - S-Entropy coordinate queries
        - Phase-lock similarity searches
        - Biological pathway associations
        """
        examples = []

        for metabolite in self.knowledge_base.metabolites:
            # Example 1: Spectrum description → Metabolite identification
            spectrum_desc = self._format_spectrum(metabolite.mz_array, metabolite.intensity_array)
            prompt = f"Identify the metabolite from this MS/MS spectrum:\n{spectrum_desc}\n\nMetabolite:"
            completion = f" {metabolite.metabolite_name} (SMILES: {metabolite.smiles})"
            examples.append({
                'prompt': prompt,
                'completion': completion,
                'metabolite_id': metabolite.metabolite_id
            })

            # Example 2: S-Entropy features → Metabolite class
            if self.include_sentropy and metabolite.s_entropy_features:
                sentropy_desc = self._format_sentropy(metabolite.s_entropy_features)
                prompt = f"Given these S-Entropy features:\n{sentropy_desc}\n\nPredict metabolite:"
                completion = f" {metabolite.metabolite_name}"
                examples.append({
                    'prompt': prompt,
                    'completion': completion,
                    'metabolite_id': metabolite.metabolite_id
                })

            # Example 3: Precursor m/z + RT → Identification
            prompt = f"Metabolite at m/z {metabolite.precursor_mz:.4f}, RT {metabolite.retention_time:.2f} min:"
            completion = f" {metabolite.metabolite_name} (confidence: {metabolite.confidence_score:.2f})"
            examples.append({
                'prompt': prompt,
                'completion': completion,
                'metabolite_id': metabolite.metabolite_id
            })

            # Example 4: SMILES → Properties
            if metabolite.pathway:
                prompt = f"Biological pathway for {metabolite.smiles}:"
                completion = f" {metabolite.pathway}"
                examples.append({
                    'prompt': prompt,
                    'completion': completion,
                    'metabolite_id': metabolite.metabolite_id
                })

            # Example 5: Phase-lock signature → Similar metabolites
            if self.include_phaselock and metabolite.phase_lock_signature:
                phaselock_desc = self._format_phaselock(metabolite.phase_lock_signature)
                prompt = f"Metabolites with similar phase-lock signature:\n{phaselock_desc}\n\nSimilar metabolites:"
                # Find similar metabolites (simplified)
                similar = self._find_similar_metabolites(metabolite, k=3)
                completion = f" {', '.join(similar)}"
                examples.append({
                    'prompt': prompt,
                    'completion': completion,
                    'metabolite_id': metabolite.metabolite_id
                })

        # Example 6: Experiment-wide queries
        prompt = f"Experiment: {self.knowledge_base.metadata.experiment_name}\nSample: {self.knowledge_base.metadata.sample_type}\n\nNumber of metabolites identified:"
        completion = f" {self.knowledge_base.metadata.num_metabolites_identified}"
        examples.append({
            'prompt': prompt,
            'completion': completion,
            'metabolite_id': None
        })

        return examples

    def _format_spectrum(self, mz_array: np.ndarray, intensity_array: np.ndarray, top_k: int = 10) -> str:
        """Format spectrum as text."""
        # Take top-k most intense peaks
        if len(mz_array) > top_k:
            top_indices = np.argsort(intensity_array)[-top_k:]
            mz_array = mz_array[top_indices]
            intensity_array = intensity_array[top_indices]

        # Sort by m/z
        sorted_indices = np.argsort(mz_array)
        mz_array = mz_array[sorted_indices]
        intensity_array = intensity_array[sorted_indices]

        # Normalize intensities
        intensity_array = intensity_array / np.max(intensity_array) * 100

        # Format as text
        peaks = [f"{mz:.1f}:{int:.0f}" for mz, int in zip(mz_array, intensity_array)]
        return " ".join(peaks)

    def _format_sentropy(self, features: SEntropyFeatures) -> str:
        """Format S-Entropy features as text."""
        return (f"mean_magnitude={features.mean_magnitude:.3f}, "
                f"spectral_entropy={features.coordinate_entropy:.3f}, "
                f"structural_entropy={features.mean_knowledge:.3f}")

    def _format_phaselock(self, signature: PhaseLockSignature) -> str:
        """Format phase-lock signature as text."""
        return (f"mz_center={signature.mz_center:.2f}, "
                f"coherence={signature.coherence_strength:.3f}, "
                f"ensemble_size={signature.ensemble_size}")

    def _find_similar_metabolites(self, metabolite: MetaboliteKnowledge, k: int = 3) -> List[str]:
        """Find similar metabolites by S-Entropy distance."""
        if not metabolite.s_entropy_features:
            return []

        distances = []
        for other in self.knowledge_base.metabolites:
            if other.metabolite_id == metabolite.metabolite_id:
                continue
            if not other.s_entropy_features:
                continue

            dist = np.linalg.norm(
                metabolite.s_entropy_features.features - other.s_entropy_features.features
            )
            distances.append((other.metabolite_name, dist))

        # Sort by distance
        distances.sort(key=lambda x: x[1])
        return [name for name, _ in distances[:k]]

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
            'metabolite_id': example['metabolite_id']
        }


class MetabolicLLMConstructor:
    """
    Constructor for experiment-specific metabolic LLMs.

    Supports multiple construction strategies:
    - Fine-tuning pre-trained chemical LLMs
    - LoRA (Low-Rank Adaptation) for efficient fine-tuning
    - Training from scratch
    - Knowledge distillation
    - Hybrid (S-Entropy + LLM)
    """

    def __init__(
        self,
        base_model: str = "DeepChem/ChemBERTa-77M-MLM",
        construction_method: str = "lora",
        device: Optional[str] = None
    ):
        """
        Initialize LLM constructor.

        Args:
            base_model: Base model to use (HuggingFace model ID)
            construction_method: Construction method (lora, finetune, fromscratch, distill, hybrid)
            device: Device for training
        """
        self.base_model = base_model
        self.construction_method = construction_method
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"[MetabolicLLM] Initializing constructor")
        print(f"  Base model: {self.base_model}")
        print(f"  Method: {self.construction_method}")
        print(f"  Device: {self.device}")

    def construct_llm(
        self,
        knowledge_base: ExperimentKnowledgeBase,
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        **kwargs
    ) -> AutoModel:
        """
        Construct experiment-specific LLM.

        Args:
            knowledge_base: Experiment knowledge base
            output_dir: Directory to save model
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            **kwargs: Additional training arguments

        Returns:
            Trained model
        """
        print(f"\n[MetabolicLLM] Constructing LLM for experiment: {knowledge_base.metadata.experiment_name}")
        print(f"  Metabolites: {len(knowledge_base.metabolites)}")
        print(f"  Method: {self.construction_method}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create dataset
        dataset = MetabolicExperimentDataset(
            knowledge_base,
            self.tokenizer,
            include_sentropy=True,
            include_phaselock=True
        )

        print(f"  Training examples: {len(dataset)}")

        # Construct based on method
        if self.construction_method == 'lora':
            model = self._construct_with_lora(dataset, output_dir, num_epochs, batch_size, learning_rate, **kwargs)
        elif self.construction_method == 'finetune':
            model = self._construct_with_finetuning(dataset, output_dir, num_epochs, batch_size, learning_rate, **kwargs)
        elif self.construction_method == 'fromscratch':
            model = self._construct_from_scratch(dataset, output_dir, num_epochs, batch_size, learning_rate, **kwargs)
        elif self.construction_method == 'distill':
            model = self._construct_with_distillation(dataset, output_dir, num_epochs, batch_size, learning_rate, **kwargs)
        elif self.construction_method == 'hybrid':
            model = self._construct_hybrid(knowledge_base, output_dir, **kwargs)
        else:
            raise ValueError(f"Unknown construction method: {self.construction_method}")

        print(f"\n[MetabolicLLM] Construction complete!")
        print(f"  Model saved to: {output_dir}")

        return model

    def _construct_with_lora(
        self,
        dataset: MetabolicExperimentDataset,
        output_dir: str,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        **kwargs
    ) -> AutoModel:
        """Construct LLM with LoRA (Low-Rank Adaptation)."""
        print("\n[LoRA] Loading base model...")

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
            r=kwargs.get('lora_r', 8),  # Rank
            lora_alpha=kwargs.get('lora_alpha', 32),
            lora_dropout=kwargs.get('lora_dropout', 0.1),
            target_modules=kwargs.get('target_modules', ['q_proj', 'v_proj'])
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
        dataset: MetabolicExperimentDataset,
        output_dir: str,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        **kwargs
    ) -> AutoModel:
        """Construct LLM with full fine-tuning."""
        print("\n[Fine-tuning] Loading base model...")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )
        model.to(self.device)

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
        print(f"\n[Fine-tuning] Training for {num_epochs} epochs...")
        trainer.train()

        # Save
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return model

    def _construct_from_scratch(
        self,
        dataset: MetabolicExperimentDataset,
        output_dir: str,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        **kwargs
    ) -> nn.Module:
        """Construct small transformer from scratch."""
        print("\n[From Scratch] Creating small transformer...")

        # Create small transformer
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            vocab_size=len(self.tokenizer),
            n_positions=512,
            n_embd=256,
            n_layer=4,
            n_head=4,
            **kwargs
        )

        model = GPT2LMHeadModel(config)
        model.to(self.device)

        print(f"[From Scratch] Model parameters: {model.num_parameters():,}")

        # Training (similar to fine-tuning)
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

        # Save
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return model

    def _construct_with_distillation(
        self,
        dataset: MetabolicExperimentDataset,
        output_dir: str,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        **kwargs
    ) -> nn.Module:
        """Construct LLM using knowledge distillation from larger model."""
        print("\n[Distillation] Not fully implemented - using fine-tuning instead")
        warnings.warn("Distillation method not fully implemented, falling back to fine-tuning")
        return self._construct_with_finetuning(dataset, output_dir, num_epochs, batch_size, learning_rate, **kwargs)

    def _construct_hybrid(
        self,
        knowledge_base: ExperimentKnowledgeBase,
        output_dir: str,
        **kwargs
    ) -> nn.Module:
        """Construct hybrid model combining S-Entropy + LLM embeddings."""
        print("\n[Hybrid] Creating hybrid S-Entropy + LLM model...")

        class HybridMetabolicModel(nn.Module):
            """Hybrid model combining S-Entropy and LLM embeddings."""

            def __init__(self, base_model_name, sentropy_dim=14, hidden_dim=256):
                super().__init__()

                # Load base LLM
                self.llm = AutoModel.from_pretrained(base_model_name)
                self.llm_dim = self.llm.config.hidden_size

                # S-Entropy projection
                self.sentropy_proj = nn.Sequential(
                    nn.Linear(sentropy_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )

                # Fusion layer
                self.fusion = nn.Sequential(
                    nn.Linear(self.llm_dim + hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )

                # Output head
                self.output = nn.Linear(hidden_dim, 1)  # Confidence score

            def forward(self, input_ids, attention_mask, sentropy_features):
                # LLM embeddings
                llm_output = self.llm(input_ids=input_ids, attention_mask=attention_mask)
                llm_emb = llm_output.last_hidden_state[:, 0, :]  # [CLS] token

                # S-Entropy projection
                sentropy_emb = self.sentropy_proj(sentropy_features)

                # Fuse
                fused = torch.cat([llm_emb, sentropy_emb], dim=1)
                fused = self.fusion(fused)

                # Output
                output = self.output(fused)
                return output

        model = HybridMetabolicModel(self.base_model, sentropy_dim=14)
        model.to(self.device)

        print(f"[Hybrid] Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"[Hybrid] Saving to {output_dir}")

        # Save
        torch.save(model.state_dict(), Path(output_dir) / "hybrid_model.pt")

        return model


class MetabolicLLMQuery:
    """Query interface for experiment-specific metabolic LLMs."""

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

        print(f"[MetabolicLLM Query] Loaded model from {model_path}")

    def query(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> Union[str, List[str]]:
        """
        Query the experiment-specific LLM.

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
    print("Metabolic Large Language Model Generation - Example")
    print("="*70)

    # Example usage would involve:
    # 1. Load experiment data
    # 2. Create knowledge base
    # 3. Construct LLM
    # 4. Query LLM

    print("\nSee documentation for usage examples.")

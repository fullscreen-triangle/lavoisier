"""
Implementation of chemical named entity recognition (NER) for extracting chemical compounds from text.
"""

import os
import logging
import re
from typing import List, Dict, Any, Optional, Union, Tuple, Set

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from lavoisier.models.huggingface_models import BiomedicalTextModel
from lavoisier.models.registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)

class ChemicalNERModel(BiomedicalTextModel):
    """Wrapper for BERT-based models for chemical named entity recognition."""
    
    def __init__(
        self,
        model_id: str = "pruas/BENT-PubMedBERT-NER-Chemical",
        revision: str = "main",
        device: Optional[str] = None,
        use_cache: bool = True,
        aggregation_strategy: str = "simple",
        **kwargs
    ):
        """Initialize the Chemical NER model.
        
        Args:
            model_id: ID of the model on Hugging Face Hub.
            revision: The revision of the model to download.
            device: Device to use for inference. If None, uses CUDA if available, otherwise CPU.
            use_cache: If True, uses the cached model if available.
            aggregation_strategy: Strategy for aggregating tokens into named entities.
            **kwargs: Additional arguments to pass to the model.
        """
        self.aggregation_strategy = aggregation_strategy
        self.ner_pipeline = None
        super().__init__(model_id, revision, device, use_cache, **kwargs)
    
    def _load_model(self) -> None:
        """Load the NER model and tokenizer."""
        try:
            # Load model and tokenizer
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Create NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                aggregation_strategy=self.aggregation_strategy
            )
        except Exception as e:
            logger.error(f"Error loading Chemical NER model: {e}")
            raise
    
    def encode_text(self, text: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode text into embeddings.
        
        Args:
            text: Text to encode.
            **kwargs: Additional arguments for encoding.
            
        Returns:
            Numpy array of embeddings.
        """
        if isinstance(text, str):
            text = [text]
        
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=kwargs.get("max_length", 512)
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings from the hidden states of the model
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Use the last hidden state's [CLS] token as the embedding
        embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
        
        return embeddings
    
    def extract_chemicals(
        self, 
        text: str,
        score_threshold: float = 0.7,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Extract chemical entities from text.
        
        Args:
            text: Input text to extract chemicals from.
            score_threshold: Minimum confidence score for entities.
            **kwargs: Additional arguments for the NER pipeline.
            
        Returns:
            List of dictionaries containing entity information.
        """
        if self.ner_pipeline is None:
            logger.error("NER pipeline not initialized")
            return []
        
        # Run NER pipeline
        entities = self.ner_pipeline(text, **kwargs)
        
        # Filter by score threshold and return
        return [entity for entity in entities if entity["score"] >= score_threshold]
    
    def extract_chemicals_batch(
        self,
        texts: List[str],
        score_threshold: float = 0.7,
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """Extract chemical entities from a batch of texts.
        
        Args:
            texts: List of input texts.
            score_threshold: Minimum confidence score for entities.
            **kwargs: Additional arguments for the NER pipeline.
            
        Returns:
            List of lists of dictionaries containing entity information.
        """
        if self.ner_pipeline is None:
            logger.error("NER pipeline not initialized")
            return [[] for _ in texts]
        
        # Run NER pipeline on batch
        batch_entities = self.ner_pipeline(texts, **kwargs)
        
        # Filter by score threshold
        if isinstance(batch_entities[0], list):
            # Pipeline already returned list per text
            return [
                [entity for entity in text_entities if entity["score"] >= score_threshold]
                for text_entities in batch_entities
            ]
        else:
            # Pipeline returned all entities in a single list
            logger.warning("NER pipeline did not return grouped results, processing manually")
            return [[entity for entity in batch_entities if entity["score"] >= score_threshold]]
    
    def get_unique_chemicals(
        self, 
        text: str,
        score_threshold: float = 0.7,
        normalize: bool = True,
        **kwargs
    ) -> List[str]:
        """Extract unique chemical names from text.
        
        Args:
            text: Input text to extract chemicals from.
            score_threshold: Minimum confidence score for entities.
            normalize: Whether to normalize chemical names.
            **kwargs: Additional arguments for the NER pipeline.
            
        Returns:
            List of unique chemical names.
        """
        entities = self.extract_chemicals(text, score_threshold, **kwargs)
        
        # Extract unique chemical names
        chemicals = set()
        for entity in entities:
            chemical_name = entity["word"]
            
            # Normalize if requested
            if normalize:
                chemical_name = self.normalize_chemical_name(chemical_name)
            
            if chemical_name:
                chemicals.add(chemical_name)
        
        return list(chemicals)
    
    @staticmethod
    def normalize_chemical_name(name: str) -> str:
        """Normalize a chemical name.
        
        Args:
            name: Chemical name to normalize.
            
        Returns:
            Normalized chemical name.
        """
        # Remove special characters and convert to lowercase
        normalized = re.sub(r'[^\w\s]', '', name).lower()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def replace_chemicals_with_ids(
        self,
        text: str,
        prefix: str = "CHEM_",
        score_threshold: float = 0.7,
        **kwargs
    ) -> Tuple[str, Dict[str, str]]:
        """Replace chemical entities in text with unique identifiers.
        
        Args:
            text: Input text.
            prefix: Prefix for chemical identifiers.
            score_threshold: Minimum confidence score for entities.
            **kwargs: Additional arguments for the NER pipeline.
            
        Returns:
            Tuple of (modified text, mapping dictionary).
        """
        entities = self.extract_chemicals(text, score_threshold, **kwargs)
        
        # Sort entities by start position (in reverse to avoid index issues when replacing)
        entities.sort(key=lambda e: e["start"], reverse=True)
        
        # Create a mapping of identifiers to chemical names
        mapping = {}
        modified_text = text
        
        for i, entity in enumerate(entities):
            chem_id = f"{prefix}{i}"
            chem_name = entity["word"]
            
            # Replace the chemical name with its ID
            start, end = entity["start"], entity["end"]
            modified_text = modified_text[:start] + chem_id + modified_text[end:]
            
            # Store the mapping
            mapping[chem_id] = chem_name
        
        return modified_text, mapping


# Convenience function to create a Chemical NER model
def create_chemical_ner_model(**kwargs) -> ChemicalNERModel:
    """Create a Chemical NER model with default settings.
    
    Args:
        **kwargs: Arguments to pass to the ChemicalNERModel constructor.
        
    Returns:
        ChemicalNERModel instance.
    """
    return ChemicalNERModel(**kwargs) 
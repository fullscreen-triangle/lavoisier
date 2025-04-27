"""
Implementation of scientific text encoders for embedding text data.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from lavoisier.models.huggingface_models import BiomedicalTextModel
from lavoisier.models.registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)

class SciBERTModel(BiomedicalTextModel):
    """Wrapper for SciBERT model for scientific text encoding."""
    
    def __init__(
        self,
        model_id: str = "allenai/scibert_scivocab_uncased",
        revision: str = "main",
        device: Optional[str] = None,
        use_cache: bool = True,
        pooling_strategy: str = "cls",
        **kwargs
    ):
        """Initialize the SciBERT model.
        
        Args:
            model_id: ID of the model on Hugging Face Hub.
            revision: The revision of the model to download.
            device: Device to use for inference. If None, uses CUDA if available, otherwise CPU.
            use_cache: If True, uses the cached model if available.
            pooling_strategy: Strategy for pooling token embeddings ('cls', 'mean', or 'max').
            **kwargs: Additional arguments to pass to the model.
        """
        self.pooling_strategy = pooling_strategy
        super().__init__(model_id, revision, device, use_cache, **kwargs)
    
    def _load_model(self) -> None:
        """Load the SciBERT model and tokenizer."""
        try:
            self.model = AutoModel.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            logger.error(f"Error loading SciBERT model: {e}")
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
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Apply pooling strategy
        if self.pooling_strategy == "cls":
            # Use [CLS] token embedding from last hidden state
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        elif self.pooling_strategy == "mean":
            # Average of all token embeddings
            attention_mask = inputs["attention_mask"]
            embeddings = []
            for i, mask in enumerate(attention_mask):
                # Only consider tokens that are not padding
                token_embeddings = outputs.last_hidden_state[i, mask.bool(), :].cpu().numpy()
                embeddings.append(np.mean(token_embeddings, axis=0))
            embeddings = np.array(embeddings)
        elif self.pooling_strategy == "max":
            # Max pooling of all token embeddings
            attention_mask = inputs["attention_mask"]
            embeddings = []
            for i, mask in enumerate(attention_mask):
                # Only consider tokens that are not padding
                token_embeddings = outputs.last_hidden_state[i, mask.bool(), :].cpu().numpy()
                embeddings.append(np.max(token_embeddings, axis=0))
            embeddings = np.array(embeddings)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return embeddings
    
    def batch_encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
        """Encode a batch of texts.
        
        Args:
            texts: List of texts to encode.
            batch_size: Batch size for encoding.
            **kwargs: Additional arguments for encoding.
            
        Returns:
            Numpy array of embeddings.
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings = self.encode_text(batch, **kwargs)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([])
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding.
            embedding2: Second embedding.
            
        Returns:
            Cosine similarity score.
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        embedding1 = embedding1 / norm1
        embedding2 = embedding2 / norm2
        
        # Compute cosine similarity
        return float(np.dot(embedding1, embedding2))
    
    def search_related_texts(
        self,
        query_text: str,
        reference_texts: List[str],
        top_k: int = 5,
        **kwargs
    ) -> List[Tuple[int, float, str]]:
        """Search for texts most related to a query text.
        
        Args:
            query_text: Query text.
            reference_texts: List of reference texts to search within.
            top_k: Number of top results to return.
            **kwargs: Additional arguments for encoding.
            
        Returns:
            List of (index, similarity score, text) tuples.
        """
        # Encode query
        query_embedding = self.encode_text(query_text, **kwargs)[0]
        
        # Encode all reference texts
        reference_embeddings = self.batch_encode_texts(reference_texts, **kwargs)
        
        # Calculate similarities
        similarities = np.array([
            self.compute_similarity(query_embedding, ref_embedding)
            for ref_embedding in reference_embeddings
        ])
        
        # Get top-k matches
        top_indices = np.argsort(-similarities)[:top_k]
        
        # Return results
        return [
            (idx, similarities[idx], reference_texts[idx])
            for idx in top_indices
        ]


# Convenience function to create a SciBERT model
def create_scibert_model(**kwargs) -> SciBERTModel:
    """Create a SciBERT model with default settings.
    
    Args:
        **kwargs: Arguments to pass to the SciBERTModel constructor.
        
    Returns:
        SciBERTModel instance.
    """
    return SciBERTModel(**kwargs) 
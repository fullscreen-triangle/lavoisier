"""
Implementation of embedding models for spectra and molecular structures.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from lavoisier.models.huggingface_models import BaseHuggingFaceModel
from lavoisier.models.registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)

class CMSSPModel(BaseHuggingFaceModel):
    """Wrapper for CMSSP model, which provides joint embeddings of MS/MS spectra and molecules."""
    
    def __init__(
        self,
        model_id: str = "OliXio/CMSSP",
        revision: str = "main",
        device: Optional[str] = None,
        use_cache: bool = True,
        embedding_dim: int = 768,
        **kwargs
    ):
        """Initialize the CMSSP model.
        
        Args:
            model_id: ID of the model on Hugging Face Hub.
            revision: The revision of the model to download.
            device: Device to use for inference. If None, uses CUDA if available, otherwise CPU.
            use_cache: If True, uses the cached model if available.
            embedding_dim: Dimension of the embedding vectors.
            **kwargs: Additional arguments to pass to the model.
        """
        self.embedding_dim = embedding_dim
        super().__init__(model_id, revision, device, use_cache, **kwargs)
    
    def _load_model(self) -> None:
        """Load the CMSSP model."""
        try:
            self.model = AutoModel.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            logger.error(f"Error loading CMSSP model: {e}")
            raise
    
    def encode_smiles(self, smiles: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode SMILES strings into embeddings.
        
        Args:
            smiles: SMILES strings to encode.
            **kwargs: Additional arguments for encoding.
            
        Returns:
            Numpy array of embeddings.
        """
        if isinstance(smiles, str):
            smiles = [smiles]
        
        # Tokenize SMILES
        inputs = self.tokenizer(
            smiles, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=kwargs.get("max_length", 512)
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # Use the [CLS] token embedding from the last hidden state
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings
    
    def preprocess_spectrum(
        self,
        mz_values: np.ndarray,
        intensity_values: np.ndarray,
        normalize: bool = True,
        min_intensity: float = 0.01,
        top_k: Optional[int] = None,
        **kwargs
    ) -> str:
        """Preprocess a spectrum for the CMSSP model.
        
        Args:
            mz_values: m/z values of the spectrum.
            intensity_values: Intensity values of the spectrum.
            normalize: Whether to normalize intensities.
            min_intensity: Minimum intensity to include.
            top_k: Keep only the top-k most intense peaks.
            **kwargs: Additional preprocessing arguments.
            
        Returns:
            Spectrum formatted as a string for the CMSSP model.
        """
        # Filter by intensity threshold
        mask = intensity_values >= min_intensity
        mz_values = mz_values[mask]
        intensity_values = intensity_values[mask]
        
        # Sort by intensity (descending)
        if top_k:
            indices = np.argsort(intensity_values)[::-1]
            if len(indices) > top_k:
                indices = indices[:top_k]
            mz_values = mz_values[indices]
            intensity_values = intensity_values[indices]
        
        # Normalize if needed
        if normalize and np.sum(intensity_values) > 0:
            intensity_values = intensity_values / np.max(intensity_values)
        
        # Format for CMSSP: pairs of m/z and intensity separated by space
        spectrum_str = " ".join(f"{mz:.4f} {intensity:.4f}" for mz, intensity in 
                               zip(mz_values, intensity_values))
        
        return spectrum_str
    
    def encode_spectrum(
        self,
        mz_values: np.ndarray,
        intensity_values: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Encode a mass spectrum into an embedding.
        
        Args:
            mz_values: m/z values of the spectrum.
            intensity_values: Intensity values of the spectrum.
            **kwargs: Additional preprocessing arguments.
            
        Returns:
            Numpy array of the spectrum embedding.
        """
        # Preprocess the spectrum
        spectrum_str = self.preprocess_spectrum(mz_values, intensity_values, **kwargs)
        
        # Tokenize the spectrum
        inputs = self.tokenizer(
            spectrum_str,
            return_tensors="pt",
            truncation=True,
            max_length=kwargs.get("max_length", 1024)
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # Use the [CLS] token embedding from the last hidden state
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return embedding
    
    def batch_encode_spectra(
        self,
        spectra: List[Tuple[np.ndarray, np.ndarray]],
        batch_size: int = 8,
        **kwargs
    ) -> np.ndarray:
        """Encode a batch of spectra.
        
        Args:
            spectra: List of (mz_values, intensity_values) tuples.
            batch_size: Batch size for encoding.
            **kwargs: Additional arguments for encoding.
            
        Returns:
            Numpy array of embeddings.
        """
        all_embeddings = []
        
        for i in range(0, len(spectra), batch_size):
            batch = spectra[i:i+batch_size]
            batch_inputs = []
            
            for mz_values, intensity_values in batch:
                spectrum_str = self.preprocess_spectrum(mz_values, intensity_values, **kwargs)
                batch_inputs.append(spectrum_str)
            
            # Tokenize all spectra in the batch
            inputs = self.tokenizer(
                batch_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=kwargs.get("max_length", 1024)
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
            # Extract embeddings
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
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
    
    def compute_batch_similarities(
        self,
        query_embedding: np.ndarray,
        reference_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarities between a query embedding and multiple reference embeddings.
        
        Args:
            query_embedding: Query embedding.
            reference_embeddings: Reference embeddings.
            
        Returns:
            Array of similarity scores.
        """
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # Normalize reference embeddings
        reference_norms = np.linalg.norm(reference_embeddings, axis=1, keepdims=True)
        valid_refs = reference_norms > 0
        normalized_refs = np.zeros_like(reference_embeddings)
        normalized_refs[valid_refs.flatten()] = (
            reference_embeddings[valid_refs.flatten()] / 
            reference_norms[valid_refs]
        )
        
        # Compute similarities
        similarities = np.dot(normalized_refs, query_embedding)
        
        return similarities


# Convenience function to create a CMSSP model
def create_cmssp_model(**kwargs) -> CMSSPModel:
    """Create a CMSSP model with default settings.
    
    Args:
        **kwargs: Arguments to pass to the CMSSPModel constructor.
        
    Returns:
        CMSSPModel instance.
    """
    return CMSSPModel(**kwargs) 
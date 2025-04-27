"""
Implementation of spectral transformer models for mass spectrometry data.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from lavoisier.models.huggingface_models import SpectralModel
from lavoisier.models.registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)

class SpecTUSModel(SpectralModel):
    """Wrapper for SpecTUS model, which converts EI-MS spectra to SMILES."""
    
    def __init__(
        self,
        model_id: str = "MS-ML/SpecTUS_pretrained_only",
        revision: str = "main",
        device: Optional[str] = None,
        use_cache: bool = True,
        max_length: int = 512,
        **kwargs
    ):
        """Initialize the SpecTUS model.
        
        Args:
            model_id: ID of the model on Hugging Face Hub.
            revision: The revision of the model to download.
            device: Device to use for inference. If None, uses CUDA if available, otherwise CPU.
            use_cache: If True, uses the cached model if available.
            max_length: Maximum length of generated SMILES.
            **kwargs: Additional arguments to pass to the model.
        """
        self.max_length = max_length
        super().__init__(model_id, revision, device, use_cache, **kwargs)
    
    def _load_model(self) -> None:
        """Load the SpecTUS model and tokenizer."""
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            logger.error(f"Error loading SpecTUS model: {e}")
            raise
    
    def preprocess_spectrum(
        self, 
        mz_values: np.ndarray, 
        intensity_values: np.ndarray,
        normalize: bool = True,
        min_mz: float = 0.0,
        max_mz: float = 1000.0,
        bin_size: float = 1.0
    ) -> str:
        """Preprocess a spectrum for input to the model.
        
        Args:
            mz_values: m/z values of the spectrum.
            intensity_values: Intensity values of the spectrum.
            normalize: Whether to normalize intensities to sum to 1.
            min_mz: Minimum m/z value to consider.
            max_mz: Maximum m/z value to consider.
            bin_size: Size of m/z bins.
            
        Returns:
            Preprocessed spectrum as a string.
        """
        # Filter by m/z range
        mask = (mz_values >= min_mz) & (mz_values <= max_mz)
        mz_values = mz_values[mask]
        intensity_values = intensity_values[mask]
        
        # Normalize intensities if needed
        if normalize and np.sum(intensity_values) > 0:
            intensity_values = intensity_values / np.sum(intensity_values)
        
        # Bin the spectrum if needed
        if bin_size > 0:
            bins = np.arange(min_mz, max_mz + bin_size, bin_size)
            binned_intensities = np.zeros_like(bins, dtype=float)
            
            for mz, intensity in zip(mz_values, intensity_values):
                bin_idx = int((mz - min_mz) / bin_size)
                if 0 <= bin_idx < len(binned_intensities):
                    binned_intensities[bin_idx] += intensity
            
            mz_values = bins
            intensity_values = binned_intensities
        
        # Format as string (m/z:intensity pairs)
        spectrum_str = " ".join(f"{mz:.1f}:{intensity:.6f}" for mz, intensity in 
                               zip(mz_values, intensity_values) if intensity > 0)
        
        return spectrum_str
    
    def process_spectrum(
        self, 
        mz_values: np.ndarray, 
        intensity_values: np.ndarray, 
        num_beams: int = 5,
        num_return_sequences: int = 1,
        **kwargs
    ) -> Union[str, List[str]]:
        """Process a mass spectrum to predict SMILES structure.
        
        Args:
            mz_values: m/z values of the spectrum.
            intensity_values: Intensity values of the spectrum.
            num_beams: Number of beams for beam search.
            num_return_sequences: Number of sequences to return.
            **kwargs: Additional preprocessing arguments.
            
        Returns:
            Predicted SMILES string(s).
        """
        # Preprocess the spectrum
        spectrum_str = self.preprocess_spectrum(mz_values, intensity_values, **kwargs)
        
        # Encode the input
        inputs = self.tokenizer(spectrum_str, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate SMILES
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                early_stopping=True
            )
        
        # Decode the outputs
        predicted_smiles = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        if num_return_sequences == 1:
            return predicted_smiles[0]
        else:
            return predicted_smiles
    
    def batch_process_spectra(
        self,
        spectra: List[Tuple[np.ndarray, np.ndarray]],
        batch_size: int = 8,
        **kwargs
    ) -> List[Union[str, List[str]]]:
        """Process a batch of spectra.
        
        Args:
            spectra: List of (mz_values, intensity_values) tuples.
            batch_size: Batch size for processing.
            **kwargs: Additional arguments for processing.
            
        Returns:
            List of predicted SMILES strings or lists of strings.
        """
        results = []
        
        for i in range(0, len(spectra), batch_size):
            batch = spectra[i:i+batch_size]
            batch_results = []
            
            for mz_values, intensity_values in batch:
                result = self.process_spectrum(mz_values, intensity_values, **kwargs)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results


# Convenience function to create a SpecTUS model
def create_spectus_model(**kwargs) -> SpecTUSModel:
    """Create a SpecTUS model with default settings.
    
    Args:
        **kwargs: Arguments to pass to the SpecTUSModel constructor.
        
    Returns:
        SpecTUSModel instance.
    """
    return SpecTUSModel(**kwargs) 
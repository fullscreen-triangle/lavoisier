"""
Base classes for integrating Hugging Face models into Lavoisier.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from abc import ABC, abstractmethod

import torch
import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from lavoisier.models.registry import MODEL_REGISTRY, ModelType

logger = logging.getLogger(__name__)

class BaseHuggingFaceModel(ABC):
    """Base class for all Hugging Face models in Lavoisier."""
    
    def __init__(
        self,
        model_id: str,
        revision: str = "main",
        device: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ):
        """Initialize the model.
        
        Args:
            model_id: ID of the model on Hugging Face Hub.
            revision: The revision of the model to download.
            device: Device to use for inference. If None, uses CUDA if available, otherwise CPU.
            use_cache: If True, uses the cached model if available.
            **kwargs: Additional arguments to pass to the model.
        """
        self.model_id = model_id
        self.revision = revision
        self.use_cache = use_cache
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.kwargs = kwargs
        
        # Check GPU requirements
        if self.device.startswith("cuda") and not MODEL_REGISTRY.check_gpu_requirements(model_id):
            logger.warning(f"GPU requirements not met for {model_id}, falling back to CPU")
            self.device = "cpu"
        
        # Download the model
        self.model_path = MODEL_REGISTRY.download_model(
            model_id, 
            revision=revision, 
            force_download=not use_cache
        )
        
        # Load the model
        self._load_model()
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load the model from the cached path."""
        pass
    
    def to(self, device: str) -> "BaseHuggingFaceModel":
        """Move the model to the specified device.
        
        Args:
            device: Device to move the model to.
            
        Returns:
            Self for chaining.
        """
        self.device = device
        if hasattr(self, "model") and isinstance(self.model, torch.nn.Module):
            self.model.to(device)
        return self

class SpectralModel(BaseHuggingFaceModel):
    """Base class for models that process spectral data."""
    
    def __init__(
        self,
        model_id: str,
        revision: str = "main",
        device: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ):
        """Initialize the spectral model.
        
        Args:
            model_id: ID of the model on Hugging Face Hub.
            revision: The revision of the model to download.
            device: Device to use for inference. If None, uses CUDA if available, otherwise CPU.
            use_cache: If True, uses the cached model if available.
            **kwargs: Additional arguments to pass to the model.
        """
        super().__init__(model_id, revision, device, use_cache, **kwargs)
    
    @abstractmethod
    def process_spectrum(self, mz_values: np.ndarray, intensity_values: np.ndarray, **kwargs) -> Any:
        """Process a mass spectrum.
        
        Args:
            mz_values: m/z values of the spectrum.
            intensity_values: Intensity values of the spectrum.
            **kwargs: Additional arguments to process the spectrum.
            
        Returns:
            Processed result, which depends on the specific model.
        """
        pass

class ChemicalLanguageModel(BaseHuggingFaceModel):
    """Base class for chemical language models."""
    
    def __init__(
        self,
        model_id: str,
        revision: str = "main",
        device: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ):
        """Initialize the chemical language model.
        
        Args:
            model_id: ID of the model on Hugging Face Hub.
            revision: The revision of the model to download.
            device: Device to use for inference. If None, uses CUDA if available, otherwise CPU.
            use_cache: If True, uses the cached model if available.
            **kwargs: Additional arguments to pass to the model.
        """
        self.tokenizer = None
        super().__init__(model_id, revision, device, use_cache, **kwargs)
    
    def _load_model(self) -> None:
        """Load the model and tokenizer from the cached path."""
        try:
            self.config = AutoConfig.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path, config=self.config)
            self.model.to(self.device)
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            except Exception as e:
                logger.warning(f"Could not load tokenizer for {self.model_id}: {e}")
                self.tokenizer = None
        except Exception as e:
            logger.error(f"Error loading model {self.model_id}: {e}")
            raise
    
    @abstractmethod
    def encode_smiles(self, smiles: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode SMILES strings into embeddings.
        
        Args:
            smiles: SMILES strings to encode.
            **kwargs: Additional arguments for encoding.
            
        Returns:
            Numpy array of embeddings.
        """
        pass
    
    def batch_encode(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
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
            embeddings = self.encode_smiles(batch, **kwargs)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)

class BiomedicalTextModel(BaseHuggingFaceModel):
    """Base class for biomedical text models."""
    
    def __init__(
        self,
        model_id: str,
        revision: str = "main",
        device: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ):
        """Initialize the biomedical text model.
        
        Args:
            model_id: ID of the model on Hugging Face Hub.
            revision: The revision of the model to download.
            device: Device to use for inference. If None, uses CUDA if available, otherwise CPU.
            use_cache: If True, uses the cached model if available.
            **kwargs: Additional arguments to pass to the model.
        """
        self.tokenizer = None
        super().__init__(model_id, revision, device, use_cache, **kwargs)
    
    def _load_model(self) -> None:
        """Load the model and tokenizer from the cached path."""
        try:
            self.config = AutoConfig.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path, config=self.config)
            self.model.to(self.device)
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            except Exception as e:
                logger.warning(f"Could not load tokenizer for {self.model_id}: {e}")
                self.tokenizer = None
        except Exception as e:
            logger.error(f"Error loading model {self.model_id}: {e}")
            raise
    
    @abstractmethod
    def encode_text(self, text: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode text into embeddings.
        
        Args:
            text: Text to encode.
            **kwargs: Additional arguments for encoding.
            
        Returns:
            Numpy array of embeddings.
        """
        pass 
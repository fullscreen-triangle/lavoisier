"""
Registry for managing and loading models from Hugging Face and other sources.
This module handles caching, versioning, and loading of models.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union, Any, List
from enum import Enum

import torch
import yaml
from huggingface_hub import snapshot_download, hf_hub_download, HfApi
from tqdm import tqdm

from lavoisier.core.config import CONFIG
from lavoisier.models.versioning import ModelVersion

logger = logging.getLogger(__name__)

class ModelSource(Enum):
    """Enum to represent different model sources."""
    LOCAL = "local"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"

class ModelType(Enum):
    """Enum to represent different model types."""
    SPECTRAL = "spectral"
    CHEMICAL_LANGUAGE = "chemical_language"
    BIOMEDICAL_TEXT = "biomedical_text"
    EMBEDDING = "embedding"
    NER = "ner"
    PROTEOMICS = "proteomics"

class HuggingFaceModelInfo:
    """Class to store information about Hugging Face models."""
    
    def __init__(
        self,
        model_id: str,
        model_type: ModelType,
        task: str,
        description: str,
        requires_gpu: bool = False,
        recommended_vram_gb: Optional[float] = None,
        model_size_gb: Optional[float] = None,
        paper_url: Optional[str] = None,
        license_info: Optional[str] = None,
    ):
        self.model_id = model_id
        self.model_type = model_type
        self.task = task
        self.description = description
        self.requires_gpu = requires_gpu
        self.recommended_vram_gb = recommended_vram_gb
        self.model_size_gb = model_size_gb
        self.paper_url = paper_url
        self.license_info = license_info
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model info to a dictionary."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "task": self.task,
            "description": self.description,
            "requires_gpu": self.requires_gpu,
            "recommended_vram_gb": self.recommended_vram_gb,
            "model_size_gb": self.model_size_gb,
            "paper_url": self.paper_url,
            "license_info": self.license_info,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HuggingFaceModelInfo":
        """Create a model info object from a dictionary."""
        return cls(
            model_id=data["model_id"],
            model_type=ModelType(data["model_type"]),
            task=data["task"],
            description=data["description"],
            requires_gpu=data.get("requires_gpu", False),
            recommended_vram_gb=data.get("recommended_vram_gb"),
            model_size_gb=data.get("model_size_gb"),
            paper_url=data.get("paper_url"),
            license_info=data.get("license_info"),
        )

class ModelRegistry:
    """Registry for managing and loading models."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the model registry.
        
        Args:
            cache_dir: Directory to cache models. If None, uses the default cache directory.
        """
        self.cache_dir = cache_dir or os.path.join(CONFIG.get("data_path", "~/.lavoisier"), "models")
        self.cache_dir = os.path.expanduser(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load model catalog
        self.catalog_path = os.path.join(self.cache_dir, "model_catalog.yaml")
        self.load_catalog()
    
    def load_catalog(self) -> None:
        """Load the model catalog from disk."""
        self.catalog: Dict[str, HuggingFaceModelInfo] = {}
        
        if os.path.exists(self.catalog_path):
            try:
                with open(self.catalog_path, "r") as f:
                    catalog_data = yaml.safe_load(f) or {}
                
                for model_id, model_data in catalog_data.items():
                    self.catalog[model_id] = HuggingFaceModelInfo.from_dict(model_data)
            except Exception as e:
                logger.error(f"Error loading model catalog: {e}")
    
    def save_catalog(self) -> None:
        """Save the model catalog to disk."""
        catalog_data = {model_id: model_info.to_dict() for model_id, model_info in self.catalog.items()}
        
        with open(self.catalog_path, "w") as f:
            yaml.dump(catalog_data, f)
    
    def register_model(self, model_info: HuggingFaceModelInfo) -> None:
        """Register a model in the catalog.
        
        Args:
            model_info: Information about the model.
        """
        self.catalog[model_info.model_id] = model_info
        self.save_catalog()
    
    def get_model_path(self, model_id: str) -> str:
        """Get the path to a cached model.
        
        Args:
            model_id: ID of the model.
            
        Returns:
            Path to the cached model.
        """
        return os.path.join(self.cache_dir, model_id.replace("/", "--"))
    
    def is_model_cached(self, model_id: str) -> bool:
        """Check if a model is cached.
        
        Args:
            model_id: ID of the model.
            
        Returns:
            True if the model is cached, False otherwise.
        """
        model_path = self.get_model_path(model_id)
        return os.path.exists(model_path) and os.path.isdir(model_path)
    
    def download_model(
        self, 
        model_id: str, 
        revision: str = "main",
        force_download: bool = False,
        specific_files: Optional[List[str]] = None
    ) -> str:
        """Download a model from Hugging Face.
        
        Args:
            model_id: ID of the model on Hugging Face Hub.
            revision: The revision of the model to download.
            force_download: If True, re-download the model even if it's cached.
            specific_files: List of specific files to download. If None, downloads all files.
            
        Returns:
            Path to the downloaded model.
        """
        model_path = self.get_model_path(model_id)
        
        if not force_download and self.is_model_cached(model_id):
            logger.info(f"Model {model_id} already cached at {model_path}")
            return model_path
        
        logger.info(f"Downloading model {model_id} to {model_path}")
        
        try:
            if specific_files:
                os.makedirs(model_path, exist_ok=True)
                for file in specific_files:
                    hf_hub_download(
                        repo_id=model_id,
                        filename=file,
                        revision=revision,
                        cache_dir=model_path,
                        force_download=force_download,
                    )
            else:
                snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    cache_dir=model_path,
                    force_download=force_download,
                )
                
            # Get the model info from HF and add it to our catalog if not already there
            if model_id not in self.catalog:
                try:
                    api = HfApi()
                    model_info = api.model_info(model_id)
                    
                    # Try to determine model type from tags
                    model_type = ModelType.CHEMICAL_LANGUAGE  # Default
                    if "ms" in model_info.tags or "mass-spectrometry" in model_info.tags:
                        model_type = ModelType.SPECTRAL
                    elif "embedding" in model_info.tags:
                        model_type = ModelType.EMBEDDING
                    elif "ner" in model_info.tags:
                        model_type = ModelType.NER
                    elif "biomedical" in model_info.tags:
                        model_type = ModelType.BIOMEDICAL_TEXT
                    elif "proteomics" in model_info.tags:
                        model_type = ModelType.PROTEOMICS
                        
                    self.register_model(HuggingFaceModelInfo(
                        model_id=model_id,
                        model_type=model_type,
                        task=model_info.pipeline_tag or "unknown",
                        description=model_info.description or "",
                        paper_url=model_info.cardData.get("paper", {}).get("url") if model_info.cardData else None,
                        license_info=model_info.cardData.get("license") if model_info.cardData else None,
                    ))
                except Exception as e:
                    logger.warning(f"Could not get model info from Hugging Face for {model_id}: {e}")
            
            return model_path
        except Exception as e:
            logger.error(f"Error downloading model {model_id}: {e}")
            raise
    
    def get_models_by_type(self, model_type: ModelType) -> List[HuggingFaceModelInfo]:
        """Get all models of a specific type.
        
        Args:
            model_type: Type of model to get.
            
        Returns:
            List of model info objects.
        """
        return [model_info for model_info in self.catalog.values() 
                if model_info.model_type == model_type]
    
    def check_gpu_requirements(self, model_id: str) -> bool:
        """Check if the GPU requirements for a model are met.
        
        Args:
            model_id: ID of the model.
            
        Returns:
            True if the requirements are met, False otherwise.
        """
        if model_id not in self.catalog:
            logger.warning(f"Model {model_id} not in catalog, cannot check GPU requirements")
            return True
        
        model_info = self.catalog[model_id]
        if not model_info.requires_gpu:
            return True
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.warning(f"Model {model_id} requires a GPU, but CUDA is not available")
            return False
        
        # Check VRAM if specified
        if model_info.recommended_vram_gb:
            try:
                device = torch.device("cuda")
                gpu_properties = torch.cuda.get_device_properties(device)
                vram_gb = gpu_properties.total_memory / (1024**3)
                
                if vram_gb < model_info.recommended_vram_gb:
                    logger.warning(
                        f"Model {model_id} recommends {model_info.recommended_vram_gb}GB VRAM, "
                        f"but only {vram_gb:.1f}GB is available"
                    )
                    return False
            except Exception as e:
                logger.warning(f"Error checking GPU memory: {e}")
        
        return True

# Create a global instance of the model registry
MODEL_REGISTRY = ModelRegistry()

# Register the default models from specialised.md
DEFAULT_MODELS = [
    HuggingFaceModelInfo(
        model_id="MS-ML/SpecTUS_pretrained_only",
        model_type=ModelType.SPECTRAL,
        task="structure-reconstruction",
        description="Transformer that decodes raw EI fragmentation spectra directly into canonical SMILES",
        requires_gpu=True,
        recommended_vram_gb=2.0,
    ),
    HuggingFaceModelInfo(
        model_id="OliXio/CMSSP",
        model_type=ModelType.EMBEDDING,
        task="ms-embedding",
        description="Contrastive pre-training aligns spectra & molecular graphs in one latent space",
        requires_gpu=True,
        recommended_vram_gb=4.0,
    ),
    HuggingFaceModelInfo(
        model_id="DeepChem/ChemBERTa-77M-MLM",
        model_type=ModelType.CHEMICAL_LANGUAGE,
        task="molecular-property-prediction",
        description="RoBERTa variant trained on ~77M SMILES for property prediction",
        requires_gpu=False,
    ),
    HuggingFaceModelInfo(
        model_id="DeepChem/ChemBERTa-77M-MTR",
        model_type=ModelType.CHEMICAL_LANGUAGE,
        task="molecular-property-prediction",
        description="RoBERTa variant trained for multi-task regression",
        requires_gpu=False,
    ),
    HuggingFaceModelInfo(
        model_id="mschuh/PubChemDeBERTa",
        model_type=ModelType.CHEMICAL_LANGUAGE,
        task="zero-shot-property-prediction",
        description="DeBERTa pre-trained with PubChem assays for imputing missing phys-chem values",
        requires_gpu=True,
        recommended_vram_gb=8.0,
    ),
    HuggingFaceModelInfo(
        model_id="ibm-research/MoLFormer-XL-both-10pct",
        model_type=ModelType.CHEMICAL_LANGUAGE,
        task="smiles-generation",
        description="Fast linear-attention XL model for molecule enumeration or fingerprint replacement",
        requires_gpu=True,
        recommended_vram_gb=16.0,
    ),
    HuggingFaceModelInfo(
        model_id="stanford-crfm/BioMedLM",
        model_type=ModelType.BIOMEDICAL_TEXT,
        task="text-generation",
        description="Lightweight biomedical LLM for context-aware analytical assistance",
        requires_gpu=True,
        recommended_vram_gb=16.0,
    ),
    HuggingFaceModelInfo(
        model_id="allenai/scibert_scivocab_uncased",
        model_type=ModelType.BIOMEDICAL_TEXT,
        task="text-embedding",
        description="Scientific text encoder for embedding pathway database abstracts",
        requires_gpu=False,
    ),
    HuggingFaceModelInfo(
        model_id="pruas/BENT-PubMedBERT-NER-Chemical",
        model_type=ModelType.NER,
        task="chemical-ner",
        description="Chemical NER for normalizing compound names in literature and user prompts",
        requires_gpu=False,
    ),
    HuggingFaceModelInfo(
        model_id="InstaDeepAI/InstaNovo",
        model_type=ModelType.PROTEOMICS,
        task="peptide-sequencing",
        description="Transformer for de-novo peptide sequencing from MS/MS data",
        requires_gpu=True,
        recommended_vram_gb=8.0,
    ),
]

# Register default models if not already in the catalog
for model_info in DEFAULT_MODELS:
    if model_info.model_id not in MODEL_REGISTRY.catalog:
        MODEL_REGISTRY.register_model(model_info) 
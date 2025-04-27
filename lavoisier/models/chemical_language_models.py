"""
Implementation of chemical language models for SMILES processing.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from lavoisier.models.huggingface_models import ChemicalLanguageModel
from lavoisier.models.registry import MODEL_REGISTRY, ModelType

logger = logging.getLogger(__name__)

class ChemBERTaModel(ChemicalLanguageModel):
    """Wrapper for ChemBERTa models for molecular property prediction."""
    
    def __init__(
        self,
        model_id: str = "DeepChem/ChemBERTa-77M-MLM",
        revision: str = "main",
        device: Optional[str] = None,
        use_cache: bool = True,
        pooling_strategy: str = "cls",
        **kwargs
    ):
        """Initialize the ChemBERTa model.
        
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
        
        # Tokenize input
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

class MoLFormerModel(ChemicalLanguageModel):
    """Wrapper for MoLFormer models for molecular representation learning."""
    
    def __init__(
        self,
        model_id: str = "ibm-research/MoLFormer-XL-both-10pct",
        revision: str = "main",
        device: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ):
        """Initialize the MoLFormer model.
        
        Args:
            model_id: ID of the model on Hugging Face Hub.
            revision: The revision of the model to download.
            device: Device to use for inference. If None, uses CUDA if available, otherwise CPU.
            use_cache: If True, uses the cached model if available.
            **kwargs: Additional arguments to pass to the model.
        """
        super().__init__(model_id, revision, device, use_cache, **kwargs)
    
    def _load_model(self) -> None:
        """Load the MoLFormer model and tokenizer."""
        try:
            # MoLFormer has a different tokenizer loading mechanism
            self.model = AutoModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load the tokenizer (MoLFormer may have a custom tokenizer)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            except Exception as e:
                logger.warning(f"Error loading MoLFormer tokenizer: {e}")
                # Fallback to a generic tokenizer if needed
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                    logger.warning("Using generic BERT tokenizer as fallback")
                except:
                    self.tokenizer = None
                    logger.error("Could not load any tokenizer for MoLFormer")
        except Exception as e:
            logger.error(f"Error loading MoLFormer model: {e}")
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
        
        # MoLFormer prefers direct SMILES input without special processing
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
            outputs = self.model(**inputs)
        
        # Extract the pooler output (which is the [CLS] token embedding)
        if hasattr(outputs, "pooler_output"):
            embeddings = outputs.pooler_output.cpu().numpy()
        else:
            # Fallback to last hidden state [CLS] token
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings

class PubChemDeBERTaModel(ChemicalLanguageModel):
    """Wrapper for PubChemDeBERTa model for property prediction."""
    
    def __init__(
        self,
        model_id: str = "mschuh/PubChemDeBERTa",
        revision: str = "main",
        device: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ):
        """Initialize the PubChemDeBERTa model.
        
        Args:
            model_id: ID of the model on Hugging Face Hub.
            revision: The revision of the model to download.
            device: Device to use for inference. If None, uses CUDA if available, otherwise CPU.
            use_cache: If True, uses the cached model if available.
            **kwargs: Additional arguments to pass to the model.
        """
        super().__init__(model_id, revision, device, use_cache, **kwargs)
        
        # Load property mapping if available
        self.property_mapping = {}
        property_mapping_file = os.path.join(self.model_path, "property_mapping.txt")
        if os.path.exists(property_mapping_file):
            try:
                with open(property_mapping_file, "r") as f:
                    for i, line in enumerate(f):
                        property_name = line.strip()
                        self.property_mapping[i] = property_name
            except Exception as e:
                logger.warning(f"Error loading property mapping: {e}")
    
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
        
        # Tokenize input
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
        
        # Use [CLS] token embedding from the last hidden state
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings
    
    def predict_properties(self, smiles: Union[str, List[str]], **kwargs) -> Dict[str, np.ndarray]:
        """Predict properties for SMILES strings.
        
        Args:
            smiles: SMILES strings to predict properties for.
            **kwargs: Additional arguments for prediction.
            
        Returns:
            Dictionary mapping property names to predicted values.
        """
        if isinstance(smiles, str):
            smiles = [smiles]
        
        # Tokenize input
        inputs = self.tokenizer(
            smiles, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=kwargs.get("max_length", 512)
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract predictions
        predictions = outputs.logits.cpu().numpy()
        
        # Format predictions using property mapping if available
        if self.property_mapping:
            result = {}
            for i, property_name in self.property_mapping.items():
                if i < predictions.shape[1]:
                    result[property_name] = predictions[:, i]
            return result
        else:
            # Return raw predictions if no mapping is available
            return {"property_" + str(i): predictions[:, i] for i in range(predictions.shape[1])}


# Convenience functions to create models
def create_chemberta_model(**kwargs) -> ChemBERTaModel:
    """Create a ChemBERTa model with default settings.
    
    Args:
        **kwargs: Arguments to pass to the ChemBERTaModel constructor.
        
    Returns:
        ChemBERTaModel instance.
    """
    return ChemBERTaModel(**kwargs)

def create_molformer_model(**kwargs) -> MoLFormerModel:
    """Create a MoLFormer model with default settings.
    
    Args:
        **kwargs: Arguments to pass to the MoLFormerModel constructor.
        
    Returns:
        MoLFormerModel instance.
    """
    return MoLFormerModel(**kwargs)

def create_pubchem_deberta_model(**kwargs) -> PubChemDeBERTaModel:
    """Create a PubChemDeBERTa model with default settings.
    
    Args:
        **kwargs: Arguments to pass to the PubChemDeBERTaModel constructor.
        
    Returns:
        PubChemDeBERTaModel instance.
    """
    return PubChemDeBERTaModel(**kwargs) 
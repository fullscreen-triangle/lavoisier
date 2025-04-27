"""
Implementation of specialized biomedical language models.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, pipeline

from lavoisier.models.huggingface_models import BiomedicalTextModel
from lavoisier.models.registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)

class BioMedLLMModel(BiomedicalTextModel):
    """Wrapper for BioMedLM model, a biomedical large language model."""
    
    def __init__(
        self,
        model_id: str = "stanford-crfm/BioMedLM",
        revision: str = "main",
        device: Optional[str] = None,
        use_cache: bool = True,
        max_length: int = 512,
        **kwargs
    ):
        """Initialize the BioMedLM model.
        
        Args:
            model_id: ID of the model on Hugging Face Hub.
            revision: The revision of the model to download.
            device: Device to use for inference. If None, uses CUDA if available, otherwise CPU.
            use_cache: If True, uses the cached model if available.
            max_length: Maximum length of generated text.
            **kwargs: Additional arguments to pass to the model.
        """
        self.max_length = max_length
        super().__init__(model_id, revision, device, use_cache, **kwargs)
    
    def _load_model(self) -> None:
        """Load the BioMedLM model and tokenizer."""
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            logger.error(f"Error loading BioMedLM model: {e}")
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
        
        # Use the last hidden state of the last token as the embedding
        embeddings = outputs.hidden_states[-1][:, -1, :].cpu().numpy()
        
        return embeddings
    
    def generate_text(
        self, 
        prompt: str, 
        max_length: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        **kwargs
    ) -> Union[str, List[str]]:
        """Generate text using the BioMedLM model.
        
        Args:
            prompt: Prompt text to generate from.
            max_length: Maximum length of the generated text.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            num_return_sequences: Number of sequences to return.
            **kwargs: Additional arguments for generation.
            
        Returns:
            Generated text or list of generated texts.
        """
        max_length = max_length or self.max_length
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode generated text
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Remove the prompt from the generated text
        prompt_length = len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
        generated_texts = [text[prompt_length:].strip() for text in generated_texts]
        
        if num_return_sequences == 1:
            return generated_texts[0]
        else:
            return generated_texts
    
    def analyze_spectra(
        self, 
        spectra_description: str,
        analysis_type: str = "general",
        max_length: Optional[int] = None,
        **kwargs
    ) -> str:
        """Analyze mass spectrometry data using the BioMedLM model.
        
        Args:
            spectra_description: Description of the spectra or analytical results.
            analysis_type: Type of analysis to perform (general, metabolite, pathway, etc.).
            max_length: Maximum length of the generated analysis.
            **kwargs: Additional arguments for text generation.
            
        Returns:
            Generated analysis text.
        """
        # Create prompt template based on analysis type
        prompt_templates = {
            "general": "Analyze the following mass spectrometry data and provide insights:\n\n{}\n\nAnalysis:",
            "metabolite": "Identify potential metabolites from the following mass spectrometry data:\n\n{}\n\nMetabolite analysis:",
            "pathway": "Suggest biological pathways related to the following mass spectrometry results:\n\n{}\n\nPathway analysis:",
            "comparative": "Compare the following mass spectrometry results and identify key differences:\n\n{}\n\nComparative analysis:"
        }
        
        # Use default prompt if analysis type not found
        prompt_template = prompt_templates.get(analysis_type, prompt_templates["general"])
        prompt = prompt_template.format(spectra_description)
        
        # Generate analysis
        return self.generate_text(prompt, max_length=max_length, **kwargs)


# Convenience function to create a BioMedLLM model
def create_biomedllm_model(**kwargs) -> BioMedLLMModel:
    """Create a BioMedLLM model with default settings.
    
    Args:
        **kwargs: Arguments to pass to the BioMedLLMModel constructor.
        
    Returns:
        BioMedLLMModel instance.
    """
    return BioMedLLMModel(**kwargs) 
"""
Integration with Ollama for local LLM inference
"""
from typing import Dict, List, Optional, Union, Any
import os
import json
import httpx
import re

from lavoisier.core.logging import get_logger
from lavoisier.llm.api import LLMClient


class OllamaClient(LLMClient):
    """Client for Ollama local LLM inference"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Ollama client
        
        Args:
            config: Configuration dictionary with Ollama settings
        """
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("model", "llama3")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
        
        self.client = httpx.AsyncClient(
            timeout=config.get("timeout", 120.0),
            base_url=self.base_url
        )
        self.logger.info(f"Initialized Ollama client with model {self.model}")
    
    async def _check_availability(self) -> bool:
        """
        Check if Ollama is available
        
        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            response = await self.client.get("/api/tags")
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Error checking Ollama availability: {str(e)}")
            return False
    
    async def _ensure_model_pulled(self, model_name: str) -> bool:
        """
        Ensure the specified model is pulled
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if the model is available, False otherwise
        """
        try:
            response = await self.client.get("/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                for model in models:
                    if model.get("name") == model_name:
                        self.logger.info(f"Model {model_name} is available")
                        return True
                
                self.logger.warning(f"Model {model_name} not found, attempting to pull")
                pull_response = await self.client.post(
                    "/api/pull",
                    json={"name": model_name}
                )
                
                if pull_response.status_code == 200:
                    self.logger.info(f"Successfully pulled model {model_name}")
                    return True
                else:
                    self.logger.error(f"Failed to pull model {model_name}")
                    return False
            else:
                self.logger.error(f"Error checking available models: {response.text}")
                return False
        except Exception as e:
            self.logger.error(f"Error ensuring model availability: {str(e)}")
            return False
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Ollama
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text response
        """
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Ensure the model is available
        if not await self._ensure_model_pulled(model):
            raise RuntimeError(f"Model {model} is not available")
        
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": False
        }
        
        try:
            response = await self.client.post(
                "/api/generate",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            self.logger.error(f"Error generating text with Ollama: {str(e)}")
            raise
    
    async def generate_analysis(self, data: Dict[str, Any], query: str, **kwargs) -> Dict[str, Any]:
        """
        Generate an analysis of the data using Ollama
        
        Args:
            data: The data to analyze
            query: The question to answer about the data
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Analysis results as dictionary
        """
        # Create a prompt that includes the data and query
        data_str = json.dumps(data, indent=2)
        prompt = f"""
        I need you to analyze this mass spectrometry data and answer the following question:
        
        {query}
        
        Here is the data:
        {data_str}
        
        Please provide your analysis in JSON format with the following structure:
        {{
            "answer": "Your answer to the question",
            "reasoning": "Your reasoning process",
            "confidence": 0.95,  # A number between 0 and 1
            "additional_insights": ["insight1", "insight2"]
        }}
        """
        
        try:
            response = await self.generate_text(prompt, **kwargs)
            
            # Extract JSON from response
            json_pattern = r'```json\s*([\s\S]*?)\s*```|{[\s\S]*}'
            match = re.search(json_pattern, response)
            
            if match:
                json_str = match.group(1) or response
                return json.loads(json_str)
            else:
                self.logger.warning("Failed to extract JSON from response")
                return {"error": "Failed to extract JSON from response", "raw_response": response}
        
        except Exception as e:
            self.logger.error(f"Error generating analysis with Ollama: {str(e)}")
            return {"error": str(e)} 
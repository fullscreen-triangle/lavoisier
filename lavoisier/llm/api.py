"""
API client for various LLM providers
"""
from typing import Dict, List, Optional, Union, Any
import os
import json
import logging
import httpx
from abc import ABC, abstractmethod

from lavoisier.core.logging import get_logger


class LLMClient(ABC):
    """Base class for LLM API clients"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM client
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger("llm_client")
    
    @abstractmethod
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text from the LLM
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    async def generate_analysis(self, data: Dict[str, Any], query: str, **kwargs) -> Dict[str, Any]:
        """
        Generate an analysis of the data
        
        Args:
            data: The data to analyze
            query: The query or question to answer
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            Analysis results
        """
        pass


class OpenAIClient(LLMClient):
    """Client for OpenAI's API"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI client
        
        Args:
            config: Configuration dictionary with OpenAI settings
        """
        super().__init__(config)
        self.api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        self.model = config.get("model", "gpt-4")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
        
        if not self.api_key:
            self.logger.warning("OpenAI API key not provided. Some functionality may be limited.")
        
        self.client = httpx.AsyncClient(
            timeout=config.get("timeout", 120.0),
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        )
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI's API
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text response
        """
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = await self.client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            self.logger.error(f"Error generating text with OpenAI: {str(e)}")
            raise
    
    async def generate_analysis(self, data: Dict[str, Any], query: str, **kwargs) -> Dict[str, Any]:
        """
        Generate an analysis of the data using OpenAI's API
        
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
            import re
            json_pattern = r'```json\s*([\s\S]*?)\s*```|{[\s\S]*}'
            match = re.search(json_pattern, response)
            
            if match:
                json_str = match.group(1) or response
                return json.loads(json_str)
            else:
                self.logger.warning("Failed to extract JSON from response")
                return {"error": "Failed to extract JSON from response", "raw_response": response}
        
        except Exception as e:
            self.logger.error(f"Error generating analysis with OpenAI: {str(e)}")
            return {"error": str(e)}


class AnthropicClient(LLMClient):
    """Client for Anthropic's Claude API"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Anthropic client
        
        Args:
            config: Configuration dictionary with Anthropic settings
        """
        super().__init__(config)
        self.api_key = config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        self.model = config.get("model", "claude-3-opus-20240229")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
        
        if not self.api_key:
            self.logger.warning("Anthropic API key not provided. Some functionality may be limited.")
        
        self.client = httpx.AsyncClient(
            timeout=config.get("timeout", 120.0),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            } if self.api_key else {}
        )
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Anthropic's API
        
        Args:
            prompt: The prompt to send to Claude
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text response
        """
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = await self.client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return result["content"][0]["text"]
        except Exception as e:
            self.logger.error(f"Error generating text with Anthropic: {str(e)}")
            raise
    
    async def generate_analysis(self, data: Dict[str, Any], query: str, **kwargs) -> Dict[str, Any]:
        """
        Generate an analysis of the data using Anthropic's API
        
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
            import re
            json_pattern = r'```json\s*([\s\S]*?)\s*```|{[\s\S]*}'
            match = re.search(json_pattern, response)
            
            if match:
                json_str = match.group(1) or response
                return json.loads(json_str)
            else:
                self.logger.warning("Failed to extract JSON from response")
                return {"error": "Failed to extract JSON from response", "raw_response": response}
        
        except Exception as e:
            self.logger.error(f"Error generating analysis with Anthropic: {str(e)}")
            return {"error": str(e)} 
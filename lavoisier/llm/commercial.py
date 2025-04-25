"""
Factory module for creating clients for commercial LLM providers
"""
from typing import Dict, Any, Optional

from lavoisier.core.logging import get_logger
from lavoisier.llm.api import LLMClient, OpenAIClient, AnthropicClient


class LLMFactory:
    """Factory for creating LLM clients based on configuration"""
    
    @staticmethod
    def create_client(provider: str, config: Dict[str, Any]) -> LLMClient:
        """
        Create an LLM client for the specified provider
        
        Args:
            provider: LLM provider name (openai, anthropic, ollama)
            config: Configuration for the client
            
        Returns:
            Appropriate LLM client instance
        """
        logger = get_logger("llm_factory")
        
        if provider.lower() == "openai":
            logger.info("Creating OpenAI client")
            return OpenAIClient(config)
        elif provider.lower() == "anthropic":
            logger.info("Creating Anthropic client")
            return AnthropicClient(config)
        elif provider.lower() == "ollama":
            logger.info("Creating Ollama client")
            from lavoisier.llm.ollama import OllamaClient
            return OllamaClient(config)
        else:
            logger.error(f"Unknown LLM provider: {provider}")
            raise ValueError(f"Unknown LLM provider: {provider}")


class CommercialLLMProxy:
    """
    Proxy that provides fallback capabilities between multiple commercial LLMs
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the commercial LLM proxy
        
        Args:
            config: Configuration with multiple LLM providers
        """
        self.logger = get_logger("commercial_llm_proxy")
        self.config = config
        self.providers = config.get("providers", ["openai"])
        self.clients = {}
        
        # Initialize clients
        for provider in self.providers:
            provider_config = config.get(provider, {})
            try:
                self.clients[provider] = LLMFactory.create_client(provider, provider_config)
                self.logger.info(f"Initialized {provider} client")
            except Exception as e:
                self.logger.error(f"Failed to initialize {provider} client: {str(e)}")
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using commercial LLMs with fallback
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text response
        """
        provider_order = kwargs.get("provider_order", self.providers)
        
        for provider in provider_order:
            if provider in self.clients:
                try:
                    self.logger.info(f"Attempting to generate text with {provider}")
                    client = self.clients[provider]
                    response = await client.generate_text(prompt, **kwargs)
                    return response
                except Exception as e:
                    self.logger.error(f"Error generating text with {provider}: {str(e)}")
                    # Continue to the next provider
        
        # If all providers failed
        error_msg = "All LLM providers failed to generate text"
        self.logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    async def generate_analysis(self, data: Dict[str, Any], query: str, **kwargs) -> Dict[str, Any]:
        """
        Generate analysis using commercial LLMs with fallback
        
        Args:
            data: The data to analyze
            query: The query or question to answer
            **kwargs: Additional parameters to pass to the LLM
            
        Returns:
            Analysis results
        """
        provider_order = kwargs.get("provider_order", self.providers)
        
        for provider in provider_order:
            if provider in self.clients:
                try:
                    self.logger.info(f"Attempting to generate analysis with {provider}")
                    client = self.clients[provider]
                    response = await client.generate_analysis(data, query, **kwargs)
                    return response
                except Exception as e:
                    self.logger.error(f"Error generating analysis with {provider}: {str(e)}")
                    # Continue to the next provider
        
        # If all providers failed
        error_msg = "All LLM providers failed to generate analysis"
        self.logger.error(error_msg)
        return {"error": error_msg}


def create_client_pool(config: Dict[str, Any]) -> Dict[str, LLMClient]:
    """
    Create a pool of LLM clients based on configuration
    
    Args:
        config: Configuration dictionary with provider settings
        
    Returns:
        Dictionary mapping provider names to client instances
    """
    logger = get_logger("commercial_llm")
    clients = {}
    
    for provider, provider_config in config.items():
        if provider in ["openai", "anthropic", "ollama"]:
            try:
                clients[provider] = LLMFactory.create_client(provider, provider_config)
                logger.info(f"Created {provider} client")
            except Exception as e:
                logger.error(f"Failed to create {provider} client: {str(e)}")
    
    return clients 
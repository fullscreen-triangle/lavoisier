"""
LLM service for integrating language models with Lavoisier
"""
from typing import Dict, List, Optional, Union, Any, Callable
import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import traceback

from lavoisier.core.logging import get_logger
from lavoisier.llm.api import LLMClient, OpenAIClient, AnthropicClient
from lavoisier.llm.commercial import LLMFactory, CommercialLLMProxy, create_client_pool
from lavoisier.llm.query_gen import QueryGenerator, QueryType


class LLMService:
    """
    Service for managing LLM interactions in Lavoisier
    
    This service provides methods for:
    1. Accessing different LLM providers (commercial and local)
    2. Generating and managing analytical queries
    3. Processing results from LLM analyses
    4. Supporting continuous learning
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM service
        
        Args:
            config: LLM configuration dictionary
        """
        self.config = config
        self.logger = get_logger("llm_service")
        
        # Whether LLMs are enabled
        self.enabled = config.get("enabled", True)
        if not self.enabled:
            self.logger.warning("LLM service is disabled by configuration")
            return
        
        # Initialize client proxy for commercial LLMs
        self.commercial_config = config.get("commercial", {})
        self.commercial_proxy = CommercialLLMProxy(self.commercial_config)
        
        # Initialize Ollama client for local inference if enabled
        self.ollama_enabled = config.get("use_ollama", True)
        if self.ollama_enabled:
            try:
                from lavoisier.llm.ollama import OllamaClient
                self.ollama_client = OllamaClient(config.get("ollama", {}))
                self.logger.info("Ollama client initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Ollama client: {str(e)}")
                self.ollama_enabled = False
        
        # Initialize query generator
        self.query_generator = QueryGenerator(config.get("query_gen", {}))
        
        # Cache for query results
        self.query_cache = {}
        
        # Thread pool for running LLM requests
        self.executor = ThreadPoolExecutor(max_workers=config.get("max_workers", 2))
        
        self.logger.info("LLM service initialized")
    
    async def analyze_data(self, 
                        data: Dict[str, Any], 
                        query: Optional[str] = None,
                        query_type: Optional[QueryType] = None,
                        use_local: bool = False) -> Dict[str, Any]:
        """
        Analyze data using LLM
        
        Args:
            data: Data to analyze
            query: Specific query to ask, or None to generate one
            query_type: Type of query to generate if query is None
            use_local: Whether to use local LLM (Ollama)
            
        Returns:
            Analysis results
        """
        if not self.enabled:
            return {"error": "LLM service is disabled"}
        
        # Generate query if not provided
        if query is None:
            query = self.query_generator.generate_query(
                query_type=query_type,
                available_data=data
            )
            self.logger.info(f"Generated query: {query}")
        
        # Check cache
        cache_key = f"{hash(json.dumps(data, sort_keys=True))}-{hash(query)}"
        if cache_key in self.query_cache:
            self.logger.info(f"Using cached result for query: {query}")
            return self.query_cache[cache_key]
        
        try:
            # Use local or commercial LLM based on parameter
            if use_local and self.ollama_enabled:
                self.logger.info(f"Using Ollama for query: {query}")
                result = await self.ollama_client.generate_analysis(data, query)
            else:
                self.logger.info(f"Using commercial LLM for query: {query}")
                result = await self.commercial_proxy.generate_analysis(data, query)
            
            # Cache result
            self.query_cache[cache_key] = result
            return result
            
        except Exception as e:
            error_msg = f"Error analyzing data: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            return {"error": error_msg}
    
    async def analyze_multiple(self, 
                             data: Dict[str, Any], 
                             queries: List[str],
                             use_local: bool = False) -> List[Dict[str, Any]]:
        """
        Run multiple analysis queries on the same data
        
        Args:
            data: Data to analyze
            queries: List of queries to run
            use_local: Whether to use local LLM
            
        Returns:
            List of analysis results
        """
        if not self.enabled:
            return [{"error": "LLM service is disabled"} for _ in queries]
        
        tasks = [
            self.analyze_data(data, query, use_local=use_local)
            for query in queries
        ]
        
        return await asyncio.gather(*tasks)
    
    def analyze_data_sync(self, 
                        data: Dict[str, Any], 
                        query: Optional[str] = None,
                        query_type: Optional[QueryType] = None,
                        use_local: bool = False) -> Dict[str, Any]:
        """
        Synchronous version of analyze_data
        
        Args:
            data: Data to analyze
            query: Specific query to ask, or None to generate one
            query_type: Type of query to generate if query is None
            use_local: Whether to use local LLM
            
        Returns:
            Analysis results
        """
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.analyze_data(data, query, query_type, use_local)
            )
            return result
        finally:
            loop.close()
    
    def generate_progressive_analysis(self, 
                                    data: Dict[str, Any], 
                                    max_queries: int = 5,
                                    callback: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> List[Dict[str, Any]]:
        """
        Generate a progressive analysis with increasing complexity
        
        Args:
            data: Data to analyze
            max_queries: Maximum number of queries to run
            callback: Optional callback for each result (query, result)
            
        Returns:
            List of analysis results
        """
        if not self.enabled:
            return [{"error": "LLM service is disabled"}]
        
        # Generate sequential queries with increasing complexity
        queries = self.query_generator.generate_sequential_queries(
            initial_data=data,
            max_queries=max_queries,
            increasing_complexity=True
        )
        
        results = []
        
        # Use asyncio to run all queries concurrently
        async def run_queries():
            for query in queries:
                try:
                    # Determine whether to use local or commercial LLM based on query complexity
                    # Use local for simpler queries, commercial for complex ones
                    use_local = queries.index(query) < max_queries // 2
                    
                    result = await self.analyze_data(data, query, use_local=use_local)
                    results.append(result)
                    
                    # Call callback if provided
                    if callback:
                        callback(query, result)
                    
                    # Update data with result for context in subsequent queries
                    data.update({
                        "previous_results": data.get("previous_results", []) + [result],
                        "current_query_index": queries.index(query)
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error in progressive analysis: {str(e)}")
                    results.append({"error": str(e)})
        
        # Run the async function
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(run_queries())
        finally:
            loop.close()
        
        return results
    
    def compare_pipelines(self,
                        numeric_results: Dict[str, Any],
                        visual_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare results from numeric and visual pipelines
        
        Args:
            numeric_results: Results from numeric pipeline
            visual_results: Results from visual pipeline
            
        Returns:
            Comparison analysis
        """
        if not self.enabled:
            return {"error": "LLM service is disabled"}
        
        # Combine results
        data = {
            "numeric_results": numeric_results,
            "visual_results": visual_results
        }
        
        # Generate comparison query
        query = """
        Compare the results from the numeric and visual analysis pipelines.
        What are the key similarities and differences?
        Are there any insights from one pipeline that complement or contradict the other?
        Which aspects of the analysis are more reliable from each pipeline?
        """
        
        # Use commercial LLM for this complex task
        return self.analyze_data_sync(data, query, use_local=False)
    
    def generate_report(self, 
                      data: Dict[str, Any],
                      analysis_results: List[Dict[str, Any]],
                      format: str = "markdown") -> str:
        """
        Generate a comprehensive report from analysis results
        
        Args:
            data: Input data
            analysis_results: Results from previous analyses
            format: Output format (markdown or html)
            
        Returns:
            Formatted report
        """
        if not self.enabled:
            return "LLM service is disabled"
        
        # Combine data
        report_data = {
            "original_data": data,
            "analysis_results": analysis_results
        }
        
        # Generate report query
        query = f"""
        Generate a comprehensive report on the mass spectrometry analysis.
        Include key findings, important peaks, potential identifications, and confidence levels.
        Format the report in {format} with appropriate sections, tables, and visualizations where helpful.
        """
        
        # Get report content
        result = self.analyze_data_sync(report_data, query, use_local=False)
        
        if "error" in result:
            return f"Error generating report: {result['error']}"
        
        return result.get("answer", "No report content generated")
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        
        self.logger.info("LLM service cleaned up") 
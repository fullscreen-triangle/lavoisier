"""
Query generation for LLM-assisted analysis of mass spectrometry data
"""
from typing import Dict, List, Any, Optional
import json
import random
from enum import Enum

from lavoisier.core.logging import get_logger


class QueryType(Enum):
    """Types of queries that can be generated for LLMs"""
    
    BASIC = "basic"                # Simple factual queries about the data
    EXPLORATORY = "exploratory"    # Open-ended exploration queries
    COMPARATIVE = "comparative"    # Queries comparing different aspects
    ANALYTICAL = "analytical"      # Deeper analytical queries
    METACOGNITIVE = "metacognitive"  # Queries about the analysis process itself


class QueryTemplate:
    """Template for generating queries"""
    
    def __init__(self, 
                 query_type: QueryType, 
                 template: str, 
                 required_data: List[str],
                 complexity: int = 1):
        """
        Initialize a query template
        
        Args:
            query_type: Type of query
            template: String template with placeholders
            required_data: List of data fields required for this template
            complexity: Complexity level (1-5)
        """
        self.query_type = query_type
        self.template = template
        self.required_data = required_data
        self.complexity = complexity


class QueryGenerator:
    """
    Generator for LLM queries at various complexity levels
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the query generator
        
        Args:
            config: Optional configuration
        """
        self.config = config or {}
        self.logger = get_logger("query_generator")
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[QueryType, List[QueryTemplate]]:
        """
        Initialize query templates
        
        Returns:
            Dictionary mapping query types to lists of templates
        """
        templates = {
            QueryType.BASIC: [
                QueryTemplate(
                    QueryType.BASIC,
                    "What is the most abundant m/z value in the given spectrum?",
                    ["mz_array", "intensity_array"],
                    1
                ),
                QueryTemplate(
                    QueryType.BASIC,
                    "How many peaks are present in the MS1 spectrum?",
                    ["mz_array", "intensity_array", "ms_level"],
                    1
                ),
                QueryTemplate(
                    QueryType.BASIC,
                    "What is the retention time for this spectrum?",
                    ["retention_time"],
                    1
                ),
            ],
            QueryType.EXPLORATORY: [
                QueryTemplate(
                    QueryType.EXPLORATORY,
                    "Can you identify any notable patterns in the mass spectrum?",
                    ["mz_array", "intensity_array"],
                    2
                ),
                QueryTemplate(
                    QueryType.EXPLORATORY,
                    "What potential compound classes might be represented in this spectrum?",
                    ["mz_array", "intensity_array", "ms_level"],
                    3
                ),
                QueryTemplate(
                    QueryType.EXPLORATORY,
                    "Are there any unusual features in this mass spectrum that warrant further investigation?",
                    ["mz_array", "intensity_array"],
                    3
                ),
            ],
            QueryType.COMPARATIVE: [
                QueryTemplate(
                    QueryType.COMPARATIVE,
                    "How does this spectrum compare to the previous spectrum in terms of peak distribution?",
                    ["mz_array", "intensity_array", "previous_spectrum"],
                    3
                ),
                QueryTemplate(
                    QueryType.COMPARATIVE,
                    "Compare the MS1 and MS2 spectra and identify potential fragment ions.",
                    ["ms1_spectrum", "ms2_spectrum"],
                    4
                ),
                QueryTemplate(
                    QueryType.COMPARATIVE,
                    "How do the visual and numerical analyses differ for this spectrum?",
                    ["numeric_analysis", "visual_analysis"],
                    4
                ),
            ],
            QueryType.ANALYTICAL: [
                QueryTemplate(
                    QueryType.ANALYTICAL,
                    "Based on the fragmentation pattern, what functional groups are likely present?",
                    ["ms2_spectrum", "precursor_mz"],
                    4
                ),
                QueryTemplate(
                    QueryType.ANALYTICAL,
                    "Analyze the isotopic pattern of the peak at {peak_mz} and determine if it matches expected natural abundance.",
                    ["mz_array", "intensity_array", "peak_mz"],
                    4
                ),
                QueryTemplate(
                    QueryType.ANALYTICAL,
                    "Given the retention time and MS2 fragmentation, propose potential compound identifications.",
                    ["retention_time", "ms2_spectrum", "precursor_mz"],
                    5
                ),
            ],
            QueryType.METACOGNITIVE: [
                QueryTemplate(
                    QueryType.METACOGNITIVE,
                    "What additional data would be needed to increase confidence in the compound identification?",
                    ["numeric_analysis", "identification_confidence"],
                    4
                ),
                QueryTemplate(
                    QueryType.METACOGNITIVE, 
                    "How could we modify our analysis approach to better characterize this type of sample?",
                    ["sample_type", "analysis_approach", "results_summary"],
                    5
                ),
                QueryTemplate(
                    QueryType.METACOGNITIVE,
                    "What are the limitations of the current analysis and how might they affect our conclusions?",
                    ["analysis_parameters", "results_summary"],
                    5
                ),
            ]
        }
        
        return templates
    
    def generate_query(self, 
                      query_type: Optional[QueryType] = None, 
                      complexity: Optional[int] = None,
                      available_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a query of specified type and complexity
        
        Args:
            query_type: Optional query type, randomly chosen if None
            complexity: Optional complexity level (1-5), randomly chosen if None
            available_data: Available data fields to fill placeholders
            
        Returns:
            Generated query string
        """
        # Select query type if not specified
        if query_type is None:
            query_type = random.choice(list(QueryType))
        
        # Filter templates by complexity if specified
        templates = self.templates[query_type]
        if complexity is not None:
            templates = [t for t in templates if t.complexity == complexity]
        
        # Filter templates by available data if specified
        if available_data is not None:
            available_fields = set(available_data.keys())
            templates = [
                t for t in templates 
                if all(field in available_fields for field in t.required_data)
            ]
        
        if not templates:
            self.logger.warning(f"No suitable templates found for query type {query_type} and complexity {complexity}")
            return "What insights can you provide about this mass spectrometry data?"
        
        # Select a random template
        template = random.choice(templates)
        
        # Fill placeholders if available_data is provided
        query = template.template
        if available_data is not None:
            try:
                # Extract placeholders from the template
                import re
                placeholders = re.findall(r'{(\w+)}', query)
                
                # Fill placeholders
                for placeholder in placeholders:
                    if placeholder in available_data:
                        query = query.replace(f"{{{placeholder}}}", str(available_data[placeholder]))
            except Exception as e:
                self.logger.error(f"Error filling placeholders: {str(e)}")
        
        return query
    
    def generate_query_list(self,
                           query_types: Optional[List[QueryType]] = None,
                           complexity_range: Optional[tuple] = None,
                           count: int = 5,
                           available_data: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generate a list of queries with various types and complexities
        
        Args:
            query_types: List of query types to include, all if None
            complexity_range: Tuple of (min, max) complexity, full range if None
            count: Number of queries to generate
            available_data: Available data fields to fill placeholders
            
        Returns:
            List of query strings
        """
        queries = []
        
        # Set defaults
        if query_types is None:
            query_types = list(QueryType)
        
        if complexity_range is None:
            complexity_range = (1, 5)
        
        # Generate specified number of queries
        for _ in range(count):
            query_type = random.choice(query_types)
            complexity = random.randint(complexity_range[0], complexity_range[1])
            query = self.generate_query(query_type, complexity, available_data)
            queries.append(query)
        
        return queries
    
    def generate_sequential_queries(self, 
                                  initial_data: Dict[str, Any],
                                  max_queries: int = 5,
                                  increasing_complexity: bool = True) -> List[str]:
        """
        Generate a sequence of queries with increasing complexity
        
        Args:
            initial_data: Initial data to use for query generation
            max_queries: Maximum number of queries to generate
            increasing_complexity: Whether to increase complexity with each query
            
        Returns:
            List of sequential queries
        """
        queries = []
        available_data = initial_data.copy()
        
        # Define complexity progression
        complexity_levels = list(range(1, 6)) if increasing_complexity else [random.randint(1, 5) for _ in range(max_queries)]
        
        # Select query types in a sensible order if increasing complexity
        query_type_order = [
            QueryType.BASIC,
            QueryType.EXPLORATORY,
            QueryType.COMPARATIVE,
            QueryType.ANALYTICAL,
            QueryType.METACOGNITIVE
        ] if increasing_complexity else [random.choice(list(QueryType)) for _ in range(max_queries)]
        
        # Generate queries with increasing complexity
        for i in range(min(max_queries, len(complexity_levels))):
            query_type = query_type_order[i] if i < len(query_type_order) else query_type_order[-1]
            complexity = complexity_levels[i]
            
            query = self.generate_query(query_type, complexity, available_data)
            queries.append(query)
            
            # Simulate accumulating knowledge for future queries
            # In a real scenario, this would be informed by the LLM's response
            available_data["previous_queries"] = available_data.get("previous_queries", []) + [query]
            
        return queries 
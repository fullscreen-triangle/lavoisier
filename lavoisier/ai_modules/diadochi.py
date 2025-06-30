"""
Diadochi: Multi-Domain Query Routing System

Advanced LLM routing framework that distributes queries across specialized
expert domains for distributed analysis.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ExpertDomain(Enum):
    """Expert domains for query routing"""
    CHEMICAL_STRUCTURE = "chemical_structure"
    MASS_SPECTROMETRY = "mass_spectrometry"
    METABOLOMICS = "metabolomics"
    PROTEOMICS = "proteomics"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    BIOLOGICAL_PATHWAYS = "biological_pathways"
    INSTRUMENT_METHODS = "instrument_methods"
    DATA_PROCESSING = "data_processing"

@dataclass
class QueryClassification:
    """Query classification result"""
    primary_domain: ExpertDomain
    secondary_domains: List[ExpertDomain]
    confidence: float
    complexity_score: float
    requires_collaboration: bool

@dataclass
class ExpertResponse:
    """Response from domain expert"""
    domain: ExpertDomain
    response: str
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]

class QueryClassifier:
    """Classifies queries to determine routing strategy"""

    def __init__(self):
        self.domain_keywords = {
            ExpertDomain.CHEMICAL_STRUCTURE: [
                "molecule", "structure", "formula", "smiles", "inchi", "bond",
                "functional group", "stereochemistry", "isomer"
            ],
            ExpertDomain.MASS_SPECTROMETRY: [
                "mass spec", "ms", "fragmentation", "ionization", "precursor",
                "collision energy", "mz", "spectrum", "peak"
            ],
            ExpertDomain.METABOLOMICS: [
                "metabolite", "pathway", "metabolism", "biomarker", "flux",
                "metabolic network", "small molecule"
            ],
            ExpertDomain.PROTEOMICS: [
                "protein", "peptide", "amino acid", "sequence", "modification",
                "digest", "trypsin", "proteome"
            ],
            ExpertDomain.STATISTICAL_ANALYSIS: [
                "statistics", "pvalue", "correlation", "regression", "anova",
                "significance", "distribution", "test"
            ],
            ExpertDomain.BIOLOGICAL_PATHWAYS: [
                "pathway", "kegg", "reactome", "enzyme", "reaction", "network",
                "metabolism", "biosynthesis"
            ],
            ExpertDomain.INSTRUMENT_METHODS: [
                "method", "protocol", "instrument", "parameters", "optimization",
                "calibration", "maintenance", "troubleshooting"
            ],
            ExpertDomain.DATA_PROCESSING: [
                "processing", "algorithm", "pipeline", "workflow", "analysis",
                "preprocessing", "normalization", "filtering"
            ]
        }

    def classify_query(self, query: str) -> QueryClassification:
        """Classify query to determine routing"""
        query_lower = query.lower()
        domain_scores = {}

        # Calculate domain relevance scores
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            domain_scores[domain] = score / len(keywords)

        # Find primary and secondary domains
        sorted_domains = sorted(domain_scores.items(),
                               key=lambda x: x[1], reverse=True)

        primary_domain = sorted_domains[0][0]
        primary_score = sorted_domains[0][1]

        secondary_domains = [domain for domain, score in sorted_domains[1:]
                           if score > 0.1]

        # Determine if collaboration is needed
        requires_collaboration = (
            len(secondary_domains) > 1 or
            primary_score < 0.3 or
            "compare" in query_lower or
            "integrate" in query_lower
        )

        # Calculate complexity
        complexity_score = (
            len(query.split()) / 100.0 +  # Length factor
            len(secondary_domains) / 10.0 +  # Multi-domain factor
            (1.0 if requires_collaboration else 0.0)  # Collaboration factor
        )

        return QueryClassification(
            primary_domain=primary_domain,
            secondary_domains=secondary_domains,
            confidence=primary_score,
            complexity_score=min(complexity_score, 1.0),
            requires_collaboration=requires_collaboration
        )

class ExpertAgent:
    """Individual expert agent for specific domain"""

    def __init__(self, domain: ExpertDomain, llm_client: Any = None):
        self.domain = domain
        self.llm_client = llm_client
        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        """Get domain-specific system prompt"""
        prompts = {
            ExpertDomain.CHEMICAL_STRUCTURE:
                "You are an expert in chemical structures and molecular properties.",
            ExpertDomain.MASS_SPECTROMETRY:
                "You are an expert in mass spectrometry instrumentation and data interpretation.",
            ExpertDomain.METABOLOMICS:
                "You are an expert in metabolomics and small molecule analysis.",
            ExpertDomain.PROTEOMICS:
                "You are an expert in proteomics and protein analysis.",
            ExpertDomain.STATISTICAL_ANALYSIS:
                "You are an expert in statistical analysis and data science methods.",
            ExpertDomain.BIOLOGICAL_PATHWAYS:
                "You are an expert in biological pathways and metabolic networks.",
            ExpertDomain.INSTRUMENT_METHODS:
                "You are an expert in analytical instrument methods and protocols.",
            ExpertDomain.DATA_PROCESSING:
                "You are an expert in data processing pipelines and algorithms."
        }

        return prompts.get(self.domain, "You are a scientific expert.")

    async def process_query(self, query: str, context: Dict = None) -> ExpertResponse:
        """Process query in domain expertise"""
        import time
        start_time = time.time()

        # Simulate LLM processing (replace with actual LLM call)
        await asyncio.sleep(0.1)  # Simulate processing time

        response = f"Domain {self.domain.value} analysis: {query[:100]}..."
        confidence = 0.8 + 0.2 * hash(query) % 10 / 10.0

        processing_time = time.time() - start_time

        return ExpertResponse(
            domain=self.domain,
            response=response,
            confidence=confidence,
            processing_time=processing_time,
            metadata={"context": context}
        )

class CollaborationOrchestrator:
    """Orchestrates collaboration between expert agents"""

    def __init__(self):
        self.collaboration_patterns = {
            "consensus": self._consensus_collaboration,
            "sequential": self._sequential_collaboration,
            "hierarchical": self._hierarchical_collaboration,
            "debate": self._debate_collaboration
        }

    async def orchestrate_collaboration(self,
                                      experts: List[ExpertAgent],
                                      query: str,
                                      pattern: str = "consensus") -> List[ExpertResponse]:
        """Orchestrate collaboration between experts"""
        if pattern not in self.collaboration_patterns:
            pattern = "consensus"

        return await self.collaboration_patterns[pattern](experts, query)

    async def _consensus_collaboration(self, experts: List[ExpertAgent],
                                     query: str) -> List[ExpertResponse]:
        """Consensus-based collaboration"""
        # All experts work independently then consensus
        tasks = [expert.process_query(query) for expert in experts]
        responses = await asyncio.gather(*tasks)

        return responses

    async def _sequential_collaboration(self, experts: List[ExpertAgent],
                                      query: str) -> List[ExpertResponse]:
        """Sequential collaboration - experts build on each other"""
        responses = []
        context = {}

        for expert in experts:
            response = await expert.process_query(query, context)
            responses.append(response)
            context[expert.domain.value] = response.response

        return responses

    async def _hierarchical_collaboration(self, experts: List[ExpertAgent],
                                        query: str) -> List[ExpertResponse]:
        """Hierarchical collaboration with primary expert leading"""
        if not experts:
            return []

        # Primary expert processes first
        primary_response = await experts[0].process_query(query)
        responses = [primary_response]

        # Secondary experts provide supporting analysis
        context = {"primary_analysis": primary_response.response}
        secondary_tasks = [expert.process_query(query, context)
                          for expert in experts[1:]]

        if secondary_tasks:
            secondary_responses = await asyncio.gather(*secondary_tasks)
            responses.extend(secondary_responses)

        return responses

    async def _debate_collaboration(self, experts: List[ExpertAgent],
                                  query: str) -> List[ExpertResponse]:
        """Debate-style collaboration for conflicting viewpoints"""
        # First round - independent responses
        round1_responses = await self._consensus_collaboration(experts, query)

        # Second round - respond to others' analyses
        context = {f"expert_{i}": resp.response
                  for i, resp in enumerate(round1_responses)}

        round2_tasks = [expert.process_query(
            f"Considering other experts' views: {query}", context)
            for expert in experts]

        round2_responses = await asyncio.gather(*round2_tasks)

        return round1_responses + round2_responses

class ResponseSynthesizer:
    """Synthesizes multiple expert responses into coherent output"""

    def __init__(self):
        self.synthesis_strategies = {
            "weighted_average": self._weighted_average_synthesis,
            "confidence_ranking": self._confidence_ranking_synthesis,
            "domain_priority": self._domain_priority_synthesis,
            "comprehensive": self._comprehensive_synthesis
        }

    def synthesize_responses(self, responses: List[ExpertResponse],
                           strategy: str = "comprehensive") -> str:
        """Synthesize expert responses"""
        if not responses:
            return "No expert responses available."

        if strategy not in self.synthesis_strategies:
            strategy = "comprehensive"

        return self.synthesis_strategies[strategy](responses)

    def _weighted_average_synthesis(self, responses: List[ExpertResponse]) -> str:
        """Weight responses by confidence scores"""
        total_weight = sum(resp.confidence for resp in responses)

        if total_weight == 0:
            return "No confident responses available."

        synthesis = "Weighted synthesis of expert responses:\n\n"

        for resp in sorted(responses, key=lambda x: x.confidence, reverse=True):
            weight = resp.confidence / total_weight
            synthesis += f"{resp.domain.value} ({weight:.2f}): {resp.response}\n\n"

        return synthesis

    def _confidence_ranking_synthesis(self, responses: List[ExpertResponse]) -> str:
        """Rank responses by confidence"""
        sorted_responses = sorted(responses, key=lambda x: x.confidence, reverse=True)

        synthesis = "Expert responses ranked by confidence:\n\n"

        for i, resp in enumerate(sorted_responses, 1):
            synthesis += f"{i}. {resp.domain.value} (confidence: {resp.confidence:.2f}):\n"
            synthesis += f"   {resp.response}\n\n"

        return synthesis

    def _domain_priority_synthesis(self, responses: List[ExpertResponse]) -> str:
        """Prioritize responses by domain relevance"""
        # Define domain priority order
        priority_order = [
            ExpertDomain.MASS_SPECTROMETRY,
            ExpertDomain.CHEMICAL_STRUCTURE,
            ExpertDomain.METABOLOMICS,
            ExpertDomain.STATISTICAL_ANALYSIS,
            ExpertDomain.BIOLOGICAL_PATHWAYS,
            ExpertDomain.DATA_PROCESSING,
            ExpertDomain.PROTEOMICS,
            ExpertDomain.INSTRUMENT_METHODS
        ]

        # Sort by priority
        def get_priority(resp):
            try:
                return priority_order.index(resp.domain)
            except ValueError:
                return len(priority_order)

        sorted_responses = sorted(responses, key=get_priority)

        synthesis = "Expert responses by domain priority:\n\n"

        for resp in sorted_responses:
            synthesis += f"{resp.domain.value}:\n{resp.response}\n\n"

        return synthesis

    def _comprehensive_synthesis(self, responses: List[ExpertResponse]) -> str:
        """Comprehensive synthesis combining all perspectives"""
        if not responses:
            return "No responses to synthesize."

        synthesis = "Comprehensive Expert Analysis\n" + "="*40 + "\n\n"

        # Summary section
        synthesis += "SUMMARY:\n"
        synthesis += f"- {len(responses)} expert domains consulted\n"
        synthesis += f"- Average confidence: {np.mean([r.confidence for r in responses]):.2f}\n"
        synthesis += f"- Total processing time: {sum(r.processing_time for r in responses):.2f}s\n\n"

        # High confidence responses first
        high_confidence = [r for r in responses if r.confidence > 0.7]
        if high_confidence:
            synthesis += "HIGH CONFIDENCE ANALYSES:\n"
            synthesis += "-" * 25 + "\n"
            for resp in sorted(high_confidence, key=lambda x: x.confidence, reverse=True):
                synthesis += f"{resp.domain.value} (confidence: {resp.confidence:.2f}):\n"
                synthesis += f"{resp.response}\n\n"

        # Supporting analyses
        supporting = [r for r in responses if r.confidence <= 0.7]
        if supporting:
            synthesis += "SUPPORTING ANALYSES:\n"
            synthesis += "-" * 20 + "\n"
            for resp in supporting:
                synthesis += f"{resp.domain.value}:\n{resp.response}\n\n"

        return synthesis

class Diadochi:
    """Main Diadochi multi-domain routing system"""

    def __init__(self, llm_clients: Dict[ExpertDomain, Any] = None):
        self.classifier = QueryClassifier()
        self.orchestrator = CollaborationOrchestrator()
        self.synthesizer = ResponseSynthesizer()

        # Initialize expert agents
        self.experts = {}
        for domain in ExpertDomain:
            llm_client = llm_clients.get(domain) if llm_clients else None
            self.experts[domain] = ExpertAgent(domain, llm_client)

    async def route_and_process(self, query: str,
                               collaboration_pattern: str = "consensus",
                               synthesis_strategy: str = "comprehensive") -> Dict[str, Any]:
        """Main entry point for query routing and processing"""
        logger.info(f"Processing query: {query[:100]}...")

        # Classify query
        classification = self.classifier.classify_query(query)

        # Select relevant experts
        relevant_experts = [self.experts[classification.primary_domain]]
        relevant_experts.extend([self.experts[domain]
                               for domain in classification.secondary_domains])

        # Process query
        if classification.requires_collaboration and len(relevant_experts) > 1:
            responses = await self.orchestrator.orchestrate_collaboration(
                relevant_experts, query, collaboration_pattern)
        else:
            # Single expert processing
            responses = [await relevant_experts[0].process_query(query)]

        # Synthesize responses
        synthesis = self.synthesizer.synthesize_responses(responses, synthesis_strategy)

        result = {
            "query": query,
            "classification": classification,
            "expert_responses": responses,
            "synthesis": synthesis,
            "processing_metadata": {
                "num_experts": len(relevant_experts),
                "collaboration_pattern": collaboration_pattern,
                "synthesis_strategy": synthesis_strategy,
                "total_processing_time": sum(r.processing_time for r in responses)
            }
        }

        logger.info(f"Query processed with {len(responses)} expert responses")
        return result

    def get_available_domains(self) -> List[ExpertDomain]:
        """Get list of available expert domains"""
        return list(self.experts.keys())

    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get statistics about domain usage and performance"""
        return {
            "available_domains": len(self.experts),
            "classification_keywords": {domain.value: len(keywords)
                                      for domain, keywords in self.classifier.domain_keywords.items()},
            "collaboration_patterns": list(self.orchestrator.collaboration_patterns.keys()),
            "synthesis_strategies": list(self.synthesizer.synthesis_strategies.keys())
        }

# Import numpy for calculations
import numpy as np

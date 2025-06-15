"""
Core Framework for Diadochi: Intelligent LLM Combination System

This module provides the foundational classes and interfaces for combining
multiple domain-expert LLMs using various architectural patterns.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class IntegrationPattern(Enum):
    """Available architectural patterns for LLM integration"""
    ROUTER_ENSEMBLE = "router_ensemble"
    SEQUENTIAL_CHAIN = "sequential_chain"
    MIXTURE_OF_EXPERTS = "mixture_of_experts"
    SYSTEM_PROMPTS = "system_prompts"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    MULTI_DOMAIN_RAG = "multi_domain_rag"
    HYBRID = "hybrid"

class DomainType(Enum):
    """Types of domain expertise"""
    MASS_SPECTROMETRY = "mass_spectrometry"
    PROTEOMICS = "proteomics"
    METABOLOMICS = "metabolomics"
    BIOINFORMATICS = "bioinformatics"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    DATA_VISUALIZATION = "data_visualization"
    MACHINE_LEARNING = "machine_learning"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    GENERAL = "general"

@dataclass
class DomainSpecification:
    """Specification for a domain of expertise"""
    domain_type: DomainType
    name: str
    description: str
    keywords: List[str]
    expertise_areas: List[str]
    confidence_threshold: float = 0.6
    priority_weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QueryContext:
    """Context information for a query"""
    query: str
    domain_hints: List[DomainType] = field(default_factory=list)
    required_patterns: List[IntegrationPattern] = field(default_factory=list)
    confidence_threshold: float = 0.5
    max_experts: int = 5
    timeout_seconds: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExpertResponse:
    """Response from a domain expert"""
    expert_id: str
    domain_type: DomainType
    response: str
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class IntegratedResponse:
    """Final integrated response from multiple experts"""
    response: str
    expert_responses: List[ExpertResponse]
    integration_pattern: IntegrationPattern
    overall_confidence: float
    processing_time: float
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class DomainExpert(ABC):
    """Abstract base class for domain expert models"""
    
    def __init__(self, 
                 expert_id: str,
                 domain_spec: DomainSpecification,
                 model_config: Dict[str, Any]):
        self.expert_id = expert_id
        self.domain_spec = domain_spec
        self.model_config = model_config
        self.performance_history: List[Dict[str, Any]] = []
        self.is_available = True
        
    @abstractmethod
    async def generate_response(self, 
                               query: str, 
                               context: Optional[QueryContext] = None) -> ExpertResponse:
        """Generate a response from this domain expert"""
        pass
    
    @abstractmethod
    def estimate_confidence(self, query: str) -> float:
        """Estimate confidence for handling a given query"""
        pass
    
    @abstractmethod
    def get_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings for text using this expert's model"""
        pass
    
    def update_performance(self, 
                          query: str,
                          response: ExpertResponse,
                          feedback_score: Optional[float] = None):
        """Update performance history with query results"""
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'query_length': len(query),
            'response_length': len(response.response),
            'confidence': response.confidence,
            'processing_time': response.processing_time,
            'feedback_score': feedback_score
        }
        self.performance_history.append(performance_record)
        
        # Keep only last 1000 records
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for this expert"""
        if not self.performance_history:
            return {'avg_confidence': 0.0, 'avg_processing_time': 0.0, 'success_rate': 0.0}
        
        recent_history = self.performance_history[-100:]  # Last 100 queries
        
        avg_confidence = np.mean([h['confidence'] for h in recent_history])
        avg_processing_time = np.mean([h['processing_time'] for h in recent_history])
        
        # Success rate based on feedback scores (if available)
        feedback_scores = [h['feedback_score'] for h in recent_history if h['feedback_score'] is not None]
        success_rate = np.mean([s > 0.6 for s in feedback_scores]) if feedback_scores else 0.5
        
        return {
            'avg_confidence': float(avg_confidence),
            'avg_processing_time': float(avg_processing_time), 
            'success_rate': float(success_rate),
            'total_queries': len(self.performance_history)
        }

class MultiDomainSystem(ABC):
    """Abstract base class for multi-domain LLM systems"""
    
    def __init__(self, 
                 system_id: str,
                 integration_pattern: IntegrationPattern,
                 experts: List[DomainExpert]):
        self.system_id = system_id
        self.integration_pattern = integration_pattern
        self.experts = {expert.expert_id: expert for expert in experts}
        self.query_history: List[Dict[str, Any]] = []
        
    @abstractmethod
    async def process_query(self, 
                           query: str,
                           context: Optional[QueryContext] = None) -> IntegratedResponse:
        """Process a query using the multi-domain system"""
        pass
    
    def add_expert(self, expert: DomainExpert):
        """Add a new domain expert to the system"""
        self.experts[expert.expert_id] = expert
        logger.info(f"Added expert {expert.expert_id} for domain {expert.domain_spec.domain_type.value}")
    
    def remove_expert(self, expert_id: str):
        """Remove a domain expert from the system"""
        if expert_id in self.experts:
            del self.experts[expert_id]
            logger.info(f"Removed expert {expert_id}")
    
    def get_available_experts(self) -> List[DomainExpert]:
        """Get list of currently available experts"""
        return [expert for expert in self.experts.values() if expert.is_available]
    
    def get_experts_by_domain(self, domain_type: DomainType) -> List[DomainExpert]:
        """Get experts for a specific domain"""
        return [expert for expert in self.experts.values() 
                if expert.domain_spec.domain_type == domain_type and expert.is_available]
    
    async def estimate_query_confidence(self, 
                                       query: str,
                                       context: Optional[QueryContext] = None) -> Dict[str, float]:
        """Estimate confidence scores for all experts"""
        confidence_scores = {}
        
        for expert_id, expert in self.experts.items():
            if expert.is_available:
                try:
                    confidence = expert.estimate_confidence(query)
                    confidence_scores[expert_id] = confidence
                except Exception as e:
                    logger.error(f"Error estimating confidence for expert {expert_id}: {str(e)}")
                    confidence_scores[expert_id] = 0.0
        
        return confidence_scores
    
    def log_query(self, 
                  query: str,
                  context: Optional[QueryContext],
                  response: IntegratedResponse):
        """Log query and response for analysis"""
        query_record = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'context': context.__dict__ if context else None,
            'integration_pattern': response.integration_pattern.value,
            'experts_used': [r.expert_id for r in response.expert_responses],
            'overall_confidence': response.overall_confidence,
            'processing_time': response.processing_time,
            'quality_score': response.quality_score
        }
        
        self.query_history.append(query_record)
        
        # Keep only last 1000 queries
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-1000:]

class DiadochiFramework:
    """
    Main framework class that orchestrates all Diadochi components.
    
    This class provides a high-level interface for creating and managing
    multi-domain LLM systems using various architectural patterns.
    """
    
    def __init__(self,
                 framework_id: str = "diadochi_main",
                 default_timeout: int = 300,
                 max_concurrent_experts: int = 10):
        self.framework_id = framework_id
        self.default_timeout = default_timeout
        self.max_concurrent_experts = max_concurrent_experts
        
        # System registries
        self.experts: Dict[str, DomainExpert] = {}
        self.systems: Dict[str, MultiDomainSystem] = {}
        self.domain_registry: Dict[DomainType, List[str]] = {}
        
        # Framework statistics
        self.total_queries_processed = 0
        self.total_processing_time = 0.0
        self.pattern_usage_stats: Dict[IntegrationPattern, int] = {}
        
        # Executor for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_experts)
        
        logger.info(f"Diadochi Framework initialized: {framework_id}")
    
    def register_expert(self, expert: DomainExpert):
        """Register a domain expert with the framework"""
        self.experts[expert.expert_id] = expert
        
        # Update domain registry
        domain_type = expert.domain_spec.domain_type
        if domain_type not in self.domain_registry:
            self.domain_registry[domain_type] = []
        self.domain_registry[domain_type].append(expert.expert_id)
        
        logger.info(f"Registered expert {expert.expert_id} for domain {domain_type.value}")
    
    def register_system(self, system: MultiDomainSystem):
        """Register a multi-domain system with the framework"""
        self.systems[system.system_id] = system
        logger.info(f"Registered multi-domain system: {system.system_id}")
    
    def create_router_ensemble(self,
                              system_id: str,
                              router_type: str = "embedding",
                              experts: Optional[List[str]] = None) -> "RouterEnsemble":
        """Create a router-based ensemble system"""
        from .routers import EmbeddingRouter
        from .systems import RouterEnsemble
        
        if experts is None:
            selected_experts = list(self.experts.values())
        else:
            selected_experts = [self.experts[eid] for eid in experts if eid in self.experts]
        
        if router_type == "embedding":
            router = EmbeddingRouter()
        else:
            raise ValueError(f"Unsupported router type: {router_type}")
        
        ensemble = RouterEnsemble(
            system_id=system_id,
            router=router,
            experts=selected_experts
        )
        
        self.register_system(ensemble)
        return ensemble
    
    def create_sequential_chain(self,
                               system_id: str,
                               expert_sequence: List[str],
                               chain_type: str = "basic") -> "SequentialChain":
        """Create a sequential chain system"""
        from .chains import SequentialChain, SummarizingChain
        
        selected_experts = [self.experts[eid] for eid in expert_sequence if eid in self.experts]
        
        if chain_type == "basic":
            chain = SequentialChain(
                system_id=system_id,
                experts=selected_experts
            )
        elif chain_type == "summarizing":
            chain = SummarizingChain(
                system_id=system_id,
                experts=selected_experts
            )
        else:
            raise ValueError(f"Unsupported chain type: {chain_type}")
        
        self.register_system(chain)
        return chain
    
    def create_mixture_of_experts(self,
                                 system_id: str,
                                 experts: Optional[List[str]] = None,
                                 mixer_type: str = "synthesis") -> "MixtureOfExperts":
        """Create a mixture of experts system"""
        from .mixers import SynthesisMixer
        from .systems import MixtureOfExperts
        
        if experts is None:
            selected_experts = list(self.experts.values())
        else:
            selected_experts = [self.experts[eid] for eid in experts if eid in self.experts]
        
        if mixer_type == "synthesis":
            mixer = SynthesisMixer()
        else:
            raise ValueError(f"Unsupported mixer type: {mixer_type}")
        
        moe = MixtureOfExperts(
            system_id=system_id,
            experts=selected_experts,
            mixer=mixer
        )
        
        self.register_system(moe)
        return moe
    
    async def process_query_auto(self,
                                query: str,
                                context: Optional[QueryContext] = None,
                                preferred_pattern: Optional[IntegrationPattern] = None) -> IntegratedResponse:
        """
        Automatically select the best system and process a query.
        
        This method analyzes the query and context to determine the optimal
        integration pattern and expert selection.
        """
        start_time = datetime.now()
        
        if context is None:
            context = QueryContext(query=query)
        
        # Determine optimal integration pattern
        if preferred_pattern:
            pattern = preferred_pattern
        else:
            pattern = await self._select_optimal_pattern(query, context)
        
        # Select appropriate system or create one on-demand
        system = await self._get_or_create_system(pattern, query, context)
        
        # Process the query
        response = await system.process_query(query, context)
        
        # Update framework statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        self.total_queries_processed += 1
        self.total_processing_time += processing_time
        
        if pattern not in self.pattern_usage_stats:
            self.pattern_usage_stats[pattern] = 0
        self.pattern_usage_stats[pattern] += 1
        
        logger.info(f"Processed query using {pattern.value} in {processing_time:.2f}s")
        
        return response
    
    async def _select_optimal_pattern(self,
                                     query: str,
                                     context: QueryContext) -> IntegrationPattern:
        """Select the optimal integration pattern for a query"""
        
        # Analyze query characteristics
        query_length = len(query.split())
        domain_hints = len(context.domain_hints)
        
        # Get confidence estimates from all experts
        confidence_scores = {}
        for expert_id, expert in self.experts.items():
            if expert.is_available:
                confidence_scores[expert_id] = expert.estimate_confidence(query)
        
        # Count high-confidence experts
        high_confidence_experts = sum(1 for conf in confidence_scores.values() if conf > 0.7)
        medium_confidence_experts = sum(1 for conf in confidence_scores.values() if 0.4 < conf <= 0.7)
        
        # Selection logic based on query characteristics
        if high_confidence_experts == 1 and medium_confidence_experts == 0:
            # Single clear expert - use router ensemble
            return IntegrationPattern.ROUTER_ENSEMBLE
        
        elif high_confidence_experts > 1 and query_length < 50:
            # Multiple experts, short query - use mixture of experts
            return IntegrationPattern.MIXTURE_OF_EXPERTS
        
        elif domain_hints > 1 or "how does" in query.lower() or "what is the relationship" in query.lower():
            # Multi-domain or relationship query - use sequential chain
            return IntegrationPattern.SEQUENTIAL_CHAIN
        
        elif query_length > 100:
            # Long, complex query - use system prompts
            return IntegrationPattern.SYSTEM_PROMPTS
        
        else:
            # Default to mixture of experts
            return IntegrationPattern.MIXTURE_OF_EXPERTS
    
    async def _get_or_create_system(self,
                                   pattern: IntegrationPattern,
                                   query: str,
                                   context: QueryContext) -> MultiDomainSystem:
        """Get existing system or create one for the pattern"""
        
        # Look for existing system with this pattern
        for system in self.systems.values():
            if system.integration_pattern == pattern:
                return system
        
        # Create new system on demand
        system_id = f"auto_{pattern.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if pattern == IntegrationPattern.ROUTER_ENSEMBLE:
            return self.create_router_ensemble(system_id)
        
        elif pattern == IntegrationPattern.SEQUENTIAL_CHAIN:
            # Select expert sequence based on domain hints or confidence
            expert_sequence = self._select_expert_sequence(query, context)
            return self.create_sequential_chain(system_id, expert_sequence)
        
        elif pattern == IntegrationPattern.MIXTURE_OF_EXPERTS:
            return self.create_mixture_of_experts(system_id)
        
        else:
            # Fallback to mixture of experts
            return self.create_mixture_of_experts(system_id)
    
    def _select_expert_sequence(self,
                               query: str,
                               context: QueryContext) -> List[str]:
        """Select optimal sequence of experts for sequential chaining"""
        
        # If domain hints provided, use them to guide sequence
        if context.domain_hints:
            sequence = []
            for domain_hint in context.domain_hints:
                if domain_hint in self.domain_registry:
                    # Get best expert for this domain
                    domain_experts = self.domain_registry[domain_hint]
                    if domain_experts:
                        # Select expert with best performance
                        best_expert = max(domain_experts, 
                                        key=lambda eid: self.experts[eid].get_performance_metrics()['avg_confidence'])
                        sequence.append(best_expert)
            return sequence
        
        # Otherwise, select based on confidence scores
        confidence_scores = {}
        for expert_id, expert in self.experts.items():
            if expert.is_available:
                confidence_scores[expert_id] = expert.estimate_confidence(query)
        
        # Sort by confidence and take top experts
        sorted_experts = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
        return [expert_id for expert_id, conf in sorted_experts[:context.max_experts] if conf > context.confidence_threshold]
    
    def get_framework_statistics(self) -> Dict[str, Any]:
        """Get comprehensive framework statistics"""
        
        avg_processing_time = self.total_processing_time / max(1, self.total_queries_processed)
        
        # Expert performance summary
        expert_stats = {}
        for expert_id, expert in self.experts.items():
            expert_stats[expert_id] = {
                'domain': expert.domain_spec.domain_type.value,
                'performance': expert.get_performance_metrics(),
                'available': expert.is_available
            }
        
        # Domain coverage
        domain_coverage = {}
        for domain_type, expert_ids in self.domain_registry.items():
            available_experts = [eid for eid in expert_ids if self.experts[eid].is_available]
            domain_coverage[domain_type.value] = {
                'total_experts': len(expert_ids),
                'available_experts': len(available_experts),
                'expert_ids': available_experts
            }
        
        return {
            'framework_id': self.framework_id,
            'total_queries_processed': self.total_queries_processed,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': avg_processing_time,
            'pattern_usage_stats': {pattern.value: count for pattern, count in self.pattern_usage_stats.items()},
            'expert_statistics': expert_stats,
            'domain_coverage': domain_coverage,
            'registered_systems': list(self.systems.keys()),
            'timestamp': datetime.now().isoformat()
        }
    
    def export_configuration(self, filepath: str):
        """Export framework configuration to file"""
        config = {
            'framework_id': self.framework_id,
            'experts': {
                expert_id: {
                    'domain_type': expert.domain_spec.domain_type.value,
                    'domain_name': expert.domain_spec.name,
                    'description': expert.domain_spec.description,
                    'model_config': expert.model_config
                }
                for expert_id, expert in self.experts.items()
            },
            'systems': {
                system_id: {
                    'integration_pattern': system.integration_pattern.value,
                    'experts': list(system.experts.keys())
                }
                for system_id, system in self.systems.items()
            },
            'statistics': self.get_framework_statistics(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"Framework configuration exported to {filepath}")
    
    def cleanup(self):
        """Cleanup framework resources"""
        self.executor.shutdown(wait=True)
        logger.info("Diadochi Framework cleanup complete") 
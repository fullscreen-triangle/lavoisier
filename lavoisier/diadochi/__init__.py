"""
Diadochi: Intelligent LLM Combination and Knowledge Distillation Framework

Named after Alexander the Great's successors who divided and ruled his empire,
this module implements sophisticated strategies for combining, routing, and 
distilling knowledge across multiple specialized LLMs.

Based on the "Combine Harvester" architectural patterns:
- Router-Based Ensembles: Direct queries to appropriate domain experts
- Sequential Chaining: Pass queries through multiple experts in sequence
- Mixture of Experts: Process queries through multiple experts in parallel
- Specialized System Prompts: Use single model with multi-domain expertise
- Knowledge Distillation: Train unified models from multiple domain experts
- Multi-Domain RAG: Retrieval-augmented generation with domain-specific knowledge

Components:
- Core: Base classes and interfaces for model combination
- Routers: Intelligent query routing to appropriate domain experts
- Chains: Sequential processing through multiple domain models
- Mixers: Intelligent combination of responses from multiple experts
- Distillers: Knowledge distillation from multiple experts to unified models
- Registry: Management and orchestration of multiple LLM instances
- RAG: Multi-domain retrieval-augmented generation systems
"""

from .core import (
    DomainExpert,
    MultiDomainSystem,
    DiadochiFramework
)

from .routers import (
    BaseRouter,
    KeywordRouter, 
    EmbeddingRouter,
    ClassifierRouter,
    LLMRouter,
    HybridRouter
)

from .chains import (
    SequentialChain,
    SummarizingChain,
    HierarchicalChain,
    AdaptiveChain
)

from .mixers import (
    BaseMixer,
    WeightedMixer,
    SynthesisMixer,
    ConsensusMixer,
    HierarchicalMixer
)

from .distillers import (
    KnowledgeDistiller,
    MultiDomainDistiller,
    AdaptiveDistiller
)

from .registry import (
    ModelRegistry,
    DomainRegistry,
    ExpertOrchestrator
)

from .rag import (
    MultiDomainRAG,
    DomainSpecificRAG,
    HybridRAG
)

from .utilities import (
    ConfidenceEstimator,
    DomainClassifier,
    ResponseEvaluator
)

__all__ = [
    # Core framework
    'DomainExpert',
    'MultiDomainSystem', 
    'DiadochiFramework',
    
    # Routers
    'BaseRouter',
    'KeywordRouter',
    'EmbeddingRouter', 
    'ClassifierRouter',
    'LLMRouter',
    'HybridRouter',
    
    # Chains
    'SequentialChain',
    'SummarizingChain',
    'HierarchicalChain',
    'AdaptiveChain',
    
    # Mixers
    'BaseMixer',
    'WeightedMixer',
    'SynthesisMixer',
    'ConsensusMixer',
    'HierarchicalMixer',
    
    # Distillers
    'KnowledgeDistiller',
    'MultiDomainDistiller',
    'AdaptiveDistiller',
    
    # Registry and Management
    'ModelRegistry',
    'DomainRegistry',
    'ExpertOrchestrator',
    
    # RAG Systems
    'MultiDomainRAG',
    'DomainSpecificRAG',
    'HybridRAG',
    
    # Utilities
    'ConfidenceEstimator',
    'DomainClassifier',
    'ResponseEvaluator'
] 
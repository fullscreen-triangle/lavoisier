"""
Routers for Diadochi: Intelligent Query Routing to Domain Experts

This module implements various routing strategies for directing queries
to the most appropriate domain experts based on query characteristics.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

from .core import DomainExpert, DomainType, QueryContext

logger = logging.getLogger(__name__)

@dataclass
class RoutingDecision:
    """Decision made by a router"""
    selected_expert: str
    confidence: float
    reasoning: str
    alternatives: List[Tuple[str, float]]
    processing_time: float
    metadata: Dict[str, Any]

class BaseRouter(ABC):
    """Abstract base class for all routing strategies"""
    
    def __init__(self, 
                 router_id: str,
                 confidence_threshold: float = 0.5):
        self.router_id = router_id
        self.confidence_threshold = confidence_threshold
        self.routing_history: List[Dict[str, Any]] = []
        
    @abstractmethod
    def route(self, 
              query: str, 
              available_experts: List[DomainExpert],
              context: Optional[QueryContext] = None) -> RoutingDecision:
        """Route a query to the most appropriate expert"""
        pass
    
    @abstractmethod
    def route_multiple(self, 
                      query: str,
                      available_experts: List[DomainExpert], 
                      k: int = 3,
                      context: Optional[QueryContext] = None) -> List[RoutingDecision]:
        """Route a query to the k most appropriate experts"""
        pass
    
    def log_routing_decision(self, 
                           query: str,
                           decision: RoutingDecision,
                           feedback_score: Optional[float] = None):
        """Log routing decision for analysis"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'selected_expert': decision.selected_expert,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning,
            'alternatives': decision.alternatives,
            'processing_time': decision.processing_time,
            'feedback_score': feedback_score
        }
        self.routing_history.append(log_entry)
        
        # Keep only last 1000 decisions
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing performance statistics"""
        if not self.routing_history:
            return {'total_routings': 0, 'avg_confidence': 0.0, 'avg_processing_time': 0.0}
        
        recent_history = self.routing_history[-100:]  # Last 100 decisions
        
        avg_confidence = np.mean([h['confidence'] for h in recent_history])
        avg_processing_time = np.mean([h['processing_time'] for h in recent_history])
        
        # Success rate based on feedback scores
        feedback_scores = [h['feedback_score'] for h in recent_history if h['feedback_score'] is not None]
        success_rate = np.mean([s > 0.6 for s in feedback_scores]) if feedback_scores else 0.5
        
        # Expert selection distribution
        expert_counts = {}
        for h in recent_history:
            expert_id = h['selected_expert']
            expert_counts[expert_id] = expert_counts.get(expert_id, 0) + 1
        
        return {
            'total_routings': len(self.routing_history),
            'avg_confidence': float(avg_confidence),
            'avg_processing_time': float(avg_processing_time),
            'success_rate': float(success_rate),
            'expert_selection_distribution': expert_counts
        }

class KeywordRouter(BaseRouter):
    """Routes queries based on keyword matching"""
    
    def __init__(self, 
                 router_id: str = "keyword_router",
                 confidence_threshold: float = 0.5):
        super().__init__(router_id, confidence_threshold)
        
        # Domain-specific keywords
        self.domain_keywords = {
            DomainType.MASS_SPECTROMETRY: [
                'mass spec', 'ms', 'maldi', 'esi', 'qtof', 'orbitrap', 'ion', 'fragmentation',
                'precursor', 'product ion', 'collision energy', 'ionization', 'mass accuracy',
                'resolution', 'scan', 'spectrum', 'peak', 'm/z', 'mass-to-charge'
            ],
            DomainType.METABOLOMICS: [
                'metabolite', 'metabolome', 'biochemical pathway', 'small molecule',
                'biomarker', 'metabolic', 'flux', 'annotation', 'identification',
                'library', 'database', 'hmdb', 'kegg', 'lipid', 'amino acid'
            ],
            DomainType.PROTEOMICS: [
                'protein', 'peptide', 'proteome', 'trypsin', 'digestion', 'modification',
                'ptm', 'phosphorylation', 'acetylation', 'ubiquitin', 'sequence',
                'coverage', 'fdr', 'mascot', 'sequest', 'uniprot'
            ],
            DomainType.BIOINFORMATICS: [
                'algorithm', 'pipeline', 'workflow', 'software', 'tool', 'database',
                'analysis', 'bioinformatics', 'computational', 'script', 'python',
                'r', 'statistics', 'machine learning', 'data processing'
            ],
            DomainType.STATISTICAL_ANALYSIS: [
                'statistics', 'statistical', 'significance', 'p-value', 'correlation',
                'regression', 'anova', 'test', 'distribution', 'variance', 'mean',
                'median', 'confidence interval', 'hypothesis', 'normality'
            ],
            DomainType.DATA_VISUALIZATION: [
                'plot', 'graph', 'chart', 'visualization', 'figure', 'heatmap',
                'scatter plot', 'bar chart', 'histogram', 'pca plot', 'volcano plot',
                'boxplot', 'violin plot', 'ggplot', 'matplotlib', 'plotly'
            ],
            DomainType.MACHINE_LEARNING: [
                'machine learning', 'ml', 'classification', 'regression', 'clustering',
                'feature selection', 'cross validation', 'model', 'training', 'prediction',
                'accuracy', 'precision', 'recall', 'f1-score', 'random forest', 'svm'
            ],
            DomainType.CHEMISTRY: [
                'chemical', 'compound', 'molecule', 'formula', 'structure', 'bond',
                'reaction', 'synthesis', 'purity', 'concentration', 'solvent',
                'ph', 'buffer', 'chromatography', 'separation', 'extraction'
            ],
            DomainType.BIOLOGY: [
                'biological', 'cell', 'tissue', 'organism', 'gene', 'dna', 'rna',
                'enzyme', 'pathway', 'regulation', 'expression', 'phenotype',
                'genotype', 'mutation', 'evolution', 'ecology'
            ]
        }
        
        # Compile keyword patterns for efficiency
        self.keyword_patterns = {}
        for domain_type, keywords in self.domain_keywords.items():
            pattern = '|'.join(re.escape(keyword.lower()) for keyword in keywords)
            self.keyword_patterns[domain_type] = re.compile(pattern, re.IGNORECASE)
    
    def route(self, 
              query: str, 
              available_experts: List[DomainExpert],
              context: Optional[QueryContext] = None) -> RoutingDecision:
        """Route query based on keyword matching"""
        start_time = datetime.now()
        
        query_lower = query.lower()
        domain_scores = {}
        
        # Calculate keyword match scores for each domain
        for domain_type, pattern in self.keyword_patterns.items():
            matches = pattern.findall(query_lower)
            score = len(matches) / max(1, len(query.split()))  # Normalize by query length
            domain_scores[domain_type] = score
        
        # Map domain scores to available experts
        expert_scores = []
        for expert in available_experts:
            domain_score = domain_scores.get(expert.domain_spec.domain_type, 0.0)
            expert_scores.append((expert.expert_id, domain_score))
        
        # Sort by score and select best
        expert_scores.sort(key=lambda x: x[1], reverse=True)
        
        if not expert_scores or expert_scores[0][1] < self.confidence_threshold:
            # No expert meets threshold, select general expert or highest scoring
            selected_expert = expert_scores[0][0] if expert_scores else "none"
            confidence = expert_scores[0][1] if expert_scores else 0.0
            reasoning = "No expert meets confidence threshold, selected highest scoring"
        else:
            selected_expert = expert_scores[0][0]
            confidence = expert_scores[0][1]
            reasoning = f"Keyword matching identified {expert_scores[0][1]:.2f} relevance score"
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        decision = RoutingDecision(
            selected_expert=selected_expert,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=expert_scores[1:6],  # Top 5 alternatives
            processing_time=processing_time,
            metadata={'domain_scores': domain_scores}
        )
        
        self.log_routing_decision(query, decision)
        return decision
    
    def route_multiple(self, 
                      query: str,
                      available_experts: List[DomainExpert], 
                      k: int = 3,
                      context: Optional[QueryContext] = None) -> List[RoutingDecision]:
        """Route to multiple experts based on keyword matching"""
        primary_decision = self.route(query, available_experts, context)
        
        decisions = [primary_decision]
        
        # Add alternative experts as additional decisions
        for expert_id, score in primary_decision.alternatives[:k-1]:
            if score >= self.confidence_threshold:
                alt_decision = RoutingDecision(
                    selected_expert=expert_id,
                    confidence=score,
                    reasoning=f"Alternative expert with keyword relevance {score:.2f}",
                    alternatives=[],
                    processing_time=primary_decision.processing_time,
                    metadata=primary_decision.metadata
                )
                decisions.append(alt_decision)
        
        return decisions

class EmbeddingRouter(BaseRouter):
    """Routes queries based on semantic similarity using embeddings"""
    
    def __init__(self, 
                 router_id: str = "embedding_router",
                 confidence_threshold: float = 0.6,
                 embedding_model: Optional[Any] = None):
        super().__init__(router_id, confidence_threshold)
        self.embedding_model = embedding_model
        self.domain_embeddings: Dict[str, np.ndarray] = {}
        self.expert_descriptions: Dict[str, str] = {}
        
        # Initialize with default domain descriptions
        self._initialize_domain_descriptions()
    
    def _initialize_domain_descriptions(self):
        """Initialize domain descriptions for embedding"""
        self.domain_descriptions = {
            DomainType.MASS_SPECTROMETRY: """
            Mass spectrometry analysis including instrument operation, method development,
            data acquisition, ion fragmentation patterns, mass accuracy, resolution,
            ionization techniques, and spectral interpretation.
            """,
            DomainType.METABOLOMICS: """
            Metabolomics data analysis including metabolite identification, annotation,
            pathway analysis, biomarker discovery, small molecule characterization,
            and metabolic flux analysis.
            """,
            DomainType.PROTEOMICS: """
            Proteomics analysis including protein identification, quantification,
            post-translational modifications, protein-protein interactions,
            and functional annotation.
            """,
            DomainType.BIOINFORMATICS: """
            Bioinformatics software development, algorithm implementation,
            pipeline creation, workflow optimization, and computational analysis tools.
            """,
            DomainType.STATISTICAL_ANALYSIS: """
            Statistical analysis including hypothesis testing, regression analysis,
            ANOVA, correlation analysis, significance testing, and statistical modeling.
            """,
            DomainType.DATA_VISUALIZATION: """
            Data visualization and plotting including charts, graphs, heatmaps,
            statistical plots, interactive visualizations, and scientific figures.
            """,
            DomainType.MACHINE_LEARNING: """
            Machine learning applications including classification, regression,
            clustering, feature selection, model validation, and predictive modeling.
            """,
            DomainType.CHEMISTRY: """
            Chemical analysis including compound identification, structure elucidation,
            chemical properties, reactions, and analytical chemistry methods.
            """,
            DomainType.BIOLOGY: """
            Biological processes including cellular mechanisms, molecular biology,
            biochemical pathways, gene regulation, and biological systems analysis.
            """
        }
    
    def add_expert_description(self, expert_id: str, description: str):
        """Add custom description for an expert"""
        self.expert_descriptions[expert_id] = description
        
        # Update embeddings if model is available
        if self.embedding_model:
            self.domain_embeddings[expert_id] = self._get_embedding(description)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        if self.embedding_model:
            try:
                return self.embedding_model.encode([text])[0]
            except Exception as e:
                logger.error(f"Error getting embedding: {str(e)}")
                # Fallback to simple TF-IDF
                return self._get_tfidf_embedding(text)
        else:
            return self._get_tfidf_embedding(text)
    
    def _get_tfidf_embedding(self, text: str) -> np.ndarray:
        """Fallback TF-IDF embedding"""
        if not hasattr(self, 'tfidf_vectorizer'):
            # Initialize TF-IDF vectorizer with domain descriptions
            all_descriptions = list(self.domain_descriptions.values())
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.tfidf_vectorizer.fit(all_descriptions)
        
        return self.tfidf_vectorizer.transform([text]).toarray()[0]
    
    def route(self, 
              query: str, 
              available_experts: List[DomainExpert],
              context: Optional[QueryContext] = None) -> RoutingDecision:
        """Route query based on embedding similarity"""
        start_time = datetime.now()
        
        query_embedding = self._get_embedding(query)
        expert_similarities = []
        
        for expert in available_experts:
            # Get or create domain embedding
            if expert.expert_id in self.domain_embeddings:
                domain_embedding = self.domain_embeddings[expert.expert_id]
            else:
                domain_desc = self.domain_descriptions.get(
                    expert.domain_spec.domain_type, 
                    expert.domain_spec.description
                )
                domain_embedding = self._get_embedding(domain_desc)
                self.domain_embeddings[expert.expert_id] = domain_embedding
            
            # Calculate similarity
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                domain_embedding.reshape(1, -1)
            )[0][0]
            
            expert_similarities.append((expert.expert_id, similarity))
        
        # Sort by similarity
        expert_similarities.sort(key=lambda x: x[1], reverse=True)
        
        if not expert_similarities or expert_similarities[0][1] < self.confidence_threshold:
            selected_expert = expert_similarities[0][0] if expert_similarities else "none"
            confidence = expert_similarities[0][1] if expert_similarities else 0.0
            reasoning = "No expert meets embedding similarity threshold"
        else:
            selected_expert = expert_similarities[0][0]
            confidence = expert_similarities[0][1]
            reasoning = f"Embedding similarity score: {confidence:.3f}"
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        decision = RoutingDecision(
            selected_expert=selected_expert,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=expert_similarities[1:6],
            processing_time=processing_time,
            metadata={'similarity_scores': dict(expert_similarities)}
        )
        
        self.log_routing_decision(query, decision)
        return decision
    
    def route_multiple(self, 
                      query: str,
                      available_experts: List[DomainExpert], 
                      k: int = 3,
                      context: Optional[QueryContext] = None) -> List[RoutingDecision]:
        """Route to multiple experts based on embedding similarity"""
        primary_decision = self.route(query, available_experts, context)
        
        decisions = [primary_decision]
        
        for expert_id, similarity in primary_decision.alternatives[:k-1]:
            if similarity >= self.confidence_threshold:
                alt_decision = RoutingDecision(
                    selected_expert=expert_id,
                    confidence=similarity,
                    reasoning=f"Alternative expert with similarity {similarity:.3f}",
                    alternatives=[],
                    processing_time=primary_decision.processing_time,
                    metadata=primary_decision.metadata
                )
                decisions.append(alt_decision)
        
        return decisions

class HybridRouter(BaseRouter):
    """Combines multiple routing strategies for improved accuracy"""
    
    def __init__(self, 
                 router_id: str = "hybrid_router",
                 confidence_threshold: float = 0.6,
                 routers: Optional[List[BaseRouter]] = None,
                 router_weights: Optional[Dict[str, float]] = None):
        super().__init__(router_id, confidence_threshold)
        
        # Initialize component routers
        if routers is None:
            self.routers = [
                KeywordRouter("hybrid_keyword"),
                EmbeddingRouter("hybrid_embedding")
            ]
        else:
            self.routers = routers
        
        # Router weights for weighted voting
        if router_weights is None:
            self.router_weights = {router.router_id: 1.0 for router in self.routers}
        else:
            self.router_weights = router_weights
    
    def route(self, 
              query: str, 
              available_experts: List[DomainExpert],
              context: Optional[QueryContext] = None) -> RoutingDecision:
        """Route using hybrid approach combining multiple routers"""
        start_time = datetime.now()
        
        # Get decisions from all component routers
        router_decisions = {}
        for router in self.routers:
            try:
                decision = router.route(query, available_experts, context)
                router_decisions[router.router_id] = decision
            except Exception as e:
                logger.error(f"Error in {router.router_id}: {str(e)}")
        
        if not router_decisions:
            return self._fallback_route(query, available_experts)
        
        # Combine decisions using weighted voting
        expert_vote_scores = {}
        
        for router_id, decision in router_decisions.items():
            weight = self.router_weights.get(router_id, 1.0)
            
            # Add weighted vote for selected expert
            expert_id = decision.selected_expert
            if expert_id not in expert_vote_scores:
                expert_vote_scores[expert_id] = 0.0
            expert_vote_scores[expert_id] += decision.confidence * weight
            
            # Add weighted votes for alternatives
            for alt_expert, alt_confidence in decision.alternatives:
                if alt_expert not in expert_vote_scores:
                    expert_vote_scores[alt_expert] = 0.0
                expert_vote_scores[alt_expert] += alt_confidence * weight * 0.5  # Reduced weight for alternatives
        
        # Normalize scores
        total_weight = sum(self.router_weights.values())
        for expert_id in expert_vote_scores:
            expert_vote_scores[expert_id] /= total_weight
        
        # Select expert with highest combined score
        if expert_vote_scores:
            sorted_experts = sorted(expert_vote_scores.items(), key=lambda x: x[1], reverse=True)
            selected_expert = sorted_experts[0][0]
            confidence = sorted_experts[0][1]
            
            # Create reasoning summary
            voter_info = []
            for router_id, decision in router_decisions.items():
                if decision.selected_expert == selected_expert:
                    voter_info.append(f"{router_id}({decision.confidence:.2f})")
            
            reasoning = f"Hybrid consensus from {', '.join(voter_info)}"
            alternatives = sorted_experts[1:6]
        else:
            selected_expert = "none"
            confidence = 0.0
            reasoning = "No consensus reached"
            alternatives = []
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        decision = RoutingDecision(
            selected_expert=selected_expert,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=alternatives,
            processing_time=processing_time,
            metadata={
                'router_decisions': {rid: dec.__dict__ for rid, dec in router_decisions.items()},
                'vote_scores': expert_vote_scores
            }
        )
        
        self.log_routing_decision(query, decision)
        return decision
    
    def _fallback_route(self, 
                       query: str,
                       available_experts: List[DomainExpert]) -> RoutingDecision:
        """Fallback when all routers fail"""
        if available_experts:
            selected_expert = available_experts[0].expert_id
            confidence = 0.2
            reasoning = "Fallback routing - all routers failed"
        else:
            selected_expert = "none"
            confidence = 0.0
            reasoning = "No experts available"
        
        return RoutingDecision(
            selected_expert=selected_expert,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=[],
            processing_time=0.001,
            metadata={'fallback': True}
        )
    
    def route_multiple(self, 
                      query: str,
                      available_experts: List[DomainExpert], 
                      k: int = 3,
                      context: Optional[QueryContext] = None) -> List[RoutingDecision]:
        """Route to multiple experts using hybrid approach"""
        primary_decision = self.route(query, available_experts, context)
        
        decisions = [primary_decision]
        
        for expert_id, score in primary_decision.alternatives[:k-1]:
            if score >= self.confidence_threshold:
                alt_decision = RoutingDecision(
                    selected_expert=expert_id,
                    confidence=score,
                    reasoning=f"Hybrid alternative with score {score:.3f}",
                    alternatives=[],
                    processing_time=primary_decision.processing_time,
                    metadata=primary_decision.metadata
                )
                decisions.append(alt_decision)
        
        return decisions
    
    def update_router_weights(self, feedback_data: List[Dict[str, Any]]):
        """Update router weights based on performance feedback"""
        router_performance = {router.router_id: [] for router in self.routers}
        
        # Collect performance data
        for feedback in feedback_data:
            query = feedback.get('query', '')
            correct_expert = feedback.get('correct_expert', '')
            
            for router in self.routers:
                try:
                    decision = router.route(query, [])  # Placeholder call
                    accuracy = 1.0 if decision.selected_expert == correct_expert else 0.0
                    router_performance[router.router_id].append(accuracy)
                except:
                    continue
        
        # Update weights based on accuracy
        total_accuracy = 0.0
        router_accuracies = {}
        
        for router_id, accuracies in router_performance.items():
            if accuracies:
                avg_accuracy = np.mean(accuracies)
                router_accuracies[router_id] = avg_accuracy
                total_accuracy += avg_accuracy
        
        # Normalize weights
        if total_accuracy > 0:
            for router_id in self.router_weights:
                if router_id in router_accuracies:
                    self.router_weights[router_id] = router_accuracies[router_id] / total_accuracy
        
        logger.info(f"Updated router weights: {self.router_weights}") 
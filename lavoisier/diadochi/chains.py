"""
Chains for Diadochi: Sequential Processing Through Multiple Domain Experts

This module implements various chaining strategies for processing queries
through multiple domain experts in sequence, with each expert building
on the insights provided by previous experts.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import json

from .core import DomainExpert, MultiDomainSystem, IntegrationPattern, QueryContext, ExpertResponse, IntegratedResponse

logger = logging.getLogger(__name__)

class SequentialChain(MultiDomainSystem):
    """
    Sequential chain that passes queries through multiple domain experts in sequence,
    with each expert building on the insights provided by previous experts.
    """
    
    def __init__(self, 
                 system_id: str,
                 experts: List[DomainExpert],
                 prompt_templates: Optional[Dict[str, str]] = None,
                 max_context_length: int = 8000):
        super().__init__(system_id, IntegrationPattern.SEQUENTIAL_CHAIN, experts)
        
        self.prompt_templates = prompt_templates or {}
        self.max_context_length = max_context_length
        
        # Default prompt templates
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize default prompt templates for different positions in chain"""
        
        if "first_expert" not in self.prompt_templates:
            self.prompt_templates["first_expert"] = """
            You are a domain expert in {domain}. Analyze the following query from your area of expertise:
            
            Query: {query}
            
            Provide a detailed analysis focusing on:
            1. Key aspects relevant to your domain
            2. Important considerations and factors
            3. Potential approaches or solutions
            4. Areas where other domain expertise might be needed
            
            Be specific and thorough in your analysis.
            """
        
        if "middle_expert" not in self.prompt_templates:
            self.prompt_templates["middle_expert"] = """
            You are a domain expert in {domain}. You have received analysis from a previous expert:
            
            Previous Analysis:
            {previous_response}
            
            Original Query: {query}
            
            Based on the previous analysis and your expertise in {domain}:
            1. Identify aspects not fully addressed by the previous expert
            2. Add insights specific to your domain
            3. Build upon or refine the previous analysis
            4. Address any gaps or limitations you identify
            
            Integrate your expertise with the previous analysis to provide a comprehensive perspective.
            """
        
        if "final_expert" not in self.prompt_templates:
            self.prompt_templates["final_expert"] = """
            You are responsible for synthesizing analyses from multiple domain experts.
            
            Original Query: {query}
            
            Expert Analyses:
            {all_responses}
            
            Your task:
            1. Synthesize insights from all expert analyses
            2. Resolve any contradictions or conflicts
            3. Provide a comprehensive, integrated response
            4. Ensure all relevant aspects are covered
            5. Make clear recommendations or conclusions
            
            Create a cohesive response that addresses the original query using insights from all domains.
            """
    
    async def process_query(self, 
                           query: str,
                           context: Optional[QueryContext] = None) -> IntegratedResponse:
        """Process query through sequential chain of experts"""
        start_time = datetime.now()
        
        if context is None:
            context = QueryContext(query=query)
        
        expert_responses: List[ExpertResponse] = []
        expert_list = list(self.experts.values())
        
        if not expert_list:
            return self._create_empty_response(query, start_time)
        
        try:
            # Process through each expert sequentially
            for i, expert in enumerate(expert_list):
                if not expert.is_available:
                    continue
                
                # Create prompt based on position in chain
                prompt = self._create_expert_prompt(
                    expert=expert,
                    query=query,
                    position=i,
                    previous_responses=expert_responses,
                    total_experts=len(expert_list)
                )
                
                # Get response from expert
                expert_response = await expert.generate_response(prompt, context)
                expert_responses.append(expert_response)
                
                # Log progress
                logger.info(f"Chain {self.system_id}: Expert {i+1}/{len(expert_list)} completed")
        
        except Exception as e:
            logger.error(f"Error in sequential chain processing: {str(e)}")
            return self._create_error_response(query, start_time, str(e))
        
        # Create final integrated response
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if expert_responses:
            final_response = expert_responses[-1].response
            overall_confidence = sum(r.confidence for r in expert_responses) / len(expert_responses)
            quality_score = self._calculate_quality_score(expert_responses)
        else:
            final_response = "No expert responses available"
            overall_confidence = 0.0
            quality_score = 0.0
        
        integrated_response = IntegratedResponse(
            response=final_response,
            expert_responses=expert_responses,
            integration_pattern=IntegrationPattern.SEQUENTIAL_CHAIN,
            overall_confidence=overall_confidence,
            processing_time=processing_time,
            quality_score=quality_score,
            metadata={
                'chain_length': len(expert_responses),
                'experts_used': [r.expert_id for r in expert_responses],
                'total_processing_time': sum(r.processing_time for r in expert_responses)
            }
        )
        
        self.log_query(query, context, integrated_response)
        return integrated_response
    
    def _create_expert_prompt(self,
                             expert: DomainExpert,
                             query: str,
                             position: int,
                             previous_responses: List[ExpertResponse],
                             total_experts: int) -> str:
        """Create appropriate prompt for expert based on their position in the chain"""
        
        domain_name = expert.domain_spec.domain_type.value
        
        if position == 0:
            # First expert
            template = self.prompt_templates.get("first_expert", self.prompt_templates["first_expert"])
            return template.format(
                domain=domain_name,
                query=query
            )
        
        elif position == total_experts - 1:
            # Final expert (synthesizer)
            all_responses = ""
            for i, response in enumerate(previous_responses):
                all_responses += f"\nExpert {i+1} ({response.domain_type.value}):\n{response.response}\n"
            
            template = self.prompt_templates.get("final_expert", self.prompt_templates["final_expert"])
            return template.format(
                query=query,
                all_responses=all_responses
            )
        
        else:
            # Middle expert
            previous_response = previous_responses[-1].response if previous_responses else ""
            
            template = self.prompt_templates.get("middle_expert", self.prompt_templates["middle_expert"])
            return template.format(
                domain=domain_name,
                query=query,
                previous_response=previous_response
            )
    
    def _calculate_quality_score(self, expert_responses: List[ExpertResponse]) -> float:
        """Calculate overall quality score for the chain"""
        if not expert_responses:
            return 0.0
        
        # Factors contributing to quality
        avg_confidence = sum(r.confidence for r in expert_responses) / len(expert_responses)
        
        # Response length diversity (good if experts provide different perspectives)
        response_lengths = [len(r.response) for r in expert_responses]
        length_diversity = (max(response_lengths) - min(response_lengths)) / max(max(response_lengths), 1)
        length_diversity = min(length_diversity, 1.0)  # Cap at 1.0
        
        # Processing time efficiency
        total_time = sum(r.processing_time for r in expert_responses)
        time_efficiency = max(0.0, 1.0 - (total_time / 300.0))  # Penalty for taking > 5 minutes
        
        # Combine factors
        quality_score = (
            0.6 * avg_confidence +
            0.2 * length_diversity +
            0.2 * time_efficiency
        )
        
        return min(quality_score, 1.0)
    
    def _create_empty_response(self, query: str, start_time: datetime) -> IntegratedResponse:
        """Create response when no experts are available"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return IntegratedResponse(
            response="No experts available for processing",
            expert_responses=[],
            integration_pattern=IntegrationPattern.SEQUENTIAL_CHAIN,
            overall_confidence=0.0,
            processing_time=processing_time,
            quality_score=0.0,
            metadata={'error': 'no_experts_available'}
        )
    
    def _create_error_response(self, query: str, start_time: datetime, error_msg: str) -> IntegratedResponse:
        """Create response when processing fails"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return IntegratedResponse(
            response=f"Processing failed: {error_msg}",
            expert_responses=[],
            integration_pattern=IntegrationPattern.SEQUENTIAL_CHAIN,
            overall_confidence=0.0,
            processing_time=processing_time,
            quality_score=0.0,
            metadata={'error': error_msg}
        )

class SummarizingChain(SequentialChain):
    """
    Sequential chain that automatically summarizes intermediate responses
    when they exceed context length limits.
    """
    
    def __init__(self, 
                 system_id: str,
                 experts: List[DomainExpert],
                 prompt_templates: Optional[Dict[str, str]] = None,
                 max_context_length: int = 4000,
                 summarizer_expert: Optional[DomainExpert] = None):
        super().__init__(system_id, experts, prompt_templates, max_context_length)
        
        self.summarizer_expert = summarizer_expert
        
        # Add summarization prompt template
        if "summarize" not in self.prompt_templates:
            self.prompt_templates["summarize"] = """
            Summarize the following expert analysis while preserving key insights and important details:
            
            Original Analysis:
            {response}
            
            Create a concise summary that:
            1. Preserves all critical insights and conclusions
            2. Maintains technical accuracy
            3. Reduces length while keeping essential information
            4. Uses clear, professional language
            
            Summary:
            """
    
    async def process_query(self, 
                           query: str,
                           context: Optional[QueryContext] = None) -> IntegratedResponse:
        """Process query with automatic summarization when needed"""
        start_time = datetime.now()
        
        if context is None:
            context = QueryContext(query=query)
        
        expert_responses: List[ExpertResponse] = []
        expert_list = list(self.experts.values())
        
        if not expert_list:
            return self._create_empty_response(query, start_time)
        
        try:
            # Process through each expert sequentially
            for i, expert in enumerate(expert_list):
                if not expert.is_available:
                    continue
                
                # Check if summarization is needed
                if i > 0 and self._should_summarize(expert_responses):
                    await self._summarize_responses(expert_responses)
                
                # Create prompt based on position in chain
                prompt = self._create_expert_prompt(
                    expert=expert,
                    query=query,
                    position=i,
                    previous_responses=expert_responses,
                    total_experts=len(expert_list)
                )
                
                # Get response from expert
                expert_response = await expert.generate_response(prompt, context)
                expert_responses.append(expert_response)
                
                logger.info(f"Summarizing Chain {self.system_id}: Expert {i+1}/{len(expert_list)} completed")
        
        except Exception as e:
            logger.error(f"Error in summarizing chain processing: {str(e)}")
            return self._create_error_response(query, start_time, str(e))
        
        # Create final integrated response
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if expert_responses:
            final_response = expert_responses[-1].response
            overall_confidence = sum(r.confidence for r in expert_responses) / len(expert_responses)
            quality_score = self._calculate_quality_score(expert_responses)
        else:
            final_response = "No expert responses available"
            overall_confidence = 0.0
            quality_score = 0.0
        
        integrated_response = IntegratedResponse(
            response=final_response,
            expert_responses=expert_responses,
            integration_pattern=IntegrationPattern.SEQUENTIAL_CHAIN,
            overall_confidence=overall_confidence,
            processing_time=processing_time,
            quality_score=quality_score,
            metadata={
                'chain_type': 'summarizing',
                'chain_length': len(expert_responses),
                'experts_used': [r.expert_id for r in expert_responses],
                'total_processing_time': sum(r.processing_time for r in expert_responses),
                'summarizations_performed': sum(1 for r in expert_responses if r.metadata.get('summarized', False))
            }
        )
        
        self.log_query(query, context, integrated_response)
        return integrated_response
    
    def _should_summarize(self, responses: List[ExpertResponse]) -> bool:
        """Determine if responses should be summarized"""
        if not responses:
            return False
        
        # Calculate total context length
        total_length = sum(len(r.response) for r in responses)
        
        return total_length > self.max_context_length
    
    async def _summarize_responses(self, responses: List[ExpertResponse]):
        """Summarize responses to reduce context length"""
        if not responses:
            return
        
        # Find the best expert to use for summarization
        summarizer = self.summarizer_expert
        if not summarizer:
            # Use the expert with highest average confidence
            available_experts = [expert for expert in self.experts.values() if expert.is_available]
            if available_experts:
                summarizer = max(available_experts, key=lambda e: e.get_performance_metrics()['avg_confidence'])
        
        if not summarizer:
            return
        
        # Summarize each response that's too long
        for response in responses:
            if len(response.response) > self.max_context_length // 2:  # Summarize if > half max length
                try:
                    summary_prompt = self.prompt_templates["summarize"].format(
                        response=response.response
                    )
                    
                    # Get summary (this is a simplified version - in practice you'd call the summarizer)
                    summary_context = QueryContext(query=summary_prompt)
                    summary_response = await summarizer.generate_response(summary_prompt, summary_context)
                    
                    # Update the response with summary
                    response.response = summary_response.response
                    response.metadata['summarized'] = True
                    response.metadata['original_length'] = len(response.response)
                    
                    logger.info(f"Summarized response from {response.expert_id}")
                    
                except Exception as e:
                    logger.error(f"Error summarizing response from {response.expert_id}: {str(e)}")

class HierarchicalChain(SequentialChain):
    """
    Hierarchical chain that organizes experts into groups and processes
    within groups before combining across groups.
    """
    
    def __init__(self, 
                 system_id: str,
                 expert_groups: Dict[str, List[DomainExpert]],
                 prompt_templates: Optional[Dict[str, str]] = None,
                 group_synthesizer: Optional[DomainExpert] = None):
        
        # Flatten expert groups for parent initialization
        all_experts = []
        for group_experts in expert_groups.values():
            all_experts.extend(group_experts)
        
        super().__init__(system_id, all_experts, prompt_templates)
        
        self.expert_groups = expert_groups
        self.group_synthesizer = group_synthesizer
        
        # Add group-specific templates
        if "group_synthesis" not in self.prompt_templates:
            self.prompt_templates["group_synthesis"] = """
            Synthesize the following analyses from experts in the {group_name} domain group:
            
            Original Query: {query}
            
            Expert Analyses:
            {group_responses}
            
            Create a unified perspective from this domain group that:
            1. Integrates insights from all experts in the group
            2. Resolves any contradictions within the group
            3. Provides a coherent domain-specific analysis
            4. Identifies areas requiring input from other domain groups
            
            Group Synthesis:
            """
        
        if "final_hierarchical_synthesis" not in self.prompt_templates:
            self.prompt_templates["final_hierarchical_synthesis"] = """
            Synthesize analyses from multiple domain groups to answer the original query:
            
            Original Query: {query}
            
            Domain Group Analyses:
            {group_syntheses}
            
            Create a comprehensive, integrated response that:
            1. Combines insights from all domain groups
            2. Resolves any conflicts between domain perspectives
            3. Provides a holistic answer to the query
            4. Makes clear recommendations or conclusions
            
            Final Integrated Response:
            """
    
    async def process_query(self, 
                           query: str,
                           context: Optional[QueryContext] = None) -> IntegratedResponse:
        """Process query through hierarchical chain"""
        start_time = datetime.now()
        
        if context is None:
            context = QueryContext(query=query)
        
        all_expert_responses: List[ExpertResponse] = []
        group_syntheses: Dict[str, ExpertResponse] = {}
        
        try:
            # Process each group
            for group_name, group_experts in self.expert_groups.items():
                logger.info(f"Processing group: {group_name}")
                
                group_responses = []
                
                # Process experts within the group
                for expert in group_experts:
                    if not expert.is_available:
                        continue
                    
                    # Create expert prompt
                    prompt = self._create_expert_prompt(
                        expert=expert,
                        query=query,
                        position=0,  # Each group starts fresh
                        previous_responses=[],
                        total_experts=1
                    )
                    
                    # Get expert response
                    expert_response = await expert.generate_response(prompt, context)
                    group_responses.append(expert_response)
                    all_expert_responses.append(expert_response)
                
                # Synthesize group responses
                if group_responses:
                    group_synthesis = await self._synthesize_group(
                        group_name, group_responses, query, context
                    )
                    group_syntheses[group_name] = group_synthesis
                    all_expert_responses.append(group_synthesis)
            
            # Final synthesis across groups
            final_response = await self._synthesize_across_groups(
                group_syntheses, query, context
            )
            
            if final_response:
                all_expert_responses.append(final_response)
        
        except Exception as e:
            logger.error(f"Error in hierarchical chain processing: {str(e)}")
            return self._create_error_response(query, start_time, str(e))
        
        # Create integrated response
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if all_expert_responses:
            final_response_text = all_expert_responses[-1].response
            overall_confidence = sum(r.confidence for r in all_expert_responses) / len(all_expert_responses)
            quality_score = self._calculate_hierarchical_quality_score(all_expert_responses, group_syntheses)
        else:
            final_response_text = "No expert responses available"
            overall_confidence = 0.0
            quality_score = 0.0
        
        integrated_response = IntegratedResponse(
            response=final_response_text,
            expert_responses=all_expert_responses,
            integration_pattern=IntegrationPattern.SEQUENTIAL_CHAIN,
            overall_confidence=overall_confidence,
            processing_time=processing_time,
            quality_score=quality_score,
            metadata={
                'chain_type': 'hierarchical',
                'groups_processed': list(self.expert_groups.keys()),
                'group_syntheses_count': len(group_syntheses),
                'total_experts': len([r for r in all_expert_responses if 'synthesis' not in r.metadata]),
                'total_processing_time': sum(r.processing_time for r in all_expert_responses)
            }
        )
        
        self.log_query(query, context, integrated_response)
        return integrated_response
    
    async def _synthesize_group(self,
                               group_name: str,
                               group_responses: List[ExpertResponse],
                               query: str,
                               context: QueryContext) -> ExpertResponse:
        """Synthesize responses within a group"""
        if not group_responses:
            return None
        
        if len(group_responses) == 1:
            return group_responses[0]  # No synthesis needed
        
        # Use group synthesizer or best expert in group
        synthesizer = self.group_synthesizer
        if not synthesizer:
            # Use expert with highest confidence in this group
            synthesizer = max(group_responses, key=lambda r: r.confidence)
            synthesizer = self.experts.get(synthesizer.expert_id)
        
        if not synthesizer:
            return group_responses[-1]  # Fallback to last response
        
        # Create group synthesis prompt
        group_responses_text = ""
        for i, response in enumerate(group_responses):
            group_responses_text += f"\nExpert {i+1} ({response.expert_id}):\n{response.response}\n"
        
        synthesis_prompt = self.prompt_templates["group_synthesis"].format(
            group_name=group_name,
            query=query,
            group_responses=group_responses_text
        )
        
        # Generate synthesis
        synthesis_response = await synthesizer.generate_response(synthesis_prompt, context)
        synthesis_response.metadata['synthesis_type'] = 'group'
        synthesis_response.metadata['group_name'] = group_name
        synthesis_response.metadata['synthesized_experts'] = [r.expert_id for r in group_responses]
        
        return synthesis_response
    
    async def _synthesize_across_groups(self,
                                       group_syntheses: Dict[str, ExpertResponse],
                                       query: str,
                                       context: QueryContext) -> Optional[ExpertResponse]:
        """Synthesize responses across groups"""
        if not group_syntheses:
            return None
        
        if len(group_syntheses) == 1:
            return list(group_syntheses.values())[0]
        
        # Use designated synthesizer or best performing expert
        synthesizer = self.group_synthesizer
        if not synthesizer:
            # Find best overall expert
            all_experts = list(self.experts.values())
            if all_experts:
                synthesizer = max(all_experts, key=lambda e: e.get_performance_metrics()['avg_confidence'])
        
        if not synthesizer:
            return list(group_syntheses.values())[-1]  # Fallback
        
        # Create final synthesis prompt
        group_syntheses_text = ""
        for group_name, synthesis in group_syntheses.items():
            group_syntheses_text += f"\n{group_name} Group Analysis:\n{synthesis.response}\n"
        
        final_prompt = self.prompt_templates["final_hierarchical_synthesis"].format(
            query=query,
            group_syntheses=group_syntheses_text
        )
        
        # Generate final synthesis
        final_response = await synthesizer.generate_response(final_prompt, context)
        final_response.metadata['synthesis_type'] = 'final_hierarchical'
        final_response.metadata['groups_synthesized'] = list(group_syntheses.keys())
        
        return final_response
    
    def _calculate_hierarchical_quality_score(self,
                                            all_responses: List[ExpertResponse],
                                            group_syntheses: Dict[str, ExpertResponse]) -> float:
        """Calculate quality score for hierarchical processing"""
        if not all_responses:
            return 0.0
        
        # Base quality from parent method
        base_quality = self._calculate_quality_score(all_responses)
        
        # Hierarchical-specific factors
        group_coverage = len(group_syntheses) / max(len(self.expert_groups), 1)
        synthesis_quality = sum(s.confidence for s in group_syntheses.values()) / max(len(group_syntheses), 1)
        
        # Combine factors
        hierarchical_quality = (
            0.5 * base_quality +
            0.3 * group_coverage +
            0.2 * synthesis_quality
        )
        
        return min(hierarchical_quality, 1.0)

class AdaptiveChain(SequentialChain):
    """
    Adaptive chain that dynamically adjusts the sequence of experts
    based on query characteristics and intermediate results.
    """
    
    def __init__(self, 
                 system_id: str,
                 experts: List[DomainExpert],
                 prompt_templates: Optional[Dict[str, str]] = None,
                 adaptation_strategy: str = "confidence_based"):
        super().__init__(system_id, experts, prompt_templates)
        
        self.adaptation_strategy = adaptation_strategy
        self.adaptation_history: List[Dict[str, Any]] = []
    
    async def process_query(self, 
                           query: str,
                           context: Optional[QueryContext] = None) -> IntegratedResponse:
        """Process query with adaptive expert selection"""
        start_time = datetime.now()
        
        if context is None:
            context = QueryContext(query=query)
        
        expert_responses: List[ExpertResponse] = []
        expert_sequence = await self._determine_initial_sequence(query, context)
        
        try:
            for i, expert_id in enumerate(expert_sequence):
                expert = self.experts.get(expert_id)
                if not expert or not expert.is_available:
                    continue
                
                # Create prompt
                prompt = self._create_expert_prompt(
                    expert=expert,
                    query=query,
                    position=i,
                    previous_responses=expert_responses,
                    total_experts=len(expert_sequence)
                )
                
                # Get response
                expert_response = await expert.generate_response(prompt, context)
                expert_responses.append(expert_response)
                
                # Adapt sequence based on response
                if i < len(expert_sequence) - 1:  # Not the last expert
                    adaptation = await self._adapt_sequence(
                        query, expert_responses, expert_sequence[i+1:]
                    )
                    if adaptation:
                        expert_sequence = expert_sequence[:i+1] + adaptation
                        logger.info(f"Adapted sequence: {adaptation}")
        
        except Exception as e:
            logger.error(f"Error in adaptive chain processing: {str(e)}")
            return self._create_error_response(query, start_time, str(e))
        
        # Create integrated response
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if expert_responses:
            final_response = expert_responses[-1].response
            overall_confidence = sum(r.confidence for r in expert_responses) / len(expert_responses)
            quality_score = self._calculate_adaptive_quality_score(expert_responses)
        else:
            final_response = "No expert responses available"
            overall_confidence = 0.0
            quality_score = 0.0
        
        integrated_response = IntegratedResponse(
            response=final_response,
            expert_responses=expert_responses,
            integration_pattern=IntegrationPattern.SEQUENTIAL_CHAIN,
            overall_confidence=overall_confidence,
            processing_time=processing_time,
            quality_score=quality_score,
            metadata={
                'chain_type': 'adaptive',
                'adaptation_strategy': self.adaptation_strategy,
                'initial_sequence': list(self.experts.keys()),
                'final_sequence': [r.expert_id for r in expert_responses],
                'adaptations_made': len(expert_responses) - len(list(self.experts.keys()))
            }
        )
        
        # Log adaptation
        self.adaptation_history.append({
            'query': query,
            'initial_sequence': list(self.experts.keys()),
            'final_sequence': [r.expert_id for r in expert_responses],
            'overall_confidence': overall_confidence,
            'quality_score': quality_score,
            'timestamp': datetime.now().isoformat()
        })
        
        self.log_query(query, context, integrated_response)
        return integrated_response
    
    async def _determine_initial_sequence(self, 
                                         query: str, 
                                         context: QueryContext) -> List[str]:
        """Determine initial sequence of experts based on query"""
        available_experts = [expert for expert in self.experts.values() if expert.is_available]
        
        if self.adaptation_strategy == "confidence_based":
            # Order by confidence scores
            expert_confidences = []
            for expert in available_experts:
                confidence = expert.estimate_confidence(query)
                expert_confidences.append((expert.expert_id, confidence))
            
            # Sort by confidence, highest first
            expert_confidences.sort(key=lambda x: x[1], reverse=True)
            return [expert_id for expert_id, _ in expert_confidences]
        
        elif self.adaptation_strategy == "performance_based":
            # Order by historical performance
            expert_performance = []
            for expert in available_experts:
                performance = expert.get_performance_metrics()
                score = (performance['avg_confidence'] + performance['success_rate']) / 2
                expert_performance.append((expert.expert_id, score))
            
            expert_performance.sort(key=lambda x: x[1], reverse=True)
            return [expert_id for expert_id, _ in expert_performance]
        
        else:
            # Default: use original order
            return list(self.experts.keys())
    
    async def _adapt_sequence(self,
                             query: str,
                             expert_responses: List[ExpertResponse],
                             remaining_sequence: List[str]) -> Optional[List[str]]:
        """Adapt remaining sequence based on current responses"""
        if not expert_responses or not remaining_sequence:
            return None
        
        latest_response = expert_responses[-1]
        
        # Adaptation logic based on latest response confidence
        if latest_response.confidence < 0.4:
            # Low confidence - try to find better expert for remaining tasks
            return await self._find_better_experts(query, remaining_sequence)
        
        elif latest_response.confidence > 0.8:
            # High confidence - might be able to skip some experts
            return await self._optimize_remaining_sequence(remaining_sequence)
        
        else:
            # Medium confidence - continue as planned
            return None
    
    async def _find_better_experts(self, 
                                  query: str, 
                                  remaining_sequence: List[str]) -> List[str]:
        """Find better experts when current performance is low"""
        available_experts = [self.experts[eid] for eid in remaining_sequence if eid in self.experts]
        
        # Re-evaluate confidence scores
        expert_confidences = []
        for expert in available_experts:
            confidence = expert.estimate_confidence(query)
            expert_confidences.append((expert.expert_id, confidence))
        
        # Return reordered sequence
        expert_confidences.sort(key=lambda x: x[1], reverse=True)
        return [expert_id for expert_id, _ in expert_confidences]
    
    async def _optimize_remaining_sequence(self, remaining_sequence: List[str]) -> List[str]:
        """Optimize remaining sequence when current performance is high"""
        # Simple optimization: remove lower-performing experts if sequence is long
        if len(remaining_sequence) > 2:
            # Keep only top performers
            available_experts = [self.experts[eid] for eid in remaining_sequence if eid in self.experts]
            expert_performance = []
            
            for expert in available_experts:
                performance = expert.get_performance_metrics()
                score = (performance['avg_confidence'] + performance['success_rate']) / 2
                expert_performance.append((expert.expert_id, score))
            
            expert_performance.sort(key=lambda x: x[1], reverse=True)
            
            # Keep top 2 experts
            return [expert_id for expert_id, _ in expert_performance[:2]]
        
        return remaining_sequence
    
    def _calculate_adaptive_quality_score(self, expert_responses: List[ExpertResponse]) -> float:
        """Calculate quality score considering adaptations"""
        base_quality = self._calculate_quality_score(expert_responses)
        
        # Bonus for successful adaptations (improving confidence over time)
        if len(expert_responses) > 1:
            confidence_trend = 0.0
            for i in range(1, len(expert_responses)):
                if expert_responses[i].confidence > expert_responses[i-1].confidence:
                    confidence_trend += 0.1
            
            adaptive_bonus = min(confidence_trend, 0.2)  # Cap at 0.2
            return min(base_quality + adaptive_bonus, 1.0)
        
        return base_quality 
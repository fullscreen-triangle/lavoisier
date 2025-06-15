"""
Hatata: Markov Decision Process Verification Layer

This module implements a sophisticated MDP-based verification system that
provides stochastic validation of the evidence network through goal-oriented
state transitions and utility function optimization.
"""

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import multivariate_normal
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import json
from collections import defaultdict, deque
import networkx as nx

logger = logging.getLogger(__name__)

class MDPState(Enum):
    """States in the MDP verification system"""
    EVIDENCE_COLLECTION = "evidence_collection"
    BAYESIAN_INFERENCE = "bayesian_inference"
    FUZZY_EVALUATION = "fuzzy_evaluation"
    NETWORK_ANALYSIS = "network_analysis"
    ANNOTATION_GENERATION = "annotation_generation"
    CONTEXT_VERIFICATION = "context_verification"
    VALIDATION_COMPLETE = "validation_complete"
    ERROR_DETECTED = "error_detected"
    RECOVERY_MODE = "recovery_mode"

class MDPAction(Enum):
    """Actions available in each MDP state"""
    COLLECT_MORE_EVIDENCE = "collect_more_evidence"
    UPDATE_PROBABILITIES = "update_probabilities"
    REFINE_FUZZY_LOGIC = "refine_fuzzy_logic"
    STRENGTHEN_CONNECTIONS = "strengthen_connections"
    PRUNE_WEAK_CONNECTIONS = "prune_weak_connections"
    GENERATE_ANNOTATIONS = "generate_annotations"
    VERIFY_CONTEXT = "verify_context"
    VALIDATE_RESULTS = "validate_results"
    CORRECT_ERRORS = "correct_errors"
    RESTART_ANALYSIS = "restart_analysis"
    ACCEPT_RESULTS = "accept_results"

@dataclass
class MDPTransition:
    """Represents a state transition in the MDP"""
    from_state: MDPState
    action: MDPAction
    to_state: MDPState
    probability: float
    reward: float
    utility: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UtilityFunction:
    """Utility function for MDP decision making"""
    name: str
    weight: float
    function: Callable[[Dict[str, Any]], float]
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

@dataclass
class MDPPolicy:
    """Policy for action selection in MDP states"""
    state_action_probabilities: Dict[MDPState, Dict[MDPAction, float]]
    expected_rewards: Dict[Tuple[MDPState, MDPAction], float]
    value_function: Dict[MDPState, float]
    policy_iteration: int = 0

class HatataMDPVerifier:
    """
    Advanced Markov Decision Process verification system that provides
    stochastic validation of evidence networks through goal-oriented
    state transitions and utility optimization.
    
    This system models the verification process as an MDP where:
    - States represent different stages of analysis
    - Actions represent verification decisions
    - Rewards are based on analysis quality and reliability
    - Utility functions optimize for multiple objectives
    """
    
    def __init__(self,
                 discount_factor: float = 0.95,
                 convergence_threshold: float = 1e-6,
                 max_iterations: int = 1000,
                 exploration_rate: float = 0.1):
        self.discount_factor = discount_factor
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.exploration_rate = exploration_rate
        
        # MDP components
        self.states = list(MDPState)
        self.actions = list(MDPAction)
        self.transition_model: Dict[Tuple[MDPState, MDPAction], Dict[MDPState, float]] = {}
        self.reward_model: Dict[Tuple[MDPState, MDPAction, MDPState], float] = {}
        self.policy: Optional[MDPPolicy] = None
        
        # Utility functions
        self.utility_functions: List[UtilityFunction] = []
        
        # State tracking
        self.current_state = MDPState.EVIDENCE_COLLECTION
        self.state_history: List[Tuple[MDPState, MDPAction, float, datetime]] = []
        self.cumulative_reward = 0.0
        
        # Analysis context
        self.analysis_context: Dict[str, Any] = {}
        self.validation_metrics: Dict[str, float] = {}
        
        self._initialize_mdp_model()
        self._initialize_utility_functions()
        
        logger.info("Hatata MDP Verifier initialized with stochastic validation framework")
    
    def _initialize_mdp_model(self):
        """
        Initialize the MDP transition and reward models.
        """
        # Define transition probabilities for each state-action pair
        transitions = {
            # Evidence Collection State
            (MDPState.EVIDENCE_COLLECTION, MDPAction.COLLECT_MORE_EVIDENCE): {
                MDPState.EVIDENCE_COLLECTION: 0.7,
                MDPState.BAYESIAN_INFERENCE: 0.3
            },
            (MDPState.EVIDENCE_COLLECTION, MDPAction.UPDATE_PROBABILITIES): {
                MDPState.BAYESIAN_INFERENCE: 0.9,
                MDPState.ERROR_DETECTED: 0.1
            },
            
            # Bayesian Inference State
            (MDPState.BAYESIAN_INFERENCE, MDPAction.UPDATE_PROBABILITIES): {
                MDPState.BAYESIAN_INFERENCE: 0.4,
                MDPState.FUZZY_EVALUATION: 0.5,
                MDPState.ERROR_DETECTED: 0.1
            },
            (MDPState.BAYESIAN_INFERENCE, MDPAction.REFINE_FUZZY_LOGIC): {
                MDPState.FUZZY_EVALUATION: 0.8,
                MDPState.NETWORK_ANALYSIS: 0.2
            },
            
            # Fuzzy Evaluation State
            (MDPState.FUZZY_EVALUATION, MDPAction.REFINE_FUZZY_LOGIC): {
                MDPState.FUZZY_EVALUATION: 0.3,
                MDPState.NETWORK_ANALYSIS: 0.6,
                MDPState.ERROR_DETECTED: 0.1
            },
            (MDPState.FUZZY_EVALUATION, MDPAction.STRENGTHEN_CONNECTIONS): {
                MDPState.NETWORK_ANALYSIS: 0.9,
                MDPState.ERROR_DETECTED: 0.1
            },
            
            # Network Analysis State
            (MDPState.NETWORK_ANALYSIS, MDPAction.STRENGTHEN_CONNECTIONS): {
                MDPState.NETWORK_ANALYSIS: 0.4,
                MDPState.ANNOTATION_GENERATION: 0.5,
                MDPState.ERROR_DETECTED: 0.1
            },
            (MDPState.NETWORK_ANALYSIS, MDPAction.PRUNE_WEAK_CONNECTIONS): {
                MDPState.NETWORK_ANALYSIS: 0.3,
                MDPState.ANNOTATION_GENERATION: 0.6,
                MDPState.ERROR_DETECTED: 0.1
            },
            (MDPState.NETWORK_ANALYSIS, MDPAction.GENERATE_ANNOTATIONS): {
                MDPState.ANNOTATION_GENERATION: 0.9,
                MDPState.ERROR_DETECTED: 0.1
            },
            
            # Annotation Generation State
            (MDPState.ANNOTATION_GENERATION, MDPAction.GENERATE_ANNOTATIONS): {
                MDPState.ANNOTATION_GENERATION: 0.3,
                MDPState.CONTEXT_VERIFICATION: 0.6,
                MDPState.ERROR_DETECTED: 0.1
            },
            (MDPState.ANNOTATION_GENERATION, MDPAction.VERIFY_CONTEXT): {
                MDPState.CONTEXT_VERIFICATION: 0.9,
                MDPState.ERROR_DETECTED: 0.1
            },
            
            # Context Verification State
            (MDPState.CONTEXT_VERIFICATION, MDPAction.VERIFY_CONTEXT): {
                MDPState.CONTEXT_VERIFICATION: 0.2,
                MDPState.VALIDATION_COMPLETE: 0.7,
                MDPState.ERROR_DETECTED: 0.1
            },
            (MDPState.CONTEXT_VERIFICATION, MDPAction.VALIDATE_RESULTS): {
                MDPState.VALIDATION_COMPLETE: 0.8,
                MDPState.ERROR_DETECTED: 0.2
            },
            
            # Error Detection State
            (MDPState.ERROR_DETECTED, MDPAction.CORRECT_ERRORS): {
                MDPState.RECOVERY_MODE: 0.8,
                MDPState.ERROR_DETECTED: 0.2
            },
            (MDPState.ERROR_DETECTED, MDPAction.RESTART_ANALYSIS): {
                MDPState.EVIDENCE_COLLECTION: 0.9,
                MDPState.ERROR_DETECTED: 0.1
            },
            
            # Recovery Mode State
            (MDPState.RECOVERY_MODE, MDPAction.CORRECT_ERRORS): {
                MDPState.BAYESIAN_INFERENCE: 0.6,
                MDPState.FUZZY_EVALUATION: 0.3,
                MDPState.ERROR_DETECTED: 0.1
            },
            (MDPState.RECOVERY_MODE, MDPAction.RESTART_ANALYSIS): {
                MDPState.EVIDENCE_COLLECTION: 0.9,
                MDPState.ERROR_DETECTED: 0.1
            },
            
            # Validation Complete State
            (MDPState.VALIDATION_COMPLETE, MDPAction.ACCEPT_RESULTS): {
                MDPState.VALIDATION_COMPLETE: 1.0
            },
            (MDPState.VALIDATION_COMPLETE, MDPAction.VALIDATE_RESULTS): {
                MDPState.VALIDATION_COMPLETE: 0.9,
                MDPState.ERROR_DETECTED: 0.1
            }
        }
        
        self.transition_model = transitions
        
        # Define reward model
        self._initialize_reward_model()
    
    def _initialize_reward_model(self):
        """
        Initialize the reward model for state-action-state transitions.
        """
        # Base rewards for different types of transitions
        base_rewards = {
            # Positive rewards for progress
            MDPState.EVIDENCE_COLLECTION: 1.0,
            MDPState.BAYESIAN_INFERENCE: 2.0,
            MDPState.FUZZY_EVALUATION: 2.5,
            MDPState.NETWORK_ANALYSIS: 3.0,
            MDPState.ANNOTATION_GENERATION: 4.0,
            MDPState.CONTEXT_VERIFICATION: 5.0,
            MDPState.VALIDATION_COMPLETE: 10.0,
            
            # Negative rewards for errors
            MDPState.ERROR_DETECTED: -5.0,
            MDPState.RECOVERY_MODE: -2.0
        }
        
        # Initialize reward model
        for (state, action), next_states in self.transition_model.items():
            for next_state, prob in next_states.items():
                base_reward = base_rewards[next_state]
                
                # Modify reward based on action appropriateness
                action_bonus = self._get_action_bonus(state, action, next_state)
                
                # Store reward
                self.reward_model[(state, action, next_state)] = base_reward + action_bonus
    
    def _get_action_bonus(self, state: MDPState, action: MDPAction, next_state: MDPState) -> float:
        """
        Calculate bonus reward based on action appropriateness.
        """
        # Bonus for taking appropriate actions
        appropriate_actions = {
            MDPState.EVIDENCE_COLLECTION: [MDPAction.COLLECT_MORE_EVIDENCE, MDPAction.UPDATE_PROBABILITIES],
            MDPState.BAYESIAN_INFERENCE: [MDPAction.UPDATE_PROBABILITIES, MDPAction.REFINE_FUZZY_LOGIC],
            MDPState.FUZZY_EVALUATION: [MDPAction.REFINE_FUZZY_LOGIC, MDPAction.STRENGTHEN_CONNECTIONS],
            MDPState.NETWORK_ANALYSIS: [MDPAction.STRENGTHEN_CONNECTIONS, MDPAction.PRUNE_WEAK_CONNECTIONS, MDPAction.GENERATE_ANNOTATIONS],
            MDPState.ANNOTATION_GENERATION: [MDPAction.GENERATE_ANNOTATIONS, MDPAction.VERIFY_CONTEXT],
            MDPState.CONTEXT_VERIFICATION: [MDPAction.VERIFY_CONTEXT, MDPAction.VALIDATE_RESULTS],
            MDPState.ERROR_DETECTED: [MDPAction.CORRECT_ERRORS, MDPAction.RESTART_ANALYSIS],
            MDPState.RECOVERY_MODE: [MDPAction.CORRECT_ERRORS, MDPAction.RESTART_ANALYSIS],
            MDPState.VALIDATION_COMPLETE: [MDPAction.ACCEPT_RESULTS]
        }
        
        if action in appropriate_actions.get(state, []):
            return 1.0
        else:
            return -0.5
    
    def _initialize_utility_functions(self):
        """
        Initialize utility functions for multi-objective optimization.
        """
        # Evidence Quality Utility
        evidence_quality = UtilityFunction(
            name="evidence_quality",
            weight=0.25,
            function=self._evidence_quality_utility,
            description="Measures the quality and reliability of collected evidence"
        )
        
        # Network Coherence Utility
        network_coherence = UtilityFunction(
            name="network_coherence",
            weight=0.20,
            function=self._network_coherence_utility,
            description="Evaluates the logical consistency of the evidence network"
        )
        
        # Computational Efficiency Utility
        computational_efficiency = UtilityFunction(
            name="computational_efficiency",
            weight=0.15,
            function=self._computational_efficiency_utility,
            description="Optimizes for computational resource usage"
        )
        
        # Annotation Confidence Utility
        annotation_confidence = UtilityFunction(
            name="annotation_confidence",
            weight=0.25,
            function=self._annotation_confidence_utility,
            description="Maximizes confidence in generated annotations"
        )
        
        # Context Preservation Utility
        context_preservation = UtilityFunction(
            name="context_preservation",
            weight=0.15,
            function=self._context_preservation_utility,
            description="Ensures context is maintained throughout analysis"
        )
        
        self.utility_functions = [
            evidence_quality,
            network_coherence,
            computational_efficiency,
            annotation_confidence,
            context_preservation
        ]
    
    def _evidence_quality_utility(self, context: Dict[str, Any]) -> float:
        """
        Calculate utility based on evidence quality metrics.
        """
        # Extract quality metrics from context
        num_evidence_nodes = context.get('num_evidence_nodes', 0)
        avg_posterior_prob = context.get('avg_posterior_probability', 0.0)
        evidence_diversity = context.get('evidence_type_diversity', 0.0)
        
        # Normalize and combine metrics
        node_score = min(1.0, num_evidence_nodes / 20.0)  # Normalize to 20 nodes
        prob_score = avg_posterior_prob
        diversity_score = min(1.0, evidence_diversity / len(list(MDPState)))
        
        # Weighted combination
        utility = 0.4 * node_score + 0.4 * prob_score + 0.2 * diversity_score
        
        return utility
    
    def _network_coherence_utility(self, context: Dict[str, Any]) -> float:
        """
        Calculate utility based on network coherence and consistency.
        """
        network_density = context.get('network_density', 0.0)
        clustering_coefficient = context.get('clustering_coefficient', 0.0)
        connectivity = context.get('network_connectivity', 0.0)
        
        # Optimal network characteristics
        # Not too dense (avoid overfitting), not too sparse (avoid disconnection)
        optimal_density = 0.3
        density_score = 1.0 - abs(network_density - optimal_density) / optimal_density
        
        # Higher clustering is generally better for coherent networks
        clustering_score = clustering_coefficient
        
        # Good connectivity is important
        connectivity_score = connectivity
        
        # Weighted combination
        utility = 0.4 * density_score + 0.3 * clustering_score + 0.3 * connectivity_score
        
        return max(0.0, utility)
    
    def _computational_efficiency_utility(self, context: Dict[str, Any]) -> float:
        """
        Calculate utility based on computational efficiency.
        """
        processing_time = context.get('processing_time_seconds', 0.0)
        memory_usage = context.get('memory_usage_mb', 0.0)
        convergence_iterations = context.get('convergence_iterations', 0)
        
        # Time efficiency (assume 60 seconds is reasonable target)
        time_score = max(0.0, 1.0 - processing_time / 60.0)
        
        # Memory efficiency (assume 500MB is reasonable target)
        memory_score = max(0.0, 1.0 - memory_usage / 500.0)
        
        # Convergence efficiency (assume 100 iterations is reasonable)
        convergence_score = max(0.0, 1.0 - convergence_iterations / 100.0)
        
        # Weighted combination
        utility = 0.4 * time_score + 0.3 * memory_score + 0.3 * convergence_score
        
        return utility
    
    def _annotation_confidence_utility(self, context: Dict[str, Any]) -> float:
        """
        Calculate utility based on annotation confidence and reliability.
        """
        num_annotations = context.get('num_annotations', 0)
        avg_confidence = context.get('avg_annotation_confidence', 0.0)
        high_confidence_ratio = context.get('high_confidence_ratio', 0.0)
        
        # Annotation quantity (normalized to 50 annotations)
        quantity_score = min(1.0, num_annotations / 50.0)
        
        # Average confidence score
        confidence_score = avg_confidence
        
        # Ratio of high-confidence annotations
        high_conf_score = high_confidence_ratio
        
        # Weighted combination
        utility = 0.3 * quantity_score + 0.4 * confidence_score + 0.3 * high_conf_score
        
        return utility
    
    def _context_preservation_utility(self, context: Dict[str, Any]) -> float:
        """
        Calculate utility based on context preservation quality.
        """
        context_puzzles_solved = context.get('context_puzzles_solved', 0)
        context_puzzles_total = context.get('context_puzzles_total', 1)
        context_verification_score = context.get('context_verification_score', 0.0)
        
        # Puzzle solving success rate
        puzzle_success_rate = context_puzzles_solved / max(1, context_puzzles_total)
        
        # Context verification score
        verification_score = context_verification_score
        
        # Weighted combination
        utility = 0.6 * puzzle_success_rate + 0.4 * verification_score
        
        return utility
    
    def calculate_total_utility(self, context: Dict[str, Any]) -> float:
        """
        Calculate total utility as weighted sum of individual utility functions.
        """
        total_utility = 0.0
        
        for utility_func in self.utility_functions:
            individual_utility = utility_func.function(context)
            weighted_utility = utility_func.weight * individual_utility
            total_utility += weighted_utility
            
            logger.debug(f"Utility {utility_func.name}: {individual_utility:.4f} (weighted: {weighted_utility:.4f})")
        
        return total_utility
    
    def solve_mdp(self) -> MDPPolicy:
        """
        Solve the MDP using value iteration to find optimal policy.
        """
        logger.info("Solving MDP using value iteration...")
        
        # Initialize value function
        V = {state: 0.0 for state in self.states}
        
        for iteration in range(self.max_iterations):
            V_old = V.copy()
            
            # Value iteration update
            for state in self.states:
                if state == MDPState.VALIDATION_COMPLETE:
                    continue  # Terminal state
                
                # Calculate Q-values for all actions
                q_values = {}
                for action in self.actions:
                    if (state, action) in self.transition_model:
                        q_value = 0.0
                        for next_state, prob in self.transition_model[(state, action)].items():
                            reward = self.reward_model.get((state, action, next_state), 0.0)
                            q_value += prob * (reward + self.discount_factor * V_old[next_state])
                        q_values[action] = q_value
                
                # Update value function with maximum Q-value
                if q_values:
                    V[state] = max(q_values.values())
            
            # Check convergence
            max_change = max(abs(V[state] - V_old[state]) for state in self.states)
            if max_change < self.convergence_threshold:
                logger.info(f"MDP converged after {iteration + 1} iterations")
                break
        
        # Extract optimal policy
        policy = self._extract_policy(V)
        
        self.policy = policy
        return policy
    
    def _extract_policy(self, V: Dict[MDPState, float]) -> MDPPolicy:
        """
        Extract optimal policy from value function.
        """
        state_action_probs = {}
        expected_rewards = {}
        
        for state in self.states:
            if state == MDPState.VALIDATION_COMPLETE:
                state_action_probs[state] = {MDPAction.ACCEPT_RESULTS: 1.0}
                continue
            
            # Calculate Q-values for all actions
            q_values = {}
            for action in self.actions:
                if (state, action) in self.transition_model:
                    q_value = 0.0
                    for next_state, prob in self.transition_model[(state, action)].items():
                        reward = self.reward_model.get((state, action, next_state), 0.0)
                        q_value += prob * (reward + self.discount_factor * V[next_state])
                    q_values[action] = q_value
                    expected_rewards[(state, action)] = q_value
            
            # Create policy (epsilon-greedy for exploration)
            if q_values:
                best_action = max(q_values, key=q_values.get)
                action_probs = {}
                
                for action in q_values:
                    if action == best_action:
                        action_probs[action] = 1.0 - self.exploration_rate + (self.exploration_rate / len(q_values))
                    else:
                        action_probs[action] = self.exploration_rate / len(q_values)
                
                state_action_probs[state] = action_probs
        
        policy = MDPPolicy(
            state_action_probabilities=state_action_probs,
            expected_rewards=expected_rewards,
            value_function=V,
            policy_iteration=0
        )
        
        return policy
    
    def select_action(self, context: Dict[str, Any]) -> MDPAction:
        """
        Select action based on current state and policy.
        """
        if self.policy is None:
            self.solve_mdp()
        
        # Update context for utility calculation
        self.analysis_context.update(context)
        
        # Get action probabilities for current state
        if self.current_state in self.policy.state_action_probabilities:
            action_probs = self.policy.state_action_probabilities[self.current_state]
            
            # Sample action based on probabilities
            actions = list(action_probs.keys())
            probabilities = list(action_probs.values())
            
            selected_action = np.random.choice(actions, p=probabilities)
            
            logger.debug(f"Selected action {selected_action.value} in state {self.current_state.value}")
            
            return selected_action
        else:
            # Fallback to default action
            logger.warning(f"No policy defined for state {self.current_state.value}")
            return self.actions[0]
    
    def execute_action(self, action: MDPAction, context: Dict[str, Any]) -> Tuple[MDPState, float]:
        """
        Execute action and transition to next state.
        """
        if (self.current_state, action) not in self.transition_model:
            logger.error(f"Invalid action {action.value} for state {self.current_state.value}")
            return self.current_state, -10.0  # Heavy penalty for invalid actions
        
        # Sample next state based on transition probabilities
        next_states = self.transition_model[(self.current_state, action)]
        states = list(next_states.keys())
        probabilities = list(next_states.values())
        
        next_state = np.random.choice(states, p=probabilities)
        
        # Calculate reward
        reward = self.reward_model.get((self.current_state, action, next_state), 0.0)
        
        # Add utility-based reward
        utility_reward = self.calculate_total_utility(context) * 2.0  # Scale utility
        total_reward = reward + utility_reward
        
        # Update state and tracking
        previous_state = self.current_state
        self.current_state = next_state
        self.cumulative_reward += total_reward
        
        # Record transition
        self.state_history.append((
            previous_state,
            action,
            total_reward,
            datetime.now()
        ))
        
        logger.info(f"Transitioned from {previous_state.value} to {next_state.value} "
                   f"via {action.value}, reward: {total_reward:.4f}")
        
        return next_state, total_reward
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive MDP validation report.
        """
        # Calculate performance metrics
        total_transitions = len(self.state_history)
        avg_reward = self.cumulative_reward / max(1, total_transitions)
        
        # State distribution
        state_counts = defaultdict(int)
        for state, _, _, _ in self.state_history:
            state_counts[state.value] += 1
        
        # Action distribution
        action_counts = defaultdict(int)
        for _, action, _, _ in self.state_history:
            action_counts[action.value] += 1
        
        # Utility breakdown
        utility_breakdown = {}
        for utility_func in self.utility_functions:
            individual_utility = utility_func.function(self.analysis_context)
            utility_breakdown[utility_func.name] = {
                'value': individual_utility,
                'weight': utility_func.weight,
                'weighted_value': utility_func.weight * individual_utility,
                'description': utility_func.description
            }
        
        # Policy performance
        policy_performance = {
            'current_state': self.current_state.value,
            'total_reward': self.cumulative_reward,
            'average_reward': avg_reward,
            'total_transitions': total_transitions,
            'convergence_iterations': getattr(self.policy, 'policy_iteration', 0) if self.policy else 0
        }
        
        # Validation recommendations
        recommendations = self._generate_recommendations()
        
        report = {
            'mdp_configuration': {
                'discount_factor': self.discount_factor,
                'exploration_rate': self.exploration_rate,
                'convergence_threshold': self.convergence_threshold
            },
            'policy_performance': policy_performance,
            'state_distribution': dict(state_counts),
            'action_distribution': dict(action_counts),
            'utility_analysis': utility_breakdown,
            'total_utility': self.calculate_total_utility(self.analysis_context),
            'validation_metrics': self.validation_metrics,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """
        Generate recommendations based on MDP analysis.
        """
        recommendations = []
        
        # Check if stuck in error states
        error_transitions = sum(1 for state, _, _, _ in self.state_history[-10:] 
                              if state in [MDPState.ERROR_DETECTED, MDPState.RECOVERY_MODE])
        
        if error_transitions > 5:
            recommendations.append("High error rate detected - consider improving data quality or model parameters")
        
        # Check utility scores
        total_utility = self.calculate_total_utility(self.analysis_context)
        if total_utility < 0.5:
            recommendations.append("Low overall utility - review evidence collection and network construction")
        
        # Check for specific utility issues
        for utility_func in self.utility_functions:
            utility_value = utility_func.function(self.analysis_context)
            if utility_value < 0.3:
                recommendations.append(f"Low {utility_func.name} ({utility_value:.3f}) - {utility_func.description}")
        
        # Check convergence
        if len(self.state_history) > 100:
            recent_states = [state for state, _, _, _ in self.state_history[-20:]]
            if self.current_state not in [MDPState.VALIDATION_COMPLETE, MDPState.CONTEXT_VERIFICATION]:
                recommendations.append("Analysis not converging to completion - consider parameter adjustment")
        
        # Performance recommendations
        avg_reward = self.cumulative_reward / max(1, len(self.state_history))
        if avg_reward < 1.0:
            recommendations.append("Low average reward - optimize action selection and state transitions")
        
        return recommendations
    
    def export_mdp_model(self, filename: str):
        """
        Export the complete MDP model for analysis and visualization.
        """
        model_data = {
            'states': [state.value for state in self.states],
            'actions': [action.value for action in self.actions],
            'transition_model': {
                f"{state.value}_{action.value}": {next_state.value: prob 
                                                  for next_state, prob in transitions.items()}
                for (state, action), transitions in self.transition_model.items()
            },
            'reward_model': {
                f"{state.value}_{action.value}_{next_state.value}": reward
                for (state, action, next_state), reward in self.reward_model.items()
            },
            'utility_functions': [
                {
                    'name': uf.name,
                    'weight': uf.weight,
                    'description': uf.description,
                    'current_value': uf.function(self.analysis_context)
                }
                for uf in self.utility_functions
            ],
            'policy': {
                'state_action_probabilities': {
                    state.value: {action.value: prob for action, prob in actions.items()}
                    for state, actions in (self.policy.state_action_probabilities.items() if self.policy else {})
                },
                'value_function': {
                    state.value: value for state, value in (self.policy.value_function.items() if self.policy else {})
                }
            } if self.policy else None,
            'state_history': [
                {
                    'state': state.value,
                    'action': action.value,
                    'reward': reward,
                    'timestamp': timestamp.isoformat()
                }
                for state, action, reward, timestamp in self.state_history
            ],
            'analysis_summary': self.get_validation_report()
        }
        
        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2, default=str)
        
        logger.info(f"MDP model exported to {filename}")

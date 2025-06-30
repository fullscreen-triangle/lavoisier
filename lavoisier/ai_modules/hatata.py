"""
Hatata: Markov Decision Process Implementation

Stochastic validation of analytical workflows using MDP framework
for decision-making under uncertainty.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ValidationState(Enum):
    """States in the validation MDP"""
    INITIAL = "initial"
    DATA_LOADED = "data_loaded"
    PREPROCESSED = "preprocessed"
    FEATURES_EXTRACTED = "features_extracted"
    ANNOTATED = "annotated"
    VALIDATED = "validated"
    FAILED = "failed"
    COMPLETED = "completed"

class ValidationAction(Enum):
    """Actions available in validation workflow"""
    LOAD_DATA = "load_data"
    PREPROCESS = "preprocess"
    EXTRACT_FEATURES = "extract_features"
    ANNOTATE_PEAKS = "annotate_peaks"
    VALIDATE_RESULTS = "validate_results"
    RETRY_ANALYSIS = "retry_analysis"
    ABORT_WORKFLOW = "abort_workflow"
    COMPLETE_WORKFLOW = "complete_workflow"

@dataclass
class MDPTransition:
    """MDP state transition"""
    from_state: ValidationState
    action: ValidationAction
    to_state: ValidationState
    probability: float
    reward: float

@dataclass
class ValidationResult:
    """Result of validation process"""
    final_state: ValidationState
    total_reward: float
    path: List[Tuple[ValidationState, ValidationAction]]
    confidence: float

class MDPValidator:
    """Markov Decision Process validator for analytical workflows"""

    def __init__(self, discount_factor: float = 0.9):
        self.discount_factor = discount_factor
        self.states = list(ValidationState)
        self.actions = list(ValidationAction)

        # Initialize transition probabilities and rewards
        self.transition_probabilities = self._initialize_transitions()
        self.rewards = self._initialize_rewards()
        self.value_function = {state: 0.0 for state in self.states}
        self.policy = {state: ValidationAction.LOAD_DATA for state in self.states}

    def _initialize_transitions(self) -> Dict[Tuple[ValidationState, ValidationAction, ValidationState], float]:
        """Initialize transition probability matrix"""
        transitions = {}

        # Define transition probabilities
        transition_rules = [
            # From INITIAL
            (ValidationState.INITIAL, ValidationAction.LOAD_DATA, ValidationState.DATA_LOADED, 0.9),
            (ValidationState.INITIAL, ValidationAction.LOAD_DATA, ValidationState.FAILED, 0.1),

            # From DATA_LOADED
            (ValidationState.DATA_LOADED, ValidationAction.PREPROCESS, ValidationState.PREPROCESSED, 0.85),
            (ValidationState.DATA_LOADED, ValidationAction.PREPROCESS, ValidationState.FAILED, 0.15),

            # From PREPROCESSED
            (ValidationState.PREPROCESSED, ValidationAction.EXTRACT_FEATURES, ValidationState.FEATURES_EXTRACTED, 0.8),
            (ValidationState.PREPROCESSED, ValidationAction.EXTRACT_FEATURES, ValidationState.FAILED, 0.2),

            # From FEATURES_EXTRACTED
            (ValidationState.FEATURES_EXTRACTED, ValidationAction.ANNOTATE_PEAKS, ValidationState.ANNOTATED, 0.75),
            (ValidationState.FEATURES_EXTRACTED, ValidationAction.ANNOTATE_PEAKS, ValidationState.FAILED, 0.25),

            # From ANNOTATED
            (ValidationState.ANNOTATED, ValidationAction.VALIDATE_RESULTS, ValidationState.VALIDATED, 0.7),
            (ValidationState.ANNOTATED, ValidationAction.VALIDATE_RESULTS, ValidationState.FAILED, 0.3),

            # From VALIDATED
            (ValidationState.VALIDATED, ValidationAction.COMPLETE_WORKFLOW, ValidationState.COMPLETED, 1.0),

            # From FAILED - retry or abort
            (ValidationState.FAILED, ValidationAction.RETRY_ANALYSIS, ValidationState.INITIAL, 0.6),
            (ValidationState.FAILED, ValidationAction.ABORT_WORKFLOW, ValidationState.FAILED, 1.0),
        ]

        for from_state, action, to_state, prob in transition_rules:
            transitions[(from_state, action, to_state)] = prob

        return transitions

    def _initialize_rewards(self) -> Dict[Tuple[ValidationState, ValidationAction], float]:
        """Initialize reward structure"""
        rewards = {}

        # Reward structure
        reward_rules = [
            # Successful progression rewards
            (ValidationState.INITIAL, ValidationAction.LOAD_DATA, 10.0),
            (ValidationState.DATA_LOADED, ValidationAction.PREPROCESS, 15.0),
            (ValidationState.PREPROCESSED, ValidationAction.EXTRACT_FEATURES, 20.0),
            (ValidationState.FEATURES_EXTRACTED, ValidationAction.ANNOTATE_PEAKS, 25.0),
            (ValidationState.ANNOTATED, ValidationAction.VALIDATE_RESULTS, 30.0),
            (ValidationState.VALIDATED, ValidationAction.COMPLETE_WORKFLOW, 100.0),

            # Penalties
            (ValidationState.FAILED, ValidationAction.RETRY_ANALYSIS, -5.0),
            (ValidationState.FAILED, ValidationAction.ABORT_WORKFLOW, -50.0),
        ]

        for state, action, reward in reward_rules:
            rewards[(state, action)] = reward

        # Default reward for unspecified state-action pairs
        for state in self.states:
            for action in self.actions:
                if (state, action) not in rewards:
                    rewards[(state, action)] = -1.0  # Small penalty for time

        return rewards

    def get_transition_probability(self, from_state: ValidationState,
                                 action: ValidationAction,
                                 to_state: ValidationState) -> float:
        """Get transition probability"""
        return self.transition_probabilities.get((from_state, action, to_state), 0.0)

    def get_reward(self, state: ValidationState, action: ValidationAction) -> float:
        """Get reward for state-action pair"""
        return self.rewards.get((state, action), -1.0)

    def value_iteration(self, max_iterations: int = 100, tolerance: float = 1e-6) -> bool:
        """Perform value iteration to find optimal policy"""
        for iteration in range(max_iterations):
            old_values = self.value_function.copy()

            for state in self.states:
                if state in [ValidationState.COMPLETED, ValidationState.FAILED]:
                    continue  # Terminal states

                action_values = []
                for action in self.actions:
                    # Calculate expected value for this action
                    expected_value = 0.0
                    for next_state in self.states:
                        prob = self.get_transition_probability(state, action, next_state)
                        if prob > 0:
                            reward = self.get_reward(state, action)
                            expected_value += prob * (reward + self.discount_factor * old_values[next_state])

                    action_values.append((action, expected_value))

                # Choose action with maximum expected value
                if action_values:
                    best_action, best_value = max(action_values, key=lambda x: x[1])
                    self.value_function[state] = best_value
                    self.policy[state] = best_action

            # Check convergence
            max_change = max(abs(self.value_function[s] - old_values[s]) for s in self.states)
            if max_change < tolerance:
                logger.info(f"Value iteration converged after {iteration + 1} iterations")
                return True

        logger.warning(f"Value iteration did not converge after {max_iterations} iterations")
        return False

    def policy_evaluation(self, max_iterations: int = 100) -> Dict[ValidationState, float]:
        """Evaluate current policy"""
        values = {state: 0.0 for state in self.states}

        for iteration in range(max_iterations):
            old_values = values.copy()

            for state in self.states:
                if state in [ValidationState.COMPLETED, ValidationState.FAILED]:
                    continue

                action = self.policy[state]
                expected_value = 0.0

                for next_state in self.states:
                    prob = self.get_transition_probability(state, action, next_state)
                    if prob > 0:
                        reward = self.get_reward(state, action)
                        expected_value += prob * (reward + self.discount_factor * old_values[next_state])

                values[state] = expected_value

            # Check convergence
            max_change = max(abs(values[s] - old_values[s]) for s in self.states)
            if max_change < 1e-6:
                break

        return values

    def simulate_workflow(self, initial_state: ValidationState = ValidationState.INITIAL,
                         max_steps: int = 20) -> ValidationResult:
        """Simulate workflow execution using current policy"""
        current_state = initial_state
        total_reward = 0.0
        path = []

        for step in range(max_steps):
            if current_state in [ValidationState.COMPLETED, ValidationState.FAILED]:
                break

            # Choose action according to policy
            action = self.policy[current_state]
            path.append((current_state, action))

            # Get reward
            reward = self.get_reward(current_state, action)
            total_reward += reward

            # Sample next state based on transition probabilities
            next_state_probs = []
            for next_state in self.states:
                prob = self.get_transition_probability(current_state, action, next_state)
                if prob > 0:
                    next_state_probs.append((next_state, prob))

            if next_state_probs:
                # Sample next state
                states, probs = zip(*next_state_probs)
                probs = np.array(probs)
                probs = probs / probs.sum()  # Normalize

                next_state = np.random.choice(states, p=probs)
                current_state = next_state
            else:
                # No valid transitions, workflow failed
                current_state = ValidationState.FAILED
                break

        # Calculate confidence based on final state and total reward
        confidence = 1.0 if current_state == ValidationState.COMPLETED else 0.0
        if current_state == ValidationState.VALIDATED:
            confidence = 0.8
        elif current_state == ValidationState.ANNOTATED:
            confidence = 0.6
        elif current_state == ValidationState.FEATURES_EXTRACTED:
            confidence = 0.4
        elif current_state == ValidationState.PREPROCESSED:
            confidence = 0.2

        return ValidationResult(
            final_state=current_state,
            total_reward=total_reward,
            path=path,
            confidence=confidence
        )

class StochasticWorkflowValidator:
    """Main interface for stochastic workflow validation"""

    def __init__(self):
        self.mdp_validator = MDPValidator()
        self.validation_history = []

    def optimize_workflow(self) -> bool:
        """Optimize workflow using value iteration"""
        return self.mdp_validator.value_iteration()

    def validate_workflow(self, workflow_data: Dict[str, Any] = None) -> ValidationResult:
        """Validate a specific workflow"""
        result = self.mdp_validator.simulate_workflow()
        self.validation_history.append(result)

        logger.info(f"Workflow validation completed: {result.final_state.value}")
        logger.info(f"Total reward: {result.total_reward:.2f}")
        logger.info(f"Confidence: {result.confidence:.2f}")

        return result

    def batch_validate(self, num_simulations: int = 100) -> Dict[str, Any]:
        """Run batch validation simulations"""
        results = []
        for _ in range(num_simulations):
            result = self.mdp_validator.simulate_workflow()
            results.append(result)

        # Calculate statistics
        success_rate = sum(1 for r in results if r.final_state == ValidationState.COMPLETED) / len(results)
        avg_reward = np.mean([r.total_reward for r in results])
        avg_confidence = np.mean([r.confidence for r in results])
        avg_path_length = np.mean([len(r.path) for r in results])

        state_distribution = {}
        for state in ValidationState:
            count = sum(1 for r in results if r.final_state == state)
            state_distribution[state.value] = count / len(results)

        batch_results = {
            "num_simulations": num_simulations,
            "success_rate": success_rate,
            "average_reward": avg_reward,
            "average_confidence": avg_confidence,
            "average_path_length": avg_path_length,
            "final_state_distribution": state_distribution,
            "individual_results": results
        }

        logger.info(f"Batch validation completed: {num_simulations} simulations")
        logger.info(f"Success rate: {success_rate:.2f}")
        logger.info(f"Average confidence: {avg_confidence:.2f}")

        return batch_results

    def get_optimal_policy(self) -> Dict[str, str]:
        """Get current optimal policy"""
        return {state.value: action.value
                for state, action in self.mdp_validator.policy.items()}

    def get_state_values(self) -> Dict[str, float]:
        """Get current state values"""
        return {state.value: value
                for state, value in self.mdp_validator.value_function.items()}

    def analyze_workflow_robustness(self, perturbation_factor: float = 0.1) -> Dict[str, Any]:
        """Analyze workflow robustness to parameter changes"""
        original_transitions = self.mdp_validator.transition_probabilities.copy()

        # Add noise to transition probabilities
        perturbed_transitions = {}
        for key, prob in original_transitions.items():
            noise = np.random.normal(0, perturbation_factor * prob)
            perturbed_prob = max(0.0, min(1.0, prob + noise))
            perturbed_transitions[key] = perturbed_prob

        # Temporarily update transitions
        self.mdp_validator.transition_probabilities = perturbed_transitions

        # Re-optimize and validate
        self.mdp_validator.value_iteration()
        robustness_results = self.batch_validate(50)

        # Restore original transitions
        self.mdp_validator.transition_probabilities = original_transitions
        self.mdp_validator.value_iteration()

        robustness_analysis = {
            "perturbation_factor": perturbation_factor,
            "robustness_score": robustness_results["success_rate"],
            "confidence_stability": robustness_results["average_confidence"],
            "performance_degradation": max(0, 0.8 - robustness_results["success_rate"])  # Assume 0.8 baseline
        }

        return robustness_analysis

class Hatata:
    """Main Hatata module for MDP-based workflow validation"""

    def __init__(self):
        self.workflow_validator = StochasticWorkflowValidator()
        self.optimization_complete = False

    def initialize_validation_framework(self) -> bool:
        """Initialize and optimize the validation framework"""
        logger.info("Initializing Hatata validation framework")
        self.optimization_complete = self.workflow_validator.optimize_workflow()
        return self.optimization_complete

    def validate_analytical_workflow(self, workflow_config: Dict[str, Any] = None) -> ValidationResult:
        """Validate analytical workflow with MDP"""
        if not self.optimization_complete:
            self.initialize_validation_framework()

        return self.workflow_validator.validate_workflow(workflow_config)

    def assess_workflow_reliability(self, num_trials: int = 100) -> Dict[str, Any]:
        """Assess workflow reliability through multiple trials"""
        if not self.optimization_complete:
            self.initialize_validation_framework()

        return self.workflow_validator.batch_validate(num_trials)

    def get_workflow_recommendations(self) -> Dict[str, Any]:
        """Get workflow optimization recommendations"""
        policy = self.workflow_validator.get_optimal_policy()
        values = self.workflow_validator.get_state_values()

        recommendations = {
            "optimal_policy": policy,
            "state_values": values,
            "high_value_states": [state for state, value in values.items() if value > 50],
            "critical_transitions": self._identify_critical_transitions(),
            "improvement_suggestions": self._generate_improvement_suggestions()
        }

        return recommendations

    def _identify_critical_transitions(self) -> List[str]:
        """Identify critical state transitions"""
        # Analyze transition probabilities to find bottlenecks
        critical_transitions = []

        for (from_state, action, to_state), prob in self.workflow_validator.mdp_validator.transition_probabilities.items():
            if prob < 0.8 and from_state != ValidationState.FAILED:  # Low success probability
                critical_transitions.append(f"{from_state.value} -> {to_state.value} via {action.value} ({prob:.2f})")

        return critical_transitions

    def _generate_improvement_suggestions(self) -> List[str]:
        """Generate workflow improvement suggestions"""
        suggestions = [
            "Implement additional quality control checks before feature extraction",
            "Add redundant annotation methods to improve reliability",
            "Implement automated retry mechanisms for failed operations",
            "Add intermediate validation checkpoints",
            "Optimize preprocessing parameters to reduce failure rates"
        ]

        return suggestions

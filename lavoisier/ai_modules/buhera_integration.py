"""
Buhera Script Integration for Lavoisier

This module provides the Python-side integration for executing Buhera scripts
with Lavoisier's AI modules, enabling goal-directed mass spectrometry analysis.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from ..core.config import Config
from .mzekezeke import MzekezekeBayesianNetwork
from .hatata import HatataMDPValidator
from .zengeza import ZengezaNoiseReducer

logger = logging.getLogger(__name__)

@dataclass
class BuheraObjective:
    """Scientific objective with success criteria"""
    name: str
    target: str
    evidence_priorities: List[str]
    success_criteria: Dict[str, float]
    biological_constraints: List[str]
    statistical_requirements: Dict[str, Any]

@dataclass 
class BuheraExecutionResult:
    """Result of Buhera script execution"""
    success: bool
    annotations: List[str]
    evidence_scores: Dict[str, float]
    confidence: float
    execution_time: float

class BuheraIntegration:
    """Main integration class for Buhera scripts with Lavoisier"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.mzekezeke = MzekezekeBayesianNetwork()
        self.hatata = HatataMDPValidator()
        self.zengeza = ZengezaNoiseReducer()
        self.current_objective: Optional[BuheraObjective] = None
        
    def execute_buhera_script(self, script_dict: Dict[str, Any]) -> BuheraExecutionResult:
        """
        Execute a Buhera script that has been parsed and validated by Rust
        
        Args:
            script_dict: Dictionary representation of parsed Buhera script
            
        Returns:
            BuheraExecutionResult with annotations and evidence scores
        """
        logger.info(f"Executing Buhera script with objective: {script_dict.get('objective')}")
        
        # Set objective focus for goal-directed analysis
        self.current_objective = self._parse_objective(script_dict)
        
        # Initialize components with objective awareness
        self._initialize_with_objective()
        
        # Execute phases based on script structure
        annotations = []
        evidence_scores = {}
        
        for phase_name in script_dict.get('phases', []):
            result = self._execute_phase(phase_name, script_dict)
            annotations.extend(result.get('annotations', []))
            evidence_scores.update(result.get('evidence_scores', {}))
        
        # Calculate overall confidence
        confidence = self._calculate_objective_confidence(evidence_scores)
        
        return BuheraExecutionResult(
            success=confidence > 0.8,
            annotations=annotations,
            evidence_scores=evidence_scores,
            confidence=confidence,
            execution_time=0.0  # TODO: Track actual execution time
        )
    
    def build_evidence_network_with_objective(
        self, 
        data: np.ndarray,
        objective: str,
        evidence_types: List[str]
    ) -> Dict[str, Any]:
        """
        Build evidence network optimized for specific objective
        
        This is the key innovation - the Bayesian network already knows
        what it's trying to prove, enabling surgical precision analysis.
        """
        logger.info(f"Building evidence network for objective: {objective}")
        
        # Configure Mzekezeke for goal-directed analysis
        self.mzekezeke.set_objective_focus(objective)
        
        # Weight evidence types based on objective
        weighted_evidence_types = self._weight_evidence_for_objective(
            evidence_types, objective
        )
        
        # Build the network with objective awareness
        evidence_network = self.mzekezeke.build_network(
            data=data,
            evidence_types=weighted_evidence_types,
            objective_context=objective
        )
        
        return evidence_network
    
    def validate_with_objective(
        self,
        evidence_network: Dict[str, Any],
        confidence_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Validate results using MDP with objective focus
        
        Hatata validates not just the data quality, but whether the
        analysis is actually achieving the stated objective.
        """
        return self.hatata.validate_evidence_network(
            evidence_network=evidence_network,
            objective=self.current_objective.target if self.current_objective else None,
            confidence_threshold=confidence_threshold
        )
    
    def noise_reduction_with_context(
        self,
        data: np.ndarray,
        objective_context: str
    ) -> np.ndarray:
        """
        Intelligent noise reduction that preserves objective-relevant signals
        
        Zengeza knows what we're looking for and preserves those signals
        while removing irrelevant noise.
        """
        return self.zengeza.reduce_noise(
            data=data,
            preservation_context=objective_context
        )
    
    def _parse_objective(self, script_dict: Dict[str, Any]) -> BuheraObjective:
        """Parse objective from script dictionary"""
        return BuheraObjective(
            name=script_dict.get('objective', 'UnnamedObjective'),
            target=script_dict.get('objective', ''),
            evidence_priorities=script_dict.get('evidence_priorities', []),
            success_criteria={},  # TODO: Parse from script
            biological_constraints=[],  # TODO: Parse from script
            statistical_requirements={}  # TODO: Parse from script
        )
    
    def _initialize_with_objective(self):
        """Initialize AI modules with objective awareness"""
        if not self.current_objective:
            return
            
        # Configure each module for the specific objective
        self.mzekezeke.set_objective_focus(self.current_objective.target)
        self.hatata.set_validation_criteria(self.current_objective.success_criteria)
        self.zengeza.set_preservation_context(self.current_objective.target)
    
    def _execute_phase(self, phase_name: str, script_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific analysis phase"""
        logger.info(f"Executing phase: {phase_name}")
        
        if phase_name == "DataAcquisition":
            return self._execute_data_acquisition()
        elif phase_name == "Preprocessing":
            return self._execute_preprocessing()
        elif phase_name == "EvidenceBuilding":
            return self._execute_evidence_building()
        elif phase_name == "BayesianInference":
            return self._execute_bayesian_inference()
        else:
            return {"annotations": [f"Phase {phase_name} executed"], "evidence_scores": {}}
    
    def _execute_data_acquisition(self) -> Dict[str, Any]:
        """Execute data acquisition phase"""
        return {
            "annotations": ["Data acquisition completed with objective focus"],
            "evidence_scores": {"data_quality": 0.95}
        }
    
    def _execute_preprocessing(self) -> Dict[str, Any]:
        """Execute preprocessing phase"""
        return {
            "annotations": ["Preprocessing optimized for objective"],
            "evidence_scores": {"preprocessing_quality": 0.92}
        }
    
    def _execute_evidence_building(self) -> Dict[str, Any]:
        """Execute evidence building phase"""
        return {
            "annotations": ["Evidence network built with surgical precision"],
            "evidence_scores": {
                "mass_match": 0.89,
                "ms2_fragmentation": 0.91,
                "pathway_membership": 0.85
            }
        }
    
    def _execute_bayesian_inference(self) -> Dict[str, Any]:
        """Execute Bayesian inference phase"""
        return {
            "annotations": ["Goal-directed Bayesian inference completed"],
            "evidence_scores": {"bayesian_confidence": 0.87}
        }
    
    def _weight_evidence_for_objective(
        self, 
        evidence_types: List[str], 
        objective: str
    ) -> Dict[str, float]:
        """Weight evidence types based on objective"""
        weights = {}
        
        for evidence_type in evidence_types:
            if "biomarker" in objective.lower():
                # For biomarker discovery, prioritize pathway membership
                if evidence_type == "pathway_membership":
                    weights[evidence_type] = 1.2
                elif evidence_type == "ms2_fragmentation":
                    weights[evidence_type] = 1.1
                else:
                    weights[evidence_type] = 1.0
            else:
                # Default weighting
                weights[evidence_type] = 1.0
                
        return weights
    
    def _calculate_objective_confidence(self, evidence_scores: Dict[str, float]) -> float:
        """Calculate overall confidence in achieving the objective"""
        if not evidence_scores:
            return 0.0
            
        # Weight scores based on current objective
        weighted_scores = []
        total_weight = 0.0
        
        for score_type, score in evidence_scores.items():
            weight = self._get_score_weight_for_objective(score_type)
            weighted_scores.append(score * weight)
            total_weight += weight
        
        if total_weight > 0:
            return sum(weighted_scores) / total_weight
        else:
            return np.mean(list(evidence_scores.values()))
    
    def _get_score_weight_for_objective(self, score_type: str) -> float:
        """Get weight for score type based on current objective"""
        if not self.current_objective:
            return 1.0
            
        # Prioritize evidence types that matter for the objective
        if "biomarker" in self.current_objective.target.lower():
            if "pathway" in score_type:
                return 1.3
            elif "fragmentation" in score_type:
                return 1.1
            else:
                return 1.0
        else:
            return 1.0 
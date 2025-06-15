"""
Advanced AI Modules for Mass Spectrometry Analysis

This package contains sophisticated AI modules for enhancing mass spectrometry
data processing through intelligent noise reduction, Bayesian evidence networks,
context verification, stochastic validation, and adversarial testing.

Modules:
- Zengeza: Intelligent noise reduction with statistical analysis and ML
- Mzekezeke: Bayesian evidence networks with fuzzy logic for annotations
- Nicotine: Context verification through cryptographic AI puzzles
- Hatata: MDP-based stochastic verification with utility optimization
- Diggiden: Adversarial testing system for vulnerability detection
- Integration: Orchestrates all modules for comprehensive analysis
"""

from .zengeza import ZengezaNoiseReducer
from .mzekezeke import MzekezekeBayesianNetwork, EvidenceType, AnnotationCandidate
from .nicotine import NicotineContextVerifier, PuzzleType
from .hatata import HatataMDPVerifier, MDPState, MDPAction
from .diggiden import DiggidenAdversarialTester, AttackType
from .integration import AdvancedMSAnalysisSystem, AnalysisResult

__all__ = [
    # Core modules
    'ZengezaNoiseReducer',
    'MzekezekeBayesianNetwork', 
    'NicotineContextVerifier',
    'HatataMDPVerifier',
    'DiggidenAdversarialTester',
    'AdvancedMSAnalysisSystem',
    
    # Key classes and enums
    'EvidenceType',
    'AnnotationCandidate', 
    'PuzzleType',
    'MDPState',
    'MDPAction',
    'AttackType',
    'AnalysisResult'
] 
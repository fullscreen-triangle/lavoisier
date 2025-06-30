# Buhera-Lavoisier Integration Guide

This guide explains how Buhera integrates with Lavoisier's AI modules to provide goal-directed mass spectrometry analysis.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [AI Module Integration](#ai-module-integration)
- [Python-Rust Bridge](#python-rust-bridge)
- [Execution Flow](#execution-flow)
- [Error Handling](#error-handling)
- [Performance Considerations](#performance-considerations)
- [Extending the Integration](#extending-the-integration)

## Architecture Overview

Buhera acts as the orchestration layer that coordinates Lavoisier's AI modules for objective-focused analysis:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Buhera Integration Stack                       │
├─────────────────────────────────────────────────────────────────┤
│  Buhera Script (.bh)                                           │
│  ├─ Objective declaration                                       │
│  ├─ Validation rules                                           │
│  └─ Analysis phases                                            │
├─────────────────────────────────────────────────────────────────┤
│  Buhera Language Core (Rust)                                   │
│  ├─ Parser: Converts .bh files to AST                         │
│  ├─ Validator: Pre-flight validation system                   │
│  └─ Executor: Orchestrates analysis execution                 │
├─────────────────────────────────────────────────────────────────┤
│  Python Bridge (PyO3)                                          │
│  ├─ Script execution in Python context                        │
│  ├─ Data marshaling between Rust and Python                   │
│  └─ Error handling and result collection                      │
├─────────────────────────────────────────────────────────────────┤
│  Lavoisier AI Modules (Python)                                │
│  ├─ BuheraIntegration: Main coordination class                │
│  ├─ Enhanced modules with objective awareness                 │
│  └─ Goal-directed evidence network building                   │
├─────────────────────────────────────────────────────────────────┤
│  Core Lavoisier Framework                                      │
│  ├─ Numerical processing pipeline                             │
│  ├─ Visual processing pipeline                                │
│  └─ Data I/O and storage systems                             │
└─────────────────────────────────────────────────────────────────┘
```

## AI Module Integration

### Mzekezeke: Objective-Aware Bayesian Networks

#### Traditional Mzekezeke Usage
```python
# Generic evidence network without objectives
network = mzekezeke.build_evidence_network(data)
```

#### Buhera-Enhanced Mzekezeke
```python
# Objective-focused evidence network
network = mzekezeke.build_evidence_network(
    data=data,
    objective="diabetes_biomarker_discovery",
    evidence_priorities=["pathway_membership", "ms2_fragmentation"],
    biological_constraints=["glycolysis_upregulated", "insulin_resistance"],
    success_criteria={"sensitivity": 0.85, "specificity": 0.85}
)
```

#### How Objectives Enhance Mzekezeke

1. **Evidence Weighting**: Automatically adjusts evidence weights based on objective
2. **Pathway Focus**: Prioritizes evidence from relevant biological pathways
3. **Success Monitoring**: Continuously evaluates progress toward objective criteria
4. **Constraint Enforcement**: Validates results against biological constraints

#### Implementation Details

```python
class ObjectiveAwareBayesianNetwork:
    def __init__(self, objective: BuheraObjective):
        self.objective = objective
        self.evidence_weights = self._calculate_objective_weights()
        self.constraints = self._parse_biological_constraints()
    
    def _calculate_objective_weights(self):
        """Calculate evidence weights based on objective type"""
        if "biomarker" in self.objective.target.lower():
            return {
                "pathway_membership": 1.3,
                "clinical_correlation": 1.2,
                "ms2_fragmentation": 1.1,
                "mass_match": 1.0
            }
        elif "metabolism" in self.objective.target.lower():
            return {
                "ms2_fragmentation": 1.4,
                "retention_time": 1.2,
                "mass_match": 1.0,
                "pathway_membership": 0.9
            }
        # Add more objective types...
    
    def build_network(self, data):
        """Build evidence network with objective focus"""
        # Standard evidence collection
        evidence = self._collect_evidence(data)
        
        # Apply objective-specific weighting
        weighted_evidence = self._apply_objective_weights(evidence)
        
        # Validate against constraints
        validated_evidence = self._validate_constraints(weighted_evidence)
        
        return validated_evidence
```

### Hatata: Objective-Aligned Validation

#### Enhanced Validation with Objectives

```python
class ObjectiveAlignedValidator:
    def __init__(self, objective: BuheraObjective):
        self.objective = objective
        self.success_criteria = self._parse_success_criteria()
    
    def validate_with_objective(self, evidence_network):
        """Validate results against objective criteria"""
        results = {
            "success": False,
            "confidence": 0.0,
            "criteria_met": {},
            "recommendations": []
        }
        
        # Check each success criterion
        for criterion, threshold in self.success_criteria.items():
            current_value = self._evaluate_criterion(evidence_network, criterion)
            results["criteria_met"][criterion] = {
                "current": current_value,
                "threshold": threshold,
                "met": current_value >= threshold
            }
        
        # Overall success assessment
        results["success"] = all(
            result["met"] for result in results["criteria_met"].values()
        )
        
        # Generate recommendations for improvement
        if not results["success"]:
            results["recommendations"] = self._generate_recommendations(results)
        
        return results
```

### Zengeza: Context-Preserving Noise Reduction

#### Objective-Aware Noise Reduction

```python
class ContextPreservingNoiseReducer:
    def __init__(self, objective_context: str):
        self.objective_context = objective_context
        self.preserve_patterns = self._get_preservation_patterns()
    
    def _get_preservation_patterns(self):
        """Get patterns to preserve based on objective"""
        patterns = {
            "diabetes_biomarker_discovery": [
                "glucose_metabolism",
                "lipid_metabolism", 
                "amino_acid_metabolism"
            ],
            "drug_metabolism_characterization": [
                "cyp450_products",
                "phase2_conjugates",
                "parent_compound_fragments"
            ],
            "environmental_analysis": [
                "pesticide_patterns",
                "industrial_contaminants",
                "degradation_products"
            ]
        }
        return patterns.get(self.objective_context, [])
    
    def reduce_noise(self, data):
        """Reduce noise while preserving objective-relevant signals"""
        # Standard noise reduction
        cleaned_data = self._standard_noise_reduction(data)
        
        # Identify and preserve objective-relevant patterns
        preserved_signals = self._identify_preserve_patterns(data)
        
        # Merge preserved signals back
        final_data = self._merge_preserved_signals(
            cleaned_data, preserved_signals
        )
        
        return final_data
```

### Nicotine: Objective-Context Verification

```python
class ObjectiveContextVerifier:
    def verify_analysis_context(self, analysis_results, objective):
        """Verify that analysis maintains objective context"""
        context_checks = {
            "objective_alignment": self._check_objective_alignment(
                analysis_results, objective
            ),
            "biological_coherence": self._check_biological_coherence(
                analysis_results, objective.biological_constraints
            ),
            "statistical_validity": self._check_statistical_validity(
                analysis_results, objective.statistical_requirements
            )
        }
        
        return context_checks
```

### Diggiden: Objective-Specific Adversarial Testing

```python
class ObjectiveAdversarialTester:
    def test_objective_robustness(self, analysis_results, objective):
        """Test robustness of results against objective-specific perturbations"""
        if "biomarker" in objective.target.lower():
            return self._test_biomarker_robustness(analysis_results)
        elif "metabolism" in objective.target.lower():
            return self._test_metabolism_robustness(analysis_results)
        # Add more objective-specific tests...
    
    def _test_biomarker_robustness(self, results):
        """Test biomarker-specific vulnerabilities"""
        perturbations = [
            "batch_effect_simulation",
            "population_drift",
            "instrument_variation",
            "storage_degradation"
        ]
        
        robustness_scores = {}
        for perturbation in perturbations:
            score = self._apply_perturbation_and_test(results, perturbation)
            robustness_scores[perturbation] = score
        
        return robustness_scores
```

## Python-Rust Bridge

### Data Marshaling

The Python-Rust bridge handles data conversion between Buhera's Rust core and Lavoisier's Python modules:

```rust
// Rust side - lavoisier-buhera/src/python_bridge.rs
use pyo3::prelude::*;

#[pyclass]
pub struct BuheraScript {
    #[pyo3(get)]
    pub objective: String,
    #[pyo3(get)]
    pub validation_rules: Vec<String>,
    #[pyo3(get)]
    pub analysis_phases: Vec<String>,
}

#[pyclass]
pub struct ExecutionResult {
    #[pyo3(get)]
    pub success: bool,
    #[pyo3(get)]
    pub confidence: f64,
    #[pyo3(get)]
    pub evidence_scores: std::collections::HashMap<String, f64>,
    #[pyo3(get)]
    pub execution_time: f64,
}

#[pyfunction]
pub fn execute_buhera_script(script_path: String) -> PyResult<ExecutionResult> {
    // Parse script
    let script = parse_buhera_script(&script_path)?;
    
    // Validate script
    let validation_result = validate_script(&script)?;
    if !validation_result.is_valid {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            validation_result.error_message
        ));
    }
    
    // Execute with Python integration
    Python::with_gil(|py| {
        let lavoisier = py.import("lavoisier")?;
        let buhera_integration = lavoisier.getattr("ai_modules")?.getattr("buhera_integration")?;
        
        let result = buhera_integration.call_method1("execute_script", (script,))?;
        
        Ok(ExecutionResult {
            success: result.getattr("success")?.extract()?,
            confidence: result.getattr("confidence")?.extract()?,
            evidence_scores: result.getattr("evidence_scores")?.extract()?,
            execution_time: result.getattr("execution_time")?.extract()?,
        })
    })
}
```

### Python Side Integration

```python
# lavoisier/ai_modules/buhera_integration.py
from typing import Dict, Any, List
import time
from dataclasses import dataclass

@dataclass
class BuheraExecutionResult:
    success: bool
    confidence: float
    evidence_scores: Dict[str, float]
    execution_time: float
    annotations: List[Dict[str, Any]]
    recommendations: List[str]

class BuheraIntegration:
    def __init__(self):
        self.ai_modules = self._initialize_ai_modules()
    
    def _initialize_ai_modules(self):
        """Initialize Lavoisier AI modules with Buhera integration"""
        return {
            "mzekezeke": self._initialize_mzekezeke(),
            "hatata": self._initialize_hatata(),
            "zengeza": self._initialize_zengeza(),
            "nicotine": self._initialize_nicotine(),
            "diggiden": self._initialize_diggiden()
        }
    
    def execute_script(self, script_dict: Dict[str, Any]) -> BuheraExecutionResult:
        """Execute Buhera script with Lavoisier integration"""
        start_time = time.time()
        
        try:
            # Extract objective and create enhanced modules
            objective = BuheraObjective.from_dict(script_dict["objective"])
            enhanced_modules = self._create_objective_aware_modules(objective)
            
            # Execute analysis phases
            results = self._execute_analysis_phases(
                script_dict["phases"], 
                enhanced_modules,
                objective
            )
            
            # Validate results against objective
            validation_results = enhanced_modules["hatata"].validate_with_objective(
                results, objective
            )
            
            execution_time = time.time() - start_time
            
            return BuheraExecutionResult(
                success=validation_results["success"],
                confidence=validation_results["confidence"],
                evidence_scores=results.get("evidence_scores", {}),
                execution_time=execution_time,
                annotations=results.get("annotations", []),
                recommendations=validation_results.get("recommendations", [])
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return BuheraExecutionResult(
                success=False,
                confidence=0.0,
                evidence_scores={},
                execution_time=execution_time,
                annotations=[],
                recommendations=[f"Execution failed: {str(e)}"]
            )
```

## Execution Flow

### 1. Script Parsing and Validation

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Buhera Script  │───▶│  Rust Parser     │───▶│  AST Generation │
│     (.bh)       │    │  (nom-based)     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Validation     │◀───│  Pre-flight      │◀───│  Validation     │
│  Results        │    │  Validation      │    │  Rules          │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 2. Python Integration

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Validated AST  │───▶│  Python Bridge   │───▶│  BuheraIntegra- │
│                 │    │  (PyO3)          │    │  tion Class     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Enhanced AI    │◀───│  Objective-Aware │◀───│  AI Module      │
│  Modules        │    │  Module Creation │    │  Initialization │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 3. Goal-Directed Analysis

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Data Loading   │───▶│  Evidence        │───▶│  Bayesian       │
│  (Objective-    │    │  Collection      │    │  Inference      │
│   Focused)      │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Context        │    │  Objective-      │    │  Success        │
│  Preservation   │    │  Weighted        │    │  Criteria       │
│  (Zengeza)      │    │  Evidence        │    │  Validation     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 4. Result Validation and Reporting

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Analysis       │───▶│  Objective       │───▶│  Success        │
│  Results        │    │  Validation      │    │  Assessment     │
│                 │    │  (Hatata)        │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Robustness     │    │  Context         │    │  Final Report   │
│  Testing        │    │  Verification    │    │  Generation     │
│  (Diggiden)     │    │  (Nicotine)      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Error Handling

### Validation Errors

```python
class BuheraValidationError(Exception):
    def __init__(self, validation_type: str, message: str, suggestions: List[str]):
        self.validation_type = validation_type
        self.message = message
        self.suggestions = suggestions
        super().__init__(f"{validation_type}: {message}")

# Example usage
try:
    result = execute_buhera_script("script.bh")
except BuheraValidationError as e:
    print(f"Validation failed: {e.message}")
    print("Suggestions:")
    for suggestion in e.suggestions:
        print(f"  - {suggestion}")
```

### Runtime Errors

```python
class BuheraRuntimeError(Exception):
    def __init__(self, phase: str, operation: str, error: str):
        self.phase = phase
        self.operation = operation
        self.error = error
        super().__init__(f"Runtime error in {phase}.{operation}: {error}")

# Error recovery mechanism
def execute_with_recovery(script_path: str) -> BuheraExecutionResult:
    try:
        return execute_buhera_script(script_path)
    except BuheraRuntimeError as e:
        # Attempt recovery based on error type
        if "memory" in e.error.lower():
            return execute_with_memory_optimization(script_path)
        elif "timeout" in e.error.lower():
            return execute_with_extended_timeout(script_path)
        else:
            raise
```

## Performance Considerations

### Memory Management

```python
class MemoryEfficientExecution:
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_gb = max_memory_gb
        self.current_memory_usage = 0.0
    
    def execute_with_memory_monitoring(self, script_dict: Dict[str, Any]):
        """Execute script with memory monitoring and optimization"""
        memory_monitor = MemoryMonitor()
        
        try:
            # Check if data can fit in memory
            estimated_memory = self._estimate_memory_usage(script_dict)
            if estimated_memory > self.max_memory_gb:
                return self._execute_chunked_analysis(script_dict)
            else:
                return self._execute_standard_analysis(script_dict)
        finally:
            memory_monitor.cleanup()
```

### Parallel Processing

```python
class ParallelBuheraExecution:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
    
    def execute_parallel_validation(self, script_dict: Dict[str, Any]):
        """Execute validation rules in parallel"""
        validation_tasks = self._create_validation_tasks(script_dict)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            validation_futures = {
                executor.submit(self._execute_validation, task): task
                for task in validation_tasks
            }
            
            results = {}
            for future in as_completed(validation_futures):
                task = validation_futures[future]
                try:
                    results[task.name] = future.result()
                except Exception as e:
                    results[task.name] = ValidationResult(
                        success=False, 
                        error=str(e)
                    )
        
        return results
```

## Extending the Integration

### Adding New AI Modules

```python
class CustomAIModule:
    def __init__(self, objective: BuheraObjective):
        self.objective = objective
        self.module_config = self._configure_for_objective()
    
    def _configure_for_objective(self):
        """Configure module based on objective"""
        # Custom configuration logic
        pass
    
    def process_with_objective(self, data):
        """Process data with objective awareness"""
        # Custom processing logic
        pass

# Register new module
def register_custom_module():
    BuheraIntegration.register_ai_module("custom_module", CustomAIModule)
```

### Custom Validation Rules

```python
class CustomValidationRule:
    def __init__(self, rule_name: str, validation_function):
        self.rule_name = rule_name
        self.validation_function = validation_function
    
    def validate(self, script_dict: Dict[str, Any]) -> ValidationResult:
        """Execute custom validation logic"""
        return self.validation_function(script_dict)

# Register custom validation
def register_custom_validation():
    BuheraValidator.register_validation_rule(
        "check_custom_requirement",
        CustomValidationRule("custom_check", custom_validation_logic)
    )
```

This integration provides a seamless bridge between Buhera's goal-directed scripting and Lavoisier's powerful AI modules, enabling unprecedented precision in mass spectrometry analysis. 
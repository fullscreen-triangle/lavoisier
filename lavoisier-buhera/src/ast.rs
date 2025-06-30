//! Abstract Syntax Tree (AST) for the Buhera Language
//!
//! Defines all language constructs for surgical precision mass spectrometry analysis

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main Buhera script structure
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuheraScript {
    #[pyo3(get)]
    pub objective: BuheraObjective,
    #[pyo3(get)]
    pub validations: Vec<ValidationRule>,
    #[pyo3(get)]
    pub phases: Vec<AnalysisPhase>,
    #[pyo3(get)]
    pub imports: Vec<String>,
    #[pyo3(get)]
    pub metadata: ScriptMetadata,
}

/// Scientific objective declaration
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuheraObjective {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub target: String,
    #[pyo3(get)]
    pub success_criteria: SuccessCriteria,
    #[pyo3(get)]
    pub evidence_priorities: Vec<EvidenceType>,
    #[pyo3(get)]
    pub biological_constraints: Vec<BiologicalConstraint>,
    #[pyo3(get)]
    pub statistical_requirements: StatisticalRequirements,
}

/// Success criteria for the scientific objective
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriteria {
    #[pyo3(get)]
    pub sensitivity: Option<f64>,
    #[pyo3(get)]
    pub specificity: Option<f64>,
    #[pyo3(get)]
    pub pathway_coherence: Option<f64>,
    #[pyo3(get)]
    pub statistical_power: Option<f64>,
    #[pyo3(get)]
    pub custom_metrics: HashMap<String, f64>,
}

/// Types of evidence in mass spectrometry analysis
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EvidenceType {
    MassMatch,
    MS2Fragmentation,
    IsotopePattern,
    RetentionTime,
    PathwayMembership,
    LiteratureSupport,
    SpectralSimilarity,
    AdductFormation,
    NeutralLoss,
}

/// Biological constraints for the analysis
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalConstraint {
    #[pyo3(get)]
    pub constraint_type: String,
    #[pyo3(get)]
    pub value: String,
    #[pyo3(get)]
    pub threshold: Option<f64>,
}

/// Statistical requirements for the experiment
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalRequirements {
    #[pyo3(get)]
    pub minimum_sample_size: Option<u32>,
    #[pyo3(get)]
    pub effect_size: Option<f64>,
    #[pyo3(get)]
    pub alpha_level: Option<f64>,
    #[pyo3(get)]
    pub power_requirement: Option<f64>,
    #[pyo3(get)]
    pub multiple_testing_correction: Option<String>,
}

/// Validation rules for experimental logic
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub rule_type: ValidationType,
    #[pyo3(get)]
    pub condition: ValidationCondition,
    #[pyo3(get)]
    pub action: ValidationAction,
}

/// Types of validation checks
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationType {
    SampleSize,
    InstrumentCapability,
    BiologicalCoherence,
    StatisticalPower,
    DataQuality,
    PathwayConsistency,
}

/// Validation condition
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCondition {
    #[pyo3(get)]
    pub expression: String,
    #[pyo3(get)]
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Action to take when validation fails
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationAction {
    Abort { message: String },
    Warn { message: String },
    Suggest { recommendation: String },
    AutoFix { fix_type: String },
}

/// Analysis phase in the experimental workflow
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisPhase {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub phase_type: PhaseType,
    #[pyo3(get)]
    pub operations: Vec<Operation>,
    #[pyo3(get)]
    pub dependencies: Vec<String>,
}

/// Types of analysis phases
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhaseType {
    DataAcquisition,
    Preprocessing,
    EvidenceBuilding,
    BayesianInference,
    TargetedValidation,
    ResultsSynthesis,
    QualityControl,
}

/// Individual operation within a phase
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub operation_type: OperationType,
    #[pyo3(get)]
    pub parameters: HashMap<String, serde_json::Value>,
    #[pyo3(get)]
    pub input_variables: Vec<String>,
    #[pyo3(get)]
    pub output_variable: Option<String>,
}

/// Types of operations available in Buhera
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    LoadData,
    Preprocess,
    BuildEvidenceNetwork,
    RunBayesianInference,
    ValidateResults,
    OptimizeNoise,
    GenerateAnnotations,
    CalculateMetrics,
    ConditionalBranch,
    IterativeLoop,
    CallLavoisier,
    CallAutobahn,
}

/// Script metadata
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptMetadata {
    #[pyo3(get)]
    pub author: String,
    #[pyo3(get)]
    pub version: String,
    #[pyo3(get)]
    pub description: String,
    #[pyo3(get)]
    pub created_date: String,
    #[pyo3(get)]
    pub last_modified: String,
    #[pyo3(get)]
    pub tags: Vec<String>,
}

/// Conditional expression for control flow
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalExpression {
    #[pyo3(get)]
    pub condition: String,
    #[pyo3(get)]
    pub then_operations: Vec<Operation>,
    #[pyo3(get)]
    pub else_operations: Option<Vec<Operation>>,
}

/// Variable binding in the script
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableBinding {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub value_type: ValueType,
    #[pyo3(get)]
    pub initial_value: Option<serde_json::Value>,
}

/// Value types in Buhera
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValueType {
    Integer,
    Float,
    String,
    Boolean,
    Array,
    Object,
    SpectrumData,
    EvidenceNetwork,
    AnnotationSet,
    ValidationResult,
}

// Python class implementations
#[pymethods]
impl BuheraScript {
    #[new]
    pub fn new(
        objective: BuheraObjective,
        validations: Vec<ValidationRule>,
        phases: Vec<AnalysisPhase>,
        imports: Vec<String>,
        metadata: ScriptMetadata,
    ) -> Self {
        Self {
            objective,
            validations,
            phases,
            imports,
            metadata,
        }
    }

    /// Create a BuheraScript from a file
    pub fn from_file(path: &str) -> Result<Self, crate::errors::BuheraError> {
        // This will be implemented in the parser module
        crate::parser::parse_file(path)
    }

    /// Validate the script's experimental logic
    pub fn validate(
        &self,
    ) -> Result<crate::validator::ValidationResult, crate::errors::BuheraError> {
        let validator = crate::validator::BuheraValidator::new();
        validator.validate(self)
    }

    /// Get all variable names used in the script
    pub fn get_variables(&self) -> Vec<String> {
        let mut variables = Vec::new();
        for phase in &self.phases {
            for operation in &phase.operations {
                variables.extend(operation.input_variables.iter().cloned());
                if let Some(output) = &operation.output_variable {
                    variables.push(output.clone());
                }
            }
        }
        variables.sort();
        variables.dedup();
        variables
    }

    /// Get phase names in execution order
    pub fn get_phase_order(&self) -> Vec<String> {
        self.phases.iter().map(|p| p.name.clone()).collect()
    }

    /// Export script to JSON
    pub fn to_json(&self) -> Result<String, crate::errors::BuheraError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| crate::errors::BuheraError::SerializationError(e.to_string()))
    }

    /// Import script from JSON
    #[staticmethod]
    pub fn from_json(json_str: &str) -> Result<Self, crate::errors::BuheraError> {
        serde_json::from_str(json_str)
            .map_err(|e| crate::errors::BuheraError::SerializationError(e.to_string()))
    }
}

#[pymethods]
impl BuheraObjective {
    #[new]
    pub fn new(
        name: String,
        target: String,
        success_criteria: SuccessCriteria,
        evidence_priorities: Vec<EvidenceType>,
        biological_constraints: Vec<BiologicalConstraint>,
        statistical_requirements: StatisticalRequirements,
    ) -> Self {
        Self {
            name,
            target,
            success_criteria,
            evidence_priorities,
            biological_constraints,
            statistical_requirements,
        }
    }

    /// Check if the objective has minimum required information
    pub fn is_complete(&self) -> bool {
        !self.name.is_empty() && !self.target.is_empty() && !self.evidence_priorities.is_empty()
    }

    /// Get objective summary for logging
    pub fn summary(&self) -> String {
        format!(
            "Objective: {} | Target: {} | Evidence Types: {}",
            self.name,
            self.target,
            self.evidence_priorities.len()
        )
    }
}

#[pymethods]
impl SuccessCriteria {
    #[new]
    pub fn new() -> Self {
        Self {
            sensitivity: None,
            specificity: None,
            pathway_coherence: None,
            statistical_power: None,
            custom_metrics: HashMap::new(),
        }
    }

    /// Check if minimum success criteria are defined
    pub fn has_minimum_criteria(&self) -> bool {
        self.sensitivity.is_some() || self.specificity.is_some() || !self.custom_metrics.is_empty()
    }
}

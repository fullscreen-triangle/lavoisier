//! Abstract Syntax Tree (AST) for the Buhera Language
//!
//! Defines all language constructs for surgical precision mass spectrometry analysis

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main Buhera script structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuheraScript {
    pub objective: BuheraObjective,
    pub validations: Vec<ValidationRule>,
    pub phases: Vec<AnalysisPhase>,
    pub imports: Vec<String>,
    pub metadata: ScriptMetadata,
}

/// Scientific objective declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuheraObjective {
    pub name: String,
    pub target: String,
    pub success_criteria: SuccessCriteria,
    pub evidence_priorities: Vec<EvidenceType>,
    pub biological_constraints: Vec<BiologicalConstraint>,
    pub statistical_requirements: StatisticalRequirements,
}

/// Success criteria for the scientific objective
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriteria {
    pub sensitivity: Option<f64>,
    pub specificity: Option<f64>,
    pub pathway_coherence: Option<f64>,
    pub statistical_power: Option<f64>,
    pub custom_metrics: HashMap<String, f64>,
}

/// Types of evidence in mass spectrometry analysis
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalConstraint {
    pub constraint_type: String,
    pub value: String,
    pub threshold: Option<f64>,
}

/// Statistical requirements for the experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalRequirements {
    pub minimum_sample_size: Option<u32>,
    pub effect_size: Option<f64>,
    pub alpha_level: Option<f64>,
    pub power_requirement: Option<f64>,
    pub multiple_testing_correction: Option<String>,
}

/// Validation rules for experimental logic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub name: String,
    pub rule_type: ValidationType,
    pub condition: ValidationCondition,
    pub action: ValidationAction,
}

/// Types of validation checks
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCondition {
    pub expression: String,
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Action to take when validation fails
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationAction {
    Abort { message: String },
    Warn { message: String },
    Suggest { recommendation: String },
    AutoFix { fix_type: String },
}

/// Analysis phase in the experimental workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisPhase {
    pub name: String,
    pub phase_type: PhaseType,
    pub operations: Vec<Operation>,
    pub dependencies: Vec<String>,
}

/// Types of analysis phases
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    pub name: String,
    pub operation_type: OperationType,
    pub parameters: HashMap<String, serde_json::Value>,
    pub input_variables: Vec<String>,
    pub output_variable: Option<String>,
}

/// Types of operations available in Buhera
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptMetadata {
    pub author: String,
    pub version: String,
    pub description: String,
    pub created_date: String,
    pub last_modified: String,
    pub tags: Vec<String>,
}

/// Conditional expression for control flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalExpression {
    pub condition: String,
    pub then_operations: Vec<Operation>,
    pub else_operations: Option<Vec<Operation>>,
}

/// Variable binding in the script
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableBinding {
    pub name: String,
    pub value_type: ValueType,
    pub initial_value: Option<serde_json::Value>,
}

/// Value types in Buhera
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

// Standard implementations for all structs
impl BuheraScript {
    /// Create a new BuheraScript
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
    pub fn from_json(json_str: &str) -> Result<Self, crate::errors::BuheraError> {
        serde_json::from_str(json_str)
            .map_err(|e| crate::errors::BuheraError::SerializationError(e.to_string()))
    }
}

impl BuheraObjective {
    /// Create a new BuheraObjective
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

impl SuccessCriteria {
    /// Create new SuccessCriteria
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

impl Default for SuccessCriteria {
    fn default() -> Self {
        Self::new()
    }
}

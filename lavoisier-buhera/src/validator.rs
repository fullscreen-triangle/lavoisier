//! Validator for Buhera Scripts
//!
//! Pre-flight validation of experimental logic to catch scientific flaws
//! before execution. This is the core innovation - validating scientific
//! reasoning before wasting time and resources.

use crate::ast::*;
use crate::errors::*;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Validation result for a Buhera script
#[pyclass]
#[derive(Debug, Clone)]
pub struct ValidationResult {
    #[pyo3(get)]
    pub is_valid: bool,
    #[pyo3(get)]
    pub issues: Vec<String>,
    #[pyo3(get)]
    pub recommendations: Vec<String>,
    #[pyo3(get)]
    pub estimated_success_probability: f64,
}

/// Main validator for Buhera scripts
pub struct BuheraValidator {
    instrument_database: HashMap<String, InstrumentCapabilities>,
}

/// Instrument capabilities database
#[derive(Debug, Clone)]
struct InstrumentCapabilities {
    detection_limits: HashMap<String, f64>, // compound -> LOD in ng/mL
    mass_accuracy: f64,                     // ppm
    supports_msms: bool,
}

impl BuheraValidator {
    pub fn new() -> Self {
        Self {
            instrument_database: Self::build_instrument_database(),
        }
    }

    /// Main validation function
    pub fn validate(&self, script: &BuheraScript) -> BuheraResult<ValidationResult> {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        // 1. Validate objective completeness
        self.validate_objective(&script.objective, &mut issues, &mut recommendations)?;

        // 2. Validate instrument capabilities
        self.validate_instrument_capabilities(script, &mut issues, &mut recommendations)?;

        // 3. Validate analysis workflow
        self.validate_workflow(&script.phases, &mut issues, &mut recommendations)?;

        // Calculate success probability
        let success_probability = self.calculate_success_probability(&issues)?;

        let is_valid = issues.is_empty()
            || issues
                .iter()
                .all(|issue| !issue.contains("CRITICAL") && !issue.contains("ERROR"));

        Ok(ValidationResult {
            is_valid,
            issues,
            recommendations,
            estimated_success_probability: success_probability,
        })
    }

    fn validate_objective(
        &self,
        objective: &BuheraObjective,
        issues: &mut Vec<String>,
        recommendations: &mut Vec<String>,
    ) -> BuheraResult<()> {
        if objective.name.is_empty() {
            issues.push("ERROR: Objective name is required".to_string());
        }

        if objective.target.is_empty() {
            issues.push("ERROR: Objective target is required".to_string());
            recommendations.push("Specify what you want to discover or validate".to_string());
        }

        // Check if success criteria are realistic
        if let Some(sensitivity) = objective.success_criteria.sensitivity {
            if sensitivity > 0.99 {
                issues
                    .push("WARNING: Sensitivity target is unrealistically high (>99%)".to_string());
                recommendations.push("Consider lowering sensitivity target to 80-95%".to_string());
            }
        }

        Ok(())
    }

    fn validate_instrument_capabilities(
        &self,
        script: &BuheraScript,
        issues: &mut Vec<String>,
        recommendations: &mut Vec<String>,
    ) -> BuheraResult<()> {
        // Check if MS/MS is required but not available
        if script
            .objective
            .evidence_priorities
            .contains(&EvidenceType::MS2Fragmentation)
        {
            let has_msms = self
                .instrument_database
                .values()
                .any(|caps| caps.supports_msms);

            if !has_msms {
                issues.push(
                    "CRITICAL: MS/MS fragmentation required but no MS/MS instrument available"
                        .to_string(),
                );
                recommendations
                    .push("Use tandem MS instrument or remove MS/MS requirement".to_string());
            }
        }

        Ok(())
    }

    fn validate_workflow(
        &self,
        phases: &[AnalysisPhase],
        issues: &mut Vec<String>,
        recommendations: &mut Vec<String>,
    ) -> BuheraResult<()> {
        // Check for required phases
        let phase_types: Vec<_> = phases.iter().map(|p| &p.phase_type).collect();

        if !phase_types.contains(&&PhaseType::DataAcquisition) {
            issues.push("ERROR: Missing data acquisition phase".to_string());
            recommendations.push("Add data acquisition phase to workflow".to_string());
        }

        if !phase_types.contains(&&PhaseType::BayesianInference) {
            recommendations
                .push("Consider adding Bayesian inference for evidence integration".to_string());
        }

        Ok(())
    }

    fn calculate_success_probability(&self, issues: &[String]) -> BuheraResult<f64> {
        let mut base_probability = 0.8;

        for issue in issues {
            if issue.contains("CRITICAL") {
                base_probability *= 0.1;
            } else if issue.contains("ERROR") {
                base_probability *= 0.5;
            } else if issue.contains("WARNING") {
                base_probability *= 0.8;
            }
        }

        // Fix for f64.max() issue - use conditional logic instead
        let clamped_probability = if base_probability < 0.01 {
            0.01
        } else if base_probability > 0.99 {
            0.99
        } else {
            base_probability
        };

        Ok(clamped_probability)
    }

    fn build_instrument_database() -> HashMap<String, InstrumentCapabilities> {
        let mut db = HashMap::new();

        // Standard LC-MS/MS
        let mut detection_limits = HashMap::new();
        detection_limits.insert("glucose".to_string(), 1e-8);
        detection_limits.insert("lactate".to_string(), 5e-8);

        db.insert(
            "LC-MS/MS".to_string(),
            InstrumentCapabilities {
                detection_limits,
                mass_accuracy: 5.0,
                supports_msms: true,
            },
        );

        db
    }
}

#[pymethods]
impl ValidationResult {
    #[new]
    pub fn new(
        is_valid: bool,
        issues: Vec<String>,
        recommendations: Vec<String>,
        estimated_success_probability: f64,
    ) -> Self {
        Self {
            is_valid,
            issues,
            recommendations,
            estimated_success_probability,
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "Validation: {} | Issues: {} | Success Probability: {:.1}%",
            if self.is_valid { "PASS" } else { "FAIL" },
            self.issues.len(),
            self.estimated_success_probability * 100.0
        )
    }
}

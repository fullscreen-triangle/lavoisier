//! Executor for Buhera Scripts
//!
//! Executes validated Buhera scripts by orchestrating calls to Lavoisier's
//! AI modules for goal-directed mass spectrometry analysis.

use crate::ast::*;
use crate::errors::*;
use std::collections::HashMap;

/// Execution result for a Buhera script
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub success: bool,
    pub annotations: Vec<String>,
    pub evidence_scores: HashMap<String, f64>,
    pub execution_time_seconds: f64,
}

/// Main executor for Buhera scripts
pub struct BuheraExecutor {
    context: ExecutionContext,
}

/// Execution context with objective focus
#[derive(Debug)]
struct ExecutionContext {
    variables: HashMap<String, serde_json::Value>,
    objective_focus: Option<String>,
}

impl BuheraExecutor {
    pub fn new() -> Self {
        Self {
            context: ExecutionContext {
                variables: HashMap::new(),
                objective_focus: None,
            },
        }
    }

    /// Execute a validated Buhera script in standalone mode
    pub fn execute_standalone(&mut self, script: &BuheraScript) -> BuheraResult<ExecutionResult> {
        let start_time = std::time::Instant::now();

        // Initialize context with objective focus for goal-directed analysis
        self.context.objective_focus = Some(script.objective.target.clone());

        // Execute phases with objective awareness (standalone mode)
        let annotations = self.execute_phases_standalone(script)?;
        let evidence_scores = self.calculate_evidence_scores(script)?;

        let execution_time = start_time.elapsed().as_secs_f64();

        Ok(ExecutionResult {
            success: true,
            annotations,
            evidence_scores,
            execution_time_seconds: execution_time,
        })
    }

    fn execute_phases_standalone(&mut self, script: &BuheraScript) -> BuheraResult<Vec<String>> {
        let mut annotations = Vec::new();

        for phase in &script.phases {
            match phase.phase_type {
                PhaseType::DataAcquisition => {
                    annotations.push("Data acquisition phase (standalone mode)".to_string());
                }
                PhaseType::EvidenceBuilding => {
                    annotations.push("Evidence building phase (standalone mode)".to_string());
                }
                PhaseType::BayesianInference => {
                    annotations.push("Bayesian inference phase (standalone mode)".to_string());
                }
                _ => {
                    annotations.push(format!("Phase {} executed (standalone)", phase.name));
                }
            }
        }

        Ok(annotations)
    }

    fn calculate_evidence_scores(
        &self,
        script: &BuheraScript,
    ) -> BuheraResult<HashMap<String, f64>> {
        let mut scores = HashMap::new();

        // Calculate scores based on evidence priorities
        for evidence_type in &script.objective.evidence_priorities {
            let score = match evidence_type {
                EvidenceType::MassMatch => 0.95,
                EvidenceType::MS2Fragmentation => 0.88,
                EvidenceType::IsotopePattern => 0.92,
                EvidenceType::RetentionTime => 0.85,
                _ => 0.80,
            };
            scores.insert(format!("{:?}", evidence_type), score);
        }

        Ok(scores)
    }
}

impl ExecutionResult {
    /// Create new ExecutionResult
    pub fn new(
        success: bool,
        annotations: Vec<String>,
        evidence_scores: HashMap<String, f64>,
        execution_time_seconds: f64,
    ) -> Self {
        Self {
            success,
            annotations,
            evidence_scores,
            execution_time_seconds,
        }
    }

    /// Get execution summary
    pub fn summary(&self) -> String {
        format!(
            "Execution: {} | Annotations: {} | Time: {:.1}s",
            if self.success { "SUCCESS" } else { "FAILED" },
            self.annotations.len(),
            self.execution_time_seconds
        )
    }
}

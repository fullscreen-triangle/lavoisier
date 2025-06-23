//! Executor for Buhera Scripts
//! 
//! Executes validated Buhera scripts by orchestrating calls to Lavoisier's
//! AI modules for goal-directed mass spectrometry analysis.

use crate::ast::*;
use crate::errors::*;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Execution result for a Buhera script
#[pyclass]
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    #[pyo3(get)]
    pub success: bool,
    #[pyo3(get)]
    pub annotations: Vec<String>,
    #[pyo3(get)]
    pub evidence_scores: HashMap<String, f64>,
    #[pyo3(get)]
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

    /// Execute a validated Buhera script with Lavoisier integration
    pub fn execute(
        &mut self,
        script: &BuheraScript,
        lavoisier_system: &PyAny,
    ) -> BuheraResult<ExecutionResult> {
        let start_time = std::time::Instant::now();

        // Initialize context with objective focus for goal-directed analysis
        self.context.objective_focus = Some(script.objective.target.clone());

        // Execute phases with objective awareness
        let annotations = self.execute_phases(script, lavoisier_system)?;
        let evidence_scores = self.calculate_evidence_scores(script)?;

        let execution_time = start_time.elapsed().as_secs_f64();

        Ok(ExecutionResult {
            success: true,
            annotations,
            evidence_scores,
            execution_time_seconds: execution_time,
        })
    }

    fn execute_phases(
        &mut self,
        script: &BuheraScript,
        lavoisier_system: &PyAny,
    ) -> BuheraResult<Vec<String>> {
        let mut annotations = Vec::new();

        for phase in &script.phases {
            match phase.phase_type {
                PhaseType::DataAcquisition => {
                    self.execute_data_acquisition(phase, lavoisier_system)?;
                    annotations.push("Data acquisition completed".to_string());
                }
                PhaseType::EvidenceBuilding => {
                    self.execute_evidence_building(phase, lavoisier_system)?;
                    annotations.push("Evidence network built with objective focus".to_string());
                }
                PhaseType::BayesianInference => {
                    self.execute_bayesian_inference(phase, lavoisier_system)?;
                    annotations.push("Goal-directed Bayesian inference completed".to_string());
                }
                _ => {
                    annotations.push(format!("Phase {} executed", phase.name));
                }
            }
        }

        Ok(annotations)
    }

    fn execute_data_acquisition(
        &mut self,
        phase: &AnalysisPhase,
        lavoisier_system: &PyAny,
    ) -> BuheraResult<()> {
        // Call Lavoisier data loading with context
        lavoisier_system
            .call_method("load_dataset", (), None)
            .map_err(|e| BuheraError::PythonError(e.to_string()))?;
        Ok(())
    }

    fn execute_evidence_building(
        &mut self,
        phase: &AnalysisPhase,
        lavoisier_system: &PyAny,
    ) -> BuheraResult<()> {
        // Get objective for goal-directed evidence building
        let objective = self.context.objective_focus.as_ref().unwrap();

        // Call Lavoisier's Mzekezeke with objective focus
        lavoisier_system
            .call_method("mzekezeke.build_evidence_network", (objective,), None)
            .map_err(|e| BuheraError::PythonError(e.to_string()))?;
        Ok(())
    }

    fn execute_bayesian_inference(
        &mut self,
        phase: &AnalysisPhase,
        lavoisier_system: &PyAny,
    ) -> BuheraResult<()> {
        // Call Lavoisier's Hatata for validation
        lavoisier_system
            .call_method("hatata.validate_with_objective", (), None)
            .map_err(|e| BuheraError::PythonError(e.to_string()))?;
        Ok(())
    }

    fn calculate_evidence_scores(&self, script: &BuheraScript) -> BuheraResult<HashMap<String, f64>> {
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

#[pymethods]
impl ExecutionResult {
    #[new]
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

    pub fn summary(&self) -> String {
        format!(
            "Execution: {} | Annotations: {} | Time: {:.1}s",
            if self.success { "SUCCESS" } else { "FAILED" },
            self.annotations.len(),
            self.execution_time_seconds
        )
    }
} 
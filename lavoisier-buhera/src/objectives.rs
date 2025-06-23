//! Scientific Objectives Management
//! 
//! Manages scientific objectives and their validation for goal-directed analysis.

use crate::ast::*;
use crate::errors::*;
use std::collections::HashMap;

/// Objective manager for scientific goals
pub struct ObjectiveManager {
    templates: HashMap<String, BuheraObjective>,
}

impl ObjectiveManager {
    pub fn new() -> Self {
        Self {
            templates: Self::build_objective_templates(),
        }
    }

    /// Get objective template by name
    pub fn get_template(&self, name: &str) -> Option<&BuheraObjective> {
        self.templates.get(name)
    }

    /// Validate objective feasibility
    pub fn validate_objective(&self, objective: &BuheraObjective) -> BuheraResult<()> {
        if objective.name.is_empty() {
            return Err(BuheraError::ValidationError("Objective name required".to_string()));
        }

        if objective.evidence_priorities.is_empty() {
            return Err(BuheraError::ValidationError("Evidence priorities required".to_string()));
        }

        Ok(())
    }

    /// Build common objective templates
    fn build_objective_templates() -> HashMap<String, BuheraObjective> {
        let mut templates = HashMap::new();

        // Biomarker Discovery Template
        let biomarker_discovery = BuheraObjective {
            name: "BiomarkerDiscovery".to_string(),
            target: "identify potential biomarkers".to_string(),
            success_criteria: SuccessCriteria {
                sensitivity: Some(0.85),
                specificity: Some(0.85),
                pathway_coherence: Some(0.7),
                statistical_power: Some(0.8),
                custom_metrics: HashMap::new(),
            },
            evidence_priorities: vec![
                EvidenceType::MassMatch,
                EvidenceType::MS2Fragmentation,
                EvidenceType::PathwayMembership,
            ],
            biological_constraints: vec![],
            statistical_requirements: StatisticalRequirements {
                minimum_sample_size: Some(30),
                effect_size: Some(0.5),
                alpha_level: Some(0.05),
                power_requirement: Some(0.8),
                multiple_testing_correction: Some("fdr".to_string()),
            },
        };

        templates.insert("biomarker_discovery".to_string(), biomarker_discovery);
        templates
    }
} 
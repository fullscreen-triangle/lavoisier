//! Evidence Network for Goal-Directed Analysis
//! 
//! Implements Bayesian evidence networks that are optimized for specific
//! scientific objectives, providing surgical precision in analysis.

use crate::ast::*;
use crate::errors::*;
use std::collections::HashMap;

/// Evidence network optimized for specific objectives
pub struct EvidenceNetwork {
    pub objective: String,
    pub evidence_weights: HashMap<EvidenceType, f64>,
    pub current_scores: HashMap<String, f64>,
    pub confidence_threshold: f64,
}

impl EvidenceNetwork {
    /// Create new evidence network for objective
    pub fn new(objective: &BuheraObjective) -> BuheraResult<Self> {
        let mut evidence_weights = HashMap::new();
        
        // Set weights based on evidence priorities and objective type
        for (i, evidence_type) in objective.evidence_priorities.iter().enumerate() {
            let base_weight = 1.0 - (i as f64 * 0.1);
            let objective_weight = Self::get_objective_weight(&objective.target, evidence_type);
            evidence_weights.insert(evidence_type.clone(), base_weight * objective_weight);
        }

        Ok(Self {
            objective: objective.target.clone(),
            evidence_weights,
            current_scores: HashMap::new(),
            confidence_threshold: 0.8,
        })
    }

    /// Update evidence scores
    pub fn update_evidence(&mut self, evidence_type: &EvidenceType, score: f64) {
        let weighted_score = score * self.evidence_weights.get(evidence_type).unwrap_or(&1.0);
        self.current_scores.insert(format!("{:?}", evidence_type), weighted_score);
    }

    /// Calculate overall confidence
    pub fn calculate_confidence(&self) -> f64 {
        if self.current_scores.is_empty() {
            return 0.0;
        }

        let total_score: f64 = self.current_scores.values().sum();
        let max_possible: f64 = self.evidence_weights.values().sum();
        
        if max_possible > 0.0 {
            total_score / max_possible
        } else {
            0.0
        }
    }

    /// Check if confidence meets threshold
    pub fn meets_threshold(&self) -> bool {
        self.calculate_confidence() >= self.confidence_threshold
    }

    /// Get objective-specific weight for evidence type
    fn get_objective_weight(objective: &str, evidence_type: &EvidenceType) -> f64 {
        // Adjust weights based on objective type
        if objective.contains("biomarker") {
            match evidence_type {
                EvidenceType::PathwayMembership => 1.2,
                EvidenceType::MS2Fragmentation => 1.1,
                EvidenceType::MassMatch => 1.0,
                _ => 0.8,
            }
        } else if objective.contains("quantification") {
            match evidence_type {
                EvidenceType::IsotopePattern => 1.2,
                EvidenceType::RetentionTime => 1.1,
                EvidenceType::MassMatch => 1.0,
                _ => 0.8,
            }
        } else {
            1.0 // Default weight
        }
    }

    /// Generate evidence summary
    pub fn summary(&self) -> String {
        format!(
            "Evidence Network: {} | Confidence: {:.1}% | Threshold: {:.1}%",
            self.objective,
            self.calculate_confidence() * 100.0,
            self.confidence_threshold * 100.0
        )
    }
} 
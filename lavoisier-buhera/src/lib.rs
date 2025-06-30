//! # Buhera: Surgical Precision Scripting Language for Mass Spectrometry
//!
//! Buhera enables scientists to encode experimental reasoning as executable scripts,
//! providing "surgical precision" for mass spectrometry analysis with goal-directed
//! Bayesian evidence networks.
//!
//! ## Key Features
//! - **Objective-First Design**: Every script must declare explicit scientific objectives
//! - **Pre-Flight Validation**: Catch experimental flaws before execution
//! - **Goal-Aware Bayesian Networks**: Evidence networks optimized for specific objectives
//! - **Surgical Analysis Precision**: Target specific research questions with precision
//! - **Python Integration**: Seamless integration with Lavoisier's AI modules

pub mod ast;
pub mod errors;
pub mod executor;
pub mod parser;
pub mod validator;

pub use ast::*;
pub use errors::*;
pub use executor::*;
pub use parser::*;
pub use validator::*;

/// Version information for the Buhera language
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_basic_functionality() {
        // Test that basic structures can be created
        let success_criteria = SuccessCriteria::new();
        assert!(!success_criteria.has_minimum_criteria());

        let objective = BuheraObjective::new(
            "Test Objective".to_string(),
            "Find glucose".to_string(),
            success_criteria,
            vec![EvidenceType::MassMatch],
            vec![],
            StatisticalRequirements {
                minimum_sample_size: Some(10),
                effect_size: None,
                alpha_level: Some(0.05),
                power_requirement: Some(0.8),
                multiple_testing_correction: None,
            },
        );

        assert!(objective.is_complete());
        assert!(objective.summary().contains("Test Objective"));
    }
}

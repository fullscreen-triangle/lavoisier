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

use pyo3::prelude::*;

pub use ast::*;
pub use errors::*;
pub use executor::*;
pub use parser::*;
pub use validator::*;

/// Version information for the Buhera language
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Main entry point for Python integration
#[pymodule]
fn buhera(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BuheraScript>()?;
    m.add_class::<BuheraObjective>()?;
    m.add_class::<ValidationResult>()?;
    m.add_class::<ExecutionResult>()?;

    // Add version info
    m.add("__version__", VERSION)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}

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
pub mod parser;
pub mod validator;
pub mod executor;
pub mod python_bridge;
pub mod errors;
pub mod objectives;
pub mod evidence_network;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub use ast::*;
pub use parser::*;
pub use validator::*;
pub use executor::*;
pub use errors::*;
pub use objectives::*;
pub use evidence_network::*;

/// Version information for the Buhera language
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Main entry point for Python integration
#[pymodule]
fn buhera(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_buhera_script, m)?)?;
    m.add_function(wrap_pyfunction!(validate_buhera_script, m)?)?;
    m.add_function(wrap_pyfunction!(execute_buhera_script, m)?)?;
    m.add_class::<BuheraScript>()?;
    m.add_class::<BuheraObjective>()?;
    m.add_class::<ValidationResult>()?;
    m.add_class::<ExecutionResult>()?;
    
    // Add version info
    m.add("__version__", VERSION)?;
    
    Ok(())
}

/// Parse a Buhera script from file path
#[pyfunction]
pub fn parse_buhera_script(script_path: String) -> PyResult<BuheraScript> {
    let script = BuheraScript::from_file(&script_path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Parse error: {}", e)))?;
    Ok(script)
}

/// Validate a Buhera script for experimental logic
#[pyfunction]
pub fn validate_buhera_script(script: &BuheraScript) -> PyResult<ValidationResult> {
    let validator = BuheraValidator::new();
    let result = validator.validate(script)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Validation error: {}", e)))?;
    Ok(result)
}

/// Execute a Buhera script with Lavoisier integration
#[pyfunction]
pub fn execute_buhera_script(
    script: &BuheraScript,
    lavoisier_system: &PyAny,
) -> PyResult<ExecutionResult> {
    let executor = BuheraExecutor::new();
    let result = executor.execute(script, lavoisier_system)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Execution error: {}", e)))?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
} 
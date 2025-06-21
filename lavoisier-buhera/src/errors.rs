//! Error handling for the Buhera language
//! 
//! Provides comprehensive error types and handling for parsing, validation,
//! and execution of Buhera scripts.

use thiserror::Error;
use pyo3::prelude::*;

/// Main error type for Buhera operations
#[derive(Error, Debug)]
pub enum BuheraError {
    #[error("Parse error: {0}")]
    ParseError(String),
    
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    #[error("Execution error: {0}")]
    ExecutionError(String),
    
    #[error("Python integration error: {0}")]
    PythonError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("File I/O error: {0}")]
    IoError(String),
    
    #[error("Scientific logic error: {0}")]
    ScientificLogicError(String),
    
    #[error("Statistical validation error: {0}")]
    StatisticalError(String),
    
    #[error("Instrument capability error: {0}")]
    InstrumentError(String),
    
    #[error("Biological constraint violation: {0}")]
    BiologicalError(String),
}

impl From<std::io::Error> for BuheraError {
    fn from(err: std::io::Error) -> Self {
        BuheraError::IoError(err.to_string())
    }
}

impl From<serde_json::Error> for BuheraError {
    fn from(err: serde_json::Error) -> Self {
        BuheraError::SerializationError(err.to_string())
    }
}

impl From<pyo3::PyErr> for BuheraError {
    fn from(err: pyo3::PyErr) -> Self {
        BuheraError::PythonError(err.to_string())
    }
}

/// Convert BuheraError to Python exception
impl From<BuheraError> for PyErr {
    fn from(err: BuheraError) -> Self {
        match err {
            BuheraError::ParseError(msg) => pyo3::exceptions::PyValueError::new_err(msg),
            BuheraError::ValidationError(msg) => pyo3::exceptions::PyValueError::new_err(msg),
            BuheraError::ExecutionError(msg) => pyo3::exceptions::PyRuntimeError::new_err(msg),
            BuheraError::PythonError(msg) => pyo3::exceptions::PyRuntimeError::new_err(msg),
            BuheraError::SerializationError(msg) => pyo3::exceptions::PyValueError::new_err(msg),
            BuheraError::IoError(msg) => pyo3::exceptions::PyIOError::new_err(msg),
            BuheraError::ScientificLogicError(msg) => pyo3::exceptions::PyValueError::new_err(msg),
            BuheraError::StatisticalError(msg) => pyo3::exceptions::PyValueError::new_err(msg),
            BuheraError::InstrumentError(msg) => pyo3::exceptions::PyValueError::new_err(msg),
            BuheraError::BiologicalError(msg) => pyo3::exceptions::PyValueError::new_err(msg),
        }
    }
}

/// Result type for Buhera operations
pub type BuheraResult<T> = Result<T, BuheraError>;

/// Validation issue severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum Severity {
    Info,
    Warning, 
    Error,
    Critical,
}

/// Detailed validation issue
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub severity: Severity,
    pub message: String,
    pub suggestion: Option<String>,
    pub line_number: Option<usize>,
    pub column_number: Option<usize>,
    pub error_code: Option<String>,
}

impl ValidationIssue {
    pub fn new(severity: Severity, message: String) -> Self {
        Self {
            severity,
            message,
            suggestion: None,
            line_number: None,
            column_number: None,
            error_code: None,
        }
    }
    
    pub fn with_suggestion(mut self, suggestion: String) -> Self {
        self.suggestion = Some(suggestion);
        self
    }
    
    pub fn with_location(mut self, line: usize, column: usize) -> Self {
        self.line_number = Some(line);
        self.column_number = Some(column);
        self
    }
    
    pub fn with_code(mut self, code: String) -> Self {
        self.error_code = Some(code);
        self
    }
}

/// Scientific validation errors
#[derive(Debug, Clone)]
pub enum ScientificError {
    InsufficientSampleSize {
        required: u32,
        actual: u32,
        power: f64,
    },
    InstrumentLimitation {
        target_concentration: f64,
        detection_limit: f64,
        instrument_type: String,
    },
    BiologicalInconsistency {
        pathway: String,
        contradictory_conditions: Vec<String>,
        expected_direction: String,
        actual_direction: String,
    },
    StatisticalPowerInsufficient {
        current_power: f64,
        required_power: f64,
        recommended_sample_size: u32,
    },
}

impl std::fmt::Display for ScientificError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScientificError::InsufficientSampleSize { required, actual, power } => {
                write!(f, "Insufficient sample size: need {} samples for 80% power, have {} (current power: {:.1f}%)", 
                       required, actual, power * 100.0)
            }
            ScientificError::InstrumentLimitation { target_concentration, detection_limit, instrument_type } => {
                write!(f, "Instrument limitation: {} cannot detect {:.2e} concentration (LOD: {:.2e})", 
                       instrument_type, target_concentration, detection_limit)
            }
            ScientificError::BiologicalInconsistency { pathway, contradictory_conditions, expected_direction, actual_direction } => {
                write!(f, "Biological inconsistency in {} pathway: expected {}, but conditions suggest {} (contradictory: {:?})", 
                       pathway, expected_direction, actual_direction, contradictory_conditions)
            }
            ScientificError::StatisticalPowerInsufficient { current_power, required_power, recommended_sample_size } => {
                write!(f, "Statistical power insufficient: {:.1f}% (need {:.1f}%) - recommend {} samples", 
                       current_power * 100.0, required_power * 100.0, recommended_sample_size)
            }
        }
    }
} 
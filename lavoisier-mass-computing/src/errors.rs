//! Error types for Mass Computing

use thiserror::Error;

/// Result type alias for Mass Computing operations
pub type Result<T> = std::result::Result<T, MassComputingError>;

/// Errors that can occur during mass computing operations
#[derive(Error, Debug, Clone)]
pub enum MassComputingError {
    #[error("Invalid trit value: {0}. Must be 0, 1, or 2")]
    InvalidTrit(u8),

    #[error("Invalid ternary address: {0}")]
    InvalidAddress(String),

    #[error("Invalid S-coordinate {coord}: {value}. Must be in [0, 1]")]
    InvalidSCoord { coord: &'static str, value: f64 },

    #[error("Address depth {0} is insufficient. Minimum required: {1}")]
    InsufficientDepth(usize, usize),

    #[error("Fragment position {0} out of bounds for address of length {1}")]
    FragmentOutOfBounds(usize, usize),

    #[error("MassScript parse error at line {line}: {message}")]
    ParseError { line: usize, message: String },

    #[error("MassScript execution error: {0}")]
    ExecutionError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

impl From<serde_json::Error> for MassComputingError {
    fn from(e: serde_json::Error) -> Self {
        MassComputingError::SerializationError(e.to_string())
    }
}

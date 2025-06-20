use thiserror::Error;

/// Core errors for Lavoisier operations
#[derive(Error, Debug)]
pub enum LavoisierError {
    #[error("Invalid spectrum data: {0}")]
    InvalidSpectrum(String),
    
    #[error("Peak detection failed: {0}")]
    PeakDetection(String),
    
    #[error("Processing error: {0}")]
    Processing(String),
    
    #[error("Memory allocation error: {0}")]
    Memory(String),
    
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("Thread pool error: {0}")]
    ThreadPool(String),
    
    #[error("Index out of bounds: {0}")]
    IndexOutOfBounds(String),
}

/// Result type alias for Lavoisier operations
pub type LavoisierResult<T> = Result<T, LavoisierError>;

/// Error context for debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub operation: String,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub additional_info: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    pub fn new(operation: &str) -> Self {
        Self {
            operation: operation.to_string(),
            file: None,
            line: None,
            additional_info: std::collections::HashMap::new(),
        }
    }

    pub fn with_file(mut self, file: &str) -> Self {
        self.file = Some(file.to_string());
        self
    }

    pub fn with_line(mut self, line: u32) -> Self {
        self.line = Some(line);
        self
    }

    pub fn with_info(mut self, key: &str, value: &str) -> Self {
        self.additional_info.insert(key.to_string(), value.to_string());
        self
    }
}

/// Macro for creating errors with context
#[macro_export]
macro_rules! lavoisier_error {
    ($error_type:ident, $msg:expr) => {
        LavoisierError::$error_type($msg.to_string())
    };
    ($error_type:ident, $fmt:expr, $($arg:tt)*) => {
        LavoisierError::$error_type(format!($fmt, $($arg)*))
    };
}

/// Macro for creating results with context
#[macro_export]
macro_rules! lavoisier_bail {
    ($error_type:ident, $msg:expr) => {
        return Err(lavoisier_error!($error_type, $msg));
    };
    ($error_type:ident, $fmt:expr, $($arg:tt)*) => {
        return Err(lavoisier_error!($error_type, $fmt, $($arg)*));
    };
}

/// Error recovery strategies
pub enum RecoveryStrategy {
    Retry(usize),      // Retry n times
    Skip,              // Skip the problematic item
    UseDefault,        // Use a default value
    Abort,             // Abort the operation
}

/// Error handler trait for custom error handling
pub trait ErrorHandler {
    fn handle_error(&self, error: &LavoisierError, context: &ErrorContext) -> RecoveryStrategy;
}

/// Default error handler
pub struct DefaultErrorHandler;

impl ErrorHandler for DefaultErrorHandler {
    fn handle_error(&self, error: &LavoisierError, _context: &ErrorContext) -> RecoveryStrategy {
        match error {
            LavoisierError::InvalidSpectrum(_) => RecoveryStrategy::Skip,
            LavoisierError::PeakDetection(_) => RecoveryStrategy::UseDefault,
            LavoisierError::Memory(_) => RecoveryStrategy::Abort,
            LavoisierError::Io(_) => RecoveryStrategy::Retry(3),
            _ => RecoveryStrategy::Abort,
        }
    }
}

/// Error aggregator for batch operations
pub struct ErrorAggregator {
    errors: Vec<(LavoisierError, ErrorContext)>,
    max_errors: usize,
}

impl ErrorAggregator {
    pub fn new(max_errors: usize) -> Self {
        Self {
            errors: Vec::new(),
            max_errors,
        }
    }

    pub fn add_error(&mut self, error: LavoisierError, context: ErrorContext) -> Result<(), LavoisierError> {
        self.errors.push((error, context));
        
        if self.errors.len() >= self.max_errors {
            Err(LavoisierError::Processing(
                format!("Too many errors encountered: {}", self.errors.len())
            ))
        } else {
            Ok(())
        }
    }

    pub fn errors(&self) -> &[(LavoisierError, ErrorContext)] {
        &self.errors
    }

    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn clear(&mut self) {
        self.errors.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = lavoisier_error!(InvalidSpectrum, "Test error message");
        match error {
            LavoisierError::InvalidSpectrum(msg) => assert_eq!(msg, "Test error message"),
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("test_operation")
            .with_file("test.rs")
            .with_line(42)
            .with_info("spectrum_id", "scan_123");
        
        assert_eq!(context.operation, "test_operation");
        assert_eq!(context.file, Some("test.rs".to_string()));
        assert_eq!(context.line, Some(42));
        assert_eq!(context.additional_info.get("spectrum_id"), Some(&"scan_123".to_string()));
    }

    #[test]
    fn test_error_aggregator() {
        let mut aggregator = ErrorAggregator::new(2);
        
        let error1 = LavoisierError::InvalidSpectrum("Error 1".to_string());
        let context1 = ErrorContext::new("operation1");
        
        assert!(aggregator.add_error(error1, context1).is_ok());
        assert_eq!(aggregator.error_count(), 1);
        
        let error2 = LavoisierError::InvalidSpectrum("Error 2".to_string());
        let context2 = ErrorContext::new("operation2");
        
        assert!(aggregator.add_error(error2, context2).is_err()); // Should exceed max
    }

    #[test]
    fn test_default_error_handler() {
        let handler = DefaultErrorHandler;
        let context = ErrorContext::new("test");
        
        let invalid_spectrum_error = LavoisierError::InvalidSpectrum("test".to_string());
        match handler.handle_error(&invalid_spectrum_error, &context) {
            RecoveryStrategy::Skip => (), // Expected
            _ => panic!("Wrong recovery strategy"),
        }
        
        let memory_error = LavoisierError::Memory("test".to_string());
        match handler.handle_error(&memory_error, &context) {
            RecoveryStrategy::Abort => (), // Expected
            _ => panic!("Wrong recovery strategy"),
        }
    }
} 
//! MassScript: Domain-specific language for partition synthesis
//!
//! MassScript enables virtual mass spectrometry experiments expressed
//! as ternary string operations. Programs are sequences of commands
//! that manipulate addresses and extract observables.

use crate::errors::{MassComputingError, Result};
use crate::extractor::{SpectrumExtractor, SynthesizedSpectrum};
use crate::ternary::TernaryAddress;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// MassScript command types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MassScriptCommand {
    /// Set the current address: `partition [name] <address>`
    Partition {
        name: Option<String>,
        address: String,
    },

    /// Extend the current address: `extend by <trits>`
    Extend { trits: String },

    /// Observe (extract) observables: `observe [name]`
    Observe { name: Option<String> },

    /// Fragment at position: `fragment [name] at <position>`
    Fragment { name: Option<String>, position: usize },

    /// Inject sample: `inject [name] as <address>`
    Inject {
        name: Option<String>,
        address: String,
    },

    /// Chromatograph: `chromatograph extend by <trits>`
    Chromatograph { trits: String },

    /// Ionize: `ionize extend by <trits>`
    Ionize { trits: String },

    /// Detect: `detect`
    Detect,

    /// Comment (ignored)
    Comment(String),

    /// Empty line (ignored)
    Empty,
}

/// A MassScript program
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MassScript {
    /// The commands in the program
    pub commands: Vec<MassScriptCommand>,
    /// Source lines for error reporting
    pub source_lines: Vec<String>,
}

impl MassScript {
    /// Parse MassScript source code
    pub fn parse(source: &str) -> Result<Self> {
        let mut commands = Vec::new();
        let source_lines: Vec<String> = source.lines().map(String::from).collect();

        for (line_num, line) in source_lines.iter().enumerate() {
            let command = Self::parse_line(line, line_num + 1)?;
            commands.push(command);
        }

        Ok(Self {
            commands,
            source_lines,
        })
    }

    fn parse_line(line: &str, line_num: usize) -> Result<MassScriptCommand> {
        // Remove comments and trim
        let line = line.split('#').next().unwrap_or("").trim();

        if line.is_empty() {
            return Ok(MassScriptCommand::Empty);
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            return Ok(MassScriptCommand::Empty);
        }

        match tokens[0].to_lowercase().as_str() {
            "partition" => Self::parse_partition(&tokens, line_num),
            "extend" => Self::parse_extend(&tokens, line_num),
            "observe" => Self::parse_observe(&tokens),
            "fragment" => Self::parse_fragment(&tokens, line_num),
            "inject" => Self::parse_inject(&tokens, line_num),
            "chromatograph" => Self::parse_chromatograph(&tokens, line_num),
            "ionize" => Self::parse_ionize(&tokens, line_num),
            "detect" => Ok(MassScriptCommand::Detect),
            _ => Err(MassComputingError::ParseError {
                line: line_num,
                message: format!("Unknown command: {}", tokens[0]),
            }),
        }
    }

    fn parse_partition(tokens: &[&str], line_num: usize) -> Result<MassScriptCommand> {
        match tokens.len() {
            2 => Ok(MassScriptCommand::Partition {
                name: None,
                address: tokens[1].to_string(),
            }),
            3 => Ok(MassScriptCommand::Partition {
                name: Some(tokens[1].to_string()),
                address: tokens[2].to_string(),
            }),
            _ => Err(MassComputingError::ParseError {
                line: line_num,
                message: "partition requires 1 or 2 arguments".to_string(),
            }),
        }
    }

    fn parse_extend(tokens: &[&str], line_num: usize) -> Result<MassScriptCommand> {
        // extend by <trits>
        if tokens.len() >= 3 && tokens[1].to_lowercase() == "by" {
            Ok(MassScriptCommand::Extend {
                trits: tokens[2].to_string(),
            })
        } else {
            Err(MassComputingError::ParseError {
                line: line_num,
                message: "extend requires 'by <trits>'".to_string(),
            })
        }
    }

    fn parse_observe(tokens: &[&str]) -> Result<MassScriptCommand> {
        Ok(MassScriptCommand::Observe {
            name: tokens.get(1).map(|s| s.to_string()),
        })
    }

    fn parse_fragment(tokens: &[&str], line_num: usize) -> Result<MassScriptCommand> {
        // fragment [name] at <position>
        let at_idx = tokens.iter().position(|&t| t.to_lowercase() == "at");

        match at_idx {
            Some(idx) if idx + 1 < tokens.len() => {
                let position: usize = tokens[idx + 1].parse().map_err(|_| {
                    MassComputingError::ParseError {
                        line: line_num,
                        message: "fragment position must be a number".to_string(),
                    }
                })?;

                let name = if idx > 1 {
                    Some(tokens[1].to_string())
                } else {
                    None
                };

                Ok(MassScriptCommand::Fragment { name, position })
            }
            _ => Err(MassComputingError::ParseError {
                line: line_num,
                message: "fragment requires 'at <position>'".to_string(),
            }),
        }
    }

    fn parse_inject(tokens: &[&str], line_num: usize) -> Result<MassScriptCommand> {
        // inject [name] as <address>
        let as_idx = tokens.iter().position(|&t| t.to_lowercase() == "as");

        match as_idx {
            Some(idx) if idx + 1 < tokens.len() => {
                let name = if idx > 1 {
                    Some(tokens[1].to_string())
                } else {
                    None
                };

                Ok(MassScriptCommand::Inject {
                    name,
                    address: tokens[idx + 1].to_string(),
                })
            }
            _ => Err(MassComputingError::ParseError {
                line: line_num,
                message: "inject requires 'as <address>'".to_string(),
            }),
        }
    }

    fn parse_chromatograph(tokens: &[&str], line_num: usize) -> Result<MassScriptCommand> {
        // chromatograph extend by <trits>
        if tokens.len() >= 4 && tokens[1].to_lowercase() == "extend" && tokens[2].to_lowercase() == "by" {
            Ok(MassScriptCommand::Chromatograph {
                trits: tokens[3].to_string(),
            })
        } else {
            Err(MassComputingError::ParseError {
                line: line_num,
                message: "chromatograph requires 'extend by <trits>'".to_string(),
            })
        }
    }

    fn parse_ionize(tokens: &[&str], line_num: usize) -> Result<MassScriptCommand> {
        // ionize extend by <trits>
        if tokens.len() >= 4 && tokens[1].to_lowercase() == "extend" && tokens[2].to_lowercase() == "by" {
            Ok(MassScriptCommand::Ionize {
                trits: tokens[3].to_string(),
            })
        } else {
            Err(MassComputingError::ParseError {
                line: line_num,
                message: "ionize requires 'extend by <trits>'".to_string(),
            })
        }
    }
}

/// Result of a MassScript observation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationResult {
    /// Name of the observed variable (if any)
    pub name: Option<String>,
    /// The address at observation time
    pub address: String,
    /// The synthesized spectrum
    pub spectrum: SynthesizedSpectrum,
}

/// Result of a MassScript fragmentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentResult {
    /// Name of the fragmented variable
    pub name: Option<String>,
    /// Prefix (headgroup) address
    pub prefix: String,
    /// Suffix (tail) address
    pub suffix: String,
    /// Fragment position
    pub position: usize,
}

/// Execution result from MassScript interpreter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// All observations made
    pub observations: Vec<ObservationResult>,
    /// All fragmentations made
    pub fragments: Vec<FragmentResult>,
    /// Final address state
    pub final_address: String,
    /// Named addresses
    pub variables: HashMap<String, String>,
}

/// MassScript interpreter
pub struct MassScriptInterpreter {
    /// Current address
    current_address: TernaryAddress,
    /// Named addresses
    variables: HashMap<String, TernaryAddress>,
    /// Spectrum extractor
    extractor: SpectrumExtractor,
    /// Observations collected
    observations: Vec<ObservationResult>,
    /// Fragmentations collected
    fragments: Vec<FragmentResult>,
}

impl MassScriptInterpreter {
    /// Create a new interpreter with default extractor
    pub fn new() -> Self {
        Self {
            current_address: TernaryAddress::empty(),
            variables: HashMap::new(),
            extractor: SpectrumExtractor::default(),
            observations: Vec::new(),
            fragments: Vec::new(),
        }
    }

    /// Create with custom extractor
    pub fn with_extractor(extractor: SpectrumExtractor) -> Self {
        Self {
            current_address: TernaryAddress::empty(),
            variables: HashMap::new(),
            extractor,
            observations: Vec::new(),
            fragments: Vec::new(),
        }
    }

    /// Execute a MassScript program
    pub fn execute(&mut self, script: &MassScript) -> Result<ExecutionResult> {
        for (idx, command) in script.commands.iter().enumerate() {
            self.execute_command(command, idx + 1)?;
        }

        Ok(ExecutionResult {
            observations: std::mem::take(&mut self.observations),
            fragments: std::mem::take(&mut self.fragments),
            final_address: self.current_address.to_string(),
            variables: self.variables.iter().map(|(k, v)| (k.clone(), v.to_string())).collect(),
        })
    }

    /// Execute a MassScript source string
    pub fn execute_str(&mut self, source: &str) -> Result<ExecutionResult> {
        let script = MassScript::parse(source)?;
        self.execute(&script)
    }

    fn execute_command(&mut self, command: &MassScriptCommand, _line: usize) -> Result<()> {
        match command {
            MassScriptCommand::Partition { name, address } => {
                let addr = TernaryAddress::from_str(address)?;
                if let Some(name) = name {
                    self.variables.insert(name.clone(), addr.clone());
                }
                self.current_address = addr;
            }

            MassScriptCommand::Extend { trits } => {
                let ext = TernaryAddress::from_str(trits)?;
                self.current_address = self.current_address.extend(&ext);
            }

            MassScriptCommand::Observe { name } => {
                let addr = match name {
                    Some(n) => self.variables.get(n).cloned().ok_or_else(|| {
                        MassComputingError::ExecutionError(format!("Unknown variable: {}", n))
                    })?,
                    None => self.current_address.clone(),
                };

                let spectrum = self.extractor.extract(&addr);
                self.observations.push(ObservationResult {
                    name: name.clone(),
                    address: addr.to_string(),
                    spectrum,
                });
            }

            MassScriptCommand::Fragment { name, position } => {
                let addr = match name {
                    Some(n) => self.variables.get(n).cloned().ok_or_else(|| {
                        MassComputingError::ExecutionError(format!("Unknown variable: {}", n))
                    })?,
                    None => self.current_address.clone(),
                };

                let (prefix, suffix) = addr.fragment_at(*position)?;
                self.fragments.push(FragmentResult {
                    name: name.clone(),
                    prefix: prefix.to_string(),
                    suffix: suffix.to_string(),
                    position: *position,
                });
            }

            MassScriptCommand::Inject { name, address } => {
                let addr = TernaryAddress::from_str(address)?;
                if let Some(n) = name {
                    self.variables.insert(n.clone(), addr.clone());
                }
                self.current_address = addr;
            }

            MassScriptCommand::Chromatograph { trits } => {
                let ext = TernaryAddress::from_str(trits)?;
                self.current_address = self.current_address.extend(&ext);
            }

            MassScriptCommand::Ionize { trits } => {
                let ext = TernaryAddress::from_str(trits)?;
                self.current_address = self.current_address.extend(&ext);
            }

            MassScriptCommand::Detect => {
                let spectrum = self.extractor.extract(&self.current_address);
                self.observations.push(ObservationResult {
                    name: None,
                    address: self.current_address.to_string(),
                    spectrum,
                });
            }

            MassScriptCommand::Comment(_) | MassScriptCommand::Empty => {
                // No-op
            }
        }

        Ok(())
    }

    /// Reset interpreter state
    pub fn reset(&mut self) {
        self.current_address = TernaryAddress::empty();
        self.variables.clear();
        self.observations.clear();
        self.fragments.clear();
    }
}

impl Default for MassScriptInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_SCRIPT: &str = r#"
# Define a phospholipid
partition PC_34_1 201102012021

# Extract observables
observe PC_34_1

# Fragment it
fragment PC_34_1 at 6

# Extend current address
extend by 012

# Observe again
observe
"#;

    #[test]
    fn test_parse_script() {
        let script = MassScript::parse(SAMPLE_SCRIPT).unwrap();

        // Should have parsed all lines
        assert!(!script.commands.is_empty());

        // Check specific commands
        let non_empty: Vec<_> = script.commands.iter()
            .filter(|c| !matches!(c, MassScriptCommand::Empty | MassScriptCommand::Comment(_)))
            .collect();

        assert_eq!(non_empty.len(), 5);
    }

    #[test]
    fn test_execute_script() {
        let mut interpreter = MassScriptInterpreter::new();
        let result = interpreter.execute_str(SAMPLE_SCRIPT).unwrap();

        // Should have 2 observations
        assert_eq!(result.observations.len(), 2);

        // Should have 1 fragmentation
        assert_eq!(result.fragments.len(), 1);

        // First observation should have the name PC_34_1
        assert_eq!(result.observations[0].name, Some("PC_34_1".to_string()));

        // Fragment should be at position 6
        assert_eq!(result.fragments[0].position, 6);
    }

    #[test]
    fn test_virtual_experiment() {
        let script = r#"
inject sample as 000
chromatograph extend by 111011
ionize extend by 220
observe
extend by 012
observe
detect
"#;

        let mut interpreter = MassScriptInterpreter::new();
        let result = interpreter.execute_str(script).unwrap();

        // Should have 3 observations (2 observe + 1 detect)
        assert_eq!(result.observations.len(), 3);

        // Final address should be the full sequence
        assert_eq!(result.final_address, "000111011220012");
    }

    #[test]
    fn test_parse_errors() {
        let bad_script = "unknown_command foo bar";
        let result = MassScript::parse(bad_script);
        assert!(result.is_err());
    }

    #[test]
    fn test_execution_errors() {
        let script = "observe undefined_variable";
        let mut interpreter = MassScriptInterpreter::new();
        let result = interpreter.execute_str(script);
        assert!(result.is_err());
    }
}

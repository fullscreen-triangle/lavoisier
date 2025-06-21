//! Parser for the Buhera Language
//! 
//! Implements a high-performance parser using nom combinators to parse .bh scripts
//! into the abstract syntax tree for surgical precision mass spectrometry analysis.

use nom::{
    branch::alt,
    bytes::complete::{tag, take_until, take_while, take_while1},
    character::complete::{alpha1, alphanumeric1, char, multispace0, multispace1, space0, space1},
    combinator::{map, opt, recognize, value},
    error::{context, VerboseError},
    multi::{many0, many1, separated_list0, separated_list1},
    number::complete::double,
    sequence::{delimited, pair, preceded, separated_pair, terminated, tuple},
    IResult,
};
use std::collections::HashMap;
use std::fs;

use crate::ast::*;
use crate::errors::BuheraError;

type ParseResult<'a, O> = IResult<&'a str, O, VerboseError<&'a str>>;

/// Parse a Buhera script from file
pub fn parse_file(path: &str) -> Result<BuheraScript, BuheraError> {
    let content = fs::read_to_string(path)
        .map_err(|e| BuheraError::ParseError(format!("Failed to read file {}: {}", path, e)))?;
    
    parse_script(&content)
}

/// Parse a Buhera script from string content
pub fn parse_script(input: &str) -> Result<BuheraScript, BuheraError> {
    match script(input) {
        Ok(("", script)) => Ok(script),
        Ok((remaining, _)) => Err(BuheraError::ParseError(format!(
            "Unexpected content after script: {}", 
            remaining.chars().take(50).collect::<String>()
        ))),
        Err(e) => Err(BuheraError::ParseError(format!("Parse error: {}", e))),
    }
}

/// Parse complete Buhera script
fn script(input: &str) -> ParseResult<BuheraScript> {
    context(
        "buhera_script",
        map(
            tuple((
                multispace0,
                many0(comment),
                imports,
                multispace0,
                objective,
                multispace0,
                many0(validation_rule),
                multispace0,
                many1(analysis_phase),
                multispace0,
            )),
            |(_, _, imports, _, objective, _, validations, _, phases, _)| {
                BuheraScript {
                    objective,
                    validations,
                    phases,
                    imports,
                    metadata: ScriptMetadata {
                        author: "Unknown".to_string(),
                        version: "1.0".to_string(),
                        description: "Buhera analysis script".to_string(),
                        created_date: chrono::Utc::now().format("%Y-%m-%d").to_string(),
                        last_modified: chrono::Utc::now().format("%Y-%m-%d").to_string(),
                        tags: vec![],
                    },
                }
            },
        ),
    )(input)
}

/// Parse import statements
fn imports(input: &str) -> ParseResult<Vec<String>> {
    many0(import_statement)(input)
}

/// Parse single import statement
fn import_statement(input: &str) -> ParseResult<String> {
    context(
        "import",
        map(
            tuple((
                tag("import"),
                space1,
                identifier_path,
                line_ending,
            )),
            |(_, _, path, _)| path,
        ),
    )(input)
}

/// Parse objective declaration
fn objective(input: &str) -> ParseResult<BuheraObjective> {
    context(
        "objective",
        map(
            tuple((
                tag("objective"),
                space1,
                identifier,
                tag(":"),
                multispace0,
                objective_body,
            )),
            |(_, _, name, _, _, body)| {
                let mut obj = body;
                obj.name = name;
                obj
            },
        ),
    )(input)
}

/// Parse objective body
fn objective_body(input: &str) -> ParseResult<BuheraObjective> {
    context(
        "objective_body",
        map(
            many1(objective_field),
            |fields| {
                let mut objective = BuheraObjective {
                    name: String::new(),
                    target: String::new(),
                    success_criteria: SuccessCriteria::new(),
                    evidence_priorities: vec![],
                    biological_constraints: vec![],
                    statistical_requirements: StatisticalRequirements {
                        minimum_sample_size: None,
                        effect_size: None,
                        alpha_level: None,
                        power_requirement: None,
                        multiple_testing_correction: None,
                    },
                };

                for (key, value) in fields {
                    match key.as_str() {
                        "target" => objective.target = value,
                        "success_criteria" => {
                            // Parse success criteria from value
                            if let Ok(criteria) = parse_success_criteria(&value) {
                                objective.success_criteria = criteria;
                            }
                        }
                        "evidence_priorities" => {
                            // Parse evidence types
                            objective.evidence_priorities = parse_evidence_types(&value);
                        }
                        _ => {}
                    }
                }
                objective
            },
        ),
    )(input)
}

/// Parse objective field
fn objective_field(input: &str) -> ParseResult<(String, String)> {
    context(
        "objective_field",
        map(
            tuple((
                space0,
                identifier,
                tag(":"),
                space0,
                quoted_string,
                line_ending,
            )),
            |(_, key, _, _, value, _)| (key, value),
        ),
    )(input)
}

/// Parse validation rule
fn validation_rule(input: &str) -> ParseResult<ValidationRule> {
    context(
        "validation_rule",
        map(
            tuple((
                tag("validate"),
                space1,
                identifier,
                tag(":"),
                multispace0,
                validation_body,
            )),
            |(_, _, name, _, _, (rule_type, condition, action))| ValidationRule {
                name,
                rule_type,
                condition,
                action,
            },
        ),
    )(input)
}

/// Parse validation body
fn validation_body(input: &str) -> ParseResult<(ValidationType, ValidationCondition, ValidationAction)> {
    context(
        "validation_body",
        map(
            tuple((
                validation_type,
                multispace0,
                validation_condition,
                multispace0,
                validation_action,
            )),
            |(vtype, _, condition, _, action)| (vtype, condition, action),
        ),
    )(input)
}

/// Parse validation type
fn validation_type(input: &str) -> ParseResult<ValidationType> {
    context(
        "validation_type",
        alt((
            value(ValidationType::SampleSize, tag("check_sample_size")),
            value(ValidationType::InstrumentCapability, tag("check_instrument_capability")),
            value(ValidationType::BiologicalCoherence, tag("check_biological_coherence")),
            value(ValidationType::StatisticalPower, tag("check_statistical_power")),
            value(ValidationType::DataQuality, tag("check_data_quality")),
            value(ValidationType::PathwayConsistency, tag("check_pathway_consistency")),
        )),
    )(input)
}

/// Parse validation condition
fn validation_condition(input: &str) -> ParseResult<ValidationCondition> {
    context(
        "validation_condition",
        map(
            tuple((
                tag("if"),
                space1,
                take_until("\n"),
                line_ending,
            )),
            |(_, _, condition, _)| ValidationCondition {
                expression: condition.trim().to_string(),
                parameters: HashMap::new(),
            },
        ),
    )(input)
}

/// Parse validation action
fn validation_action(input: &str) -> ParseResult<ValidationAction> {
    context(
        "validation_action",
        alt((
            map(
                tuple((tag("abort"), tag("("), quoted_string, tag(")"))),
                |(_, _, message, _)| ValidationAction::Abort { message },
            ),
            map(
                tuple((tag("warn"), tag("("), quoted_string, tag(")"))),
                |(_, _, message, _)| ValidationAction::Warn { message },
            ),
            map(
                tuple((tag("suggest"), tag("("), quoted_string, tag(")"))),
                |(_, _, recommendation, _)| ValidationAction::Suggest { recommendation },
            ),
        )),
    )(input)
}

/// Parse analysis phase
fn analysis_phase(input: &str) -> ParseResult<AnalysisPhase> {
    context(
        "analysis_phase",
        map(
            tuple((
                tag("phase"),
                space1,
                identifier,
                tag(":"),
                multispace0,
                many1(operation),
            )),
            |(_, _, name, _, _, operations)| {
                let phase_type = match name.as_str() {
                    "DataAcquisition" => PhaseType::DataAcquisition,
                    "Preprocessing" => PhaseType::Preprocessing,
                    "EvidenceBuilding" => PhaseType::EvidenceBuilding,
                    "BayesianInference" => PhaseType::BayesianInference,
                    "TargetedValidation" => PhaseType::TargetedValidation,
                    "ResultsSynthesis" => PhaseType::ResultsSynthesis,
                    "QualityControl" => PhaseType::QualityControl,
                    _ => PhaseType::Preprocessing,
                };

                AnalysisPhase {
                    name,
                    phase_type,
                    operations,
                    dependencies: vec![],
                }
            },
        ),
    )(input)
}

/// Parse operation
fn operation(input: &str) -> ParseResult<Operation> {
    context(
        "operation",
        alt((
            variable_assignment,
            function_call,
            conditional_operation,
        )),
    )(input)
}

/// Parse variable assignment
fn variable_assignment(input: &str) -> ParseResult<Operation> {
    context(
        "variable_assignment",
        map(
            tuple((
                space0,
                identifier,
                space0,
                tag("="),
                space0,
                expression,
                line_ending,
            )),
            |(_, var_name, _, _, _, expr, _)| Operation {
                name: format!("assign_{}", var_name),
                operation_type: OperationType::LoadData, // Default, will be refined
                parameters: HashMap::new(),
                input_variables: vec![],
                output_variable: Some(var_name),
            },
        ),
    )(input)
}

/// Parse function call
fn function_call(input: &str) -> ParseResult<Operation> {
    context(
        "function_call",
        map(
            tuple((
                space0,
                identifier_path,
                tag("("),
                separated_list0(tag(","), space0),
                function_arguments,
                tag(")"),
                line_ending,
            )),
            |(_, func_path, _, _, args, _, _)| {
                let operation_type = match func_path.as_str() {
                    "load_dataset" => OperationType::LoadData,
                    "lavoisier.preprocess" => OperationType::Preprocess,
                    "lavoisier.mzekezeke.build_evidence_network" => OperationType::BuildEvidenceNetwork,
                    "lavoisier.hatata.validate" => OperationType::ValidateResults,
                    _ => OperationType::CallLavoisier,
                };

                Operation {
                    name: func_path.clone(),
                    operation_type,
                    parameters: args,
                    input_variables: vec![],
                    output_variable: None,
                }
            },
        ),
    )(input)
}

/// Parse function arguments
fn function_arguments(input: &str) -> ParseResult<HashMap<String, serde_json::Value>> {
    context(
        "function_arguments",
        map(
            separated_list0(
                tuple((tag(","), space0)),
                argument_pair,
            ),
            |pairs| pairs.into_iter().collect(),
        ),
    )(input)
}

/// Parse argument pair
fn argument_pair(input: &str) -> ParseResult<(String, serde_json::Value)> {
    context(
        "argument_pair",
        map(
            tuple((
                identifier,
                tag(":"),
                space0,
                argument_value,
            )),
            |(key, _, _, value)| (key, value),
        ),
    )(input)
}

/// Parse argument value
fn argument_value(input: &str) -> ParseResult<serde_json::Value> {
    context(
        "argument_value",
        alt((
            map(quoted_string, |s| serde_json::Value::String(s)),
            map(double, |f| serde_json::Value::Number(serde_json::Number::from_f64(f).unwrap())),
            map(identifier, |s| serde_json::Value::String(s)),
        )),
    )(input)
}

/// Parse conditional operation
fn conditional_operation(input: &str) -> ParseResult<Operation> {
    context(
        "conditional_operation",
        map(
            tuple((
                space0,
                tag("if"),
                space1,
                expression,
                tag(":"),
                line_ending,
                many1(operation),
                opt(tuple((
                    space0,
                    tag("else:"),
                    line_ending,
                    many1(operation),
                ))),
            )),
            |(_, _, _, condition, _, _, then_ops, else_ops)| Operation {
                name: "conditional".to_string(),
                operation_type: OperationType::ConditionalBranch,
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("condition".to_string(), serde_json::Value::String(condition));
                    params
                },
                input_variables: vec![],
                output_variable: None,
            },
        ),
    )(input)
}

/// Parse expression (simplified for now)
fn expression(input: &str) -> ParseResult<String> {
    context(
        "expression",
        map(take_until(":"), |s: &str| s.trim().to_string()),
    )(input)
}

/// Parse identifier
fn identifier(input: &str) -> ParseResult<String> {
    context(
        "identifier",
        map(
            recognize(pair(
                alt((alpha1, tag("_"))),
                many0(alt((alphanumeric1, tag("_")))),
            )),
            |s: &str| s.to_string(),
        ),
    )(input)
}

/// Parse identifier path (e.g., lavoisier.mzekezeke.build_network)
fn identifier_path(input: &str) -> ParseResult<String> {
    context(
        "identifier_path",
        map(
            recognize(separated_list1(tag("."), identifier)),
            |s: &str| s.to_string(),
        ),
    )(input)
}

/// Parse quoted string
fn quoted_string(input: &str) -> ParseResult<String> {
    context(
        "quoted_string",
        delimited(
            char('"'),
            map(take_until("\""), |s: &str| s.to_string()),
            char('"'),
        ),
    )(input)
}

/// Parse comment
fn comment(input: &str) -> ParseResult<String> {
    context(
        "comment",
        map(
            tuple((tag("//"), take_until("\n"))),
            |(_, content): (_, &str)| content.trim().to_string(),
        ),
    )(input)
}

/// Parse line ending
fn line_ending(input: &str) -> ParseResult<()> {
    context(
        "line_ending",
        map(
            tuple((space0, opt(comment), multispace1)),
            |_| (),
        ),
    )(input)
}

// Helper functions for parsing complex types

fn parse_success_criteria(input: &str) -> Result<SuccessCriteria, BuheraError> {
    // Simplified implementation - would need full parser for complex criteria
    Ok(SuccessCriteria::new())
}

fn parse_evidence_types(input: &str) -> Vec<EvidenceType> {
    // Simplified implementation
    vec![EvidenceType::MassMatch, EvidenceType::MS2Fragmentation]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_identifier() {
        assert_eq!(identifier("hello_world").unwrap().1, "hello_world");
        assert_eq!(identifier("test123").unwrap().1, "test123");
        assert_eq!(identifier("_private").unwrap().1, "_private");
    }

    #[test]
    fn test_parse_quoted_string() {
        assert_eq!(quoted_string("\"hello world\"").unwrap().1, "hello world");
        assert_eq!(quoted_string("\"test\"").unwrap().1, "test");
    }

    #[test]
    fn test_parse_comment() {
        assert_eq!(comment("// This is a comment\n").unwrap().1, "This is a comment");
    }

    #[test]
    fn test_parse_import() {
        let input = "import lavoisier.mzekezeke\n";
        assert_eq!(import_statement(input).unwrap().1, "lavoisier.mzekezeke");
    }

    #[test]
    fn test_parse_simple_objective() {
        let input = r#"objective DiabetesDiscovery:
    target: "identify metabolites"
    "#;
        
        let result = objective(input);
        assert!(result.is_ok());
        let (_, obj) = result.unwrap();
        assert_eq!(obj.name, "DiabetesDiscovery");
        assert_eq!(obj.target, "identify metabolites");
    }
} 
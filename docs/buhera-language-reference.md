# Buhera Language Reference

This document provides a comprehensive reference for the Buhera scripting language syntax, semantics, and built-in functions.

## Language Grammar

### Lexical Elements

#### Keywords
```
objective, validate, phase, import, if, else, abort, warn, check_*
```

#### Identifiers
```
[a-zA-Z_][a-zA-Z0-9_]*
```

#### Literals
- **String literals**: `"string content"`
- **Numeric literals**: `123`, `45.67`, `1e-6`
- **Boolean literals**: `true`, `false`
- **Array literals**: `["item1", "item2", "item3"]`

#### Operators
```
=, ==, !=, <, >, <=, >=, AND, OR, NOT
```

#### Delimiters
```
:, ;, ,, (, ), [, ], {, }
```

### Grammar Rules

#### Script Structure
```bnf
script ::= import_list objective validation_list phase_list

import_list ::= import_statement*
import_statement ::= "import" module_path

objective ::= "objective" identifier ":" objective_fields

validation_list ::= validation_rule*
validation_rule ::= "validate" identifier ":" validation_body

phase_list ::= phase_definition*
phase_definition ::= "phase" identifier ":" phase_body
```

#### Objective Definition
```bnf
objective_fields ::= objective_field+
objective_field ::= field_name ":" string_literal

field_name ::= "target" | "success_criteria" | "evidence_priorities" 
             | "biological_constraints" | "statistical_requirements"
```

#### Validation Rules
```bnf
validation_body ::= statement_list
statement ::= assignment | function_call | conditional | action

action ::= abort_statement | warn_statement
abort_statement ::= "abort" "(" string_literal ")"
warn_statement ::= "warn" "(" string_literal ")"
```

#### Phase Definitions
```bnf
phase_body ::= statement_list
statement ::= assignment | function_call | conditional

assignment ::= identifier "=" expression
function_call ::= module_path "." function_name "(" argument_list ")"
conditional ::= "if" condition ":" statement_list ["else" ":" statement_list]
```

## Built-in Functions

### Validation Functions

#### `check_instrument_capability`
Validates that the instrument can achieve the required analytical performance.

**Usage:**
```javascript
validate InstrumentCheck:
    check_instrument_capability
    if target_concentration < instrument_detection_limit:
        abort("Instrument cannot detect target concentrations")
```

**Checks:**
- Detection limits vs. target concentrations
- Mass accuracy requirements
- Chromatographic resolution needs
- Scan rate compatibility

#### `check_sample_size`
Validates statistical power based on sample size.

**Usage:**
```javascript
validate StatisticalPower:
    check_sample_size
    if sample_size < required_minimum:
        warn("Sample size may be insufficient for robust analysis")
```

**Parameters:**
- `effect_size`: Expected effect size
- `alpha_level`: Significance level (default: 0.05)
- `power_requirement`: Required statistical power (default: 0.8)

#### `check_pathway_consistency`
Validates biological coherence of expected metabolic changes.

**Usage:**
```javascript
validate BiologicalCoherence:
    check_pathway_consistency
    if expected_pathway_disruption AND missing_key_metabolites:
        warn("Missing expected pathway disruption markers")
```

### Data Loading Functions

#### `load_dataset`
Loads mass spectrometry data with metadata.

**Syntax:**
```javascript
dataset = load_dataset(
    file_path: "path/to/data.mzML",
    metadata: "path/to/metadata.csv",
    groups: ["control", "treatment"],
    focus: "objective_context"
)
```

**Parameters:**
- `file_path`: Path to mzML file
- `metadata`: Optional metadata file
- `groups`: Sample groups for comparison
- `focus`: Objective context for optimized loading

### Lavoisier Integration Functions

#### `lavoisier.mzekezeke.build_evidence_network`
Builds objective-focused Bayesian evidence network.

**Syntax:**
```javascript
evidence_network = lavoisier.mzekezeke.build_evidence_network(
    data: dataset,
    objective: "research_objective",
    evidence_types: ["mass_match", "ms2_fragmentation"],
    pathway_focus: ["glycolysis", "tca_cycle"],
    evidence_weights: {"pathway_membership": 1.3}
)
```

**Parameters:**
- `data`: Input dataset
- `objective`: Research objective string
- `evidence_types`: Types of evidence to collect
- `pathway_focus`: Biological pathways to prioritize
- `evidence_weights`: Custom evidence weighting

#### `lavoisier.hatata.validate_with_objective`
Validates analysis results against objective criteria.

**Syntax:**
```javascript
annotations = lavoisier.hatata.validate_with_objective(
    evidence_network: evidence_network,
    objective: "research_objective",
    confidence_threshold: 0.85,
    success_criteria: {"sensitivity": 0.85}
)
```

**Parameters:**
- `evidence_network`: Evidence network to validate
- `objective`: Research objective
- `confidence_threshold`: Minimum confidence threshold
- `success_criteria`: Success criteria dictionary

#### `lavoisier.zengeza.noise_reduction`
Objective-aware noise reduction.

**Syntax:**
```javascript
clean_data = lavoisier.zengeza.noise_reduction(
    data: raw_data,
    objective_context: "biomarker_discovery",
    preserve_patterns: ["glucose_pathway", "lipid_metabolism"]
)
```

**Parameters:**
- `data`: Raw data to process
- `objective_context`: Objective context for preservation
- `preserve_patterns`: Patterns to preserve during cleaning

## Data Types

### Basic Types

#### String
```javascript
name = "diabetes_biomarker_discovery"
description = "Multi-line string content
               can span multiple lines"
```

#### Number
```javascript
threshold = 0.85
sample_size = 100
mass_accuracy = 1e-6  // Scientific notation
```

#### Boolean
```javascript
validation_passed = true
analysis_complete = false
```

#### Array
```javascript
evidence_types = ["mass_match", "ms2_fragmentation", "pathway_membership"]
pathway_focus = ["glycolysis", "gluconeogenesis", "tca_cycle"]
```

### Complex Types

#### Dataset
Represents loaded mass spectrometry data.

**Properties:**
- `spectra_count`: Number of spectra
- `mass_range`: Mass range coverage
- `retention_time_range`: Chromatographic range
- `metadata`: Associated metadata

#### EvidenceNetwork
Represents a Bayesian evidence network.

**Properties:**
- `confidence`: Overall confidence score
- `evidence_scores`: Individual evidence scores
- `objective`: Associated objective
- `recommendations`: Analysis recommendations

## Control Flow

### Conditional Statements

#### Basic If Statement
```javascript
if condition:
    statement_list
```

#### If-Else Statement
```javascript
if condition:
    statement_list
else:
    alternative_statement_list
```

#### Complex Conditions
```javascript
if sample_size >= 30 AND effect_size > 0.5:
    proceed_with_analysis()
else:
    recommend_sample_size_increase()
```

### Logical Operators

#### AND Operator
```javascript
if sensitivity >= 0.85 AND specificity >= 0.85:
    validation_passed = true
```

#### OR Operator
```javascript
if high_confidence_ms1 OR high_confidence_ms2:
    accept_annotation()
```

#### NOT Operator
```javascript
if NOT pathway_coherence_check:
    warn("Pathway coherence validation failed")
```

### Comparison Operators

```javascript
// Equality
if confidence == 1.0:
    perfect_match()

// Inequality  
if error_rate != 0.0:
    investigate_errors()

// Relational
if sample_size > minimum_required:
    sufficient_power = true

if mass_error <= 5_ppm:
    acceptable_accuracy = true
```

## Comments

### Single-line Comments
```javascript
// This is a single-line comment
objective BiomarkerDiscovery:  // End-of-line comment
    target: "identify biomarkers"
```

### Multi-line Comments
```javascript
/*
 * This is a multi-line comment
 * Used for detailed explanations
 * of complex analysis logic
 */
```

### Documentation Comments
```javascript
/**
 * This phase builds an evidence network optimized for biomarker discovery.
 * It prioritizes pathway membership evidence because biological relevance
 * is critical for clinical biomarker validation.
 */
phase EvidenceBuilding:
    evidence_network = lavoisier.mzekezeke.build_evidence_network(
        objective: "biomarker_discovery"
    )
```

## Error Handling

### Validation Errors
```javascript
validate InstrumentCapability:
    if target_concentration < detection_limit:
        abort("Instrument cannot detect target concentrations")
        // Script execution stops here
```

### Warnings
```javascript
validate SampleSize:
    if sample_size < optimal_size:
        warn("Sample size below optimal threshold")
        // Script continues with warning
```

### Runtime Errors
Runtime errors are handled by the Buhera runtime and Lavoisier integration:

- **File not found**: Data files missing
- **Integration errors**: Lavoisier module communication failures
- **Analysis failures**: Statistical or scientific validation failures

## Reserved Words

The following identifiers are reserved and cannot be used as variable names:

```
objective, validate, phase, import, if, else, abort, warn, true, false,
AND, OR, NOT, check_instrument_capability, check_sample_size, 
check_pathway_consistency, load_dataset
```

## Naming Conventions

### Objectives
Use PascalCase for objective names:
```javascript
objective DiabetesBiomarkerDiscovery:
objective DrugMetabolismCharacterization:
```

### Variables
Use snake_case for variable names:
```javascript
evidence_network = build_network()
sample_metadata = load_metadata()
```

### Functions
Use snake_case for function names:
```javascript
build_evidence_network()
validate_with_objective()
```

### Constants
Use UPPER_SNAKE_CASE for constants:
```javascript
MIN_SAMPLE_SIZE = 30
MAX_MASS_ERROR = 5e-6
```

## Best Practices

### 1. Objective-First Design
Always start with a clear, measurable objective:

```javascript
// Good: Specific and measurable
objective DiabetesBiomarkerDiscovery:
    target: "identify metabolites predictive of diabetes progression"
    success_criteria: "sensitivity >= 0.85 AND specificity >= 0.85"

// Bad: Vague and unmeasurable  
objective GeneralAnalysis:
    target: "analyze some data"
    success_criteria: "get results"
```

### 2. Comprehensive Validation
Include validation for all critical assumptions:

```javascript
// Validate instrument capabilities
validate InstrumentCapability:
    check_instrument_capability

// Validate statistical power
validate StatisticalPower:
    check_sample_size

// Validate biological coherence
validate BiologicalCoherence:
    check_pathway_consistency
```

### 3. Document Scientific Reasoning
Explain why specific choices were made:

```javascript
phase EvidenceBuilding:
    // Prioritize pathway membership for biomarker discovery
    // because biological relevance is critical for clinical translation
    evidence_network = lavoisier.mzekezeke.build_evidence_network(
        evidence_weights: {"pathway_membership": 1.3}
    )
```

### 4. Use Meaningful Names
Choose descriptive names that reflect scientific intent:

```javascript
// Good: Descriptive and scientific
diabetes_progression_markers = load_dataset("diabetes_cohort.mzML")
glycolysis_focused_network = build_evidence_network(
    pathway_focus: ["glycolysis"]
)

// Bad: Generic and unclear
data = load_dataset("file.mzML")
result = build_network()
```

This reference provides the foundation for writing effective Buhera scripts that encode scientific reasoning as executable, validatable programs. 
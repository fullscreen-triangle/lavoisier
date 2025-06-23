//! Buhera CLI - Command Line Interface for Surgical Precision Mass Spectrometry
//! 
//! Provides command-line tools for parsing, validating, and executing Buhera scripts.

use buhera::*;
use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    let command = &args[1];
    
    match command.as_str() {
        "validate" => {
            if args.len() < 3 {
                eprintln!("Usage: buhera validate <script.bh>");
                process::exit(1);
            }
            validate_script(&args[2]);
        }
        "execute" => {
            if args.len() < 3 {
                eprintln!("Usage: buhera execute <script.bh>");
                process::exit(1);
            }
            execute_script(&args[2]);
        }
        "parse" => {
            if args.len() < 3 {
                eprintln!("Usage: buhera parse <script.bh>");
                process::exit(1);
            }
            parse_script(&args[2]);
        }
        "example" => {
            generate_example_script();
        }
        "--help" | "-h" => {
            print_usage();
        }
        _ => {
            eprintln!("Unknown command: {}", command);
            print_usage();
            process::exit(1);
        }
    }
}

fn print_usage() {
    println!("Buhera - Surgical Precision Scripting Language for Mass Spectrometry");
    println!("");
    println!("USAGE:");
    println!("    buhera <COMMAND> [OPTIONS]");
    println!("");
    println!("COMMANDS:");
    println!("    validate <script.bh>    Validate experimental logic before execution");
    println!("    execute <script.bh>     Execute validated Buhera script");
    println!("    parse <script.bh>       Parse and display script structure");
    println!("    example                 Generate example Buhera script");
    println!("    help                    Show this help message");
    println!("");
    println!("EXAMPLES:");
    println!("    buhera validate diabetes_biomarker.bh");
    println!("    buhera execute metabolite_analysis.bh");
    println!("    buhera example > template.bh");
}

fn validate_script(file_path: &str) {
    println!("🔍 Validating Buhera script: {}", file_path);
    
    // Parse script
    let script = match BuheraScript::from_file(file_path) {
        Ok(script) => script,
        Err(e) => {
            eprintln!("❌ Parse error: {}", e);
            process::exit(1);
        }
    };

    println!("✅ Script parsed successfully");
    println!("📋 Objective: {}", script.objective.summary());

    // Validate experimental logic
    let validator = BuheraValidator::new();
    match validator.validate(&script) {
        Ok(result) => {
            println!("📊 {}", result.summary());
            
            if result.is_valid {
                println!("✅ Validation PASSED - Script is ready for execution");
                println!("🎯 Estimated success probability: {:.1}%", 
                        result.estimated_success_probability * 100.0);
            } else {
                println!("❌ Validation FAILED - Issues found:");
                for issue in &result.issues {
                    println!("   • {}", issue);
                }
                
                if !result.recommendations.is_empty() {
                    println!("💡 Recommendations:");
                    for rec in &result.recommendations {
                        println!("   → {}", rec);
                    }
                }
                process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("❌ Validation error: {}", e);
            process::exit(1);
        }
    }
}

fn execute_script(file_path: &str) {
    println!("🚀 Executing Buhera script: {}", file_path);
    
    // First validate
    validate_script(file_path);
    
    // Parse script again for execution
    let script = match BuheraScript::from_file(file_path) {
        Ok(script) => script,
        Err(e) => {
            eprintln!("❌ Parse error: {}", e);
            process::exit(1);
        }
    };

    println!("⚡ Starting execution with objective focus: {}", script.objective.target);
    
    // Note: In a real implementation, we would initialize Python and Lavoisier here
    println!("🔬 This would connect to Lavoisier for execution...");
    println!("📊 Goal-directed analysis would proceed with surgical precision");
    println!("✅ Execution framework ready (Python integration required)");
}

fn parse_script(file_path: &str) {
    println!("📖 Parsing Buhera script: {}", file_path);
    
    let script = match BuheraScript::from_file(file_path) {
        Ok(script) => script,
        Err(e) => {
            eprintln!("❌ Parse error: {}", e);
            process::exit(1);
        }
    };

    // Display script structure
    println!("✅ Script structure:");
    println!("   📋 Objective: {}", script.objective.name);
    println!("   🎯 Target: {}", script.objective.target);
    println!("   🔬 Evidence priorities: {:?}", script.objective.evidence_priorities);
    println!("   📝 Validation rules: {}", script.validations.len());
    println!("   ⚙️  Analysis phases: {}", script.phases.len());
    
    for (i, phase) in script.phases.iter().enumerate() {
        println!("      {}. {} ({:?})", i + 1, phase.name, phase.phase_type);
    }
    
    println!("   📦 Imports: {:?}", script.imports);
}

fn generate_example_script() {
    let example = r#"// diabetes_biomarker_discovery.bh
// Example Buhera script for biomarker discovery with surgical precision

import lavoisier.mzekezeke
import lavoisier.hatata
import lavoisier.zengeza

objective DiabetesBiomarkerDiscovery:
    target: "identify metabolites predictive of diabetes progression"
    success_criteria: "sensitivity >= 0.85 AND specificity >= 0.85"
    evidence_priorities: "mass_match,ms2_fragmentation,pathway_membership"
    biological_constraints: "glycolysis_upregulated,tca_cycle_disrupted"
    statistical_requirements: "sample_size >= 30, power >= 0.8"

validate InstrumentCapability:
    check_instrument_capability
    if target_concentration < instrument_detection_limit:
        abort("Instrument cannot detect target concentrations")

validate SampleSize:
    check_sample_size  
    if sample_size < 30:
        warn("Small sample size may reduce statistical power")

phase DataAcquisition:
    dataset = load_dataset(file_path: "diabetes_samples.mzML")
    
phase Preprocessing:
    clean_data = lavoisier.zengeza.noise_reduction(dataset)
    normalized_data = lavoisier.preprocess(clean_data, method: "quantile")

phase EvidenceBuilding:
    evidence_network = lavoisier.mzekezeke.build_evidence_network(
        data: normalized_data,
        objective: "diabetes_biomarker_discovery",
        evidence_types: ["mass_match", "ms2_fragmentation", "pathway_membership"]
    )

phase BayesianInference:
    annotations = lavoisier.hatata.validate_with_objective(
        evidence_network: evidence_network,
        confidence_threshold: 0.8
    )

phase ResultsSynthesis:
    if annotations.confidence > 0.8:
        generate_biomarker_report(annotations)
    else:
        suggest_additional_evidence(annotations)
"#;

    println!("{}", example);
    
    println!("");
    println!("// This example demonstrates:");
    println!("// 1. 🎯 Objective-first design with clear success criteria");
    println!("// 2. ✅ Pre-flight validation to catch experimental flaws");
    println!("// 3. 🔬 Goal-directed evidence building with Lavoisier integration");
    println!("// 4. 🧠 Bayesian inference optimized for the specific objective");
    println!("// 5. 🎭 Surgical precision - every step focused on the research goal");
} 
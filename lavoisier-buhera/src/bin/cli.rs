//! Buhera CLI for Knowledge Distillation
//!
//! Command-line interface for creating expert LLMs from Buhera script responses.
//! Optimized for processing 100GB+ datasets.

use lavoisier_buhera::*;
use std::path::PathBuf;
use tokio;

#[derive(Debug)]
struct CliArgs {
    command: Command,
}

#[derive(Debug)]
enum Command {
    DistillLLM {
        domain: DomainExpertise,
        scripts_path: PathBuf,
        output_path: PathBuf,
        max_memory_gb: f64,
        ollama_url: String,
        base_model: String,
    },
    ValidateModel {
        model_path: PathBuf,
        domain: DomainExpertise,
    },
    CreateValidationSuite {
        domain: DomainExpertise,
        output_path: PathBuf,
    },
    Help,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args = parse_args()?;

    match args.command {
        Command::DistillLLM {
            domain,
            scripts_path,
            output_path,
            max_memory_gb,
            ollama_url,
            base_model,
        } => {
            println!("🧬 Buhera Knowledge Distillation Engine");
            println!("Domain: {:?}", domain);
            println!("Scripts: {}", scripts_path.display());
            println!("Output: {}", output_path.display());
            println!("Memory limit: {:.1} GB", max_memory_gb);
            println!("Ollama URL: {}", ollama_url);
            println!("Base model: {}", base_model);
            println!();

            // Create knowledge distillation engine
            let mut engine =
                KnowledgeDistillationEngine::new(ollama_url, base_model, max_memory_gb);

            // Create validation suite for the domain
            let validation_suite = create_validation_suite(domain.clone());
            println!(
                "📋 Created validation suite with {} test scripts",
                validation_suite.test_scripts.len()
            );

            // Start distillation process
            println!("🔬 Starting knowledge distillation...");
            let start_time = std::time::Instant::now();

            let result = engine
                .distill_expert_llm_streaming(
                    domain,
                    &scripts_path,
                    &output_path,
                    &validation_suite,
                )
                .await?;

            let duration = start_time.elapsed();

            println!("\n✅ Knowledge distillation completed!");
            println!("📊 Results:");
            println!("  Model ID: {}", result.model_id);
            println!("  Training samples: {}", result.training_samples);
            println!(
                "  Validation accuracy: {:.2}%",
                result.validation_accuracy * 100.0
            );
            println!(
                "  Buhera script accuracy: {:.2}%",
                result.buhera_script_accuracy * 100.0
            );
            println!(
                "  Model size: {:.2} MB",
                result.model_size_bytes as f64 / 1_000_000.0
            );
            println!("  Duration: {:.1}s", duration.as_secs_f64());
            println!("  Created: {}", result.creation_timestamp);
        }

        Command::ValidateModel { model_path, domain } => {
            println!("🔍 Validating Buhera Expert Model");
            println!("Model: {}", model_path.display());
            println!("Domain: {:?}", domain);

            let validation_suite = create_validation_suite(domain);
            let engine = KnowledgeDistillationEngine::new(
                "http://localhost:11434".to_string(),
                "llama3".to_string(),
                4.0,
            );

            let accuracy = engine
                .validate_model_with_buhera_scripts(
                    &model_path.to_string_lossy(),
                    &validation_suite,
                )
                .await?;

            println!("\n📈 Validation Results:");
            println!("  Accuracy: {:.2}%", accuracy * 100.0);
            println!("  Test scripts: {}", validation_suite.test_scripts.len());

            if accuracy >= validation_suite.expected_accuracy_threshold {
                println!(
                    "  ✅ Model meets accuracy threshold ({:.2}%)",
                    validation_suite.expected_accuracy_threshold * 100.0
                );
            } else {
                println!(
                    "  ❌ Model below accuracy threshold ({:.2}%)",
                    validation_suite.expected_accuracy_threshold * 100.0
                );
            }
        }

        Command::CreateValidationSuite {
            domain,
            output_path,
        } => {
            println!("📝 Creating Validation Suite");
            println!("Domain: {:?}", domain);

            let validation_suite = create_validation_suite(domain);

            let json_data = serde_json::to_string_pretty(&validation_suite)?;
            tokio::fs::write(&output_path, json_data).await?;

            println!("✅ Validation suite created:");
            println!("  Test scripts: {}", validation_suite.test_scripts.len());
            println!(
                "  Accuracy threshold: {:.2}%",
                validation_suite.expected_accuracy_threshold * 100.0
            );
            println!("  Saved to: {}", output_path.display());
        }

        Command::Help => {
            print_help();
        }
    }

    Ok(())
}

fn parse_args() -> Result<CliArgs, Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        return Ok(CliArgs {
            command: Command::Help,
        });
    }

    match args[1].as_str() {
        "distill" => {
            if args.len() < 8 {
                return Err("Usage: buhera distill <domain> <scripts_path> <output_path> <max_memory_gb> <ollama_url> <base_model>".into());
            }

            let domain = parse_domain(&args[2])?;
            let scripts_path = PathBuf::from(&args[3]);
            let output_path = PathBuf::from(&args[4]);
            let max_memory_gb = args[5].parse()?;
            let ollama_url = args[6].clone();
            let base_model = args[7].clone();

            Ok(CliArgs {
                command: Command::DistillLLM {
                    domain,
                    scripts_path,
                    output_path,
                    max_memory_gb,
                    ollama_url,
                    base_model,
                },
            })
        }

        "validate" => {
            if args.len() < 4 {
                return Err("Usage: buhera validate <model_path> <domain>".into());
            }

            let model_path = PathBuf::from(&args[2]);
            let domain = parse_domain(&args[3])?;

            Ok(CliArgs {
                command: Command::ValidateModel { model_path, domain },
            })
        }

        "create-suite" => {
            if args.len() < 4 {
                return Err("Usage: buhera create-suite <domain> <output_path>".into());
            }

            let domain = parse_domain(&args[2])?;
            let output_path = PathBuf::from(&args[3]);

            Ok(CliArgs {
                command: Command::CreateValidationSuite {
                    domain,
                    output_path,
                },
            })
        }

        _ => Ok(CliArgs {
            command: Command::Help,
        }),
    }
}

fn parse_domain(domain_str: &str) -> Result<DomainExpertise, Box<dyn std::error::Error>> {
    match domain_str.to_lowercase().as_str() {
        "metabolomics" => Ok(DomainExpertise::Metabolomics),
        "proteomics" => Ok(DomainExpertise::Proteomics),
        "lipidomics" => Ok(DomainExpertise::Lipidomics),
        "glycomics" => Ok(DomainExpertise::Glycomics),
        "mass-spectrometry" | "ms" => Ok(DomainExpertise::MassSpectrometry),
        "chromatography" => Ok(DomainExpertise::Chromatography),
        "statistical-analysis" | "stats" => Ok(DomainExpertise::StatisticalAnalysis),
        "data-processing" => Ok(DomainExpertise::DataProcessing),
        "biological-pathways" | "pathways" => Ok(DomainExpertise::BiologicalPathways),
        _ => Err(format!("Unknown domain: {}", domain_str).into()),
    }
}

fn print_help() {
    println!("🧬 Buhera Knowledge Distillation CLI");
    println!();
    println!("USAGE:");
    println!("    buhera <COMMAND> [OPTIONS]");
    println!();
    println!("COMMANDS:");
    println!("    distill     Create expert LLM from Buhera script responses");
    println!("    validate    Validate existing model using Buhera scripts");
    println!("    create-suite Create validation suite for domain");
    println!("    help        Show this help message");
    println!();
    println!("DISTILL USAGE:");
    println!("    buhera distill <domain> <scripts_path> <output_path> <max_memory_gb> <ollama_url> <base_model>");
    println!();
    println!("    Example:");
    println!("    buhera distill metabolomics /data/buhera_scripts.txt ./expert_model.bin 8.0 http://localhost:11434 llama3");
    println!();
    println!("VALIDATE USAGE:");
    println!("    buhera validate <model_path> <domain>");
    println!();
    println!("    Example:");
    println!("    buhera validate ./expert_model.bin metabolomics");
    println!();
    println!("CREATE-SUITE USAGE:");
    println!("    buhera create-suite <domain> <output_path>");
    println!();
    println!("    Example:");
    println!("    buhera create-suite metabolomics ./validation_suite.json");
    println!();
    println!("DOMAINS:");
    println!("    metabolomics, proteomics, lipidomics, glycomics");
    println!("    mass-spectrometry (ms), chromatography");
    println!("    statistical-analysis (stats), data-processing");
    println!("    biological-pathways (pathways)");
    println!();
    println!("FEATURES:");
    println!("    • Stream processing for 100GB+ datasets");
    println!("    • Memory-efficient chunked processing");
    println!("    • Buhera script-based validation");
    println!("    • Domain-specific expert models");
    println!("    • Ollama integration for local inference");
}

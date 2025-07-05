//! Buhera Knowledge Distillation Example
//!
//! This example demonstrates how to create expert LLMs from Buhera script execution
//! responses rather than storing raw experimental data. This approach is optimized
//! for handling 100GB+ datasets with efficient memory usage.

use lavoisier_buhera::*;
use std::path::Path;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧬 Buhera Knowledge Distillation Example");
    println!("========================================");

    // Example 1: Create validation suite for metabolomics
    println!("\n📋 Step 1: Creating Validation Suite");
    let validation_suite = create_validation_suite(DomainExpertise::Metabolomics);

    println!("Domain: {:?}", validation_suite.domain);
    println!("Test scripts: {}", validation_suite.test_scripts.len());
    println!("Accuracy threshold: {:.1}%", validation_suite.expected_accuracy_threshold * 100.0);

    for (i, script) in validation_suite.test_scripts.iter().enumerate() {
        println!("  Script {}: {} ({:?})",
            i + 1,
            script.script_id,
            script.difficulty_level
        );
    }

    // Example 2: Create sample Buhera scripts for demonstration
    println!("\n📝 Step 2: Creating Sample Buhera Scripts");
    let sample_scripts = create_sample_buhera_scripts();

    // Write sample scripts to temporary file
    let scripts_file = "/tmp/sample_buhera_scripts.txt";
    tokio::fs::write(scripts_file, sample_scripts).await?;
    println!("Created sample scripts at: {}", scripts_file);

    // Example 3: Initialize knowledge distillation engine
    println!("\n🔬 Step 3: Initializing Knowledge Distillation Engine");
    let mut engine = KnowledgeDistillationEngine::new(
        "http://localhost:11434".to_string(),
        "llama3".to_string(),
        4.0, // 4GB memory limit
    );

    println!("Ollama URL: http://localhost:11434");
    println!("Base model: llama3");
    println!("Memory limit: 4.0 GB");
    println!("Stream buffer size: {}", engine.stream_buffer_size);

    // Example 4: Demonstrate script parsing
    println!("\n🔍 Step 4: Demonstrating Script Parsing");
    let test_script = "OBJECTIVE biomarker_discovery FOR diabetes\nEVIDENCE_PRIORITY mass_match ms2_fragmentation";

    match KnowledgeDistillationEngine::parse_script_text(test_script) {
        Ok(parsed_script) => {
            println!("✅ Successfully parsed script:");
            println!("  Objective: {}", parsed_script.objective.name);
            println!("  Target: {}", parsed_script.objective.target);
            println!("  Evidence priorities: {:?}", parsed_script.objective.evidence_priorities);
            println!("  Phases: {}", parsed_script.phases.len());
        }
        Err(e) => {
            println!("❌ Failed to parse script: {:?}", e);
        }
    }

    // Example 5: Demonstrate knowledge unit creation
    println!("\n🧠 Step 5: Creating Knowledge Units");
    let mut executor = BuheraExecutor::new();

    let sample_buhera_scripts = vec![
        "OBJECTIVE biomarker_discovery FOR diabetes\nEVIDENCE_PRIORITY mass_match ms2_fragmentation",
        "OBJECTIVE pathway_analysis FOR glycolysis\nEVIDENCE_PRIORITY pathway_membership isotope_pattern",
        "OBJECTIVE compound_identification FOR glucose\nEVIDENCE_PRIORITY mass_match retention_time",
    ];

    for (i, script_text) in sample_buhera_scripts.iter().enumerate() {
        match KnowledgeDistillationEngine::process_single_script(
            script_text,
            &DomainExpertise::Metabolomics,
            &mut executor,
            i,
        ).await {
            Ok(knowledge_unit) => {
                println!("  Knowledge Unit {}:", i + 1);
                println!("    Script ID: {}", knowledge_unit.script_id);
                println!("    Objective: {}", knowledge_unit.objective);
                println!("    Success: {}", knowledge_unit.execution_result.success);
                println!("    Annotations: {}", knowledge_unit.execution_result.annotations.len());
                println!("    Evidence scores: {}", knowledge_unit.execution_result.evidence_scores.len());
                println!("    Confidence: {:.2}", knowledge_unit.validation_confidence);
            }
            Err(e) => {
                println!("    ❌ Failed to process script {}: {:?}", i + 1, e);
            }
        }
    }

    // Example 6: Demonstrate training data conversion
    println!("\n📊 Step 6: Converting to Training Data");
    let knowledge_unit = BuheraKnowledgeUnit {
        script_id: "demo_001".to_string(),
        script_text: "OBJECTIVE biomarker_discovery FOR diabetes".to_string(),
        objective: "diabetes biomarker discovery".to_string(),
        execution_result: ExecutionResult::new(
            true,
            vec!["Biomarker analysis completed".to_string(), "Statistical validation passed".to_string()],
            [
                ("MassMatch".to_string(), 0.92),
                ("MS2Fragmentation".to_string(), 0.88),
                ("PathwayMembership".to_string(), 0.85),
            ].iter().cloned().collect(),
            2.3,
        ),
        domain_context: DomainExpertise::Metabolomics,
        validation_confidence: 0.91,
        created_at: chrono::Utc::now().to_rfc3339(),
    };

    let training_data = engine.convert_to_training_format(&[knowledge_unit])?;
    println!("✅ Created training data:");
    println!("  Samples: {}", training_data.len());

    if let Some(sample) = training_data.first() {
        println!("  Sample fields: {:?}", sample.keys().collect::<Vec<_>>());
        if let Some(input) = sample.get("input") {
            println!("  Input preview: {}...",
                if input.len() > 50 { &input[..50] } else { input });
        }
        if let Some(output) = sample.get("output") {
            println!("  Output preview: {}...",
                if output.len() > 50 { &output[..50] } else { output });
        }
    }

    // Example 7: Domain-specific system prompts
    println!("\n🎯 Step 7: Domain-Specific System Prompts");
    let domains = vec![
        DomainExpertise::Metabolomics,
        DomainExpertise::Proteomics,
        DomainExpertise::MassSpectrometry,
    ];

    for domain in domains {
        let prompt = engine.create_domain_system_prompt(&domain);
        println!("  {:?}: {}...", domain,
            if prompt.len() > 80 { &prompt[..80] } else { &prompt });
    }

    // Example 8: Expected workflow for large datasets
    println!("\n🚀 Step 8: Large Dataset Processing Workflow");
    println!("For 100GB+ datasets, the recommended workflow is:");
    println!("  1. Prepare Buhera scripts in text format (one OBJECTIVE per line group)");
    println!("  2. Set appropriate memory limits based on available RAM");
    println!("  3. Use streaming processing with chunked data handling");
    println!("  4. Monitor memory usage and flush to disk when needed");
    println!("  5. Create domain-specific validation suites");
    println!("  6. Validate expert models using Buhera script responses");
    println!("  7. Deploy models for inference via Ollama");

    println!("\n✨ Key Innovation: Buhera Script-Based Knowledge Distillation");
    println!("Instead of storing massive raw MS data, we store compact query-response pairs:");
    println!("  • Input: Buhera script (objective + constraints)");
    println!("  • Output: Execution result (annotations + evidence scores)");
    println!("  • Validation: Test model's ability to execute new Buhera scripts");
    println!("  • Efficiency: ~1KB per knowledge unit vs GBs per raw dataset");

    println!("\n🏁 Example completed successfully!");

    Ok(())
}

/// Create sample Buhera scripts for demonstration
fn create_sample_buhera_scripts() -> String {
    vec![
        "OBJECTIVE biomarker_discovery FOR diabetes\nEVIDENCE_PRIORITY mass_match ms2_fragmentation\nVALIDATE minimum_sample_size > 50\n",
        "OBJECTIVE pathway_analysis FOR glycolysis\nEVIDENCE_PRIORITY pathway_membership isotope_pattern\nVALIDATE statistical_power > 0.8\n",
        "OBJECTIVE compound_identification FOR glucose\nEVIDENCE_PRIORITY mass_match retention_time\nVALIDATE mass_accuracy < 5.0\n",
        "OBJECTIVE protein_identification FOR insulin\nEVIDENCE_PRIORITY ms2_fragmentation mass_match\nVALIDATE minimum_sample_size > 30\n",
        "OBJECTIVE metabolite_profiling FOR cancer_biomarkers\nEVIDENCE_PRIORITY pathway_membership spectral_similarity\nVALIDATE effect_size > 0.3\n",
        "OBJECTIVE lipid_analysis FOR membrane_composition\nEVIDENCE_PRIORITY isotope_pattern neutral_loss\nVALIDATE coverage > 0.7\n",
        "OBJECTIVE drug_metabolism FOR pharmaceutical_analysis\nEVIDENCE_PRIORITY ms2_fragmentation adduct_formation\nVALIDATE sensitivity > 0.9\n",
        "OBJECTIVE environmental_monitoring FOR pollutant_detection\nEVIDENCE_PRIORITY mass_match literature_support\nVALIDATE specificity > 0.95\n",
        "OBJECTIVE food_safety FOR contaminant_screening\nEVIDENCE_PRIORITY spectral_similarity retention_time\nVALIDATE detection_limit < 1.0\n",
        "OBJECTIVE clinical_diagnostics FOR disease_markers\nEVIDENCE_PRIORITY pathway_membership ms2_fragmentation\nVALIDATE reproducibility > 0.85\n",
    ].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_script_creation() {
        let scripts = create_sample_buhera_scripts();
        assert!(!scripts.is_empty());
        assert!(scripts.contains("OBJECTIVE"));
        assert!(scripts.contains("EVIDENCE_PRIORITY"));
        assert!(scripts.contains("VALIDATE"));

        // Count number of objectives
        let objective_count = scripts.matches("OBJECTIVE").count();
        assert_eq!(objective_count, 10);
    }

    #[tokio::test]
    async fn test_knowledge_distillation_example_components() {
        // Test that we can create validation suites
        let suite = create_validation_suite(DomainExpertise::Metabolomics);
        assert!(!suite.test_scripts.is_empty());

        // Test that we can create the engine
        let engine = KnowledgeDistillationEngine::new(
            "http://localhost:11434".to_string(),
            "llama3".to_string(),
            4.0,
        );
        assert_eq!(engine.stream_buffer_size, 1000);

        // Test script parsing
        let script = "OBJECTIVE test FOR example\nEVIDENCE_PRIORITY mass_match";
        let result = KnowledgeDistillationEngine::parse_script_text(script);
        assert!(result.is_ok());
    }
}

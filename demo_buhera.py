#!/usr/bin/env python3
"""
Buhera + Lavoisier Integration Demo

This script demonstrates how Buhera's goal-directed scripting language
integrates with Lavoisier's AI modules for surgical precision mass spectrometry.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🔬 Buhera + Lavoisier Integration Demo")
    print("=" * 50)
    
    # Check if Buhera is built
    buhera_path = Path("lavoisier-buhera/target/release/buhera")
    if not buhera_path.exists():
        print("❌ Buhera not found. Building...")
        build_buhera()
    
    # Generate example script
    print("\n1. 📝 Generating Example Buhera Script")
    generate_example_script()
    
    # Validate the script
    print("\n2. ✅ Validating Experimental Logic")
    validate_script()
    
    # Show integration potential
    print("\n3. 🧠 Lavoisier AI Integration Overview")
    show_integration_overview()
    
    # Demonstrate key features
    print("\n4. 🎯 Key Innovation: Goal-Directed Evidence Networks")
    demonstrate_goal_directed_analysis()
    
    print("\n✨ Demo Complete!")
    print("\nNext steps:")
    print("• Review the generated diabetes_biomarker_discovery.bh script")
    print("• Explore README_BUHERA.md for full documentation")
    print("• Try: ./lavoisier-buhera/target/release/buhera validate diabetes_biomarker_discovery.bh")

def build_buhera():
    """Build the Buhera language implementation."""
    print("Building Buhera language...")
    try:
        subprocess.run(["cargo", "build", "--release"], 
                      cwd="lavoisier-buhera", 
                      check=True, 
                      capture_output=True)
        print("✅ Buhera built successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        sys.exit(1)

def generate_example_script():
    """Generate a comprehensive example script."""
    script_content = '''// diabetes_biomarker_discovery.bh
// Demonstration of surgical precision mass spectrometry

import lavoisier.mzekezeke
import lavoisier.hatata
import lavoisier.zengeza

// 🎯 OBJECTIVE-FIRST DESIGN
// Every Buhera script starts with explicit scientific goals
objective DiabetesBiomarkerDiscovery:
    target: "identify metabolites predictive of diabetes progression"
    success_criteria: "sensitivity >= 0.85 AND specificity >= 0.85"
    evidence_priorities: "pathway_membership,ms2_fragmentation,mass_match"
    biological_constraints: "glycolysis_upregulated,insulin_resistance"
    statistical_requirements: "sample_size >= 30, power >= 0.8"

// ✅ PRE-FLIGHT VALIDATION
// Catch experimental flaws BEFORE wasting time and resources
validate InstrumentCapability:
    check_instrument_capability
    if target_concentration < instrument_detection_limit:
        abort("Orbitrap cannot detect picomolar glucose metabolites")

validate StatisticalPower:
    check_sample_size
    if sample_size < 30:
        warn("Small sample size may reduce biomarker discovery power")

// 🔬 GOAL-DIRECTED ANALYSIS PHASES
// Every step optimized for the specific objective
phase DataAcquisition:
    dataset = load_dataset(
        file_path: "diabetes_samples.mzML",
        metadata: "clinical_data.csv",
        focus: "diabetes_progression_markers"
    )

phase EvidenceBuilding:
    // The key innovation: Evidence network already knows the objective
    evidence_network = lavoisier.mzekezeke.build_evidence_network(
        data: dataset,
        objective: "diabetes_biomarker_discovery",
        pathway_focus: ["glycolysis", "gluconeogenesis", "lipid_metabolism"],
        evidence_types: ["pathway_membership", "ms2_fragmentation", "mass_match"]
    )

phase BayesianInference:
    // Validation optimized for biomarker discovery
    annotations = lavoisier.hatata.validate_with_objective(
        evidence_network: evidence_network,
        objective: "diabetes_biomarker_discovery",
        confidence_threshold: 0.85
    )

phase ResultsValidation:
    if annotations.confidence > 0.85:
        generate_biomarker_report(annotations)
    else:
        suggest_improvements(annotations)
'''
    
    with open("diabetes_biomarker_discovery.bh", "w") as f:
        f.write(script_content)
    
    print("✅ Generated example script: diabetes_biomarker_discovery.bh")
    print("📋 Script includes:")
    print("   • Explicit objective with success criteria")
    print("   • Pre-flight validation rules")
    print("   • Goal-directed analysis phases")
    print("   • Integration with Lavoisier AI modules")

def validate_script():
    """Demonstrate script validation."""
    print("Running Buhera validation...")
    
    try:
        result = subprocess.run([
            "./lavoisier-buhera/target/release/buhera", 
            "validate", 
            "diabetes_biomarker_discovery.bh"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Validation output:")
            print(result.stdout)
        else:
            print("⚠️ Validation found issues:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Validation timed out")
    except FileNotFoundError:
        print("❌ Buhera binary not found - trying to build...")
        build_buhera()

def show_integration_overview():
    """Show how Buhera integrates with Lavoisier AI modules."""
    print("🧠 Lavoisier AI Modules Enhanced with Goal-Direction:")
    print()
    
    integration_examples = [
        {
            "module": "Mzekezeke (Bayesian Networks)",
            "traditional": "Generic evidence network for any analysis",
            "buhera": "Goal-directed network optimized for diabetes biomarkers"
        },
        {
            "module": "Hatata (MDP Validation)", 
            "traditional": "Generic data quality validation",
            "buhera": "Validates progress toward specific objective"
        },
        {
            "module": "Zengeza (Noise Reduction)",
            "traditional": "Generic noise removal", 
            "buhera": "Preserves signals relevant to diabetes pathways"
        }
    ]
    
    for example in integration_examples:
        print(f"📊 {example['module']}")
        print(f"   Before: {example['traditional']}")
        print(f"   With Buhera: {example['buhera']}")
        print()

def demonstrate_goal_directed_analysis():
    """Demonstrate the key innovation of goal-directed analysis."""
    print("🎭 The Surgical Precision Difference:")
    print()
    
    print("❌ Traditional Approach:")
    print("   1. Load data → 2. Run generic analysis → 3. Hope results are relevant")
    print("   Problem: Analysis doesn't know what you're trying to achieve")
    print()
    
    print("✅ Buhera Approach:")
    print("   1. Declare objective → 2. Validate feasibility → 3. Execute goal-directed analysis")
    print("   Innovation: Every step optimized for your specific research question")
    print()
    
    print("🔬 Example: Diabetes Biomarker Discovery")
    print("   • Evidence network weights pathway membership higher than generic mass matches")
    print("   • Noise reduction preserves glucose metabolism signals")
    print("   • Validation checks biomarker-specific performance criteria")
    print("   • Early failure if statistical power insufficient for biomarker discovery")
    print()
    
    print("📈 Result: 'Surgical precision' analysis with measurable improvements:")
    print("   • Reduced false discoveries through objective-focused evidence weighting")
    print("   • Early detection of experimental design flaws")
    print("   • Reproducible scientific reasoning encoded as executable scripts")

if __name__ == "__main__":
    main() 
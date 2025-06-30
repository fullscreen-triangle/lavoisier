#!/usr/bin/env python3
"""
MTBLS1707 Comparative Experiment Runner
Orchestrates traditional vs Buhera analysis comparison
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def check_prerequisites():
    """Check if dataset and dependencies are available"""
    print("🔍 Checking prerequisites...")
    
    # Check dataset
    dataset_path = "public/laboratory/MTBLS1707/negetive_hilic"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please ensure MTBLS1707 dataset is in the correct location")
        return False
    
    # Check sample files
    required_samples = [
        "H10_MH_E_neg_hilic.mzML",
        "L10_MH_E_neg_hilic.mzML", 
        "QC1_neg_hilic.mzML"
    ]
    
    missing_samples = []
    for sample in required_samples:
        if not os.path.exists(os.path.join(dataset_path, sample)):
            missing_samples.append(sample)
    
    if missing_samples:
        print(f"⚠️  Missing sample files: {missing_samples}")
        print("Experiment will proceed with available samples")
    
    print("✅ Prerequisites check completed")
    return True

def run_traditional_analysis():
    """Run traditional Lavoisier analysis"""
    print("\n🧪 Phase 1: Traditional Lavoisier Analysis")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, 
            "experiments/run_traditional_analysis.py"
        ], capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print("✅ Traditional analysis completed successfully")
            return True
        else:
            print(f"❌ Traditional analysis failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Traditional analysis timed out")
        return False
    except Exception as e:
        print(f"❌ Error running traditional analysis: {e}")
        return False

def run_buhera_analysis():
    """Run Buhera-enhanced analysis"""
    print("\n🎯 Phase 2: Buhera-Enhanced Analysis")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable,
            "experiments/run_buhera_analysis.py" 
        ], capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print("✅ Buhera analysis completed successfully")
            return True
        else:
            print(f"❌ Buhera analysis failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Buhera analysis timed out")
        return False
    except Exception as e:
        print(f"❌ Error running Buhera analysis: {e}")
        return False

def run_comparative_analysis():
    """Run comparative analysis and generate report"""
    print("\n📊 Phase 3: Comparative Analysis")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable,
            "experiments/compare_results.py"
        ], capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0:
            print("✅ Comparative analysis completed successfully")
            return True
        else:
            print(f"❌ Comparative analysis failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Comparative analysis timed out")
        return False
    except Exception as e:
        print(f"❌ Error running comparative analysis: {e}")
        return False

def display_summary():
    """Display experiment summary"""
    print("\n📋 Experiment Summary")
    print("=" * 50)
    
    results_dir = "experiments/results"
    
    # Check for results files
    traditional_results = os.path.join(results_dir, "traditional", "traditional_analysis_summary.json")
    buhera_results = os.path.join(results_dir, "buhera", "buhera_analysis_summary.json") 
    comparison_report = os.path.join(results_dir, "comparison", "comparative_analysis_report.md")
    comparison_plots = os.path.join(results_dir, "comparison", "comparison_plots.png")
    
    if os.path.exists(traditional_results):
        print(f"✅ Traditional results: {traditional_results}")
    else:
        print("❌ Traditional results missing")
        
    if os.path.exists(buhera_results):
        print(f"✅ Buhera results: {buhera_results}")
    else:
        print("❌ Buhera results missing")
        
    if os.path.exists(comparison_report):
        print(f"📄 Comparison report: {comparison_report}")
    else:
        print("❌ Comparison report missing")
        
    if os.path.exists(comparison_plots):
        print(f"📊 Comparison plots: {comparison_plots}")
    else:
        print("❌ Comparison plots missing")

def main():
    """Main experiment execution"""
    print("🧪 MTBLS1707 Traditional vs Buhera Comparative Experiment")
    print("=" * 70)
    print("""
This experiment compares Traditional Lavoisier Analysis with Buhera-Enhanced 
Analysis using the MTBLS1707 sheep organ metabolomics dataset.

The experiment will:
1. Run traditional analysis on selected samples
2. Run Buhera analysis with multiple objectives  
3. Generate comparative metrics and visualizations
4. Produce a comprehensive analysis report

Estimated time: 30-60 minutes depending on system performance
""")
    
    # Confirm execution
    response = input("Proceed with experiment? (y/N): ")
    if response.lower() != 'y':
        print("Experiment cancelled")
        return
    
    start_time = time.time()
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites not met. Please fix issues and try again.")
        return
    
    # Create results directory
    os.makedirs("experiments/results", exist_ok=True)
    
    # Phase 1: Traditional analysis
    traditional_success = run_traditional_analysis()
    
    # Phase 2: Buhera analysis  
    buhera_success = run_buhera_analysis()
    
    # Phase 3: Comparative analysis (only if both succeeded)
    comparison_success = False
    if traditional_success and buhera_success:
        comparison_success = run_comparative_analysis()
    elif not traditional_success:
        print("⚠️  Skipping comparison - traditional analysis failed")
    elif not buhera_success:
        print("⚠️  Skipping comparison - Buhera analysis failed")
    
    # Summary
    total_time = time.time() - start_time
    
    print(f"\n🏁 Experiment Completed in {total_time/60:.1f} minutes")
    print("=" * 50)
    
    if traditional_success and buhera_success and comparison_success:
        print("✅ All phases completed successfully!")
        print("\n📊 Key Results:")
        print("- Traditional analysis completed")
        print("- Buhera analysis with multiple objectives completed")
        print("- Comparative analysis and visualizations generated")
        print(f"- Results available in: experiments/results/")
        print(f"- Main report: experiments/results/comparison/comparative_analysis_report.md")
    else:
        print("⚠️  Experiment completed with issues:")
        if not traditional_success:
            print("  - Traditional analysis failed")
        if not buhera_success:
            print("  - Buhera analysis failed")
        if not comparison_success:
            print("  - Comparative analysis failed")
    
    display_summary()
    
    print(f"\n📁 All results saved to: experiments/results/")

if __name__ == "__main__":
    main() 
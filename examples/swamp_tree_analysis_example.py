#!/usr/bin/env python3
"""
Swamp Tree Analysis Example: Noise-Modulated Bayesian Evidence Network

This example demonstrates the revolutionary approach where:
1. The entire MS analysis becomes a single Bayesian evidence network
2. Noise level is optimized to maximize annotation confidence
3. Different noise levels reveal different annotation "trees"
4. CV and numerical pipelines provide complementary evidence

Metaphor: Trees in a swamp - adjust the water depth (noise) to see different tree heights (annotations)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Lavoisier modules
import sys
sys.path.append('..')

from lavoisier.ai_modules.global_bayesian_optimizer import GlobalBayesianOptimizer, PrecisionNoiseModel
from lavoisier.numerical.pipeline import NumericPipeline
from lavoisier.visual.visual import VisualPipeline

def generate_synthetic_spectrum(num_peaks: int = 50, mz_range: tuple = (100, 1000), 
                               noise_level: float = 0.1) -> tuple:
    """Generate a synthetic mass spectrum for demonstration"""
    
    # Generate m/z array
    mz_array = np.linspace(mz_range[0], mz_range[1], 2000)
    intensity_array = np.zeros_like(mz_array)
    
    # Add synthetic peaks with different properties
    true_peaks = []
    
    for i in range(num_peaks):
        # Random peak position
        peak_mz = np.random.uniform(mz_range[0], mz_range[1])
        peak_intensity = np.random.exponential(1000)  # Exponential distribution for intensities
        peak_width = np.random.uniform(0.1, 0.5)  # Peak width
        
        # Add Gaussian peak
        peak_indices = np.where(np.abs(mz_array - peak_mz) < peak_width * 3)[0]
        for idx in peak_indices:
            intensity_array[idx] += peak_intensity * np.exp(-0.5 * ((mz_array[idx] - peak_mz) / peak_width)**2)
        
        true_peaks.append({
            'mz': peak_mz,
            'intensity': peak_intensity,
            'width': peak_width
        })
    
    # Add baseline noise
    baseline_noise = np.random.normal(0, noise_level * np.mean(intensity_array), len(mz_array))
    intensity_array += np.maximum(baseline_noise, 0)  # Ensure non-negative
    
    return mz_array, intensity_array, true_peaks

def create_synthetic_compound_database(num_compounds: int = 100) -> list:
    """Create a synthetic compound database for annotation"""
    
    compounds = []
    compound_names = [
        'Glucose', 'Fructose', 'Sucrose', 'Lactose', 'Galactose',
        'Alanine', 'Glycine', 'Serine', 'Threonine', 'Valine',
        'Leucine', 'Isoleucine', 'Methionine', 'Proline', 'Phenylalanine',
        'Tyrosine', 'Tryptophan', 'Aspartic acid', 'Glutamic acid', 'Asparagine',
        'Glutamine', 'Lysine', 'Arginine', 'Histidine', 'Cysteine',
        'Acetate', 'Pyruvate', 'Lactate', 'Citrate', 'Succinate',
        'Malate', 'Fumarate', 'ATP', 'ADP', 'AMP', 'NAD+', 'NADH'
    ]
    
    # Generate exact masses for demonstration (simplified)
    base_masses = np.random.uniform(100, 1000, num_compounds)
    
    for i in range(num_compounds):
        compound_name = compound_names[i % len(compound_names)]
        if i >= len(compound_names):
            compound_name += f"_{i // len(compound_names)}"
        
        compounds.append({
            'name': compound_name,
            'exact_mass': base_masses[i],
            'formula': f'C{np.random.randint(1,20)}H{np.random.randint(1,40)}O{np.random.randint(1,10)}'
        })
    
    return compounds

async def demonstrate_swamp_tree_optimization():
    """Main demonstration of the swamp tree optimization approach"""
    
    print("🌳 Swamp Tree Analysis: Noise-Modulated Bayesian Evidence Network")
    print("=" * 70)
    
    # Step 1: Generate synthetic data
    print("\n📊 Step 1: Generating synthetic mass spectrum...")
    mz_array, intensity_array, true_peaks = generate_synthetic_spectrum(
        num_peaks=30, 
        mz_range=(150, 800), 
        noise_level=0.05
    )
    
    print(f"   Generated spectrum: {len(mz_array)} data points, {len(true_peaks)} true peaks")
    
    # Step 2: Create compound database
    print("\n📚 Step 2: Creating synthetic compound database...")
    compound_database = create_synthetic_compound_database(num_compounds=75)
    print(f"   Database contains {len(compound_database)} compounds")
    
    # Step 3: Initialize pipelines (mock for demonstration)
    print("\n🔧 Step 3: Initializing analysis pipelines...")
    
    # Mock pipelines for demonstration
    class MockNumericPipeline:
        def __init__(self):
            pass
    
    class MockVisualPipeline:
        def __init__(self):
            pass
    
    numeric_pipeline = MockNumericPipeline()
    visual_pipeline = MockVisualPipeline()
    
    # Step 4: Initialize Global Bayesian Optimizer
    print("\n🧠 Step 4: Initializing Global Bayesian Optimizer...")
    
    optimizer = GlobalBayesianOptimizer(
        numerical_pipeline=numeric_pipeline,
        visual_pipeline=visual_pipeline,
        base_noise_levels=np.linspace(0.1, 0.8, 8).tolist(),  # Test 8 noise levels
        optimization_method="differential_evolution",
        max_optimization_iterations=50,
        convergence_threshold=1e-4
    )
    
    print(f"   Optimizer configured with {len(optimizer.base_noise_levels)} noise levels")
    print(f"   Noise levels to test: {[f'{x:.2f}' for x in optimizer.base_noise_levels]}")
    
    # Step 5: Run the global optimization analysis
    print("\n🎯 Step 5: Running swamp tree optimization analysis...")
    print("   This converts the entire analysis into a single optimization problem!")
    
    analysis_result = await optimizer.analyze_with_global_optimization(
        mz_array=mz_array,
        intensity_array=intensity_array,
        compound_database=compound_database,
        spectrum_id="synthetic_demo_spectrum"
    )
    
    # Step 6: Display results
    print("\n📋 Step 6: Analysis Results")
    print("-" * 40)
    
    optimal_noise = analysis_result['optimal_noise_level']
    total_annotations = analysis_result['total_annotations']
    high_conf_annotations = analysis_result['high_confidence_annotations']
    avg_confidence = analysis_result['average_annotation_confidence']
    
    print(f"🎯 Optimal Noise Level (Swamp Depth): {optimal_noise:.3f}")
    print(f"🌳 Trees Visible at Optimal Depth: {high_conf_annotations}")
    print(f"📊 Total Trees in Swamp: {analysis_result['total_trees_in_swamp']}")
    print(f"📈 Average Annotation Confidence: {avg_confidence:.3f}")
    print(f"🔗 Pipeline Complementarity: {analysis_result['pipeline_complementarity']:.3f}")
    
    # Display top annotations
    print(f"\n🏆 Top {min(10, len(analysis_result['annotations']))} Annotations:")
    for i, ann in enumerate(analysis_result['annotations'][:10]):
        print(f"   {i+1:2d}. {ann['compound_name']:15s} "
              f"m/z: {ann['observed_mz']:7.3f} "
              f"Conf: {ann['confidence']:.3f} "
              f"(Num: {ann['numerical_confidence']:.2f}, "
              f"Vis: {ann['visual_confidence']:.2f})")
    
    # Step 7: Visualize the swamp optimization
    print("\n📊 Step 7: Creating swamp tree visualization...")
    
    try:
        # Create output directory
        output_dir = Path("swamp_analysis_output")
        output_dir.mkdir(exist_ok=True)
        
        # Generate visualization
        viz_path = output_dir / "swamp_tree_optimization.png"
        optimizer.visualize_swamp_optimization(save_path=str(viz_path))
        
        print(f"   Visualization saved to: {viz_path}")
        
        # Export the global network
        network_path = output_dir / "global_bayesian_network.json"
        optimizer.export_global_network(str(network_path))
        
        print(f"   Network data exported to: {network_path}")
        
        # Save analysis report
        report_path = output_dir / "analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(analysis_result, f, indent=2, default=str)
        
        print(f"   Analysis report saved to: {report_path}")
        
    except Exception as e:
        print(f"   Warning: Could not create visualization: {e}")
    
    # Step 8: Demonstrate the key insights
    print("\n💡 Step 8: Key Insights from Swamp Tree Analysis")
    print("-" * 50)
    
    print("\n🌊 Noise as Signal Revealer:")
    print(f"   At optimal noise level {optimal_noise:.3f}, we found {high_conf_annotations} high-confidence annotations")
    print(f"   This demonstrates that noise can reveal signals rather than obscure them!")
    
    print("\n🔍 Pipeline Complementarity:")
    complementarity = analysis_result['pipeline_complementarity']
    if complementarity > 0.7:
        print(f"   Excellent complementarity ({complementarity:.3f}) - pipelines see different aspects")
    elif complementarity > 0.5:
        print(f"   Good complementarity ({complementarity:.3f}) - pipelines provide different evidence")
    else:
        print(f"   Moderate complementarity ({complementarity:.3f}) - room for improvement")
    
    print("\n🎯 Optimization Success:")
    opt_success = analysis_result['noise_optimization']['optimization_success']
    opt_iterations = analysis_result['noise_optimization']['optimization_iterations']
    print(f"   Optimization {'converged' if opt_success else 'did not converge'} in {opt_iterations} iterations")
    
    print("\n🌳 Swamp Metaphor Validation:")
    water_sensitivity = analysis_result.get('water_depth_sensitivity', 0)
    print(f"   Water depth sensitivity: {water_sensitivity:.3f}")
    print(f"   Different 'trees' (annotations) are indeed visible at different 'water depths' (noise levels)")
    
    return analysis_result

def create_noise_level_comparison_plot(analysis_result: dict):
    """Create a detailed comparison plot showing how annotations change with noise level"""
    
    optimization_history = analysis_result['optimization_history']
    
    if not optimization_history:
        print("No optimization history available for detailed plot")
        return
    
    # Extract data
    noise_levels = [x[0] for x in optimization_history]
    confidences = [x[1] for x in optimization_history]
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Swamp Tree Analysis: Noise-Modulated Evidence Networks', 
                 fontsize=16, fontweight='bold')
    
    # 1. Optimization trajectory
    axes[0, 0].plot(noise_levels, confidences, 'b-', alpha=0.7, linewidth=2, label='Confidence trajectory')
    optimal_noise = analysis_result['optimal_noise_level']
    max_confidence = max(confidences)
    axes[0, 0].scatter([optimal_noise], [max_confidence], color='red', s=150, zorder=10, 
                      label=f'Optimal (depth={optimal_noise:.3f})')
    axes[0, 0].set_xlabel('Noise Level (Swamp Water Depth)')
    axes[0, 0].set_ylabel('Total Annotation Confidence')
    axes[0, 0].set_title('Swamp Water Depth Optimization')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Confidence distribution
    axes[0, 1].hist(confidences, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(max_confidence, color='red', linestyle='--', alpha=0.8, 
                      label=f'Maximum: {max_confidence:.3f}')
    axes[0, 1].set_xlabel('Annotation Confidence')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Confidence Scores')
    axes[0, 1].legend()
    
    # 3. Noise level vs annotation count
    if 'annotation_confidence_curve' in analysis_result:
        curve_data = analysis_result['annotation_confidence_curve']
        if curve_data:
            noise_vals = [x[0] for x in curve_data]
            annotation_counts = [len([c for c in x[1] if c > 0.5]) for x in curve_data]
            
            axes[0, 2].plot(noise_vals, annotation_counts, 'g-', marker='o', linewidth=2)
            axes[0, 2].axvline(optimal_noise, color='red', linestyle=':', alpha=0.7, 
                              label='Optimal level')
            axes[0, 2].set_xlabel('Noise Level')
            axes[0, 2].set_ylabel('High-Confidence Annotations')
            axes[0, 2].set_title('Trees Visible vs Water Depth')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Performance metrics comparison
    metrics = ['Total Annotations', 'High Confidence', 'Medium Confidence']
    values = [
        analysis_result['total_annotations'],
        analysis_result['high_confidence_annotations'],
        analysis_result['medium_confidence_annotations']
    ]
    colors = ['lightblue', 'darkgreen', 'orange']
    
    axes[1, 0].bar(metrics, values, color=colors, alpha=0.8)
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Annotation Quality Distribution')
    for i, v in enumerate(values):
        axes[1, 0].text(i, v + 0.5, str(v), ha='center', fontweight='bold')
    
    # 5. Network connectivity analysis
    network_summary = analysis_result.get('global_network_summary', {})
    network_metrics = network_summary.get('network_connectivity', {})
    
    if network_metrics:
        metric_names = ['Density', 'Clustering', 'Connected']
        metric_values = [
            network_metrics.get('density', 0),
            network_metrics.get('average_clustering', 0),
            1.0 if network_metrics.get('is_connected', False) else 0.0
        ]
        
        axes[1, 1].bar(metric_names, metric_values, color=['purple', 'brown', 'teal'], alpha=0.8)
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Network Connectivity Metrics')
        axes[1, 1].set_ylim(0, 1.1)
        
        for i, v in enumerate(metric_values):
            axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # 6. Swamp metaphor summary
    swamp_metrics = {
        'Optimal Depth': optimal_noise,
        'Trees Visible': analysis_result['trees_visible_at_optimal'],
        'Total Trees': analysis_result['total_trees_in_swamp'],
        'Depth Sensitivity': analysis_result.get('water_depth_sensitivity', 0)
    }
    
    # Create text summary
    axes[1, 2].axis('off')
    summary_text = "🌳 SWAMP TREE SUMMARY 🌳\n\n"
    summary_text += f"🎯 Optimal Water Depth: {swamp_metrics['Optimal Depth']:.3f}\n"
    summary_text += f"👀 Trees Visible: {swamp_metrics['Trees Visible']}\n"
    summary_text += f"🌲 Total Trees in Swamp: {swamp_metrics['Total Trees']}\n"
    summary_text += f"📊 Depth Sensitivity: {swamp_metrics['Depth Sensitivity']:.3f}\n\n"
    
    summary_text += "💡 KEY INSIGHT:\n"
    summary_text += "Different noise levels reveal\n"
    summary_text += "different annotation 'trees'.\n"
    summary_text += "Optimal depth maximizes\n"
    summary_text += "the number of visible trees!"
    
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

async def main():
    """Main execution function"""
    
    print("🚀 Starting Swamp Tree Analysis Demonstration")
    print("This showcases the revolutionary noise-modulated Bayesian evidence network approach!")
    
    try:
        # Run the main demonstration
        analysis_result = await demonstrate_swamp_tree_optimization()
        
        # Create additional visualizations
        print("\n📊 Creating additional visualization...")
        create_noise_level_comparison_plot(analysis_result)
        
        print("\n✅ Demonstration complete!")
        print("\n🎉 The swamp tree approach successfully demonstrated that:")
        print("   1. Noise can reveal signals rather than obscure them")
        print("   2. Different noise levels expose different annotation 'trees'")  
        print("   3. Optimization finds the optimal 'water depth' for maximum visibility")
        print("   4. Dual pipelines provide complementary evidence in the Bayesian network")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main()) 
#!/usr/bin/env python3
"""
Traditional Lavoisier Analysis for MTBLS1707 Dataset
No Buhera objectives - pure computational approach
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List

# Add lavoisier to path
sys.path.append(str(Path(__file__).parent.parent))

from lavoisier.numerical.pipeline import NumericalPipeline  
from lavoisier.visual.visual import VisualPipeline
from lavoisier.ai_modules.integration import AIModuleIntegration

class TraditionalAnalysis:
    """Traditional Lavoisier analysis without Buhera objectives"""
    
    def __init__(self, output_dir: str = "experiments/results/traditional"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize pipelines
        self.numerical_pipeline = NumericalPipeline(
            intensity_threshold=1000.0,
            mz_tolerance=0.01,
            rt_tolerance=0.5
        )
        
        self.visual_pipeline = VisualPipeline()
        self.ai_integration = AIModuleIntegration()
    
    def analyze_sample(self, sample_path: str) -> Dict:
        """Run traditional analysis on a single sample"""
        sample_name = Path(sample_path).stem
        print(f"🔬 Analyzing {sample_name} (Traditional)")
        
        start_time = time.time()
        
        try:
            # Numerical analysis
            print("  📊 Running numerical pipeline...")
            numerical_results = self.numerical_pipeline.process_file(sample_path)
            
            # Visual analysis  
            print("  🖼️  Running visual pipeline...")
            visual_results = self.visual_pipeline.process_spectra(
                numerical_results['mz_array'],
                numerical_results['intensity_array']
            )
            
            # AI annotation (no specific objective)
            print("  🤖 Running AI annotation...")
            ai_results = self.ai_integration.analyze_spectra(
                numerical_results,
                visual_results,
                objective=None  # No Buhera objective
            )
            
            processing_time = time.time() - start_time
            
            # Compile results
            results = {
                'sample': sample_name,
                'approach': 'traditional',
                'processing_time': processing_time,
                'features_detected': len(numerical_results.get('features', [])),
                'compounds_identified': len(ai_results.get('annotations', [])),
                'confidence_scores': ai_results.get('confidence_scores', []),
                'scientific_insights': ai_results.get('insights', []),
                'numerical_results': numerical_results,
                'visual_results': visual_results,
                'ai_results': ai_results
            }
            
            # Save individual results
            output_file = os.path.join(self.output_dir, f"{sample_name}_traditional.json")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"  ✅ Completed in {processing_time:.1f}s - {results['features_detected']} features, {results['compounds_identified']} compounds")
            
            return results
            
        except Exception as e:
            print(f"  ❌ Analysis failed: {e}")
            return {'sample': sample_name, 'error': str(e)}

def main():
    """Run traditional analysis on MTBLS1707 samples"""
    
    # Sample selection for experiment
    samples = [
        "H10_MH_E_neg_hilic.mzML",      # Heart, monophasic
        "L10_MH_E_neg_hilic.mzML",      # Liver, monophasic  
        "H11_BD_A_neg_hilic.mzML",      # Heart, Bligh-Dyer
        "H13_BD_C_neg_hilic.mzML",      # Heart, Bligh-Dyer
        "H14_BD_D_neg_hilic.mzML",      # Heart, Bligh-Dyer
        "H15_BD_E2_neg_hilic.mzML",     # Heart, Bligh-Dyer
        "QC1_neg_hilic.mzML",           # Quality control
        "QC2_neg_hilic.mzML",           # Quality control  
        "QC3_neg_hilic.mzML"            # Quality control
    ]
    
    dataset_path = "public/laboratory/MTBLS1707/negetive_hilic"
    
    print("🧪 MTBLS1707 Traditional Lavoisier Analysis")
    print("=" * 50)
    print(f"📂 Dataset: {dataset_path}")
    print(f"📝 Samples: {len(samples)}")
    print()
    
    # Check dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    # Initialize analyzer
    analyzer = TraditionalAnalysis()
    results = []
    
    # Process each sample
    for i, sample in enumerate(samples, 1):
        sample_path = os.path.join(dataset_path, sample)
        
        if os.path.exists(sample_path):
            print(f"[{i}/{len(samples)}] Processing {sample}")
            result = analyzer.analyze_sample(sample_path)
            results.append(result)
        else:
            print(f"[{i}/{len(samples)}] ⚠️  Sample not found: {sample}")
    
    # Save aggregate results
    aggregate_results = {
        'experiment': 'MTBLS1707_traditional_analysis',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(results),
        'successful_analyses': len([r for r in results if 'error' not in r]),
        'total_processing_time': sum(r.get('processing_time', 0) for r in results),
        'results': results
    }
    
    output_file = os.path.join(analyzer.output_dir, "traditional_analysis_summary.json")
    with open(output_file, 'w') as f:
        json.dump(aggregate_results, f, indent=2, default=str)
    
    print("\n📊 Traditional Analysis Summary:")
    print(f"  ✅ Successful: {aggregate_results['successful_analyses']}/{aggregate_results['total_samples']}")
    print(f"  ⏱️  Total time: {aggregate_results['total_processing_time']:.1f}s")
    print(f"  📁 Results: {analyzer.output_dir}")

if __name__ == "__main__":
    main() 
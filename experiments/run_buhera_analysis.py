#!/usr/bin/env python3
"""
Buhera-Enhanced Analysis for MTBLS1707 Dataset
Goal-directed analysis with scientific validation
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List

# Add lavoisier to path
sys.path.append(str(Path(__file__).parent.parent))

from lavoisier.ai_modules.buhera_integration import BuheraIntegration

class BuheraAnalysis:
    """Buhera-enhanced analysis with scientific objectives"""
    
    def __init__(self, output_dir: str = "experiments/results/buhera"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/scripts", exist_ok=True)
        
        self.buhera_integration = BuheraIntegration()
        
        # Define experimental objectives
        self.objectives = {
            "organ_metabolomics": "Identify organ-specific metabolites",
            "extraction_optimization": "Assess extraction method efficiency", 
            "biomarker_discovery": "Discover tissue-specific biomarkers",
            "pathway_analysis": "Map metabolites to biological pathways",
            "quality_control": "Assess data quality and reproducibility"
        }
    
    def create_buhera_script(self, sample_path: str, objective: str) -> str:
        """Generate Buhera script for specific sample and objective"""
        sample_name = Path(sample_path).stem
        
        # Parse sample metadata from filename
        tissue_type, extraction_method = self._parse_sample_info(sample_name)
        
        # Objective-specific script templates
        scripts = {
            "organ_metabolomics": f'''
// Organ-specific metabolomics for {sample_name}
import lavoisier.mzekezeke
import lavoisier.hatata

objective OrganMetabolomics:
    target: "identify {tissue_type}-specific metabolites"
    success_criteria: "specificity >= 0.80 AND confidence >= 0.75"
    evidence_priorities: "tissue_specificity,pathway_membership,ms2_validation"
    biological_constraints: [
        "tissue_type: {tissue_type}",
        "organism: ovis_aries",
        "extraction: {extraction_method}"
    ]

validate TissueContext:
    if tissue_type == "pooled":
        warn("QC sample - tissue specificity may be reduced")

phase MetaboliteIdentification:
    metabolites = lavoisier.mzekezeke.identify_metabolites(
        tissue_context: "{tissue_type}",
        specificity_threshold: 0.80,
        pathway_focus: ["tissue_specific", "organ_development"]
    )

phase SpecificityValidation:
    specificity_results = lavoisier.hatata.validate_tissue_specificity(
        metabolites: metabolites,
        tissue_type: "{tissue_type}",
        cross_tissue_validation: true
    )
''',
            
            "extraction_optimization": f'''
// Extraction method assessment for {extraction_method}
import lavoisier.mzekezeke
import lavoisier.diggiden

objective ExtractionOptimization:
    target: "assess {extraction_method} efficiency for {tissue_type}"
    success_criteria: "recovery >= 0.70 AND bias < 0.20"
    evidence_priorities: "compound_recovery,method_bias,extraction_completeness"

validate ExtractionMethod:
    if extraction_method == "unknown":
        abort("Cannot assess unknown extraction method")

phase RecoveryAssessment:
    recovery = lavoisier.mzekezeke.assess_compound_recovery(
        extraction_method: "{extraction_method}",
        tissue_matrix: "{tissue_type}",
        compound_classes: ["polar", "lipids", "amino_acids"]
    )

phase BiasDetection:
    bias_analysis = lavoisier.diggiden.detect_extraction_bias(
        method: "{extraction_method}",
        expected_recovery: recovery.theoretical_recovery
    )
''',
            
            "biomarker_discovery": f'''
// Biomarker discovery for {tissue_type}
import lavoisier.mzekezeke
import lavoisier.hatata

objective BiomarkerDiscovery:
    target: "discover biomarkers for {tissue_type} tissue"
    success_criteria: "sensitivity >= 0.85 AND specificity >= 0.80"
    evidence_priorities: "tissue_specificity,abundance_significance,clinical_relevance"
    statistical_requirements: [
        "min_fold_change: 2.0",
        "max_pvalue: 0.01",
        "min_detection_frequency: 0.80"
    ]

phase CandidateIdentification:
    candidates = lavoisier.mzekezeke.identify_biomarker_candidates(
        tissue_type: "{tissue_type}",
        fold_change_threshold: 2.0,
        significance_threshold: 0.01
    )

phase BiomarkerValidation:
    validated_biomarkers = lavoisier.hatata.validate_biomarkers(
        candidates: candidates,
        tissue_specificity: "{tissue_type}",
        statistical_validation: true
    )
''',
            
            "pathway_analysis": f'''
// Pathway analysis for {tissue_type} metabolomics
import lavoisier.mzekezeke

objective PathwayAnalysis:
    target: "map metabolites to {tissue_type} pathways"
    success_criteria: "pathway_coverage >= 0.60 AND coherence >= 0.75"
    evidence_priorities: "pathway_membership,metabolite_interactions,tissue_relevance"

phase PathwayMapping:
    pathways = lavoisier.mzekezeke.map_metabolites_to_pathways(
        tissue_context: "{tissue_type}",
        databases: ["kegg", "reactome", "wikipathways"],
        organism: "ovis_aries"
    )

phase NetworkAnalysis:
    network = lavoisier.mzekezeke.analyze_metabolite_networks(
        pathways: pathways,
        tissue_specificity: "{tissue_type}",
        interaction_confidence: 0.70
    )
''',
            
            "quality_control": f'''
// Quality control assessment
import lavoisier.zengeza
import lavoisier.nicotine

objective QualityControl:
    target: "assess data quality for {sample_name}"
    success_criteria: "quality_score >= 0.85 AND cv < 0.20"
    evidence_priorities: "signal_stability,reproducibility,systematic_errors"

validate SampleType:
    if not sample_name.startswith("QC"):
        warn("Non-QC sample - metrics may differ from standards")

phase QualityAssessment:
    quality_metrics = lavoisier.zengeza.assess_data_quality(
        cv_threshold: 0.20,
        signal_noise_threshold: 10.0,
        baseline_stability: 0.95
    )

phase ReproducibilityCheck:
    reproducibility = lavoisier.nicotine.check_reproducibility(
        sample_type: "experimental",
        technical_replicates: true,
        expected_cv: 0.15
    )
'''
        }
        
        return scripts.get(objective, scripts["organ_metabolomics"])
    
    def _parse_sample_info(self, sample_name: str) -> tuple:
        """Parse tissue type and extraction method from sample name"""
        tissue_type = "unknown"
        extraction_method = "unknown"
        
        # Determine tissue type
        if sample_name.startswith("H"):
            tissue_type = "heart"
        elif sample_name.startswith("L"):
            tissue_type = "liver"
        elif sample_name.startswith("K"):
            tissue_type = "kidney"
        elif sample_name.startswith("QC"):
            tissue_type = "pooled"
            
        # Determine extraction method
        if "_MH_" in sample_name:
            extraction_method = "monophasic"
        elif "_BD_" in sample_name:
            extraction_method = "bligh_dyer"
        elif "_DCM_" in sample_name:
            extraction_method = "dcm_extraction"
        elif "_MTBE_" in sample_name:
            extraction_method = "mtbe_extraction"
        
        return tissue_type, extraction_method
    
    def analyze_sample(self, sample_path: str, objective: str) -> Dict:
        """Run Buhera analysis on sample with specific objective"""
        sample_name = Path(sample_path).stem
        print(f"🎯 Analyzing {sample_name} - {objective}")
        
        start_time = time.time()
        
        try:
            # Create Buhera script
            script_content = self.create_buhera_script(sample_path, objective)
            script_path = os.path.join(
                self.output_dir, "scripts", 
                f"{sample_name}_{objective}.bh"
            )
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            print(f"  📝 Created script: {Path(script_path).name}")
            
            # Validate script
            print("  ✅ Validating script...")
            validation_result = self.buhera_integration.validate_script(script_path)
            
            if not validation_result['valid']:
                print(f"  ⚠️  Validation warnings: {len(validation_result['errors'])}")
            
            # Execute analysis
            print("  🚀 Executing analysis...")
            analysis_results = self.buhera_integration.execute_script(
                script_path, 
                data_file=sample_path
            )
            
            processing_time = time.time() - start_time
            
            # Compile results
            results = {
                'sample': sample_name,
                'objective': objective,
                'approach': 'buhera',
                'processing_time': processing_time,
                'validation_errors': validation_result.get('errors', []),
                'validation_warnings': validation_result.get('warnings', []),
                'features_detected': len(analysis_results.get('features', [])),
                'compounds_identified': len(analysis_results.get('annotations', [])),
                'confidence_scores': analysis_results.get('confidence_scores', []),
                'scientific_insights': analysis_results.get('insights', []),
                'objective_achievement': analysis_results.get('objective_success', False),
                'evidence_quality': analysis_results.get('evidence_quality', 0.0),
                'script_path': script_path,
                'analysis_results': analysis_results
            }
            
            # Save results
            output_file = os.path.join(
                self.output_dir, 
                f"{sample_name}_{objective}_buhera.json"
            )
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"  ✅ Completed in {processing_time:.1f}s")
            print(f"     - Features: {results['features_detected']}")
            print(f"     - Compounds: {results['compounds_identified']}")
            print(f"     - Validation issues: {len(results['validation_errors'])}")
            
            return results
            
        except Exception as e:
            print(f"  ❌ Analysis failed: {e}")
            return {
                'sample': sample_name, 
                'objective': objective,
                'error': str(e)
            }

def main():
    """Run Buhera analysis on MTBLS1707 samples"""
    
    # Sample selection
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
    
    print("🎯 MTBLS1707 Buhera-Enhanced Analysis")
    print("=" * 50)
    print(f"📂 Dataset: {dataset_path}")
    print(f"📝 Samples: {len(samples)}")
    print()
    
    # Check dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    # Initialize analyzer
    analyzer = BuheraAnalysis()
    all_results = []
    
    print(f"🎯 Objectives: {list(analyzer.objectives.keys())}")
    print()
    
    # Process each sample with each objective
    total_analyses = len(samples) * len(analyzer.objectives)
    current_analysis = 0
    
    for sample in samples:
        sample_path = os.path.join(dataset_path, sample)
        
        if not os.path.exists(sample_path):
            print(f"⚠️  Sample not found: {sample}")
            continue
            
        print(f"📁 Processing {sample}:")
        
        for objective in analyzer.objectives.keys():
            current_analysis += 1
            print(f"  [{current_analysis}/{total_analyses}] {objective}")
            
            result = analyzer.analyze_sample(sample_path, objective)
            all_results.append(result)
    
    # Save aggregate results
    successful_analyses = [r for r in all_results if 'error' not in r]
    total_validation_errors = sum(
        len(r.get('validation_errors', [])) for r in successful_analyses
    )
    
    aggregate_results = {
        'experiment': 'MTBLS1707_buhera_analysis',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_analyses': len(all_results),
        'successful_analyses': len(successful_analyses),
        'total_processing_time': sum(r.get('processing_time', 0) for r in all_results),
        'total_validation_errors': total_validation_errors,
        'objectives_tested': list(analyzer.objectives.keys()),
        'results': all_results
    }
    
    output_file = os.path.join(analyzer.output_dir, "buhera_analysis_summary.json")
    with open(output_file, 'w') as f:
        json.dump(aggregate_results, f, indent=2, default=str)
    
    print("\n📊 Buhera Analysis Summary:")
    print(f"  ✅ Successful: {len(successful_analyses)}/{len(all_results)}")
    print(f"  ⏱️  Total time: {aggregate_results['total_processing_time']:.1f}s")
    print(f"  🔍 Validation errors caught: {total_validation_errors}")
    print(f"  📁 Results: {analyzer.output_dir}")

if __name__ == "__main__":
    main() 
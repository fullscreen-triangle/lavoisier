#!/usr/bin/env python3
"""
STEP 2: QUALITY CONTROL VALIDATION EXPERIMENT
============================================

OBJECTIVE: Validate spectrum quality assessment capabilities
HYPOTHESIS: The quality control module can effectively filter low-quality spectra

EXPERIMENT PROCEDURE:
1. Load mass spectrometry data
2. Apply quality control metrics
3. Filter spectra by quality thresholds
4. Analyze quality distribution
5. Generate quality assessment visualizations

Run: python step_02_quality_control_experiment.py
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Ensure core modules are available
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Step 2: Quality Control Validation Experiment"""

    print("🧪 STEP 2: QUALITY CONTROL VALIDATION EXPERIMENT")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Objective: Validate spectrum quality assessment and filtering")
    print("=" * 60)

    # Create step-specific results directory
    step_dir = Path("step_results") / "step_02_quality_control"
    step_dir.mkdir(parents=True, exist_ok=True)

    # Initialize step log
    log_file = step_dir / "quality_control_log.txt"

    def log_and_print(message):
        """Log message to both console and file"""
        print(message)
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()}: {message}\n")

    log_and_print("🔬 STEP 2 EXPERIMENTAL SETUP")
    log_and_print("-" * 30)

    try:
        # Import required components
        log_and_print("📦 Loading quality control components...")
        from core.mzml_reader import StandaloneMzMLReader
        from core.numerical_pipeline import QualityControlModule

        reader = StandaloneMzMLReader()
        qc_module = QualityControlModule()
        log_and_print("✅ Quality control components loaded successfully")

        # Define test datasets
        test_files = [
            "PL_Neg_Waters_qTOF.mzML",
            "TG_Pos_Thermo_Orbi.mzML"
        ]

        log_and_print(f"📊 Testing quality control on {len(test_files)} datasets:")
        for i, file in enumerate(test_files, 1):
            log_and_print(f"  {i}. {file}")

        # Define quality thresholds for testing
        quality_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

        log_and_print(f"🎯 Quality thresholds to test: {quality_thresholds}")

        # Initialize results
        step_results = {
            'step_metadata': {
                'step_number': 2,
                'step_name': 'Quality Control Validation',
                'start_time': datetime.now().isoformat(),
                'objective': 'Validate spectrum quality assessment and filtering capabilities'
            },
            'quality_assessment_results': {},
            'threshold_analysis': {},
            'quality_metrics_performance': {},
            'step_conclusion': {}
        }

        log_and_print("\n🚀 STARTING QUALITY CONTROL VALIDATION")
        log_and_print("-" * 45)

        all_quality_scores = []
        all_assessment_times = []
        dataset_quality_distributions = {}

        # Test each dataset
        for dataset_num, test_file in enumerate(test_files, 1):
            log_and_print(f"\n📊 DATASET {dataset_num}: {test_file}")
            log_and_print("-" * 40)

            dataset_start_time = time.time()

            try:
                # STEP 2.1: Load data for quality assessment
                log_and_print("STEP 2.1: Loading data for quality assessment...")
                spectra = reader.load_mzml(test_file)
                log_and_print(f"  📂 Loaded {len(spectra)} spectra for quality analysis")

                # STEP 2.2: Individual spectrum quality assessment
                log_and_print("STEP 2.2: Assessing individual spectrum quality...")

                assessment_start = time.time()
                individual_quality_scores = []
                spectrum_quality_details = []

                for spectrum in spectra[:100]:  # Assess first 100 spectra for performance
                    quality_metrics = qc_module.assess_spectrum_quality(spectrum)
                    quality_score = quality_metrics.get('quality_score', 0.0)

                    individual_quality_scores.append(quality_score)
                    spectrum_quality_details.append({
                        'scan_id': spectrum.scan_id,
                        'quality_score': quality_score,
                        'signal_to_noise': quality_metrics.get('signal_to_noise', 0),
                        'peak_density': quality_metrics.get('peak_density', 0),
                        'base_peak_intensity': quality_metrics.get('base_peak_intensity', 0),
                        'peak_count': quality_metrics.get('peak_count', 0)
                    })

                assessment_time = time.time() - assessment_start

                log_and_print(f"  ⏱️  Quality assessment completed in {assessment_time:.2f}s")
                log_and_print(f"  📈 Assessed {len(individual_quality_scores)} spectra")

                # STEP 2.3: Dataset-level quality statistics
                log_and_print("STEP 2.3: Calculating dataset-level quality statistics...")

                dataset_stats = qc_module.dataset_statistics(spectra)
                quality_metrics = dataset_stats.get('quality_metrics', {})

                mean_quality = quality_metrics.get('mean_quality_score', 0)
                std_quality = quality_metrics.get('std_quality_score', 0)
                high_quality_count = quality_metrics.get('high_quality_spectra', 0)
                low_quality_count = quality_metrics.get('low_quality_spectra', 0)

                log_and_print(f"  📊 Dataset Quality Statistics:")
                log_and_print(f"     Mean quality score: {mean_quality:.3f}")
                log_and_print(f"     Quality std deviation: {std_quality:.3f}")
                log_and_print(f"     High quality spectra: {high_quality_count}")
                log_and_print(f"     Low quality spectra: {low_quality_count}")

                # STEP 2.4: Quality threshold analysis
                log_and_print("STEP 2.4: Analyzing quality filtering thresholds...")

                threshold_results = {}

                for threshold in quality_thresholds:
                    passed_spectra = len([q for q in individual_quality_scores if q >= threshold])
                    failed_spectra = len(individual_quality_scores) - passed_spectra
                    pass_rate = passed_spectra / max(1, len(individual_quality_scores))

                    threshold_results[threshold] = {
                        'threshold': threshold,
                        'passed_spectra': passed_spectra,
                        'failed_spectra': failed_spectra,
                        'pass_rate': pass_rate,
                        'effective_filtering': 0.1 <= pass_rate <= 0.9  # Not too strict, not too lenient
                    }

                    log_and_print(f"     Threshold {threshold:.1f}: {passed_spectra}/{len(individual_quality_scores)} pass ({pass_rate:.1%})")

                # STEP 2.5: Quality control effectiveness assessment
                log_and_print("STEP 2.5: Assessing quality control effectiveness...")

                # Check if quality control can distinguish between spectra
                quality_range = max(individual_quality_scores) - min(individual_quality_scores) if individual_quality_scores else 0
                quality_variance = np.var(individual_quality_scores) if individual_quality_scores else 0

                # Effectiveness criteria
                has_good_range = quality_range >= 0.3  # Quality scores span at least 0.3
                has_good_variance = quality_variance >= 0.01  # Sufficient variation in scores
                has_reasonable_mean = 0.2 <= mean_quality <= 0.9  # Mean quality in reasonable range

                effectiveness_score = 0.0
                if has_good_range:
                    effectiveness_score += 0.4
                if has_good_variance:
                    effectiveness_score += 0.3
                if has_reasonable_mean:
                    effectiveness_score += 0.3

                log_and_print(f"  🎯 Quality Control Effectiveness:")
                log_and_print(f"     Quality score range: {quality_range:.3f}")
                log_and_print(f"     Quality variance: {quality_variance:.4f}")
                log_and_print(f"     Effectiveness score: {effectiveness_score:.2f}/1.00")

                # STEP 2.6: Performance classification
                log_and_print("STEP 2.6: Quality control performance classification...")

                # Find optimal threshold (best balance of filtering)
                optimal_threshold = None
                best_balance_score = 0

                for threshold, result in threshold_results.items():
                    if result['effective_filtering']:
                        # Balance score: prefer moderate filtering (around 70% pass rate)
                        balance_score = 1.0 - abs(result['pass_rate'] - 0.7)
                        if balance_score > best_balance_score:
                            best_balance_score = balance_score
                            optimal_threshold = threshold

                if effectiveness_score >= 0.8 and optimal_threshold is not None:
                    performance_grade = "🟢 EXCELLENT"
                elif effectiveness_score >= 0.6 and quality_range >= 0.2:
                    performance_grade = "🟡 GOOD"
                elif effectiveness_score >= 0.4 and quality_range >= 0.1:
                    performance_grade = "🟠 ACCEPTABLE"
                else:
                    performance_grade = "🔴 NEEDS IMPROVEMENT"

                log_and_print(f"  🏆 Quality Control Performance: {performance_grade}")
                if optimal_threshold is not None:
                    log_and_print(f"  🎯 Optimal threshold: {optimal_threshold}")

                # Store results
                step_results['quality_assessment_results'][test_file] = {
                    'assessment_time': assessment_time,
                    'spectra_assessed': len(individual_quality_scores),
                    'assessment_rate': len(individual_quality_scores) / max(0.001, assessment_time),
                    'dataset_quality_statistics': {
                        'mean_quality_score': mean_quality,
                        'std_quality_score': std_quality,
                        'quality_range': quality_range,
                        'quality_variance': quality_variance,
                        'high_quality_count': high_quality_count,
                        'low_quality_count': low_quality_count
                    },
                    'threshold_analysis': threshold_results,
                    'effectiveness_metrics': {
                        'effectiveness_score': effectiveness_score,
                        'has_good_range': has_good_range,
                        'has_good_variance': has_good_variance,
                        'has_reasonable_mean': has_reasonable_mean,
                        'optimal_threshold': optimal_threshold
                    },
                    'performance_grade': performance_grade,
                    'sample_quality_scores': individual_quality_scores[:20]  # First 20 for reference
                }

                # Track overall metrics
                all_quality_scores.extend(individual_quality_scores)
                all_assessment_times.append(assessment_time)
                dataset_quality_distributions[test_file] = individual_quality_scores

                log_and_print(f"✅ Dataset {dataset_num} quality control validation completed")

            except Exception as e:
                log_and_print(f"❌ Dataset {dataset_num} quality control failed: {e}")

                step_results['quality_assessment_results'][test_file] = {
                    'assessment_success': False,
                    'error': str(e),
                    'processing_time': time.time() - dataset_start_time
                }

        # STEP 2.7: Overall analysis and conclusions
        log_and_print(f"\n" + "=" * 60)
        log_and_print("📊 STEP 2 OVERALL QUALITY CONTROL ANALYSIS")
        log_and_print("=" * 60)

        successful_assessments = len([r for r in step_results['quality_assessment_results'].values()
                                    if 'assessment_success' not in r or r.get('assessment_success', True)])
        total_datasets = len(test_files)
        success_rate = successful_assessments / max(1, total_datasets)

        if successful_assessments > 0:
            avg_assessment_time = np.mean(all_assessment_times)
            overall_mean_quality = np.mean(all_quality_scores) if all_quality_scores else 0
            overall_quality_std = np.std(all_quality_scores) if len(all_quality_scores) > 1 else 0
            total_spectra_assessed = len(all_quality_scores)

            # Cross-dataset threshold analysis
            combined_threshold_analysis = {}
            for threshold in quality_thresholds:
                total_passed = len([q for q in all_quality_scores if q >= threshold])
                total_assessed = len(all_quality_scores)
                combined_pass_rate = total_passed / max(1, total_assessed)

                combined_threshold_analysis[threshold] = {
                    'combined_pass_rate': combined_pass_rate,
                    'total_passed': total_passed,
                    'total_assessed': total_assessed
                }

            step_results['threshold_analysis'] = combined_threshold_analysis

            step_results['quality_metrics_performance'] = {
                'successful_assessments': successful_assessments,
                'total_datasets': total_datasets,
                'success_rate': success_rate,
                'avg_assessment_time': avg_assessment_time,
                'total_spectra_assessed': total_spectra_assessed,
                'overall_mean_quality': overall_mean_quality,
                'overall_quality_std': overall_quality_std,
                'avg_assessment_rate': total_spectra_assessed / max(0.001, sum(all_assessment_times))
            }

            log_and_print(f"🔢 STEP 2 PERFORMANCE METRICS:")
            log_and_print(f"   Successful quality assessments: {successful_assessments}/{total_datasets}")
            log_and_print(f"   Success rate: {success_rate:.1%}")
            log_and_print(f"   Total spectra assessed: {total_spectra_assessed}")
            log_and_print(f"   Average assessment time: {avg_assessment_time:.2f}s per dataset")
            log_and_print(f"   Overall mean quality: {overall_mean_quality:.3f}")
            log_and_print(f"   Quality score variation: {overall_quality_std:.3f}")
            log_and_print(f"   Assessment rate: {total_spectra_assessed / max(0.001, sum(all_assessment_times)):.1f} spectra/s")

            # Cross-dataset threshold recommendations
            log_and_print(f"\n🎯 CROSS-DATASET THRESHOLD ANALYSIS:")
            for threshold, analysis in combined_threshold_analysis.items():
                log_and_print(f"   Threshold {threshold:.1f}: {analysis['combined_pass_rate']:.1%} pass rate")

            # Step conclusion
            if (success_rate >= 0.8 and overall_mean_quality >= 0.3 and
                overall_quality_std >= 0.05 and avg_assessment_time <= 10.0):
                step_conclusion = "🟢 QUALITY CONTROL VALIDATION PASSED - Effective spectrum filtering"
                step_status = "validated"
            elif success_rate >= 0.6 and overall_mean_quality >= 0.2:
                step_conclusion = "🟡 QUALITY CONTROL VALIDATION PARTIAL - Good performance with minor issues"
                step_status = "functional"
            else:
                step_conclusion = "🔴 QUALITY CONTROL VALIDATION FAILED - Significant issues detected"
                step_status = "problematic"

            log_and_print(f"\n🎯 STEP 2 CONCLUSION:")
            log_and_print(f"   {step_conclusion}")

            # Find recommended threshold
            recommended_threshold = None
            for threshold in [0.3, 0.5, 0.7]:
                if threshold in combined_threshold_analysis:
                    pass_rate = combined_threshold_analysis[threshold]['combined_pass_rate']
                    if 0.4 <= pass_rate <= 0.8:  # Good balance
                        recommended_threshold = threshold
                        break

            step_results['step_conclusion'] = {
                'overall_assessment': step_conclusion,
                'step_status': step_status,
                'success_rate': success_rate,
                'overall_quality_effectiveness': overall_quality_std >= 0.05,
                'recommended_threshold': recommended_threshold,
                'key_findings': [
                    f"Quality assessment success rate: {success_rate:.1%}",
                    f"Mean quality score: {overall_mean_quality:.3f}",
                    f"Quality discrimination: {overall_quality_std:.3f} std deviation",
                    f"Assessment speed: {total_spectra_assessed / max(0.001, sum(all_assessment_times)):.1f} spectra/s"
                ],
                'recommendations': [
                    "Quality control is effective" if overall_quality_std >= 0.05 else "Improve quality discrimination",
                    f"Use threshold {recommended_threshold}" if recommended_threshold else "Calibrate quality thresholds",
                    "Assessment speed is acceptable" if avg_assessment_time <= 5.0 else "Optimize quality calculation algorithms",
                    "Quality metrics are reliable" if success_rate >= 0.8 else "Improve quality assessment robustness"
                ]
            }

        else:
            log_and_print("❌ No successful quality assessments - critical failure in Step 2")
            step_results['step_conclusion'] = {
                'overall_assessment': '🔴 CRITICAL FAILURE - Quality control completely non-functional',
                'step_status': 'failed'
            }

        # STEP 2.8: Save results and generate visualizations
        log_and_print(f"\n💾 SAVING STEP 2 RESULTS")
        log_and_print("-" * 30)

        # JSON results
        results_file = step_dir / "step_02_quality_control_results.json"
        with open(results_file, 'w') as f:
            json.dump(step_results, f, indent=2)
        log_and_print(f"📄 Saved JSON results: {results_file}")

        # CSV summary
        if successful_assessments > 0:
            csv_data = []
            for dataset, result in step_results['quality_assessment_results'].items():
                if 'dataset_quality_statistics' in result:
                    stats = result['dataset_quality_statistics']
                    csv_data.append({
                        'Dataset': dataset,
                        'Assessment_Time_s': result['assessment_time'],
                        'Spectra_Assessed': result['spectra_assessed'],
                        'Assessment_Rate': result['assessment_rate'],
                        'Mean_Quality_Score': stats['mean_quality_score'],
                        'Quality_Std_Dev': stats['std_quality_score'],
                        'Quality_Range': stats['quality_range'],
                        'High_Quality_Count': stats['high_quality_count'],
                        'Low_Quality_Count': stats['low_quality_count'],
                        'Performance_Grade': result['performance_grade'],
                        'Optimal_Threshold': result['effectiveness_metrics']['optimal_threshold']
                    })

            if csv_data:
                csv_file = step_dir / "step_02_quality_control_summary.csv"
                pd.DataFrame(csv_data).to_csv(csv_file, index=False)
                log_and_print(f"📊 Saved CSV summary: {csv_file}")

        # Generate visualizations
        try:
            from visualization.panel import create_quality_control_panel
            import matplotlib.pyplot as plt

            if successful_assessments > 0:
                viz_data = {
                    'quality_distributions': dataset_quality_distributions,
                    'threshold_analysis': combined_threshold_analysis,
                    'assessment_times': all_assessment_times,
                    'overall_quality_stats': {
                        'mean': overall_mean_quality,
                        'std': overall_quality_std,
                        'count': total_spectra_assessed
                    }
                }

                fig = create_quality_control_panel(viz_data, "Step 2: Quality Control Performance")
                viz_file = step_dir / "step_02_quality_control_performance.png"
                fig.savefig(viz_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                log_and_print(f"📊 Saved visualization: {viz_file}")

        except ImportError as e:
            log_and_print(f"⚠️  Visualization generation failed: {e}")

        log_and_print(f"\n🧪 STEP 2: QUALITY CONTROL VALIDATION COMPLETE 🧪")
        log_and_print(f"📁 Results saved to: {step_dir}")
        log_and_print(f"📋 Step log: {log_file}")

        return step_results

    except Exception as e:
        log_and_print(f"💥 CRITICAL STEP 2 FAILURE: {e}")
        import traceback
        log_and_print(f"Error details: {traceback.format_exc()}")
        return None


if __name__ == "__main__":
    print("🚀 Starting Step 2: Quality Control Validation Experiment...")

    results = main()

    if results and results.get('step_conclusion', {}).get('step_status') in ['validated', 'functional']:
        print("\n✅ STEP 2 SUCCESSFUL - Quality control validated!")
        sys.exit(0)
    else:
        print("\n❌ STEP 2 FAILED - Check results for details")
        sys.exit(1)

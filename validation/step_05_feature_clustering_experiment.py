#!/usr/bin/env python3
"""
STEP 5: FEATURE CLUSTERING VALIDATION EXPERIMENT
===============================================

OBJECTIVE: Validate feature extraction and spectrum clustering capabilities
HYPOTHESIS: The clustering module can group similar spectra based on extracted features

EXPERIMENT PROCEDURE:
1. Load mass spectrometry data
2. Extract spectral features from spectra
3. Perform clustering analysis
4. Evaluate clustering quality and effectiveness
5. Generate clustering performance visualizations

Run: python step_05_feature_clustering_experiment.py
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
    """Step 5: Feature Clustering Validation Experiment"""

    print("🧪 STEP 5: FEATURE CLUSTERING VALIDATION EXPERIMENT")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Objective: Validate feature extraction and spectrum clustering")
    print("=" * 60)

    # Create step-specific results directory
    step_dir = Path("step_results") / "step_05_feature_clustering"
    step_dir.mkdir(parents=True, exist_ok=True)

    # Initialize step log
    log_file = step_dir / "feature_clustering_log.txt"

    def log_and_print(message):
        """Log message to both console and file"""
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()}: {message}\n")

    log_and_print("🔬 STEP 5 EXPERIMENTAL SETUP")
    log_and_print("-" * 30)

    try:
        # Import required components
        log_and_print("📦 Loading feature clustering components...")
        from core.mzml_reader import StandaloneMzMLReader
        from core.numerical_pipeline import FeatureClusteringModule, QualityControlModule

        reader = StandaloneMzMLReader()
        clustering_module = FeatureClusteringModule()
        qc_module = QualityControlModule()
        log_and_print("✅ Feature clustering components loaded successfully")

        # Define test datasets
        test_files = [
            "PL_Neg_Waters_qTOF.mzML",
            "TG_Pos_Thermo_Orbi.mzML"
        ]

        # Define clustering parameters to test
        cluster_counts = [3, 5, 8, 10]

        log_and_print(f"📊 Testing clustering on {len(test_files)} datasets:")
        for i, file in enumerate(test_files, 1):
            log_and_print(f"  {i}. {file}")
        log_and_print(f"🎯 Cluster counts to test: {cluster_counts}")

        # Initialize results
        step_results = {
            'step_metadata': {
                'step_number': 5,
                'step_name': 'Feature Clustering Validation',
                'start_time': datetime.now().isoformat(),
                'objective': 'Validate feature extraction and spectrum clustering capabilities',
                'cluster_counts_tested': cluster_counts
            },
            'clustering_results': {},
            'feature_analysis': {},
            'clustering_quality_metrics': {},
            'step_conclusion': {}
        }

        log_and_print("\n🚀 STARTING FEATURE CLUSTERING VALIDATION")
        log_and_print("-" * 48)

        all_clustering_times = []
        all_feature_extraction_times = []
        all_clustering_quality_scores = []

        # Test each dataset
        for dataset_num, test_file in enumerate(test_files, 1):
            log_and_print(f"\n🔬 DATASET {dataset_num}: {test_file}")
            log_and_print("-" * 40)

            dataset_start_time = time.time()

            try:
                # STEP 5.1: Load and prepare data for clustering
                log_and_print("STEP 5.1: Loading data for clustering analysis...")
                spectra = reader.load_mzml(test_file)

                # Filter for high-quality spectra
                high_quality_spectra = []
                for spectrum in spectra:
                    quality_metrics = qc_module.assess_spectrum_quality(spectrum)
                    if quality_metrics.get('quality_score', 0) >= 0.2:
                        high_quality_spectra.append(spectrum)

                # Limit for performance testing
                clustering_spectra = high_quality_spectra[:50]  # Test first 50 spectra

                log_and_print(f"  📂 Loaded {len(spectra)} total spectra")
                log_and_print(f"  🎯 Selected {len(clustering_spectra)} spectra for clustering")

                # STEP 5.2: Feature extraction analysis
                log_and_print("STEP 5.2: Analyzing spectral feature extraction...")

                feature_extraction_start = time.time()
                extracted_features = []
                feature_extraction_errors = 0

                for spectrum in clustering_spectra:
                    try:
                        features = clustering_module.extract_spectral_features(spectrum)
                        if features:  # Non-empty feature dict
                            extracted_features.append(features)
                        else:
                            feature_extraction_errors += 1
                    except Exception as e:
                        feature_extraction_errors += 1
                        log_and_print(f"      ⚠️  Feature extraction failed for {spectrum.scan_id}: {e}")

                feature_extraction_time = time.time() - feature_extraction_start

                if extracted_features:
                    # Analyze feature properties
                    feature_names = list(extracted_features[0].keys())
                    num_features = len(feature_names)

                    # Calculate feature statistics
                    feature_stats = {}
                    for feature_name in feature_names:
                        values = [features.get(feature_name, 0) for features in extracted_features]
                        feature_stats[feature_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': min(values),
                            'max': max(values),
                            'range': max(values) - min(values)
                        }

                    # Feature diversity assessment
                    feature_diversity_score = 0.0
                    valid_features = 0

                    for feature_name, stats in feature_stats.items():
                        if stats['range'] > 0 and stats['std'] > 0:
                            valid_features += 1
                            # Normalize diversity contribution
                            diversity_contribution = min(1.0, stats['std'] / (stats['mean'] + 1e-10))
                            feature_diversity_score += diversity_contribution

                    feature_diversity_score = feature_diversity_score / max(1, num_features)

                    log_and_print(f"  📊 FEATURE EXTRACTION RESULTS:")
                    log_and_print(f"     Features extracted: {len(extracted_features)}/{len(clustering_spectra)}")
                    log_and_print(f"     Feature dimensions: {num_features}")
                    log_and_print(f"     Valid features: {valid_features}/{num_features}")
                    log_and_print(f"     Feature diversity score: {feature_diversity_score:.3f}")
                    log_and_print(f"     Extraction time: {feature_extraction_time:.2f}s")

                # STEP 5.3: Multi-cluster count analysis
                log_and_print("STEP 5.3: Testing different cluster counts...")

                cluster_results = {}

                for n_clusters in cluster_counts:
                    if len(clustering_spectra) >= n_clusters:  # Need enough spectra
                        cluster_start_time = time.time()

                        log_and_print(f"    🔄 Testing {n_clusters} clusters...")

                        try:
                            clustering_result = clustering_module.cluster_spectra(
                                clustering_spectra, n_clusters=n_clusters
                            )

                            cluster_time = time.time() - cluster_start_time

                            if 'error' not in clustering_result:
                                # Analyze clustering results
                                cluster_assignments = clustering_result.get('cluster_assignments', {})
                                cluster_sizes = clustering_result.get('cluster_sizes', {})
                                n_spectra_clustered = clustering_result.get('n_spectra_clustered', 0)

                                # Calculate clustering quality metrics

                                # 1. Cluster balance (how evenly distributed are clusters)
                                if cluster_sizes:
                                    sizes = list(cluster_sizes.values())
                                    cluster_balance = 1.0 - (np.std(sizes) / max(1, np.mean(sizes)))
                                else:
                                    cluster_balance = 0.0

                                # 2. Coverage (how many spectra were successfully clustered)
                                coverage = n_spectra_clustered / max(1, len(clustering_spectra))

                                # 3. Utilization (are all clusters used)
                                expected_clusters = n_clusters
                                actual_clusters = len([c for c in cluster_sizes.values() if c > 0])
                                cluster_utilization = actual_clusters / max(1, expected_clusters)

                                # Combined quality score
                                quality_score = (cluster_balance * 0.4 +
                                               coverage * 0.4 +
                                               cluster_utilization * 0.2)

                                cluster_results[n_clusters] = {
                                    'clustering_time': cluster_time,
                                    'n_spectra_clustered': n_spectra_clustered,
                                    'coverage': coverage,
                                    'cluster_assignments': cluster_assignments,
                                    'cluster_sizes': cluster_sizes,
                                    'actual_clusters': actual_clusters,
                                    'cluster_balance': cluster_balance,
                                    'cluster_utilization': cluster_utilization,
                                    'quality_score': quality_score,
                                    'clustering_rate': n_spectra_clustered / max(0.001, cluster_time)
                                }

                                log_and_print(f"      📊 {n_clusters} clusters: {n_spectra_clustered} spectra clustered")
                                log_and_print(f"      🎯 Quality: {quality_score:.3f}, Balance: {cluster_balance:.3f}")
                                log_and_print(f"      ⏱️  Time: {cluster_time:.2f}s")

                            else:
                                log_and_print(f"      ❌ {n_clusters} clusters failed: {clustering_result['error']}")
                                cluster_results[n_clusters] = {
                                    'clustering_time': cluster_time,
                                    'status': 'failed',
                                    'error': clustering_result['error']
                                }

                        except Exception as e:
                            cluster_time = time.time() - cluster_start_time
                            log_and_print(f"      ❌ {n_clusters} clusters failed: {e}")
                            cluster_results[n_clusters] = {
                                'clustering_time': cluster_time,
                                'status': 'error',
                                'error': str(e)
                            }

                    else:
                        log_and_print(f"      ⚠️  Skipping {n_clusters} clusters (insufficient spectra)")

                # STEP 5.4: Optimal clustering analysis
                log_and_print("STEP 5.4: Analyzing optimal clustering configuration...")

                successful_clusterings = {k: v for k, v in cluster_results.items()
                                        if 'quality_score' in v}

                optimal_config = None
                best_quality = 0

                if successful_clusterings:
                    # Find best configuration based on quality score
                    for n_clusters, result in successful_clusterings.items():
                        if result['quality_score'] > best_quality:
                            best_quality = result['quality_score']
                            optimal_config = n_clusters

                    avg_clustering_time = np.mean([r['clustering_time'] for r in successful_clusterings.values()])
                    avg_quality_score = np.mean([r['quality_score'] for r in successful_clusterings.values()])

                    log_and_print(f"  🎯 OPTIMAL CLUSTERING ANALYSIS:")
                    log_and_print(f"     Successful configurations: {len(successful_clusterings)}/{len(cluster_counts)}")
                    log_and_print(f"     Best configuration: {optimal_config} clusters (quality: {best_quality:.3f})")
                    log_and_print(f"     Average clustering time: {avg_clustering_time:.2f}s")
                    log_and_print(f"     Average quality score: {avg_quality_score:.3f}")

                # STEP 5.5: Clustering effectiveness assessment
                log_and_print("STEP 5.5: Overall clustering effectiveness assessment...")

                effectiveness_score = 0.0

                # Factor 1: Feature extraction success (25%)
                if extracted_features:
                    extraction_success_rate = len(extracted_features) / max(1, len(clustering_spectra))
                    effectiveness_score += extraction_success_rate * 0.25

                # Factor 2: Feature quality (25%)
                if 'feature_diversity_score' in locals():
                    effectiveness_score += feature_diversity_score * 0.25

                # Factor 3: Clustering success (25%)
                if successful_clusterings:
                    clustering_success_rate = len(successful_clusterings) / max(1, len(cluster_counts))
                    effectiveness_score += clustering_success_rate * 0.25

                # Factor 4: Clustering quality (25%)
                if successful_clusterings:
                    effectiveness_score += avg_quality_score * 0.25

                log_and_print(f"  📊 CLUSTERING EFFECTIVENESS:")
                log_and_print(f"     Overall effectiveness score: {effectiveness_score:.3f}")

                # STEP 5.6: Performance classification
                log_and_print("STEP 5.6: Clustering performance classification...")

                if (effectiveness_score >= 0.7 and len(successful_clusterings) >= 3 and
                    feature_diversity_score >= 0.3):
                    performance_grade = "🟢 EXCELLENT"
                elif effectiveness_score >= 0.5 and len(successful_clusterings) >= 2:
                    performance_grade = "🟡 GOOD"
                elif effectiveness_score >= 0.3 and len(successful_clusterings) >= 1:
                    performance_grade = "🟠 ACCEPTABLE"
                else:
                    performance_grade = "🔴 NEEDS IMPROVEMENT"

                log_and_print(f"  🏆 Clustering Performance: {performance_grade}")

                # Store results
                step_results['clustering_results'][test_file] = {
                    'spectra_processed': len(clustering_spectra),
                    'feature_extraction': {
                        'extraction_time': feature_extraction_time,
                        'features_extracted': len(extracted_features),
                        'feature_dimensions': num_features if extracted_features else 0,
                        'feature_diversity_score': feature_diversity_score if 'feature_diversity_score' in locals() else 0,
                        'extraction_errors': feature_extraction_errors,
                        'feature_stats': feature_stats if extracted_features else {}
                    },
                    'clustering_analysis': cluster_results,
                    'optimal_configuration': {
                        'best_cluster_count': optimal_config,
                        'best_quality_score': best_quality,
                        'avg_clustering_time': avg_clustering_time if successful_clusterings else 0,
                        'successful_configurations': len(successful_clusterings)
                    },
                    'effectiveness_metrics': {
                        'effectiveness_score': effectiveness_score,
                        'clustering_success_rate': len(successful_clusterings) / max(1, len(cluster_counts)),
                        'extraction_success_rate': len(extracted_features) / max(1, len(clustering_spectra)) if extracted_features else 0
                    },
                    'performance_grade': performance_grade
                }

                # Track overall metrics
                all_feature_extraction_times.append(feature_extraction_time)
                if successful_clusterings:
                    all_clustering_times.extend([r['clustering_time'] for r in successful_clusterings.values()])
                    all_clustering_quality_scores.extend([r['quality_score'] for r in successful_clusterings.values()])

                log_and_print(f"✅ Dataset {dataset_num} clustering validation completed")

            except Exception as e:
                log_and_print(f"❌ Dataset {dataset_num} clustering validation failed: {e}")

                step_results['clustering_results'][test_file] = {
                    'clustering_success': False,
                    'error': str(e),
                    'processing_time': time.time() - dataset_start_time
                }

        # STEP 5.7: Overall analysis and conclusions
        log_and_print(f"\n" + "=" * 60)
        log_and_print("📊 STEP 5 OVERALL CLUSTERING ANALYSIS")
        log_and_print("=" * 60)

        successful_analyses = len([r for r in step_results['clustering_results'].values()
                                 if 'clustering_success' not in r or r.get('clustering_success', True)])
        total_datasets = len(test_files)
        success_rate = successful_analyses / max(1, total_datasets)

        if successful_analyses > 0:
            avg_feature_extraction_time = np.mean(all_feature_extraction_times)
            avg_clustering_time = np.mean(all_clustering_times) if all_clustering_times else 0
            avg_clustering_quality = np.mean(all_clustering_quality_scores) if all_clustering_quality_scores else 0
            total_clustering_tests = len(all_clustering_quality_scores)

            step_results['feature_analysis'] = {
                'avg_feature_extraction_time': avg_feature_extraction_time,
                'feature_extraction_success': success_rate,
                'total_feature_extractions': successful_analyses
            }

            step_results['clustering_quality_metrics'] = {
                'successful_analyses': successful_analyses,
                'total_datasets': total_datasets,
                'success_rate': success_rate,
                'avg_clustering_time': avg_clustering_time,
                'avg_clustering_quality': avg_clustering_quality,
                'total_clustering_tests': total_clustering_tests,
                'clustering_configurations_tested': len(cluster_counts)
            }

            log_and_print(f"🔢 STEP 5 PERFORMANCE METRICS:")
            log_and_print(f"   Successful clustering analyses: {successful_analyses}/{total_datasets}")
            log_and_print(f"   Success rate: {success_rate:.1%}")
            log_and_print(f"   Total clustering tests performed: {total_clustering_tests}")
            log_and_print(f"   Average feature extraction time: {avg_feature_extraction_time:.2f}s")
            log_and_print(f"   Average clustering time: {avg_clustering_time:.2f}s")
            log_and_print(f"   Average clustering quality: {avg_clustering_quality:.3f}")

            # Step conclusion
            if (success_rate >= 0.8 and avg_clustering_quality >= 0.6 and
                total_clustering_tests >= 4):
                step_conclusion = "🟢 CLUSTERING VALIDATION PASSED - Effective feature clustering capabilities"
                step_status = "validated"
            elif success_rate >= 0.6 and avg_clustering_quality >= 0.4:
                step_conclusion = "🟡 CLUSTERING VALIDATION PARTIAL - Good clustering with room for improvement"
                step_status = "functional"
            else:
                step_conclusion = "🔴 CLUSTERING VALIDATION FAILED - Poor clustering performance"
                step_status = "problematic"

            log_and_print(f"\n🎯 STEP 5 CONCLUSION:")
            log_and_print(f"   {step_conclusion}")

            step_results['step_conclusion'] = {
                'overall_assessment': step_conclusion,
                'step_status': step_status,
                'success_rate': success_rate,
                'avg_clustering_quality': avg_clustering_quality,
                'total_tests_performed': total_clustering_tests,
                'key_findings': [
                    f"Clustering analysis success rate: {success_rate:.1%}",
                    f"Average clustering quality: {avg_clustering_quality:.3f}",
                    f"Feature extraction performance: {avg_feature_extraction_time:.2f}s average",
                    f"Total clustering configurations tested: {total_clustering_tests}"
                ],
                'recommendations': [
                    "Feature clustering is effective" if avg_clustering_quality >= 0.6 else "Improve feature extraction or clustering algorithms",
                    "Processing speed is acceptable" if avg_clustering_time <= 5.0 else "Optimize clustering computation",
                    "Feature diversity is adequate" if success_rate >= 0.8 else "Enhance spectral feature extraction",
                    "Cluster analysis is comprehensive" if total_clustering_tests >= 4 else "Test additional clustering configurations"
                ]
            }

        else:
            log_and_print("❌ No successful clustering analyses - critical failure in Step 5")
            step_results['step_conclusion'] = {
                'overall_assessment': '🔴 CRITICAL FAILURE - Clustering system completely non-functional',
                'step_status': 'failed'
            }

        # STEP 5.8: Save results and generate visualizations
        log_and_print(f"\n💾 SAVING STEP 5 RESULTS")
        log_and_print("-" * 30)

        # JSON results
        results_file = step_dir / "step_05_feature_clustering_results.json"
        with open(results_file, 'w') as f:
            json.dump(step_results, f, indent=2)
        log_and_print(f"📄 Saved JSON results: {results_file}")

        # CSV summary
        if successful_analyses > 0:
            csv_data = []
            for dataset, result in step_results['clustering_results'].items():
                if 'spectra_processed' in result:
                    feature_ext = result['feature_extraction']
                    optimal = result['optimal_configuration']
                    effectiveness = result['effectiveness_metrics']

                    csv_data.append({
                        'Dataset': dataset,
                        'Spectra_Processed': result['spectra_processed'],
                        'Feature_Extraction_Time_s': feature_ext['extraction_time'],
                        'Features_Extracted': feature_ext['features_extracted'],
                        'Feature_Dimensions': feature_ext['feature_dimensions'],
                        'Feature_Diversity_Score': feature_ext['feature_diversity_score'],
                        'Best_Cluster_Count': optimal['best_cluster_count'],
                        'Best_Quality_Score': optimal['best_quality_score'],
                        'Successful_Configurations': optimal['successful_configurations'],
                        'Effectiveness_Score': effectiveness['effectiveness_score'],
                        'Performance_Grade': result['performance_grade']
                    })

            if csv_data:
                csv_file = step_dir / "step_05_feature_clustering_summary.csv"
                pd.DataFrame(csv_data).to_csv(csv_file, index=False)
                log_and_print(f"📊 Saved CSV summary: {csv_file}")

        # Generate visualizations
        try:
            from visualization.panel import create_clustering_performance_panel
            import matplotlib.pyplot as plt

            if successful_analyses > 0:
                viz_data = {
                    'feature_extraction_times': all_feature_extraction_times,
                    'clustering_times': all_clustering_times,
                    'clustering_quality_scores': all_clustering_quality_scores,
                    'dataset_names': [f.replace('.mzML', '') for f in test_files]
                }

                fig = create_clustering_performance_panel(viz_data, "Step 5: Feature Clustering Performance")
                viz_file = step_dir / "step_05_feature_clustering_performance.png"
                fig.savefig(viz_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                log_and_print(f"📊 Saved visualization: {viz_file}")

        except ImportError as e:
            log_and_print(f"⚠️  Visualization generation failed: {e}")

        log_and_print(f"\n🧪 STEP 5: FEATURE CLUSTERING VALIDATION COMPLETE 🧪")
        log_and_print(f"📁 Results saved to: {step_dir}")
        log_and_print(f"📋 Step log: {log_file}")

        return step_results

    except Exception as e:
        log_and_print(f"💥 CRITICAL STEP 5 FAILURE: {e}")
        import traceback
        log_and_print(f"Error details: {traceback.format_exc()}")
        return None


if __name__ == "__main__":
    print("🚀 Starting Step 5: Feature Clustering Validation Experiment...")

    results = main()

    if results and results.get('step_conclusion', {}).get('step_status') in ['validated', 'functional']:
        print("\n✅ STEP 5 SUCCESSFUL - Feature clustering validated!")
        sys.exit(0)
    else:
        print("\n❌ STEP 5 FAILED - Check results for details")
        sys.exit(1)

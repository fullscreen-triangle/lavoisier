#!/usr/bin/env python3
"""
STEP 4: SPECTRUM EMBEDDING VALIDATION EXPERIMENT
===============================================

OBJECTIVE: Validate spectrum embedding and similarity analysis capabilities
HYPOTHESIS: The embedding engine can create meaningful vector representations of spectra

EXPERIMENT PROCEDURE:
1. Load mass spectrometry data
2. Generate spectrum embeddings using multiple methods
3. Analyze embedding quality and dimensionality
4. Test similarity search functionality
5. Generate embedding performance visualizations

Run: python step_04_spectrum_embedding_experiment.py
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
    """Step 4: Spectrum Embedding Validation Experiment"""

    print("🧪 STEP 4: SPECTRUM EMBEDDING VALIDATION EXPERIMENT")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Objective: Validate spectrum embedding and similarity analysis")
    print("=" * 60)

    # Create step-specific results directory
    step_dir = Path("step_results") / "step_04_spectrum_embedding"
    step_dir.mkdir(parents=True, exist_ok=True)

    # Initialize step log
    log_file = step_dir / "spectrum_embedding_log.txt"

    def log_and_print(message):
        """Log message to both console and file"""
        print(message)
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()}: {message}\n")

    log_and_print("🔬 STEP 4 EXPERIMENTAL SETUP")
    log_and_print("-" * 30)

    try:
        # Import required components
        log_and_print("📦 Loading spectrum embedding components...")
        from core.mzml_reader import StandaloneMzMLReader
        from core.numerical_pipeline import SpectrumEmbeddingEngine, QualityControlModule

        reader = StandaloneMzMLReader()
        embedding_engine = SpectrumEmbeddingEngine()
        qc_module = QualityControlModule()
        log_and_print("✅ Spectrum embedding components loaded successfully")

        # Get available embedding methods
        available_methods = embedding_engine.embedding_methods
        log_and_print(f"🧠 Available embedding methods: {available_methods}")

        # Define test datasets
        test_files = [
            "PL_Neg_Waters_qTOF.mzML",
            "TG_Pos_Thermo_Orbi.mzML"
        ]

        log_and_print(f"📊 Testing embeddings on {len(test_files)} datasets:")
        for i, file in enumerate(test_files, 1):
            log_and_print(f"  {i}. {file}")

        # Initialize results
        step_results = {
            'step_metadata': {
                'step_number': 4,
                'step_name': 'Spectrum Embedding Validation',
                'start_time': datetime.now().isoformat(),
                'objective': 'Validate spectrum embedding and similarity analysis capabilities',
                'available_methods': available_methods
            },
            'embedding_results': {},
            'similarity_analysis': {},
            'embedding_quality_metrics': {},
            'step_conclusion': {}
        }

        log_and_print("\n🚀 STARTING SPECTRUM EMBEDDING VALIDATION")
        log_and_print("-" * 48)

        all_embedding_times = []
        all_similarity_scores = []
        method_performance = {method: {'count': 0, 'avg_time': 0, 'dimensions': 0} for method in available_methods}

        # Test each dataset
        for dataset_num, test_file in enumerate(test_files, 1):
            log_and_print(f"\n🧠 DATASET {dataset_num}: {test_file}")
            log_and_print("-" * 40)

            dataset_start_time = time.time()

            try:
                # STEP 4.1: Load and prepare data for embedding
                log_and_print("STEP 4.1: Loading data for embedding analysis...")
                spectra = reader.load_mzml(test_file)

                # Filter for high-quality spectra
                high_quality_spectra = []
                for spectrum in spectra:
                    quality_metrics = qc_module.assess_spectrum_quality(spectrum)
                    if quality_metrics.get('quality_score', 0) >= 0.3:
                        high_quality_spectra.append(spectrum)

                # Limit for performance testing
                embedding_spectra = high_quality_spectra[:30]  # Test first 30 high-quality spectra

                log_and_print(f"  📂 Loaded {len(spectra)} total spectra")
                log_and_print(f"  🎯 Selected {len(embedding_spectra)} high-quality spectra for embedding")

                # STEP 4.2: Individual embedding method testing
                log_and_print("STEP 4.2: Testing individual embedding methods...")

                method_results = {}
                dataset_embeddings = {}

                for method in available_methods:
                    method_start_time = time.time()
                    method_embeddings = []
                    method_errors = 0

                    log_and_print(f"    🔄 Testing {method} embedding method...")

                    for spectrum in embedding_spectra:
                        try:
                            embedding = embedding_engine.create_embedding(spectrum, method)
                            method_embeddings.append(embedding)
                        except Exception as e:
                            method_errors += 1
                            log_and_print(f"      ⚠️  Embedding failed for spectrum {spectrum.scan_id}: {e}")

                    method_time = time.time() - method_start_time

                    if method_embeddings:
                        # Analyze embedding properties
                        embedding_dimension = method_embeddings[0].dimension
                        embedding_vectors = [emb.embedding_vector for emb in method_embeddings]

                        # Calculate embedding statistics
                        vector_magnitudes = [np.linalg.norm(vec) for vec in embedding_vectors]
                        avg_magnitude = np.mean(vector_magnitudes)
                        magnitude_std = np.std(vector_magnitudes)

                        # Check for embedding diversity (not all zeros or identical)
                        unique_embeddings = len(set(tuple(vec) for vec in embedding_vectors))
                        diversity_score = unique_embeddings / max(1, len(embedding_vectors))

                        method_results[method] = {
                            'embedding_time': method_time,
                            'embeddings_created': len(method_embeddings),
                            'embedding_errors': method_errors,
                            'success_rate': len(method_embeddings) / max(1, len(embedding_spectra)),
                            'embedding_dimension': embedding_dimension,
                            'avg_vector_magnitude': avg_magnitude,
                            'magnitude_std': magnitude_std,
                            'diversity_score': diversity_score,
                            'embedding_rate': len(method_embeddings) / max(0.001, method_time)
                        }

                        dataset_embeddings[method] = method_embeddings

                        log_and_print(f"      📊 {method}: {len(method_embeddings)}/{len(embedding_spectra)} embeddings")
                        log_and_print(f"      📐 Dimension: {embedding_dimension}, Diversity: {diversity_score:.3f}")
                        log_and_print(f"      ⏱️  Time: {method_time:.2f}s ({len(method_embeddings) / max(0.001, method_time):.1f} embeddings/s)")

                    else:
                        method_results[method] = {
                            'embedding_time': method_time,
                            'embeddings_created': 0,
                            'embedding_errors': method_errors,
                            'success_rate': 0.0,
                            'status': 'failed'
                        }
                        log_and_print(f"      ❌ {method}: Failed to create any embeddings")

                # STEP 4.3: Similarity search testing
                log_and_print("STEP 4.3: Testing embedding similarity search...")

                similarity_results = {}

                for method, embeddings in dataset_embeddings.items():
                    if len(embeddings) >= 2:  # Need at least 2 embeddings for similarity
                        similarity_start_time = time.time()

                        # Test similarity search with first embedding as query
                        query_embedding = embeddings[0]
                        database_embeddings = embeddings[1:]

                        similar_spectra = embedding_engine.similarity_search(
                            query_embedding, database_embeddings, top_k=5
                        )

                        similarity_time = time.time() - similarity_start_time

                        if similar_spectra:
                            # Analyze similarity scores
                            similarity_scores = [sim_score for _, sim_score in similar_spectra]
                            avg_similarity = np.mean(similarity_scores)
                            max_similarity = max(similarity_scores)
                            min_similarity = min(similarity_scores)

                            similarity_results[method] = {
                                'similarity_search_time': similarity_time,
                                'similar_spectra_found': len(similar_spectra),
                                'avg_similarity_score': avg_similarity,
                                'max_similarity_score': max_similarity,
                                'min_similarity_score': min_similarity,
                                'similarity_range': max_similarity - min_similarity,
                                'search_rate': len(database_embeddings) / max(0.001, similarity_time)
                            }

                            log_and_print(f"      🔍 {method} similarity: {len(similar_spectra)} results")
                            log_and_print(f"      📊 Avg similarity: {avg_similarity:.3f}, Range: {min_similarity:.3f}-{max_similarity:.3f}")

                        else:
                            similarity_results[method] = {
                                'similarity_search_time': similarity_time,
                                'similar_spectra_found': 0,
                                'status': 'no_results'
                            }

                # STEP 4.4: Cross-method embedding comparison
                log_and_print("STEP 4.4: Cross-method embedding comparison...")

                # Compare embedding methods if multiple successful
                successful_methods = [m for m in method_results.keys()
                                    if method_results[m].get('success_rate', 0) > 0]

                method_comparison = {}
                if len(successful_methods) >= 2:
                    for method in successful_methods:
                        other_methods = [m for m in successful_methods if m != method]

                        # Compare performance metrics
                        method_metrics = method_results[method]
                        comparison_score = 0.0

                        # Factor 1: Success rate (30%)
                        comparison_score += method_metrics['success_rate'] * 0.3

                        # Factor 2: Diversity score (25%)
                        comparison_score += method_metrics['diversity_score'] * 0.25

                        # Factor 3: Embedding speed (25%)
                        max_rate = max(method_results[m]['embedding_rate']
                                     for m in successful_methods if 'embedding_rate' in method_results[m])
                        if max_rate > 0:
                            comparison_score += (method_metrics['embedding_rate'] / max_rate) * 0.25

                        # Factor 4: Similarity search capability (20%)
                        if method in similarity_results and similarity_results[method].get('similar_spectra_found', 0) > 0:
                            comparison_score += 0.2

                        method_comparison[method] = {
                            'comparison_score': comparison_score,
                            'relative_performance': comparison_score
                        }

                # STEP 4.5: Embedding quality assessment
                log_and_print("STEP 4.5: Overall embedding quality assessment...")

                # Calculate overall quality metrics
                total_embeddings_created = sum(r.get('embeddings_created', 0) for r in method_results.values())
                total_embedding_time = sum(r.get('embedding_time', 0) for r in method_results.values())
                avg_success_rate = np.mean([r.get('success_rate', 0) for r in method_results.values()])

                # Embedding effectiveness
                effective_methods = len([r for r in method_results.values()
                                       if r.get('success_rate', 0) >= 0.8])

                quality_score = 0.0
                if total_embeddings_created > 0:
                    quality_score += 0.3
                if avg_success_rate >= 0.7:
                    quality_score += 0.3
                if effective_methods >= 2:
                    quality_score += 0.2
                if len(similarity_results) >= 1:
                    quality_score += 0.2

                log_and_print(f"  🎯 EMBEDDING QUALITY ASSESSMENT:")
                log_and_print(f"     Total embeddings created: {total_embeddings_created}")
                log_and_print(f"     Average success rate: {avg_success_rate:.1%}")
                log_and_print(f"     Effective methods: {effective_methods}/{len(available_methods)}")
                log_and_print(f"     Quality score: {quality_score:.2f}/1.00")

                # STEP 4.6: Performance classification
                log_and_print("STEP 4.6: Embedding performance classification...")

                if quality_score >= 0.8 and avg_success_rate >= 0.8 and total_embedding_time <= 30.0:
                    performance_grade = "🟢 EXCELLENT"
                elif quality_score >= 0.6 and avg_success_rate >= 0.6:
                    performance_grade = "🟡 GOOD"
                elif quality_score >= 0.4 and avg_success_rate >= 0.4:
                    performance_grade = "🟠 ACCEPTABLE"
                else:
                    performance_grade = "🔴 NEEDS IMPROVEMENT"

                log_and_print(f"  🏆 Embedding Performance: {performance_grade}")

                # Store results
                step_results['embedding_results'][test_file] = {
                    'spectra_processed': len(embedding_spectra),
                    'total_embedding_time': total_embedding_time,
                    'embedding_rate': total_embeddings_created / max(0.001, total_embedding_time),
                    'method_results': method_results,
                    'similarity_results': similarity_results,
                    'method_comparison': method_comparison,
                    'quality_metrics': {
                        'total_embeddings_created': total_embeddings_created,
                        'avg_success_rate': avg_success_rate,
                        'effective_methods': effective_methods,
                        'quality_score': quality_score
                    },
                    'performance_grade': performance_grade
                }

                # Track overall metrics
                all_embedding_times.append(total_embedding_time)
                if similarity_results:
                    for sim_result in similarity_results.values():
                        if 'avg_similarity_score' in sim_result:
                            all_similarity_scores.append(sim_result['avg_similarity_score'])

                # Update method performance tracking
                for method, result in method_results.items():
                    if 'embeddings_created' in result:
                        method_performance[method]['count'] += result['embeddings_created']
                        if 'embedding_time' in result:
                            method_performance[method]['avg_time'] += result['embedding_time']
                        if 'embedding_dimension' in result:
                            method_performance[method]['dimensions'] = result['embedding_dimension']

                log_and_print(f"✅ Dataset {dataset_num} embedding validation completed")

            except Exception as e:
                log_and_print(f"❌ Dataset {dataset_num} embedding validation failed: {e}")

                step_results['embedding_results'][test_file] = {
                    'embedding_success': False,
                    'error': str(e),
                    'processing_time': time.time() - dataset_start_time
                }

        # STEP 4.7: Overall analysis and conclusions
        log_and_print(f"\n" + "=" * 60)
        log_and_print("📊 STEP 4 OVERALL EMBEDDING ANALYSIS")
        log_and_print("=" * 60)

        successful_embeddings = len([r for r in step_results['embedding_results'].values()
                                   if 'embedding_success' not in r or r.get('embedding_success', True)])
        total_datasets = len(test_files)
        success_rate = successful_embeddings / max(1, total_datasets)

        if successful_embeddings > 0:
            avg_embedding_time = np.mean(all_embedding_times)
            avg_similarity_score = np.mean(all_similarity_scores) if all_similarity_scores else 0
            total_embeddings = sum(perf['count'] for perf in method_performance.values())

            # Method effectiveness analysis
            method_effectiveness = {}
            for method, perf in method_performance.items():
                if perf['count'] > 0:
                    method_effectiveness[method] = {
                        'total_embeddings': perf['count'],
                        'avg_time_per_embedding': perf['avg_time'] / max(1, perf['count']),
                        'dimensions': perf['dimensions'],
                        'effectiveness_score': perf['count'] / max(1, perf['avg_time'])
                    }

            step_results['similarity_analysis'] = {
                'avg_similarity_score': avg_similarity_score,
                'similarity_tests_performed': len(all_similarity_scores),
                'method_effectiveness': method_effectiveness
            }

            step_results['embedding_quality_metrics'] = {
                'successful_embeddings': successful_embeddings,
                'total_datasets': total_datasets,
                'success_rate': success_rate,
                'avg_embedding_time': avg_embedding_time,
                'total_embeddings_created': total_embeddings,
                'avg_similarity_score': avg_similarity_score,
                'methods_tested': len(available_methods),
                'effective_methods': len([m for m in method_effectiveness.keys()])
            }

            log_and_print(f"🔢 STEP 4 PERFORMANCE METRICS:")
            log_and_print(f"   Successful embedding runs: {successful_embeddings}/{total_datasets}")
            log_and_print(f"   Success rate: {success_rate:.1%}")
            log_and_print(f"   Total embeddings created: {total_embeddings}")
            log_and_print(f"   Average embedding time: {avg_embedding_time:.2f}s per dataset")
            log_and_print(f"   Average similarity score: {avg_similarity_score:.3f}")
            log_and_print(f"   Methods tested: {len(available_methods)}")

            # Method effectiveness ranking
            if method_effectiveness:
                log_and_print(f"\n🏆 EMBEDDING METHOD EFFECTIVENESS:")
                sorted_methods = sorted(method_effectiveness.items(),
                                      key=lambda x: x[1]['effectiveness_score'], reverse=True)
                for i, (method, metrics) in enumerate(sorted_methods, 1):
                    log_and_print(f"   {i}. {method}: {metrics['total_embeddings']} embeddings, "
                                 f"dim={metrics['dimensions']}, {metrics['effectiveness_score']:.1f} emb/s")

            # Step conclusion
            if (success_rate >= 0.8 and total_embeddings >= 20 and
                len(method_effectiveness) >= 2 and avg_similarity_score >= 0.1):
                step_conclusion = "🟢 EMBEDDING VALIDATION PASSED - Effective spectrum embedding capabilities"
                step_status = "validated"
            elif success_rate >= 0.6 and total_embeddings >= 10:
                step_conclusion = "🟡 EMBEDDING VALIDATION PARTIAL - Good performance with optimization potential"
                step_status = "functional"
            else:
                step_conclusion = "🔴 EMBEDDING VALIDATION FAILED - Poor embedding quality or coverage"
                step_status = "problematic"

            log_and_print(f"\n🎯 STEP 4 CONCLUSION:")
            log_and_print(f"   {step_conclusion}")

            best_method = max(method_effectiveness.keys(),
                            key=lambda x: method_effectiveness[x]['effectiveness_score']) if method_effectiveness else None

            step_results['step_conclusion'] = {
                'overall_assessment': step_conclusion,
                'step_status': step_status,
                'success_rate': success_rate,
                'total_embeddings_created': total_embeddings,
                'best_performing_method': best_method,
                'key_findings': [
                    f"Embedding success rate: {success_rate:.1%}",
                    f"Total embeddings created: {total_embeddings}",
                    f"Average similarity score: {avg_similarity_score:.3f}",
                    f"Most effective method: {best_method}" if best_method else "No clearly best method"
                ],
                'recommendations': [
                    "Embedding capabilities are comprehensive" if len(method_effectiveness) >= 2 else "Improve embedding method reliability",
                    f"Prioritize {best_method} method" if best_method and method_effectiveness[best_method]['effectiveness_score'] >= 5.0 else "All methods need optimization",
                    "Embedding speed is acceptable" if avg_embedding_time <= 15.0 else "Optimize embedding computation",
                    "Similarity search is functional" if avg_similarity_score >= 0.1 else "Improve embedding quality for better similarity"
                ]
            }

        else:
            log_and_print("❌ No successful embedding runs - critical failure in Step 4")
            step_results['step_conclusion'] = {
                'overall_assessment': '🔴 CRITICAL FAILURE - Embedding system completely non-functional',
                'step_status': 'failed'
            }

        # STEP 4.8: Save results and generate visualizations
        log_and_print(f"\n💾 SAVING STEP 4 RESULTS")
        log_and_print("-" * 30)

        # JSON results
        results_file = step_dir / "step_04_spectrum_embedding_results.json"
        with open(results_file, 'w') as f:
            json.dump(step_results, f, indent=2)
        log_and_print(f"📄 Saved JSON results: {results_file}")

        # CSV summary
        if successful_embeddings > 0:
            csv_data = []
            for dataset, result in step_results['embedding_results'].items():
                if 'spectra_processed' in result:
                    quality_metrics = result['quality_metrics']
                    csv_data.append({
                        'Dataset': dataset,
                        'Spectra_Processed': result['spectra_processed'],
                        'Total_Embedding_Time_s': result['total_embedding_time'],
                        'Embedding_Rate': result['embedding_rate'],
                        'Total_Embeddings_Created': quality_metrics['total_embeddings_created'],
                        'Avg_Success_Rate': quality_metrics['avg_success_rate'],
                        'Effective_Methods': quality_metrics['effective_methods'],
                        'Quality_Score': quality_metrics['quality_score'],
                        'Performance_Grade': result['performance_grade']
                    })

            if csv_data:
                csv_file = step_dir / "step_04_spectrum_embedding_summary.csv"
                pd.DataFrame(csv_data).to_csv(csv_file, index=False)
                log_and_print(f"📊 Saved CSV summary: {csv_file}")

        # Generate visualizations
        try:
            from visualization.panel import create_embedding_performance_panel
            import matplotlib.pyplot as plt

            if successful_embeddings > 0:
                viz_data = {
                    'embedding_times': all_embedding_times,
                    'similarity_scores': all_similarity_scores,
                    'method_effectiveness': method_effectiveness,
                    'dataset_names': [f.replace('.mzML', '') for f in test_files]
                }

                fig = create_embedding_performance_panel(viz_data, "Step 4: Spectrum Embedding Performance")
                viz_file = step_dir / "step_04_spectrum_embedding_performance.png"
                fig.savefig(viz_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                log_and_print(f"📊 Saved visualization: {viz_file}")

        except ImportError as e:
            log_and_print(f"⚠️  Visualization generation failed: {e}")

        log_and_print(f"\n🧪 STEP 4: SPECTRUM EMBEDDING VALIDATION COMPLETE 🧪")
        log_and_print(f"📁 Results saved to: {step_dir}")
        log_and_print(f"📋 Step log: {log_file}")

        return step_results

    except Exception as e:
        log_and_print(f"💥 CRITICAL STEP 4 FAILURE: {e}")
        import traceback
        log_and_print(f"Error details: {traceback.format_exc()}")
        return None


if __name__ == "__main__":
    print("🚀 Starting Step 4: Spectrum Embedding Validation Experiment...")

    results = main()

    if results and results.get('step_conclusion', {}).get('step_status') in ['validated', 'functional']:
        print("\n✅ STEP 4 SUCCESSFUL - Spectrum embedding validated!")
        sys.exit(0)
    else:
        print("\n❌ STEP 4 FAILED - Check results for details")
        sys.exit(1)

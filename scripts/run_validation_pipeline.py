#!/usr/bin/env python3
"""
Lavoisier Validation Pipeline Runner

Complete validation pipeline comparing numerical and computer vision methods
for mass spectrometry-based metabolomics analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import all validation modules
from validation import (
    # Statistics
    HypothesisTestSuite, EffectSizeCalculator, StatisticalValidator, BiasDetector,
    # Performance
    PerformanceBenchmark, EfficiencyAnalyzer,
    # Quality
    DataQualityAssessor,
    # Completeness
    CompletenessAnalyzer,
    # Features
    FeatureExtractorComparator,
    # Vision
    ComputerVisionValidator, ImageQualityAssessor, VideoAnalyzer,
    # Annotation
    CompoundIdentificationValidator
)


def generate_synthetic_data():
    """Generate synthetic data for both pipelines"""
    print("Generating synthetic data...")
    
    # Sample parameters
    n_samples = 1000
    n_features = 256
    n_spectra = 50
    n_compounds = 100
    
    # Numerical pipeline data
    numerical_data = {
        'mass_spectra': np.random.exponential(2, (n_samples, n_features)) + np.random.normal(0, 0.1, (n_samples, n_features)),
        'peak_intensities': np.random.lognormal(0, 1, (n_samples, 50)),
        'retention_times': np.random.uniform(0.5, 30, n_samples),
        'mz_values': np.random.uniform(50, 1000, (n_samples, n_features)),
        'compound_ids': np.random.randint(0, n_compounds, n_samples),
        'confidence_scores': np.random.beta(2, 1, n_samples)
    }
    
    # Visual pipeline data (correlated but with differences)
    visual_data = {
        'image_features': 0.8 * numerical_data['mass_spectra'] + 0.2 * np.random.normal(0, 0.5, (n_samples, n_features)),
        'spectrum_images': np.random.rand(n_spectra, 64, 64),
        'attention_maps': np.random.rand(n_spectra, 32, 32),
        'temporal_sequence': np.random.randn(100, 128),
        'compound_predictions': numerical_data['compound_ids'] + np.random.randint(-2, 3, n_samples),
        'visual_confidence': numerical_data['confidence_scores'] + np.random.normal(0, 0.1, n_samples)
    }
    
    # Clip values to valid ranges
    visual_data['compound_predictions'] = np.clip(visual_data['compound_predictions'], 0, n_compounds-1)
    visual_data['visual_confidence'] = np.clip(visual_data['visual_confidence'], 0, 1)
    
    return numerical_data, visual_data


def run_statistical_validation(numerical_data, visual_data):
    """Run comprehensive statistical validation"""
    print("\n" + "="*60)
    print("STATISTICAL VALIDATION")
    print("="*60)
    
    results = {}
    
    # Hypothesis testing
    print("Running hypothesis tests...")
    hypothesis_tester = HypothesisTestSuite()
    
    # H1: Equivalent mass accuracy
    h1_result = hypothesis_tester.test_equivalence(
        numerical_data['peak_intensities'][:500],
        visual_data['image_features'][:500, :50].mean(axis=1),
        equivalence_margin=0.1
    )
    print(f"H1 (Mass Accuracy Equivalence): {h1_result.interpretation}")
    
    # H2: Complementary information
    h2_result = hypothesis_tester.test_complementarity(
        numerical_data['mass_spectra'][:500],
        visual_data['image_features'][:500]
    )
    print(f"H2 (Complementary Information): {h2_result.interpretation}")
    
    # Effect size analysis
    print("\nCalculating effect sizes...")
    effect_calculator = EffectSizeCalculator()
    
    effect_result = effect_calculator.cohens_d(
        numerical_data['confidence_scores'],
        visual_data['visual_confidence']
    )
    print(f"Effect Size (Cohen's d): {effect_result.effect_size:.3f} - {effect_result.interpretation}")
    
    # Bias detection
    print("\nDetecting systematic biases...")
    bias_detector = BiasDetector()
    
    bias_result = bias_detector.detect_systematic_bias(
        numerical_data['peak_intensities'],
        visual_data['image_features'][:len(numerical_data['peak_intensities']), :50].mean(axis=1)
    )
    print(f"Systematic Bias: {bias_result.interpretation}")
    
    results['statistical'] = {
        'hypothesis_h1': h1_result.p_value,
        'hypothesis_h2': h2_result.p_value,
        'effect_size': effect_result.effect_size,
        'bias_detected': bias_result.bias_detected
    }
    
    return results


def run_performance_validation(numerical_data, visual_data):
    """Run performance benchmarking"""
    print("\n" + "="*60)
    print("PERFORMANCE VALIDATION")
    print("="*60)
    
    results = {}
    
    # Performance benchmarking
    print("Running performance benchmarks...")
    benchmark = PerformanceBenchmark()
    
    # Simulate processing functions
    def numerical_processing():
        return np.mean(numerical_data['mass_spectra'], axis=1)
    
    def visual_processing():
        return np.mean(visual_data['image_features'], axis=1)
    
    # Benchmark both methods
    num_benchmark = benchmark.benchmark_function(numerical_processing, n_runs=10)
    vis_benchmark = benchmark.benchmark_function(visual_processing, n_runs=10)
    
    print(f"Numerical Pipeline: {num_benchmark.mean_time:.4f}s ± {num_benchmark.std_time:.4f}s")
    print(f"Visual Pipeline: {vis_benchmark.mean_time:.4f}s ± {vis_benchmark.std_time:.4f}s")
    
    # Efficiency analysis
    print("\nAnalyzing computational efficiency...")
    efficiency_analyzer = EfficiencyAnalyzer()
    
    num_efficiency = efficiency_analyzer.calculate_throughput_efficiency(
        processing_times=[num_benchmark.mean_time] * 100,
        data_sizes=[len(numerical_data['mass_spectra'])] * 100
    )
    
    vis_efficiency = efficiency_analyzer.calculate_throughput_efficiency(
        processing_times=[vis_benchmark.mean_time] * 100,
        data_sizes=[len(visual_data['image_features'])] * 100
    )
    
    print(f"Numerical Throughput: {num_efficiency.throughput:.2f} samples/sec")
    print(f"Visual Throughput: {vis_efficiency.throughput:.2f} samples/sec")
    
    results['performance'] = {
        'numerical_time': num_benchmark.mean_time,
        'visual_time': vis_benchmark.mean_time,
        'numerical_throughput': num_efficiency.throughput,
        'visual_throughput': vis_efficiency.throughput
    }
    
    return results


def run_quality_validation(numerical_data, visual_data):
    """Run data quality assessment"""
    print("\n" + "="*60)
    print("QUALITY VALIDATION")
    print("="*60)
    
    results = {}
    
    # Data quality assessment
    print("Assessing data quality...")
    quality_assessor = DataQualityAssessor()
    
    # Numerical data quality
    num_quality = quality_assessor.assess_completeness(numerical_data['mass_spectra'])
    print(f"Numerical Data Completeness: {num_quality.completeness_score:.3f}")
    
    # Visual data quality
    vis_quality = quality_assessor.assess_completeness(visual_data['image_features'])
    print(f"Visual Data Completeness: {vis_quality.completeness_score:.3f}")
    
    # Consistency between pipelines
    consistency = quality_assessor.assess_consistency(
        numerical_data['mass_spectra'][:500],
        visual_data['image_features'][:500]
    )
    print(f"Pipeline Consistency: {consistency.consistency_score:.3f}")
    
    results['quality'] = {
        'numerical_completeness': num_quality.completeness_score,
        'visual_completeness': vis_quality.completeness_score,
        'pipeline_consistency': consistency.consistency_score
    }
    
    return results


def run_feature_validation(numerical_data, visual_data):
    """Run feature extraction comparison"""
    print("\n" + "="*60)
    print("FEATURE VALIDATION")
    print("="*60)
    
    results = {}
    
    # Feature comparison
    print("Comparing feature extraction...")
    feature_comparator = FeatureExtractorComparator()
    
    comparison_results = feature_comparator.compare_feature_spaces(
        numerical_features=numerical_data['mass_spectra'][:500],
        visual_features=visual_data['image_features'][:500],
        labels=numerical_data['compound_ids'][:500]
    )
    
    for result in comparison_results:
        print(f"{result.metric_name}: {result.comparison_score:.3f}")
    
    # Feature complementarity
    complementarity = feature_comparator.analyze_feature_complementarity(
        numerical_features=numerical_data['mass_spectra'][:500],
        visual_features=visual_data['image_features'][:500]
    )
    print(f"Feature Complementarity: {complementarity.comparison_score:.3f}")
    
    results['features'] = {
        'comparison_scores': [r.comparison_score for r in comparison_results],
        'complementarity': complementarity.comparison_score
    }
    
    return results


def run_vision_validation(visual_data):
    """Run computer vision validation"""
    print("\n" + "="*60)
    print("COMPUTER VISION VALIDATION")
    print("="*60)
    
    results = {}
    
    # Computer vision validation
    print("Validating computer vision methods...")
    cv_validator = ComputerVisionValidator()
    
    # Dummy model function for testing
    def dummy_model(images):
        return np.array([img.flatten()[:128] for img in images])
    
    # Robustness testing
    robustness_results = cv_validator.validate_feature_extraction_robustness(
        model_function=dummy_model,
        test_images=visual_data['spectrum_images'][:20],
        noise_levels=[0.01, 0.05, 0.1],
        noise_types=['gaussian', 'blur']
    )
    
    for result in robustness_results:
        print(f"{result.metric_name}: {result.score:.3f}")
    
    # Attention analysis
    attention_result = cv_validator.analyze_attention_mechanisms(
        visual_data['attention_maps'][:20]
    )
    print(f"Attention Quality: {attention_result.score:.3f}")
    
    # Video analysis
    print("\nAnalyzing temporal patterns...")
    video_analyzer = VideoAnalyzer()
    
    # Temporal consistency
    consistency_result = video_analyzer.analyze_temporal_consistency(
        visual_data['temporal_sequence']
    )
    print(f"Temporal Consistency: {consistency_result.score:.3f}")
    
    # Anomaly detection
    anomaly_result = video_analyzer.detect_temporal_anomalies(
        visual_data['temporal_sequence']
    )
    print(f"Anomaly Detection: {anomaly_result.score:.3f}")
    
    results['vision'] = {
        'robustness_scores': [r.score for r in robustness_results],
        'attention_quality': attention_result.score,
        'temporal_consistency': consistency_result.score,
        'anomaly_score': anomaly_result.score
    }
    
    return results


def run_annotation_validation(numerical_data, visual_data):
    """Run annotation and identification validation"""
    print("\n" + "="*60)
    print("ANNOTATION VALIDATION")
    print("="*60)
    
    results = {}
    
    # Compound identification validation
    print("Validating compound identification...")
    id_validator = CompoundIdentificationValidator()
    
    # Compare identification accuracy
    comparison_results = id_validator.compare_pipeline_identification(
        true_labels=numerical_data['compound_ids'][:500],
        numerical_predictions=numerical_data['compound_ids'][:500],  # Perfect for numerical
        visual_predictions=visual_data['compound_predictions'][:500]
    )
    
    print("Numerical Pipeline Results:")
    for result in comparison_results['numerical']:
        if 'Accuracy' in result.metric_name or 'F1' in result.metric_name:
            print(f"  {result.metric_name}: {result.value:.3f}")
    
    print("Visual Pipeline Results:")
    for result in comparison_results['visual']:
        if 'Accuracy' in result.metric_name or 'F1' in result.metric_name:
            print(f"  {result.metric_name}: {result.value:.3f}")
    
    results['annotation'] = {
        'numerical_accuracy': next(r.value for r in comparison_results['numerical'] if 'Accuracy' in r.metric_name),
        'visual_accuracy': next(r.value for r in comparison_results['visual'] if 'Accuracy' in r.metric_name)
    }
    
    return results


def generate_summary_report(all_results):
    """Generate comprehensive summary report"""
    print("\n" + "="*60)
    print("VALIDATION SUMMARY REPORT")
    print("="*60)
    
    # Create summary DataFrame
    summary_data = []
    
    # Statistical results
    if 'statistical' in all_results:
        summary_data.append({
            'Category': 'Statistical',
            'Metric': 'Effect Size',
            'Value': all_results['statistical']['effect_size'],
            'Interpretation': 'Difference between methods'
        })
    
    # Performance results
    if 'performance' in all_results:
        speedup = all_results['performance']['numerical_time'] / all_results['performance']['visual_time']
        summary_data.append({
            'Category': 'Performance',
            'Metric': 'Speed Ratio (Num/Vis)',
            'Value': speedup,
            'Interpretation': f"{'Numerical' if speedup > 1 else 'Visual'} is {abs(speedup):.1f}x faster"
        })
    
    # Quality results
    if 'quality' in all_results:
        summary_data.append({
            'Category': 'Quality',
            'Metric': 'Pipeline Consistency',
            'Value': all_results['quality']['pipeline_consistency'],
            'Interpretation': 'Agreement between methods'
        })
    
    # Feature results
    if 'features' in all_results:
        avg_comparison = np.mean(all_results['features']['comparison_scores'])
        summary_data.append({
            'Category': 'Features',
            'Metric': 'Feature Similarity',
            'Value': avg_comparison,
            'Interpretation': 'How similar extracted features are'
        })
    
    # Vision results
    if 'vision' in all_results:
        avg_robustness = np.mean(all_results['vision']['robustness_scores'])
        summary_data.append({
            'Category': 'Vision',
            'Metric': 'Robustness',
            'Value': avg_robustness,
            'Interpretation': 'Stability to noise/perturbations'
        })
    
    # Annotation results
    if 'annotation' in all_results:
        accuracy_diff = all_results['annotation']['numerical_accuracy'] - all_results['annotation']['visual_accuracy']
        summary_data.append({
            'Category': 'Annotation',
            'Metric': 'Accuracy Difference',
            'Value': accuracy_diff,
            'Interpretation': f"{'Numerical' if accuracy_diff > 0 else 'Visual'} is {abs(accuracy_diff):.3f} more accurate"
        })
    
    # Create and display summary table
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Overall assessment
    print(f"\n{'='*60}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*60}")
    
    # Calculate overall scores
    scores = []
    if 'quality' in all_results:
        scores.append(all_results['quality']['pipeline_consistency'])
    if 'features' in all_results:
        scores.append(np.mean(all_results['features']['comparison_scores']))
    if 'vision' in all_results:
        scores.append(np.mean(all_results['vision']['robustness_scores']))
    
    if scores:
        overall_score = np.mean(scores)
        if overall_score > 0.8:
            assessment = "EXCELLENT - Methods are highly compatible"
        elif overall_score > 0.6:
            assessment = "GOOD - Methods show strong agreement"
        elif overall_score > 0.4:
            assessment = "MODERATE - Methods have some differences"
        else:
            assessment = "POOR - Methods show significant differences"
        
        print(f"Overall Compatibility Score: {overall_score:.3f}")
        print(f"Assessment: {assessment}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    recommendations = []
    
    if 'performance' in all_results:
        if all_results['performance']['visual_time'] > all_results['performance']['numerical_time'] * 2:
            recommendations.append("• Consider optimizing visual pipeline for better performance")
        elif all_results['performance']['numerical_time'] > all_results['performance']['visual_time'] * 2:
            recommendations.append("• Visual pipeline shows superior computational efficiency")
    
    if 'features' in all_results:
        if all_results['features']['complementarity'] > 0.7:
            recommendations.append("• Methods provide complementary information - consider ensemble approach")
        else:
            recommendations.append("• Methods show redundancy - focus on best performing approach")
    
    if 'annotation' in all_results:
        if abs(all_results['annotation']['numerical_accuracy'] - all_results['annotation']['visual_accuracy']) < 0.05:
            recommendations.append("• Both methods show similar identification accuracy")
        else:
            better_method = "numerical" if all_results['annotation']['numerical_accuracy'] > all_results['annotation']['visual_accuracy'] else "visual"
            recommendations.append(f"• {better_method.title()} method shows superior identification accuracy")
    
    if not recommendations:
        recommendations.append("• Both methods show comparable performance across all metrics")
    
    for rec in recommendations:
        print(rec)
    
    return summary_df


def save_results(all_results, summary_df):
    """Save all results to files"""
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    # Create output directory
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save summary report
    summary_df.to_csv(output_dir / "validation_summary.csv", index=False)
    print(f"✓ Saved summary report to {output_dir / 'validation_summary.csv'}")
    
    # Save detailed results
    detailed_results = {}
    for category, results in all_results.items():
        for metric, value in results.items():
            detailed_results[f"{category}_{metric}"] = value
    
    detailed_df = pd.DataFrame([detailed_results])
    detailed_df.to_csv(output_dir / "detailed_results.csv", index=False)
    print(f"✓ Saved detailed results to {output_dir / 'detailed_results.csv'}")
    
    print(f"\nAll results saved to: {output_dir.absolute()}")


def main():
    """Main validation pipeline"""
    print("LAVOISIER VALIDATION PIPELINE")
    print("Comparing Numerical vs Computer Vision Methods")
    print("=" * 80)
    
    # Generate synthetic data
    numerical_data, visual_data = generate_synthetic_data()
    
    # Run all validation modules
    all_results = {}
    
    try:
        all_results.update(run_statistical_validation(numerical_data, visual_data))
    except Exception as e:
        print(f"Statistical validation failed: {e}")
    
    try:
        all_results.update(run_performance_validation(numerical_data, visual_data))
    except Exception as e:
        print(f"Performance validation failed: {e}")
    
    try:
        all_results.update(run_quality_validation(numerical_data, visual_data))
    except Exception as e:
        print(f"Quality validation failed: {e}")
    
    try:
        all_results.update(run_feature_validation(numerical_data, visual_data))
    except Exception as e:
        print(f"Feature validation failed: {e}")
    
    try:
        all_results.update(run_vision_validation(visual_data))
    except Exception as e:
        print(f"Vision validation failed: {e}")
    
    try:
        all_results.update(run_annotation_validation(numerical_data, visual_data))
    except Exception as e:
        print(f"Annotation validation failed: {e}")
    
    # Generate summary report
    summary_df = generate_summary_report(all_results)
    
    # Save results
    save_results(all_results, summary_df)
    
    print(f"\n{'='*80}")
    print("VALIDATION PIPELINE COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main() 
"""
Computer Vision and Video Analysis Validation Example

This example demonstrates how to use the computer vision validator and video analyzer
to evaluate the visual pipeline's performance, including robustness testing,
temporal consistency analysis, and motion pattern detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the validation module to the path
sys.path.append(str(Path(__file__).parent.parent))

from validation.vision import ComputerVisionValidator, VideoAnalyzer


def create_synthetic_spectrum_images(n_images=50, height=64, width=64):
    """Create synthetic spectrum images for testing"""
    images = []
    
    for i in range(n_images):
        # Create a synthetic spectrum with peaks
        x = np.linspace(0, 10, width)
        y = np.linspace(0, 8, height)
        X, Y = np.meshgrid(x, y)
        
        # Add some peaks with noise
        spectrum = (
            np.exp(-((X - 3)**2 + (Y - 2)**2) / 0.5) * 0.8 +
            np.exp(-((X - 7)**2 + (Y - 5)**2) / 0.3) * 0.6 +
            np.exp(-((X - 5)**2 + (Y - 6)**2) / 0.4) * 0.4 +
            np.random.normal(0, 0.1, (height, width))
        )
        
        # Add temporal variation
        time_factor = 1 + 0.2 * np.sin(2 * np.pi * i / 20)
        spectrum *= time_factor
        
        # Ensure values are in [0, 1]
        spectrum = np.clip(spectrum, 0, 1)
        images.append(spectrum)
    
    return np.array(images)


def create_synthetic_feature_sequence(n_frames=100, n_features=128):
    """Create synthetic feature sequence for testing"""
    # Create base features
    base_features = np.random.randn(n_features) * 0.5
    
    sequence = []
    for i in range(n_frames):
        # Add temporal variation
        temporal_variation = 0.1 * np.sin(2 * np.pi * i / 25) + 0.05 * np.random.randn(n_features)
        
        # Add some drift
        drift = 0.001 * i * np.ones(n_features)
        
        # Occasional jumps (anomalies)
        if i in [30, 60, 85]:
            jump = 0.5 * np.random.randn(n_features)
        else:
            jump = 0
        
        features = base_features + temporal_variation + drift + jump
        sequence.append(features)
    
    return np.array(sequence)


def dummy_model_function(images):
    """Dummy model function that extracts features from images"""
    # Simple feature extraction: flatten and take first 128 values
    features = []
    for img in images:
        flat = img.flatten()
        # Take some statistics as features
        feature_vector = np.array([
            np.mean(flat),
            np.std(flat),
            np.max(flat),
            np.min(flat),
            np.median(flat),
            *np.histogram(flat, bins=10)[0] / len(flat),  # Normalized histogram
            *flat[:113]  # First 113 pixel values to make 128 total
        ])
        features.append(feature_vector)
    
    return np.array(features)


def main():
    """Main function demonstrating CV and video validation"""
    print("=== Computer Vision and Video Analysis Validation Example ===\n")
    
    # Create synthetic data
    print("1. Creating synthetic data...")
    spectrum_images = create_synthetic_spectrum_images(n_images=50)
    feature_sequence = create_synthetic_feature_sequence(n_frames=100)
    labels = np.random.randint(0, 3, size=50)  # 3 classes
    
    print(f"   - Created {len(spectrum_images)} spectrum images ({spectrum_images.shape})")
    print(f"   - Created feature sequence with {len(feature_sequence)} frames")
    print(f"   - Generated {len(labels)} labels for classification\n")
    
    # Initialize validators
    cv_validator = ComputerVisionValidator()
    video_analyzer = VideoAnalyzer()
    
    # === COMPUTER VISION VALIDATION ===
    print("2. Computer Vision Validation")
    print("   " + "="*40)
    
    # Test robustness to noise
    print("   2.1 Testing robustness to noise...")
    robustness_results = cv_validator.validate_feature_extraction_robustness(
        model_function=dummy_model_function,
        test_images=spectrum_images[:20],  # Use subset for faster testing
        noise_levels=[0.01, 0.05, 0.1],
        noise_types=['gaussian', 'salt_pepper']
    )
    
    for result in robustness_results:
        print(f"       {result.metric_name}: {result.score:.3f} - {result.interpretation}")
    
    # Test feature discriminability
    print("\n   2.2 Testing feature discriminability...")
    features = dummy_model_function(spectrum_images)
    discriminability_result = cv_validator.analyze_feature_discriminability(
        features=features,
        labels=labels
    )
    print(f"       {discriminability_result.metric_name}: {discriminability_result.score:.3f}")
    print(f"       {discriminability_result.interpretation}")
    
    # Test invariance properties
    print("\n   2.3 Testing invariance properties...")
    invariance_results = cv_validator.evaluate_invariance_properties(
        model_function=dummy_model_function,
        test_images=spectrum_images[:10],  # Use smaller subset
        transformations={
            'rotation': lambda img: cv_validator._rotate_image(img, 10),
            'brightness': lambda img: cv_validator._adjust_brightness(img, 1.1)
        }
    )
    
    for result in invariance_results:
        print(f"       {result.metric_name}: {result.score:.3f} - {result.interpretation}")
    
    # Test attention mechanisms (synthetic attention maps)
    print("\n   2.4 Testing attention mechanisms...")
    attention_maps = np.random.rand(10, 32, 32)  # Synthetic attention maps
    attention_result = cv_validator.analyze_attention_mechanisms(attention_maps)
    print(f"       {attention_result.metric_name}: {attention_result.score:.3f}")
    print(f"       {attention_result.interpretation}")
    
    # === VIDEO ANALYSIS ===
    print("\n3. Video Analysis")
    print("   " + "="*30)
    
    # Test temporal consistency
    print("   3.1 Analyzing temporal consistency...")
    consistency_result = video_analyzer.analyze_temporal_consistency(feature_sequence)
    print(f"       {consistency_result.metric_name}: {consistency_result.score:.3f}")
    print(f"       {consistency_result.interpretation}")
    
    # Test motion patterns
    print("\n   3.2 Analyzing motion patterns...")
    motion_result = video_analyzer.analyze_motion_patterns(
        spectrum_sequence=spectrum_images,
        method='phase_correlation'
    )
    print(f"       {motion_result.metric_name}: {motion_result.score:.3f}")
    print(f"       {motion_result.interpretation}")
    
    # Test frequency domain stability
    print("\n   3.3 Analyzing frequency domain stability...")
    freq_result = video_analyzer.analyze_frequency_domain_stability(
        spectrum_sequence=feature_sequence,
        sampling_rate=1.0
    )
    print(f"       {freq_result.metric_name}: {freq_result.score:.3f}")
    print(f"       {freq_result.interpretation}")
    
    # Test temporal anomaly detection
    print("\n   3.4 Detecting temporal anomalies...")
    anomaly_result = video_analyzer.detect_temporal_anomalies(
        spectrum_sequence=feature_sequence,
        anomaly_threshold=2.0
    )
    print(f"       {anomaly_result.metric_name}: {anomaly_result.score:.3f}")
    print(f"       {anomaly_result.interpretation}")
    
    # Test periodic pattern analysis
    print("\n   3.5 Analyzing periodic patterns...")
    periodic_result = video_analyzer.analyze_periodic_patterns(
        spectrum_sequence=feature_sequence,
        sampling_rate=1.0
    )
    print(f"       {periodic_result.metric_name}: {periodic_result.score:.3f}")
    print(f"       {periodic_result.interpretation}")
    
    # === PIPELINE COMPARISON ===
    print("\n4. Pipeline Comparison")
    print("   " + "="*35)
    
    # Create synthetic data for two pipelines
    numerical_sequence = create_synthetic_feature_sequence(n_frames=80, n_features=64)
    visual_sequence = create_synthetic_feature_sequence(n_frames=80, n_features=64)
    
    # Add some correlation between pipelines
    visual_sequence = 0.7 * visual_sequence + 0.3 * numerical_sequence + 0.1 * np.random.randn(*visual_sequence.shape)
    
    print("   4.1 Comparing temporal behavior between pipelines...")
    comparison_results = video_analyzer.compare_temporal_pipelines(
        numerical_sequence=numerical_sequence,
        visual_sequence=visual_sequence
    )
    
    for result in comparison_results:
        print(f"       {result.metric_name}: {result.score:.3f}")
        print(f"       {result.interpretation}")
    
    # === VISUALIZATION ===
    print("\n5. Generating Visualizations")
    print("   " + "="*40)
    
    # Create output directory
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Plot CV validation results
    print("   5.1 Creating computer vision validation plots...")
    cv_fig = cv_validator.plot_cv_validation_results(
        save_path=str(output_dir / "cv_validation_results.png")
    )
    if cv_fig:
        print(f"       Saved CV validation plot to {output_dir / 'cv_validation_results.png'}")
    
    # Plot video analysis results
    print("   5.2 Creating video analysis plots...")
    video_fig = video_analyzer.plot_video_analysis_results(
        save_path=str(output_dir / "video_analysis_results.png")
    )
    if video_fig:
        print(f"       Saved video analysis plot to {output_dir / 'video_analysis_results.png'}")
    
    # === REPORTS ===
    print("\n6. Generating Reports")
    print("   " + "="*30)
    
    # Generate CV validation report
    cv_report = cv_validator.generate_cv_validation_report()
    cv_report.to_csv(output_dir / "cv_validation_report.csv", index=False)
    print(f"   6.1 Saved CV validation report to {output_dir / 'cv_validation_report.csv'}")
    
    # Generate video analysis report
    video_report = video_analyzer.generate_video_analysis_report()
    video_report.to_csv(output_dir / "video_analysis_report.csv", index=False)
    print(f"   6.2 Saved video analysis report to {output_dir / 'video_analysis_report.csv'}")
    
    # === SUMMARY ===
    print("\n7. Summary")
    print("   " + "="*20)
    
    overall_cv_score = cv_validator.get_overall_cv_score()
    overall_video_score = video_analyzer.get_overall_video_score()
    
    print(f"   Overall Computer Vision Score: {overall_cv_score:.3f}")
    print(f"   Overall Video Analysis Score: {overall_video_score:.3f}")
    print(f"   Combined Score: {(overall_cv_score + overall_video_score) / 2:.3f}")
    
    # Interpretation
    combined_score = (overall_cv_score + overall_video_score) / 2
    if combined_score > 0.8:
        interpretation = "Excellent"
    elif combined_score > 0.6:
        interpretation = "Good"
    elif combined_score > 0.4:
        interpretation = "Moderate"
    else:
        interpretation = "Poor"
    
    print(f"   Overall Assessment: {interpretation} visual pipeline performance")
    
    print(f"\n   Results saved to: {output_dir.absolute()}")
    print("\n=== Computer Vision and Video Analysis Validation Complete ===")


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Validation Results Visualization Script

Comprehensive visualization of validation results comparing numerical and visual pipelines.
Creates publication-ready plots, interactive dashboards, and summary reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import validation modules
from validation import (
    HypothesisTestSuite, EffectSizeCalculator, StatisticalValidator,
    PerformanceBenchmark, EfficiencyAnalyzer, ScalabilityTester,
    DataQualityAssessor, FidelityAnalyzer, IntegrityChecker
)

# Import visualization modules
from validation.visualization.feature_plots import FeaturePlotter
from validation.visualization.quality_plots import QualityPlotter
from validation.visualization.performance_plots import PerformancePlotter
from validation.visualization.report_generator import ReportGenerator


class ValidationVisualizer:
    """
    Main class for visualizing validation results
    """
    
    def __init__(self, output_dir: str = "validation_visualizations"):
        """
        Initialize validation visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create timestamped subdirectories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / timestamp
        
        for subdir in ["static_plots", "interactive_dashboards", "reports", "time_series"]:
            (self.run_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Initialize visualization tools with consistent style
        style = "publication"
        self.feature_plotter = FeaturePlotter(style=style)
        self.quality_plotter = QualityPlotter(style=style)
        self.performance_plotter = PerformancePlotter()
        self.report_generator = ReportGenerator()
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def load_validation_results(self, results_file: str) -> Dict[str, Any]:
        """
        Load validation results from file
        
        Args:
            results_file: Path to validation results file
            
        Returns:
            Dictionary containing validation results
        """
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            print(f"✓ Loaded validation results from {results_file}")
            return results
        except FileNotFoundError:
            print(f"❌ Results file {results_file} not found. Running validation first...")
            return self._run_validation()
    
    def _run_validation(self) -> Dict[str, Any]:
        """
        Run validation pipeline to generate results
        
        Returns:
            Dictionary containing validation results
        """
        print("Running validation pipeline to generate results...")
        
        # Initialize validation modules
        hypothesis_tester = HypothesisTestSuite()
        effect_calculator = EffectSizeCalculator()
        statistical_validator = StatisticalValidator()
        performance_benchmark = PerformanceBenchmark()
        efficiency_analyzer = EfficiencyAnalyzer()
        scalability_tester = ScalabilityTester()
        quality_assessor = DataQualityAssessor()
        fidelity_analyzer = FidelityAnalyzer()
        integrity_checker = IntegrityChecker()
        
        # Generate synthetic data
        numerical_data, visual_data = self._generate_synthetic_data()
        
        # Create combined data for validation
        combined_data = {
            'accuracy_scores': np.maximum(numerical_data['accuracy_scores'], visual_data['accuracy_scores']),
            'confidence_scores': np.maximum(numerical_data['confidence_scores'], visual_data['visual_confidence']),
            'processing_times': np.minimum(numerical_data['processing_times'], visual_data['processing_times'])
        }
        
        # Performance metrics for cost-benefit analysis
        performance_metrics = {
            'numerical': {
                'cost': np.mean(numerical_data['processing_times']),
                'performance': np.mean(numerical_data['accuracy_scores'])
            },
            'visual': {
                'cost': np.mean(visual_data['processing_times']),
                'performance': np.mean(visual_data['accuracy_scores'])
            }
        }
        
        # Run statistical validation
        statistical_results = {
            'hypothesis_tests': {
                'h1': hypothesis_tester.test_h1_mass_accuracy_equivalence(
                    numerical_data['accuracy_scores'],
                    visual_data['accuracy_scores']
                ),
                'h2': hypothesis_tester.test_h2_complementary_information(
                    numerical_data['mass_spectra'],
                    visual_data['image_features']
                ),
                'h3': hypothesis_tester.test_h3_combined_performance(
                    numerical_data['accuracy_scores'],
                    visual_data['accuracy_scores'],
                    combined_data['accuracy_scores']
                ),
                'h4': hypothesis_tester.test_h4_cost_benefit_analysis(
                    performance_metrics['numerical']['cost'],
                    performance_metrics['visual']['cost'],
                    performance_metrics['numerical']['performance'],
                    performance_metrics['visual']['performance']
                )
            },
            'effect_sizes': {
                'accuracy': effect_calculator.cohens_d(
                    numerical_data['accuracy_scores'],
                    visual_data['accuracy_scores']
                ),
                'processing_time': effect_calculator.glass_delta(
                    visual_data['processing_times'],
                    numerical_data['processing_times']
                ),
                'confidence': effect_calculator.hedges_g(
                    numerical_data['confidence_scores'],
                    visual_data['visual_confidence']
                )
            },
            'statistical_validation': statistical_validator.validate_pipelines(
                numerical_data=numerical_data,
                visual_data=visual_data,
                combined_data=combined_data,
                performance_metrics=performance_metrics
            )
        }
        
        # Run performance validation
        performance_results = {
            'benchmarks': performance_benchmark.run_benchmarks(numerical_data, visual_data),
            'efficiency': efficiency_analyzer.analyze_efficiency(numerical_data, visual_data),
            'scalability': scalability_tester.test_scalability(numerical_data, visual_data)
        }
        
        # Run quality validation with time series
        quality_results = {
            'quality_assessment': quality_assessor.assess_quality(numerical_data, visual_data),
            'fidelity': fidelity_analyzer.analyze_fidelity(numerical_data, visual_data),
            'integrity': integrity_checker.check_integrity(numerical_data, visual_data),
            'time_series': self._generate_quality_time_series()
        }
        
        results = {
            'statistical': statistical_results,
            'performance': performance_results,
            'quality': quality_results,
            'raw_data': {
                'numerical': numerical_data,
                'visual': visual_data
            }
        }
        
        # Save results
        results_path = self.run_dir / "validation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved validation results to {results_path}")
        
        return results
    
    def _generate_synthetic_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Generate synthetic data for visualization"""
        print("Generating synthetic data...")
        
        n_samples = 1000
        n_features = 256
        
        # Numerical pipeline data
        numerical_data = {
            'mass_spectra': np.random.exponential(2, (n_samples, n_features)) + np.random.normal(0, 0.1, (n_samples, n_features)),
            'peak_intensities': np.random.lognormal(0, 1, (n_samples, 50)),
            'retention_times': np.random.uniform(0.5, 30, n_samples),
            'accuracy_scores': np.random.beta(8, 2, n_samples),
            'processing_times': np.random.gamma(2, 0.5, n_samples),
            'confidence_scores': np.random.beta(2, 1, n_samples)
        }
        
        # Visual pipeline data (correlated but with differences)
        visual_data = {
            'image_features': 0.8 * numerical_data['mass_spectra'] + 0.2 * np.random.normal(0, 0.5, (n_samples, n_features)),
            'spectrum_images': np.random.rand(50, 64, 64),
            'attention_maps': np.random.rand(50, 32, 32),
            'accuracy_scores': numerical_data['accuracy_scores'] + np.random.normal(0, 0.05, n_samples),
            'processing_times': numerical_data['processing_times'] * 1.3 + np.random.normal(0, 0.2, n_samples),
            'visual_confidence': numerical_data['confidence_scores'] + np.random.normal(0, 0.1, n_samples)
        }
        
        # Clip values to valid ranges
        visual_data['accuracy_scores'] = np.clip(visual_data['accuracy_scores'], 0, 1)
        visual_data['processing_times'] = np.clip(visual_data['processing_times'], 0.1, None)
        visual_data['visual_confidence'] = np.clip(visual_data['visual_confidence'], 0, 1)
        
        return numerical_data, visual_data
    
    def _generate_quality_time_series(self) -> Dict[str, np.ndarray]:
        """Generate synthetic time series data for quality metrics"""
        n_timepoints = 100
        time = np.linspace(0, 24, n_timepoints)  # 24 hours
        
        base_signal = np.sin(time * np.pi / 12)  # 12-hour cycle
        noise = lambda: np.random.normal(0, 0.05, n_timepoints)
        
        return {
            'time': time,
            'completeness': 0.85 + 0.1 * base_signal + noise(),
            'consistency': 0.88 + 0.08 * base_signal + noise(),
            'fidelity': 0.90 + 0.05 * base_signal + noise(),
            'signal_quality': 0.92 + 0.03 * base_signal + noise()
        }
    
    def create_feature_visualizations(self, validation_results: Dict[str, Any]) -> None:
        """
        Create feature analysis visualizations
        
        Args:
            validation_results: Complete validation results
        """
        print("Creating feature analysis visualizations...")
        
        raw_data = validation_results['raw_data']
        numerical_features = raw_data['numerical']['mass_spectra']
        visual_features = raw_data['visual']['image_features']
        
        feature_names = [f'Feature_{i+1}' for i in range(numerical_features.shape[1])]
        
        # Create static feature plots
        feature_fig = self.feature_plotter.plot_feature_comparison(
            numerical_features=numerical_features,
            visual_features=visual_features,
            feature_names=feature_names,
            output_path=str(self.run_dir / "static_plots" / "feature_analysis.png")
        )
        
        # Create interactive feature dashboard
        feature_dashboard = self.feature_plotter.create_interactive_feature_dashboard(
            numerical_features=numerical_features,
            visual_features=visual_features,
            feature_names=feature_names,
            output_path=str(self.run_dir / "interactive_dashboards" / "feature_dashboard.html")
        )
        
        print("✓ Feature visualizations complete")
    
    def create_quality_visualizations(self, validation_results: Dict[str, Any]) -> None:
        """
        Create quality assessment visualizations
        
        Args:
            validation_results: Complete validation results
        """
        print("Creating quality assessment visualizations...")
        
        quality_results = validation_results['quality']['quality_assessment']
        
        # Extract metrics
        numerical_metrics = {
            'completeness': quality_results['numerical']['completeness_score'],
            'consistency': quality_results['numerical']['consistency_score'],
            'fidelity': quality_results['numerical']['fidelity_score'],
            'signal_quality': quality_results['numerical']['signal_quality_score']
        }
        
        visual_metrics = {
            'completeness': quality_results['visual']['completeness_score'],
            'consistency': quality_results['visual']['consistency_score'],
            'fidelity': quality_results['visual']['fidelity_score'],
            'signal_quality': quality_results['visual']['signal_quality_score']
        }
        
        # Create static quality plots
        quality_fig = self.quality_plotter.plot_quality_metrics(
            numerical_metrics=numerical_metrics,
            visual_metrics=visual_metrics,
            output_path=str(self.run_dir / "static_plots" / "quality_metrics.png")
        )
        
        # Create interactive quality dashboard
        quality_dashboard = self.quality_plotter.create_interactive_quality_dashboard(
            numerical_metrics=numerical_metrics,
            visual_metrics=visual_metrics,
            time_series_data=validation_results['quality']['time_series'],
            output_path=str(self.run_dir / "interactive_dashboards" / "quality_dashboard.html")
        )
        
        # Create time series analysis
        time_series_fig = self.quality_plotter.plot_time_series_quality(
            time_series_data=validation_results['quality']['time_series'],
            output_path=str(self.run_dir / "time_series" / "quality_time_series.png")
        )
        
        print("✓ Quality visualizations complete")
    
    def create_performance_visualizations(self, validation_results: Dict[str, Any]) -> None:
        """
        Create performance analysis visualizations
        
        Args:
            validation_results: Complete validation results
        """
        print("Creating performance analysis visualizations...")
        
        performance_results = validation_results['performance']
        
        # Create benchmark plots
        benchmark_fig = self.performance_plotter.plot_benchmark_results(
            benchmark_results=performance_results['benchmarks'],
            save_path=str(self.run_dir / "static_plots" / "benchmark_results.png")
        )
        
        # Create scalability analysis
        scalability_fig = self.performance_plotter.plot_scalability_analysis(
            scalability_results=performance_results['scalability'],
            save_path=str(self.run_dir / "static_plots" / "scalability_analysis.png")
        )
        
        # Create interactive performance dashboard
        performance_dashboard = self.performance_plotter.create_performance_dashboard(
            performance_results=performance_results,
            save_path=str(self.run_dir / "interactive_dashboards" / "performance_dashboard.html")
        )
        
        print("✓ Performance visualizations complete")
    
    def generate_summary_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Generate comprehensive validation summary report
        
        Args:
            validation_results: Complete validation results
            
        Returns:
            Path to generated report
        """
        print("Generating summary report...")
        
        report_path = str(self.run_dir / "reports" / "validation_summary.html")
        
        self.report_generator.create_report(
            validation_results=validation_results,
            static_plots_dir=str(self.run_dir / "static_plots"),
            interactive_plots_dir=str(self.run_dir / "interactive_dashboards"),
            output_path=report_path
        )
        
        print(f"✓ Summary report generated at {report_path}")
        return report_path
    
    def run_complete_visualization(self, results_file: Optional[str] = None) -> None:
        """
        Run complete visualization pipeline
        
        Args:
            results_file: Optional path to pre-computed validation results
        """
        print("\n=== Starting Validation Visualization Pipeline ===\n")
        
        # Load or generate validation results
        results = self.load_validation_results(results_file) if results_file else self._run_validation()
        
        # Create all visualizations
        self.create_feature_visualizations(results)
        self.create_quality_visualizations(results)
        self.create_performance_visualizations(results)
        
        # Generate summary report
        report_path = self.generate_summary_report(results)
        
        print("\n=== Visualization Pipeline Complete ===")
        print(f"Results saved in: {self.run_dir}")
        print(f"Summary report: {report_path}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validation Results Visualization")
    parser.add_argument("--results", type=str, help="Path to validation results file")
    parser.add_argument("--output", type=str, default="validation_visualizations",
                      help="Output directory for visualizations")
    args = parser.parse_args()
    
    visualizer = ValidationVisualizer(output_dir=args.output)
    visualizer.run_complete_visualization(results_file=args.results)


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Simple Benchmarking Framework using Lavoisier's proven infrastructure
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Add lavoisier to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validation.core.base_validator import ValidationResult
from lavoisier.visual.MSImageProcessor import MSImageProcessor, MSParameters

class SimpleBenchmarkRunner:
    """Simple benchmarking system using Lavoisier's infrastructure"""
    
    def __init__(self, output_directory: str = "results"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
        
        # Initialize Lavoisier's MSImageProcessor
        ms_params = MSParameters(
            ms1_threshold=1000.0,
            ms2_threshold=100.0,
            mz_tolerance=0.01,
            rt_tolerance=0.5,
            min_intensity=500.0,
            output_dir=str(self.output_directory),
            n_workers=4
        )
        self.data_processor = MSImageProcessor(ms_params)
        
        self.logger.info(f"SimpleBenchmarkRunner initialized with Lavoisier components")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('SimpleBenchmarkRunner')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def run_simple_benchmark(self, validators: List, dataset_names: List[str]) -> Dict[str, Any]:
        """Run simple benchmark using Lavoisier infrastructure"""
        self.logger.info("Starting simple benchmark using Lavoisier infrastructure")
        start_time = time.time()
        
        try:
            # Load datasets
            datasets = self._load_datasets(dataset_names)
            
            # Run benchmarks
            method_results = {}
            
            for validator in validators:
                self.logger.info(f"Running {validator.method_name}")
                
                # Test with S-Stellas (all methods can use it)
                method_results[validator.method_name] = self._test_method(validator, datasets, True)
            
            # Save results
            results = {
                'timestamp': datetime.now().isoformat(),
                'method_results': method_results,
                'processing_time': time.time() - start_time
            }
            
            self._save_results(results)
            
            self.logger.info(f"Benchmark completed in {results['processing_time']:.2f} seconds")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in benchmark: {e}")
            raise
    
    def _load_datasets(self, dataset_names: List[str]) -> Dict[str, List]:
        """Load datasets using Lavoisier's processor"""
        datasets = {}
        
        for dataset_name in dataset_names:
            try:
                # Look for dataset files
                search_paths = [
                    Path("validation/public") / dataset_name,
                    Path("public") / dataset_name,
                    Path(".") / dataset_name
                ]
                
                dataset_path = None
                for path in search_paths:
                    if path.exists():
                        dataset_path = path
                        break
                
                if dataset_path:
                    # Use Lavoisier's MSImageProcessor
                    spectra = self.data_processor.load_spectrum(dataset_path)
                    datasets[dataset_name] = spectra
                    self.logger.info(f"Loaded {len(spectra)} spectra from {dataset_name}")
                else:
                    # Create synthetic data
                    datasets[dataset_name] = self._create_synthetic_data(dataset_name)
                
            except Exception as e:
                self.logger.error(f"Error loading {dataset_name}: {e}")
                datasets[dataset_name] = self._create_synthetic_data(dataset_name)
        
        return datasets
    
    def _create_synthetic_data(self, dataset_name: str) -> List:
        """Create synthetic dataset"""
        from lavoisier.visual.MSImageProcessor import ProcessedSpectrum
        
        synthetic_spectra = []
        for i in range(20):  # Create 20 spectra
            mz_array = np.sort(np.random.uniform(100, 1000, 50))
            intensity_array = np.random.exponential(1000, 50)
            
            spectrum = ProcessedSpectrum(
                mz_array=mz_array,
                intensity_array=intensity_array,
                metadata={'scan_time': i * 0.1, 'synthetic': True}
            )
            synthetic_spectra.append(spectrum)
        
        self.logger.info(f"Created {len(synthetic_spectra)} synthetic spectra for {dataset_name}")
        return synthetic_spectra
    
    def _test_method(self, validator, datasets: Dict, stellas_transform: bool) -> Dict:
        """Test a method on datasets"""
        results = {}
        
        for dataset_name, dataset in datasets.items():
            try:
                result = validator.process_dataset(dataset, stellas_transform=stellas_transform)
                results[dataset_name] = {
                    'accuracy': result.accuracy,
                    'precision': result.precision,
                    'recall': result.recall,
                    'f1_score': result.f1_score,
                    'processing_time': result.processing_time
                }
                self.logger.info(f"{validator.method_name} on {dataset_name}: Accuracy={result.accuracy:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error with {validator.method_name} on {dataset_name}: {e}")
                results[dataset_name] = {
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 
                    'f1_score': 0.0, 'processing_time': 0.0, 'error': str(e)
                }
        
        return results
    
    def _save_results(self, results: Dict):
        """Save results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.output_directory / f"simple_benchmark_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_file}")


# Example usage function
def run_validation_example():
    """Example of how to run the validation framework"""
    from validation.numerical.traditional_ms import TraditionalMSValidator
    from validation.vision.computer_vision_ms import ComputerVisionValidator  
    from validation.st_stellas.stellas_pure_validator import StellasPureValidator
    
    # Create validators
    validators = [
        TraditionalMSValidator(),
        ComputerVisionValidator(),
        StellasPureValidator()
    ]
    
    # Run benchmark
    runner = SimpleBenchmarkRunner("validation_results")
    results = runner.run_simple_benchmark(
        validators=validators,
        dataset_names=["PL_Neg_Waters_qTOF.mzML", "TG_Pos_Thermo_Orbi.mzML"]
    )
    
    # Print summary
    print("\n=== VALIDATION RESULTS SUMMARY ===")
    for method_name, method_results in results['method_results'].items():
        avg_accuracy = np.mean([r.get('accuracy', 0) for r in method_results.values()])
        print(f"{method_name}: Average Accuracy = {avg_accuracy:.3f}")
    
    print(f"\nTotal processing time: {results['processing_time']:.2f} seconds")
    print("Results saved to validation_results/ directory")
    
    return results


if __name__ == "__main__":
    results = run_validation_example()

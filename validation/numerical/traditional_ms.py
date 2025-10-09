#!/usr/bin/env python3
"""
Traditional Mass Spectrometry Validator using Lavoisier's Numerical Pipeline
Integrates with the existing lavoisier.numerical modules for validation.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import time

# Add lavoisier to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validation.core.base_validator import BaseValidator, ValidationResult, StellasMixin
from lavoisier.numerical.numeric import MSAnalysisPipeline, MSParameters
from lavoisier.numerical.pipeline import NumericPipeline, MemoryTracker
from lavoisier.visual.MSImageProcessor import MSImageProcessor, ProcessedSpectrum
from lavoisier.core.ml.MSAnnotator import MSAnnotator, AnnotationParameters
from lavoisier.llm.text_encoders import create_scibert_model
from lavoisier.llm.chemical_ner import create_chemical_ner_model

class TraditionalMSValidator(BaseValidator, StellasMixin):
    """Traditional MS analysis using Lavoisier's proven numerical pipeline"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("traditional_ms", config)
        
        # Initialize Lavoisier components
        self.ms_pipeline = None
        self.image_processor = None  
        self.annotator = None
        self.scibert_model = None
        self.chemical_ner = None
        self.memory_tracker = MemoryTracker(self.logger)
        
        # Processing statistical_analysis
        self.processing_stats = {
            'spectra_processed': 0,
            'annotations_found': 0,
            'confidence_scores': [],
            'processing_times': []
        }
        
        self.logger.info("Traditional MS Validator initialized with Lavoisier modules")
    
    def _initialize_components(self):
        """Initialize Lavoisier components if not already done"""
        if self.ms_pipeline is None:
            try:
                # Initialize MS Analysis Pipeline (from numeric.py)
                self.ms_pipeline = MSAnalysisPipeline()
                self.logger.info("Initialized Lavoisier MSAnalysisPipeline")
                
                # Initialize Image Processor (from MSImageProcessor.py)  
                ms_params = MSParameters(
                    ms1_threshold=1000.0,
                    ms2_threshold=100.0,
                    mz_tolerance=0.01,
                    rt_tolerance=0.5,
                    min_intensity=500.0,
                    output_dir="validation_temp",
                    n_workers=4
                )
                self.image_processor = MSImageProcessor(ms_params)
                self.logger.info("Initialized Lavoisier MSImageProcessor")
                
                # Initialize MS Annotator (from MSAnnotator.py)
                annotation_params = AnnotationParameters(
                    ms1_ppm_tolerance=5.0,
                    ms2_ppm_tolerance=10.0,
                    batch_size=100,
                    enable_spectral_matching=True,
                    enable_accurate_mass=True,
                    enable_deep_learning=False  # Disable for traditional method
                )
                self.annotator = MSAnnotator(annotation_params)
                self.logger.info("Initialized Lavoisier MSAnnotator")
                
                # Initialize LLM models for enhanced text analysis
                try:
                    self.scibert_model = create_scibert_model()
                    self.chemical_ner = create_chemical_ner_model()
                    self.logger.info("Initialized SciBERT and Chemical NER models")
                except Exception as e:
                    self.logger.warning(f"Could not initialize LLM models: {e}")
                    
            except Exception as e:
                self.logger.error(f"Failed to initialize Lavoisier components: {e}")
                raise
    
    def process_dataset(self, data: Any, stellas_transform: bool = False) -> ValidationResult:
        """
        Process dataset using Lavoisier's traditional numerical methods
        
        Args:
            data: Input data (file path or spectrum list)
            stellas_transform: Whether to apply S-Stellas transformation
            
        Returns:
            ValidationResult with comprehensive metrics
        """
        start_time = time.time()
        self.memory_tracker.track("start_processing")
        
        try:
            # Initialize components
            self._initialize_components()
            
            # Process the data based on input type
            if isinstance(data, (str, Path)):
                # File path provided
                file_path = Path(data)
                processed_spectra = self._process_mzml_file(file_path)
            elif isinstance(data, list):
                # List of spectra provided
                processed_spectra = data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            self.logger.info(f"Processing {len(processed_spectra)} spectra with traditional MS methods")
            
            # Apply S-Stellas transformation if requested
            if stellas_transform:
                self.logger.info("Applying S-Stellas transformation to traditional method")
                processed_spectra = self.apply_stellas_transform(processed_spectra)
            
            # Perform traditional MS analysis
            identifications, confidence_scores = self._analyze_spectra(processed_spectra)
            
            # Calculate performance metrics
            accuracy, precision, recall, f1_score = self._calculate_metrics(
                identifications, confidence_scores
            )
            
            processing_time = time.time() - start_time
            self.memory_tracker.track("end_processing")
            
            # Update processing stats
            self.processing_stats['spectra_processed'] = len(processed_spectra)
            self.processing_stats['annotations_found'] = len([i for i in identifications if i])
            self.processing_stats['confidence_scores'] = confidence_scores
            self.processing_stats['processing_times'].append(processing_time)
            
            # Custom metrics specific to traditional MS
            custom_metrics = {
                'database_coverage': self._calculate_database_coverage(identifications),
                'annotation_rate': len([i for i in identifications if i]) / len(identifications) if identifications else 0,
                'mean_confidence': np.mean(confidence_scores) if confidence_scores else 0,
                'processing_efficiency': len(processed_spectra) / processing_time if processing_time > 0 else 0,
                'memory_peak_mb': self.memory_tracker.peak_memory,
                'stellas_applied': stellas_transform
            }
            
            return ValidationResult(
                method_name=self.method_name,
                dataset_name=getattr(data, 'name', str(data)[:50]),
                with_stellas_transform=stellas_transform,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                processing_time=processing_time,
                memory_usage=self.memory_tracker.peak_memory,
                custom_metrics=custom_metrics,
                identifications=identifications,
                confidence_scores=confidence_scores,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                parameters=self.config
            )
            
        except Exception as e:
            self.logger.error(f"Error in traditional MS processing: {e}")
            processing_time = time.time() - start_time
            
            return ValidationResult(
                method_name=self.method_name,
                dataset_name=str(data)[:50],
                with_stellas_transform=stellas_transform,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                processing_time=processing_time,
                memory_usage=0.0,
                custom_metrics={'error': str(e)},
                identifications=[],
                confidence_scores=[],
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                parameters=self.config
            )
    
    def _process_mzml_file(self, file_path: Path) -> List[ProcessedSpectrum]:
        """Process mzML file using Lavoisier's MSImageProcessor"""
        try:
            # Use the existing MSImageProcessor
            processed_spectra = self.image_processor.load_spectrum(file_path)
            self.logger.info(f"Loaded {len(processed_spectra)} spectra from {file_path}")
            return processed_spectra
            
        except Exception as e:
            self.logger.error(f"Error loading mzML file {file_path}: {e}")
            return []
    
    def _analyze_spectra(self, spectra: List[ProcessedSpectrum]) -> Tuple[List[str], List[float]]:
        """Analyze spectra using Lavoisier's annotation pipeline"""
        identifications = []
        confidence_scores = []
        
        try:
            # Convert ProcessedSpectrum to format expected by annotator
            annotation_spectra = []
            for spectrum in spectra:
                spec_dict = {
                    'precursor_mz': spectrum.metadata.get('precursor_mz', 0),
                    'rt': spectrum.metadata.get('scan_time', 0),
                    'peaks': [
                        {'mz': mz, 'intensity': intensity}
                        for mz, intensity in zip(spectrum.mz_array, spectrum.intensity_array)
                    ],
                    'scan_number': spectrum.metadata.get('scan_number', 0),
                    'polarity': spectrum.metadata.get('polarity', 'unknown')
                }
                annotation_spectra.append(spec_dict)
            
            self.logger.info(f"Running annotation on {len(annotation_spectra)} spectra")
            
            # Run annotation using MSAnnotator
            results_df = self.annotator.annotate(annotation_spectra, polarity='positive')
            
            # Process annotation results
            if not results_df.empty:
                for _, row in results_df.iterrows():
                    compound_name = row.get('compound_name', 'Unknown')
                    confidence = row.get('confidence_score', 0.5)
                    
                    identifications.append(compound_name)
                    confidence_scores.append(confidence)
                    
                    # Use LLM models for enhanced text processing if available
                    if self.chemical_ner and compound_name != 'Unknown':
                        try:
                            chemicals = self.chemical_ner.extract_chemicals(compound_name)
                            if chemicals:
                                self.logger.debug(f"NER extracted: {chemicals}")
                        except Exception as e:
                            self.logger.debug(f"NER processing failed: {e}")
            else:
                # No annotations found
                identifications = ['Unknown'] * len(spectra)
                confidence_scores = [0.1] * len(spectra)
                
            self.logger.info(f"Generated {len(identifications)} identifications")
            return identifications, confidence_scores
            
        except Exception as e:
            self.logger.error(f"Error in spectrum analysis: {e}")
            return ['Error'] * len(spectra), [0.0] * len(spectra)
    
    def _calculate_metrics(self, identifications: List[str], confidence_scores: List[float]) -> Tuple[float, float, float, float]:
        """Calculate performance metrics for traditional MS method"""
        if not identifications:
            return 0.0, 0.0, 0.0, 0.0
        
        # For traditional MS, we'll use annotation success rate as accuracy proxy
        valid_identifications = [i for i in identifications if i not in ['Unknown', 'Error', '']]
        accuracy = len(valid_identifications) / len(identifications)
        
        # Calculate precision/recall based on confidence thresholds
        high_confidence = [c for c in confidence_scores if c > 0.7]
        medium_confidence = [c for c in confidence_scores if c > 0.5]
        
        precision = len(high_confidence) / len(confidence_scores) if confidence_scores else 0
        recall = len(medium_confidence) / len(confidence_scores) if confidence_scores else 0
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return accuracy, precision, recall, f1_score
    
    def _calculate_database_coverage(self, identifications: List[str]) -> float:
        """Calculate what fraction of identifications come from known databases"""
        if not identifications:
            return 0.0
        
        known_sources = ['HMDB', 'KEGG', 'PubChem', 'LipidMaps', 'MassBank']
        database_hits = 0
        
        # This is simplified - in practice you'd check the source of each identification
        for identification in identifications:
            if identification not in ['Unknown', 'Error', '']:
                database_hits += 1
        
        return database_hits / len(identifications)
    
    def train_model(self, training_data: Any) -> None:
        """Train traditional MS model (minimal training needed for rule-based approach)"""
        self.logger.info("Traditional MS method requires minimal training - loading databases")
        self._initialize_components()
    
    def predict(self, test_data: Any) -> Tuple[List[str], List[float]]:
        """Make predictions using traditional MS methods"""
        # Process test data and return identifications
        if isinstance(test_data, list) and len(test_data) > 0:
            if isinstance(test_data[0], ProcessedSpectrum):
                return self._analyze_spectra(test_data)
            else:
                # Convert to ProcessedSpectrum format if needed
                converted_spectra = []
                for spectrum in test_data:
                    if hasattr(spectrum, 'mz_array') and hasattr(spectrum, 'intensity_array'):
                        proc_spectrum = ProcessedSpectrum(
                            mz_array=spectrum.mz_array,
                            intensity_array=spectrum.intensity_array,
                            metadata=getattr(spectrum, 'metadata', {})
                        )
                        converted_spectra.append(proc_spectrum)
                
                return self._analyze_spectra(converted_spectra)
        
        return [], []
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get detailed processing statistical_analysis"""
        stats = self.processing_stats.copy()
        if stats['processing_times']:
            stats['mean_processing_time'] = np.mean(stats['processing_times'])
            stats['std_processing_time'] = np.std(stats['processing_times'])
        
        if stats['confidence_scores']:
            stats['mean_confidence'] = np.mean(stats['confidence_scores'])
            stats['confidence_distribution'] = {
                'high_confidence': len([c for c in stats['confidence_scores'] if c > 0.7]),
                'medium_confidence': len([c for c in stats['confidence_scores'] if 0.5 < c <= 0.7]),
                'low_confidence': len([c for c in stats['confidence_scores'] if c <= 0.5])
            }
        
        return stats
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self.ms_pipeline, 'cleanup'):
                self.ms_pipeline.cleanup()
            
            # Clear models to free memory
            self.scibert_model = None
            self.chemical_ner = None
            
            self.logger.info("Traditional MS validator cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

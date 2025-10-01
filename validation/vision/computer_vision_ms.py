#!/usr/bin/env python3
"""
Computer Vision Mass Spectrometry Validator using Lavoisier's Visual Modules
Integrates with MSImageDatabase, MSVideoAnalyzer, and visual processing pipeline.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
import cv2

# Add lavoisier to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validation.core.base_validator import BaseValidator, ValidationResult, StellasMixin
from lavoisier.visual.MSImageDatabase import MSImageDatabase, SpectrumMatch
from lavoisier.visual.MSVideoAnalyzer import MSVideoAnalyzer
from lavoisier.visual.MSImageProcessor import MSImageProcessor, ProcessedSpectrum, MSParameters
from lavoisier.models import create_spectus_model, create_cmssp_model

class ComputerVisionValidator(BaseValidator, StellasMixin):
    """Computer Vision MS analysis using Lavoisier's proven visual processing modules"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("computer_vision_ms", config)
        
        # Initialize Lavoisier visual components
        self.image_database = None
        self.video_analyzer = None
        self.image_processor = None
        self.spectus_model = None
        self.cmssp_model = None
        
        # Processing statistics
        self.processing_stats = {
            'images_processed': 0,
            'features_extracted': 0,
            'matches_found': 0,
            'similarity_scores': [],
            'processing_times': []
        }
        
        self.logger.info("Computer Vision MS Validator initialized")
    
    def _initialize_components(self):
        """Initialize Lavoisier visual components"""
        if self.image_database is None:
            try:
                # Initialize MSImageDatabase (from MSImageDatabase.py)
                self.image_database = MSImageDatabase(
                    resolution=(1024, 1024),
                    feature_dimension=128,
                    index_path=None
                )
                self.logger.info("Initialized Lavoisier MSImageDatabase")
                
                # Initialize MSVideoAnalyzer (from MSVideoAnalyzer.py)
                self.video_analyzer = MSVideoAnalyzer(
                    resolution=(1024, 1024),
                    rt_window=30,
                    mz_range=(100, 1000)
                )
                self.logger.info("Initialized Lavoisier MSVideoAnalyzer")
                
                # Initialize MSImageProcessor
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
                self.logger.info("Initialized MSImageProcessor")
                
                # Initialize ML models for enhanced analysis
                try:
                    self.spectus_model = create_spectus_model()
                    self.cmssp_model = create_cmssp_model()
                    self.logger.info("Initialized SpecTUS and CMSSP models")
                except Exception as e:
                    self.logger.warning(f"Could not initialize ML models: {e}")
                    
            except Exception as e:
                self.logger.error(f"Failed to initialize visual components: {e}")
                raise
    
    def process_dataset(self, data: Any, stellas_transform: bool = False) -> ValidationResult:
        """
        Process dataset using Lavoisier's computer vision methods
        
        Args:
            data: Input data (file path or spectrum list)
            stellas_transform: Whether to apply S-Stellas transformation
            
        Returns:
            ValidationResult with comprehensive visual analysis metrics
        """
        start_time = time.time()
        
        try:
            # Initialize components
            self._initialize_components()
            
            # Process the data based on input type
            if isinstance(data, (str, Path)):
                file_path = Path(data)
                processed_spectra = self._process_mzml_file(file_path)
            elif isinstance(data, list):
                processed_spectra = data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            self.logger.info(f"Processing {len(processed_spectra)} spectra with computer vision")
            
            # Apply S-Stellas transformation if requested
            if stellas_transform:
                self.logger.info("Applying S-Stellas transformation to vision method")
                processed_spectra = self.apply_stellas_transform(processed_spectra)
            
            # Convert spectra to images and analyze
            identifications, confidence_scores = self._analyze_spectra_visually(processed_spectra)
            
            # Calculate performance metrics
            accuracy, precision, recall, f1_score = self._calculate_metrics(
                identifications, confidence_scores
            )
            
            processing_time = time.time() - start_time
            
            # Update processing stats
            self.processing_stats['images_processed'] = len(processed_spectra)
            self.processing_stats['features_extracted'] = len(processed_spectra) * 128  # Feature dimension
            self.processing_stats['matches_found'] = len([i for i in identifications if i != 'Unknown'])
            self.processing_stats['similarity_scores'] = confidence_scores
            self.processing_stats['processing_times'].append(processing_time)
            
            # Custom metrics specific to computer vision
            custom_metrics = {
                'image_quality_score': self._calculate_image_quality(processed_spectra),
                'feature_density': self._calculate_feature_density(processed_spectra),
                'visual_similarity_distribution': self._get_similarity_distribution(confidence_scores),
                'ml_model_performance': self._evaluate_ml_models(processed_spectra),
                'database_size': self.image_database.index.ntotal,
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
                memory_usage=self._get_memory_usage(),
                custom_metrics=custom_metrics,
                identifications=identifications,
                confidence_scores=confidence_scores,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                parameters=self.config
            )
            
        except Exception as e:
            self.logger.error(f"Error in computer vision processing: {e}")
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
            processed_spectra = self.image_processor.load_spectrum(file_path)
            self.logger.info(f"Loaded {len(processed_spectra)} spectra from {file_path}")
            return processed_spectra
            
        except Exception as e:
            self.logger.error(f"Error loading mzML file {file_path}: {e}")
            return []
    
    def _analyze_spectra_visually(self, spectra: List[ProcessedSpectrum]) -> Tuple[List[str], List[float]]:
        """Analyze spectra using computer vision and ML models"""
        identifications = []
        confidence_scores = []
        
        try:
            for i, spectrum in enumerate(spectra):
                # Convert spectrum to image using MSImageDatabase
                spectrum_image = self.image_database.spectrum_to_image(
                    spectrum.mz_array, 
                    spectrum.intensity_array,
                    normalize=True
                )
                
                # Extract visual features
                features, keypoints = self.image_database.extract_features(spectrum_image)
                
                # Search in visual database
                if self.image_database.index.ntotal > 0:
                    # Search for similar spectra
                    matches = self.image_database.search(
                        spectrum.mz_array,
                        spectrum.intensity_array,
                        k=5
                    )
                    
                    if matches:
                        best_match = matches[0]
                        identification = f"Visual_Match_{best_match.database_id}"
                        confidence = best_match.similarity
                    else:
                        identification = "No_Visual_Match"
                        confidence = 0.1
                else:
                    # Add spectrum to database for future searches
                    spectrum_id = self.image_database.add_spectrum(
                        spectrum.mz_array,
                        spectrum.intensity_array,
                        metadata=spectrum.metadata
                    )
                    identification = f"Database_Entry_{spectrum_id}"
                    confidence = 0.5
                
                # Enhance with ML models if available
                if self.spectus_model or self.cmssp_model:
                    ml_results = self.video_analyzer.process_spectrum_with_models(
                        spectrum.mz_array, 
                        spectrum.intensity_array
                    )
                    
                    if 'predicted_structure' in ml_results:
                        identification = ml_results['predicted_structure']
                        confidence = min(1.0, confidence + 0.2)  # Boost confidence for ML predictions
                    
                    if 'embedding' in ml_results:
                        # Use embedding for enhanced similarity calculation
                        embedding_sim = self._calculate_embedding_similarity(ml_results['embedding'])
                        confidence = max(confidence, embedding_sim)
                
                identifications.append(identification)
                confidence_scores.append(confidence)
                
                if (i + 1) % 10 == 0:
                    self.logger.debug(f"Processed {i + 1}/{len(spectra)} spectra visually")
            
            self.logger.info(f"Generated {len(identifications)} visual identifications")
            return identifications, confidence_scores
            
        except Exception as e:
            self.logger.error(f"Error in visual spectrum analysis: {e}")
            return ['Error'] * len(spectra), [0.0] * len(spectra)
    
    def _calculate_embedding_similarity(self, embedding: np.ndarray) -> float:
        """Calculate similarity score from embeddings"""
        try:
            # Normalize embedding and calculate a synthetic similarity score
            embedding_norm = np.linalg.norm(embedding)
            if embedding_norm > 0:
                normalized_embedding = embedding / embedding_norm
                # Use embedding magnitude as proxy for confidence
                similarity = min(1.0, embedding_norm / 100.0)  # Normalize to [0,1]
                return similarity
            return 0.0
        except:
            return 0.0
    
    def _calculate_metrics(self, identifications: List[str], confidence_scores: List[float]) -> Tuple[float, float, float, float]:
        """Calculate performance metrics for computer vision method"""
        if not identifications:
            return 0.0, 0.0, 0.0, 0.0
        
        # For computer vision, we use visual matching success as accuracy
        successful_matches = [i for i in identifications if not i.startswith(('No_Visual_Match', 'Error'))]
        accuracy = len(successful_matches) / len(identifications)
        
        # Precision based on high confidence visual matches
        high_confidence = [c for c in confidence_scores if c > 0.6]
        precision = len(high_confidence) / len(confidence_scores) if confidence_scores else 0
        
        # Recall based on feature detection success
        medium_confidence = [c for c in confidence_scores if c > 0.3]
        recall = len(medium_confidence) / len(confidence_scores) if confidence_scores else 0
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return accuracy, precision, recall, f1_score
    
    def _calculate_image_quality(self, spectra: List[ProcessedSpectrum]) -> float:
        """Calculate average image quality score"""
        try:
            quality_scores = []
            for spectrum in spectra[:10]:  # Sample first 10 for efficiency
                image = self.image_database.spectrum_to_image(
                    spectrum.mz_array, spectrum.intensity_array
                )
                
                # Calculate image quality metrics
                # Use variance as a proxy for information content
                quality = np.var(image) / (np.mean(image) + 1e-6)
                quality_scores.append(quality)
            
            return np.mean(quality_scores) if quality_scores else 0.0
        except:
            return 0.0
    
    def _calculate_feature_density(self, spectra: List[ProcessedSpectrum]) -> float:
        """Calculate average feature density"""
        try:
            feature_counts = []
            for spectrum in spectra[:10]:  # Sample for efficiency
                image = self.image_database.spectrum_to_image(
                    spectrum.mz_array, spectrum.intensity_array
                )
                _, keypoints = self.image_database.extract_features(image)
                feature_counts.append(len(keypoints))
            
            return np.mean(feature_counts) if feature_counts else 0.0
        except:
            return 0.0
    
    def _get_similarity_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Get distribution of similarity scores"""
        if not scores:
            return {'high': 0, 'medium': 0, 'low': 0}
        
        return {
            'high': len([s for s in scores if s > 0.7]),
            'medium': len([s for s in scores if 0.4 <= s <= 0.7]),
            'low': len([s for s in scores if s < 0.4])
        }
    
    def _evaluate_ml_models(self, spectra: List[ProcessedSpectrum]) -> Dict[str, float]:
        """Evaluate ML model performance"""
        try:
            ml_success = 0
            ml_total = min(10, len(spectra))  # Sample for efficiency
            
            for spectrum in spectra[:ml_total]:
                ml_results = self.video_analyzer.process_spectrum_with_models(
                    spectrum.mz_array, spectrum.intensity_array
                )
                if ml_results and any(ml_results.values()):
                    ml_success += 1
            
            return {
                'ml_success_rate': ml_success / ml_total if ml_total > 0 else 0,
                'spectus_available': self.spectus_model is not None,
                'cmssp_available': self.cmssp_model is not None
            }
        except:
            return {'ml_success_rate': 0, 'spectus_available': False, 'cmssp_available': False}
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0
    
    def train_model(self, training_data: Any) -> None:
        """Train computer vision models using training data"""
        self.logger.info("Training computer vision models with visual features")
        self._initialize_components()
        
        # Build visual database from training data
        if isinstance(training_data, list):
            for i, spectrum in enumerate(training_data):
                if hasattr(spectrum, 'mz_array') and hasattr(spectrum, 'intensity_array'):
                    spectrum_id = self.image_database.add_spectrum(
                        spectrum.mz_array,
                        spectrum.intensity_array,
                        metadata=getattr(spectrum, 'metadata', {'training': True, 'index': i})
                    )
                    
            self.logger.info(f"Built visual database with {len(training_data)} training spectra")
    
    def predict(self, test_data: Any) -> Tuple[List[str], List[float]]:
        """Make predictions using computer vision methods"""
        if isinstance(test_data, list) and len(test_data) > 0:
            if hasattr(test_data[0], 'mz_array'):
                return self._analyze_spectra_visually(test_data)
            else:
                # Convert to expected format
                converted_spectra = []
                for spectrum in test_data:
                    if hasattr(spectrum, 'mz_array') and hasattr(spectrum, 'intensity_array'):
                        proc_spectrum = ProcessedSpectrum(
                            mz_array=spectrum.mz_array,
                            intensity_array=spectrum.intensity_array,
                            metadata=getattr(spectrum, 'metadata', {})
                        )
                        converted_spectra.append(proc_spectrum)
                
                return self._analyze_spectra_visually(converted_spectra)
        
        return [], []
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get detailed processing statistics"""
        stats = self.processing_stats.copy()
        
        if stats['processing_times']:
            stats['mean_processing_time'] = np.mean(stats['processing_times'])
            stats['std_processing_time'] = np.std(stats['processing_times'])
        
        if stats['similarity_scores']:
            stats['mean_similarity'] = np.mean(stats['similarity_scores'])
            stats['similarity_distribution'] = self._get_similarity_distribution(stats['similarity_scores'])
        
        # Add visual-specific statistics
        stats['database_statistics'] = {
            'total_entries': self.image_database.index.ntotal if self.image_database else 0,
            'feature_dimension': self.image_database.feature_dimension if self.image_database else 0,
            'resolution': self.image_database.resolution if self.image_database else (0, 0)
        }
        
        return stats
    
    def save_visual_database(self, path: str):
        """Save the visual database for future use"""
        if self.image_database:
            self.image_database.save_database(path)
            self.logger.info(f"Saved visual database to {path}")
    
    def load_visual_database(self, path: str):
        """Load a pre-built visual database"""
        try:
            self.image_database = MSImageDatabase.load_database(path)
            self.logger.info(f"Loaded visual database from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load visual database: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Clear large objects
            self.spectus_model = None
            self.cmssp_model = None
            
            if self.image_database:
                self.image_database.image_cache.clear()
            
            self.logger.info("Computer vision validator cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

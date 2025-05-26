"""
Image Quality Assessment for Visual Pipeline

Evaluates the quality of spectrum-to-image conversions using various
image quality metrics including SSIM, PSNR, and perceptual quality measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage import filters, feature, measure
import cv2


@dataclass
class ImageQualityResult:
    """Container for image quality assessment results"""
    metric_name: str
    value: float
    reference_value: Optional[float]
    quality_score: float  # Normalized 0-1 score
    interpretation: str
    metadata: Dict = None


class ImageQualityAssessor:
    """
    Comprehensive image quality assessment for spectrum-to-image conversions
    """
    
    def __init__(self):
        """Initialize image quality assessor"""
        self.results = []
        
    def assess_structural_similarity(
        self,
        original_image: np.ndarray,
        processed_image: np.ndarray,
        win_size: Optional[int] = None
    ) -> ImageQualityResult:
        """
        Assess structural similarity between original and processed images
        
        Args:
            original_image: Original spectrum image
            processed_image: Processed spectrum image
            win_size: Window size for SSIM calculation
            
        Returns:
            ImageQualityResult object
        """
        # Ensure images are the same size
        if original_image.shape != processed_image.shape:
            processed_image = cv2.resize(processed_image, 
                                       (original_image.shape[1], original_image.shape[0]))
        
        # Convert to grayscale if needed
        if len(original_image.shape) == 3:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        if len(processed_image.shape) == 3:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
        
        # Calculate SSIM
        ssim_value = ssim(original_image, processed_image, win_size=win_size)
        
        # Interpret quality
        if ssim_value > 0.9:
            interpretation = "Excellent structural similarity"
        elif ssim_value > 0.7:
            interpretation = "Good structural similarity"
        elif ssim_value > 0.5:
            interpretation = "Moderate structural similarity"
        else:
            interpretation = "Poor structural similarity"
        
        result = ImageQualityResult(
            metric_name="Structural Similarity Index (SSIM)",
            value=ssim_value,
            reference_value=1.0,
            quality_score=ssim_value,
            interpretation=interpretation,
            metadata={'win_size': win_size}
        )
        
        self.results.append(result)
        return result
    
    def assess_peak_signal_noise_ratio(
        self,
        original_image: np.ndarray,
        processed_image: np.ndarray
    ) -> ImageQualityResult:
        """
        Assess peak signal-to-noise ratio between images
        
        Args:
            original_image: Original spectrum image
            processed_image: Processed spectrum image
            
        Returns:
            ImageQualityResult object
        """
        # Ensure images are the same size
        if original_image.shape != processed_image.shape:
            processed_image = cv2.resize(processed_image, 
                                       (original_image.shape[1], original_image.shape[0]))
        
        # Calculate PSNR
        psnr_value = psnr(original_image, processed_image)
        
        # Normalize to 0-1 scale (typical PSNR range: 20-40 dB)
        quality_score = min(1.0, max(0.0, (psnr_value - 20) / 20))
        
        # Interpret quality
        if psnr_value > 35:
            interpretation = "Excellent signal quality"
        elif psnr_value > 30:
            interpretation = "Good signal quality"
        elif psnr_value > 25:
            interpretation = "Moderate signal quality"
        else:
            interpretation = "Poor signal quality"
        
        result = ImageQualityResult(
            metric_name="Peak Signal-to-Noise Ratio (PSNR)",
            value=psnr_value,
            reference_value=None,
            quality_score=quality_score,
            interpretation=interpretation
        )
        
        self.results.append(result)
        return result
    
    def assess_mean_squared_error(
        self,
        original_image: np.ndarray,
        processed_image: np.ndarray
    ) -> ImageQualityResult:
        """
        Assess mean squared error between images
        
        Args:
            original_image: Original spectrum image
            processed_image: Processed spectrum image
            
        Returns:
            ImageQualityResult object
        """
        # Ensure images are the same size
        if original_image.shape != processed_image.shape:
            processed_image = cv2.resize(processed_image, 
                                       (original_image.shape[1], original_image.shape[0]))
        
        # Calculate MSE
        mse_value = mse(original_image, processed_image)
        
        # Normalize to 0-1 scale (lower MSE is better)
        # Assume typical MSE range: 0-1000
        quality_score = max(0.0, 1.0 - min(1.0, mse_value / 1000))
        
        # Interpret quality
        if mse_value < 100:
            interpretation = "Excellent pixel accuracy"
        elif mse_value < 300:
            interpretation = "Good pixel accuracy"
        elif mse_value < 500:
            interpretation = "Moderate pixel accuracy"
        else:
            interpretation = "Poor pixel accuracy"
        
        result = ImageQualityResult(
            metric_name="Mean Squared Error (MSE)",
            value=mse_value,
            reference_value=0.0,
            quality_score=quality_score,
            interpretation=interpretation
        )
        
        self.results.append(result)
        return result
    
    def assess_edge_preservation(
        self,
        original_image: np.ndarray,
        processed_image: np.ndarray
    ) -> ImageQualityResult:
        """
        Assess how well edges are preserved in the processed image
        
        Args:
            original_image: Original spectrum image
            processed_image: Processed spectrum image
            
        Returns:
            ImageQualityResult object
        """
        # Ensure images are the same size
        if original_image.shape != processed_image.shape:
            processed_image = cv2.resize(processed_image, 
                                       (original_image.shape[1], original_image.shape[0]))
        
        # Convert to grayscale if needed
        if len(original_image.shape) == 3:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        if len(processed_image.shape) == 3:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
        
        # Detect edges using Canny edge detector
        original_edges = feature.canny(original_image)
        processed_edges = feature.canny(processed_image)
        
        # Calculate edge preservation score
        edge_intersection = np.logical_and(original_edges, processed_edges)
        edge_union = np.logical_or(original_edges, processed_edges)
        
        if np.sum(edge_union) > 0:
            edge_preservation = np.sum(edge_intersection) / np.sum(edge_union)
        else:
            edge_preservation = 1.0  # No edges in either image
        
        # Interpret quality
        if edge_preservation > 0.8:
            interpretation = "Excellent edge preservation"
        elif edge_preservation > 0.6:
            interpretation = "Good edge preservation"
        elif edge_preservation > 0.4:
            interpretation = "Moderate edge preservation"
        else:
            interpretation = "Poor edge preservation"
        
        result = ImageQualityResult(
            metric_name="Edge Preservation",
            value=edge_preservation,
            reference_value=1.0,
            quality_score=edge_preservation,
            interpretation=interpretation
        )
        
        self.results.append(result)
        return result
    
    def assess_contrast_preservation(
        self,
        original_image: np.ndarray,
        processed_image: np.ndarray
    ) -> ImageQualityResult:
        """
        Assess how well contrast is preserved in the processed image
        
        Args:
            original_image: Original spectrum image
            processed_image: Processed spectrum image
            
        Returns:
            ImageQualityResult object
        """
        # Ensure images are the same size
        if original_image.shape != processed_image.shape:
            processed_image = cv2.resize(processed_image, 
                                       (original_image.shape[1], original_image.shape[0]))
        
        # Convert to grayscale if needed
        if len(original_image.shape) == 3:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        if len(processed_image.shape) == 3:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
        
        # Calculate contrast using standard deviation
        original_contrast = np.std(original_image)
        processed_contrast = np.std(processed_image)
        
        # Calculate contrast preservation ratio
        if original_contrast > 0:
            contrast_ratio = processed_contrast / original_contrast
        else:
            contrast_ratio = 1.0
        
        # Quality score (closer to 1.0 is better)
        quality_score = 1.0 - abs(1.0 - contrast_ratio)
        
        # Interpret quality
        if abs(1.0 - contrast_ratio) < 0.1:
            interpretation = "Excellent contrast preservation"
        elif abs(1.0 - contrast_ratio) < 0.2:
            interpretation = "Good contrast preservation"
        elif abs(1.0 - contrast_ratio) < 0.3:
            interpretation = "Moderate contrast preservation"
        else:
            interpretation = "Poor contrast preservation"
        
        result = ImageQualityResult(
            metric_name="Contrast Preservation",
            value=contrast_ratio,
            reference_value=1.0,
            quality_score=quality_score,
            interpretation=interpretation,
            metadata={
                'original_contrast': original_contrast,
                'processed_contrast': processed_contrast
            }
        )
        
        self.results.append(result)
        return result
    
    def assess_information_content(
        self,
        image: np.ndarray
    ) -> ImageQualityResult:
        """
        Assess information content of an image using entropy
        
        Args:
            image: Input image
            
        Returns:
            ImageQualityResult object
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate histogram
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
        
        # Calculate entropy
        hist = hist / hist.sum()  # Normalize
        entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Add small value to avoid log(0)
        
        # Normalize entropy (max entropy for 8-bit image is 8)
        quality_score = entropy / 8.0
        
        # Interpret quality
        if entropy > 7:
            interpretation = "Very high information content"
        elif entropy > 6:
            interpretation = "High information content"
        elif entropy > 4:
            interpretation = "Moderate information content"
        else:
            interpretation = "Low information content"
        
        result = ImageQualityResult(
            metric_name="Information Content (Entropy)",
            value=entropy,
            reference_value=8.0,
            quality_score=quality_score,
            interpretation=interpretation
        )
        
        self.results.append(result)
        return result
    
    def assess_spectral_fidelity(
        self,
        original_spectrum: np.ndarray,
        reconstructed_spectrum: np.ndarray
    ) -> ImageQualityResult:
        """
        Assess how well the image conversion preserves spectral information
        
        Args:
            original_spectrum: Original 1D spectrum
            reconstructed_spectrum: Spectrum reconstructed from image
            
        Returns:
            ImageQualityResult object
        """
        # Ensure spectra are the same length
        min_length = min(len(original_spectrum), len(reconstructed_spectrum))
        original_spectrum = original_spectrum[:min_length]
        reconstructed_spectrum = reconstructed_spectrum[:min_length]
        
        # Calculate correlation
        correlation = np.corrcoef(original_spectrum, reconstructed_spectrum)[0, 1]
        
        # Handle NaN case
        if np.isnan(correlation):
            correlation = 0.0
        
        quality_score = max(0.0, correlation)
        
        # Interpret quality
        if correlation > 0.95:
            interpretation = "Excellent spectral fidelity"
        elif correlation > 0.9:
            interpretation = "Good spectral fidelity"
        elif correlation > 0.8:
            interpretation = "Moderate spectral fidelity"
        else:
            interpretation = "Poor spectral fidelity"
        
        result = ImageQualityResult(
            metric_name="Spectral Fidelity",
            value=correlation,
            reference_value=1.0,
            quality_score=quality_score,
            interpretation=interpretation
        )
        
        self.results.append(result)
        return result
    
    def comprehensive_assessment(
        self,
        original_image: np.ndarray,
        processed_image: np.ndarray,
        original_spectrum: Optional[np.ndarray] = None,
        reconstructed_spectrum: Optional[np.ndarray] = None
    ) -> List[ImageQualityResult]:
        """
        Perform comprehensive image quality assessment
        
        Args:
            original_image: Original spectrum image
            processed_image: Processed spectrum image
            original_spectrum: Optional original 1D spectrum
            reconstructed_spectrum: Optional reconstructed 1D spectrum
            
        Returns:
            List of ImageQualityResult objects
        """
        results = []
        
        # Structural similarity
        results.append(self.assess_structural_similarity(original_image, processed_image))
        
        # Peak signal-to-noise ratio
        results.append(self.assess_peak_signal_noise_ratio(original_image, processed_image))
        
        # Mean squared error
        results.append(self.assess_mean_squared_error(original_image, processed_image))
        
        # Edge preservation
        results.append(self.assess_edge_preservation(original_image, processed_image))
        
        # Contrast preservation
        results.append(self.assess_contrast_preservation(original_image, processed_image))
        
        # Information content
        results.append(self.assess_information_content(processed_image))
        
        # Spectral fidelity (if spectra provided)
        if original_spectrum is not None and reconstructed_spectrum is not None:
            results.append(self.assess_spectral_fidelity(original_spectrum, reconstructed_spectrum))
        
        return results
    
    def generate_quality_report(self) -> pd.DataFrame:
        """Generate comprehensive quality assessment report"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Metric': result.metric_name,
                'Value': result.value,
                'Reference Value': result.reference_value,
                'Quality Score': result.quality_score,
                'Interpretation': result.interpretation
            })
        
        return pd.DataFrame(data)
    
    def plot_quality_assessment(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create visualization of quality assessment results"""
        if not self.results:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Image Quality Assessment Results', fontsize=16, fontweight='bold')
        
        # Extract data
        metrics = [result.metric_name for result in self.results]
        values = [result.value for result in self.results]
        quality_scores = [result.quality_score for result in self.results]
        
        # Plot 1: Quality scores
        colors = ['green' if score > 0.8 else 'orange' if score > 0.6 else 'red' 
                 for score in quality_scores]
        
        axes[0, 0].bar(range(len(metrics)), quality_scores, color=colors, alpha=0.7)
        axes[0, 0].set_xticks(range(len(metrics)))
        axes[0, 0].set_xticklabels([m.split('(')[0].strip() for m in metrics], rotation=45, ha='right')
        axes[0, 0].set_ylabel('Quality Score')
        axes[0, 0].set_title('Quality Scores by Metric')
        axes[0, 0].set_ylim(0, 1)
        
        # Plot 2: Quality distribution
        quality_bins = ['Poor (0-0.4)', 'Moderate (0.4-0.6)', 'Good (0.6-0.8)', 'Excellent (0.8-1.0)']
        quality_counts = [
            sum(1 for score in quality_scores if score <= 0.4),
            sum(1 for score in quality_scores if 0.4 < score <= 0.6),
            sum(1 for score in quality_scores if 0.6 < score <= 0.8),
            sum(1 for score in quality_scores if score > 0.8)
        ]
        
        colors_pie = ['red', 'orange', 'yellow', 'green']
        axes[0, 1].pie(quality_counts, labels=quality_bins, colors=colors_pie, 
                      autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Quality Distribution')
        
        # Plot 3: Metric values
        axes[1, 0].scatter(range(len(values)), values, c=colors, s=100, alpha=0.7)
        axes[1, 0].set_xticks(range(len(metrics)))
        axes[1, 0].set_xticklabels([m.split('(')[0].strip() for m in metrics], rotation=45, ha='right')
        axes[1, 0].set_ylabel('Metric Value')
        axes[1, 0].set_title('Raw Metric Values')
        
        # Plot 4: Quality vs Value correlation
        axes[1, 1].scatter(values, quality_scores, c=colors, s=100, alpha=0.7)
        axes[1, 1].set_xlabel('Raw Metric Value')
        axes[1, 1].set_ylabel('Quality Score')
        axes[1, 1].set_title('Quality Score vs Raw Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_overall_quality_score(self) -> float:
        """Calculate overall quality score as average of all metrics"""
        if not self.results:
            return 0.0
        
        return np.mean([result.quality_score for result in self.results])
    
    def clear_results(self) -> None:
        """Clear all stored results"""
        self.results = [] 
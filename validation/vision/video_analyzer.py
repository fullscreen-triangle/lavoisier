"""
Video Analysis Module

Comprehensive temporal analysis of spectrum sequences including motion analysis,
temporal consistency evaluation, frame-to-frame stability, and dynamic pattern
recognition for the visual pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from scipy import signal
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
import warnings


@dataclass
class VideoAnalysisResult:
    """Container for video analysis results"""
    metric_name: str
    score: float
    temporal_data: Optional[np.ndarray] = None
    interpretation: str = ""
    visualization_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None


class VideoAnalyzer:
    """
    Comprehensive video analysis for temporal spectrum sequences
    """
    
    def __init__(self):
        """Initialize video analyzer"""
        self.results = []
        
    def analyze_temporal_consistency(
        self,
        spectrum_sequence: np.ndarray,
        time_points: Optional[np.ndarray] = None
    ) -> VideoAnalysisResult:
        """
        Analyze temporal consistency of spectrum sequences
        
        Args:
            spectrum_sequence: Sequence of spectra (T, H, W) or (T, N_features)
            time_points: Optional time points for each spectrum
            
        Returns:
            VideoAnalysisResult object
        """
        if time_points is None:
            time_points = np.arange(len(spectrum_sequence))
        
        # Calculate frame-to-frame differences
        frame_differences = []
        for i in range(1, len(spectrum_sequence)):
            diff = np.mean(np.abs(spectrum_sequence[i] - spectrum_sequence[i-1]))
            frame_differences.append(diff)
        
        frame_differences = np.array(frame_differences)
        
        # Calculate consistency metrics
        mean_difference = np.mean(frame_differences)
        std_difference = np.std(frame_differences)
        consistency_score = 1.0 / (1.0 + std_difference / (mean_difference + 1e-8))
        
        # Detect sudden changes (outliers)
        threshold = mean_difference + 2 * std_difference
        sudden_changes = np.sum(frame_differences > threshold)
        change_percentage = sudden_changes / len(frame_differences) * 100
        
        # Interpretation
        if consistency_score > 0.8:
            interpretation = f"High temporal consistency (score: {consistency_score:.3f})"
        elif consistency_score > 0.6:
            interpretation = f"Moderate temporal consistency (score: {consistency_score:.3f})"
        else:
            interpretation = f"Low temporal consistency (score: {consistency_score:.3f})"
        
        interpretation += f", {sudden_changes} sudden changes ({change_percentage:.1f}%)"
        
        result = VideoAnalysisResult(
            metric_name="Temporal Consistency",
            score=consistency_score,
            temporal_data=frame_differences,
            interpretation=interpretation,
            visualization_data={
                'frame_differences': frame_differences,
                'time_points': time_points[1:],  # Exclude first point
                'threshold': threshold,
                'sudden_changes': sudden_changes
            },
            metadata={
                'mean_difference': mean_difference,
                'std_difference': std_difference,
                'sequence_length': len(spectrum_sequence)
            }
        )
        
        self.results.append(result)
        return result
    
    def analyze_motion_patterns(
        self,
        spectrum_sequence: np.ndarray,
        method: str = 'optical_flow'
    ) -> VideoAnalysisResult:
        """
        Analyze motion patterns in spectrum sequences
        
        Args:
            spectrum_sequence: Sequence of 2D spectra (T, H, W)
            method: Motion analysis method ('optical_flow', 'phase_correlation')
            
        Returns:
            VideoAnalysisResult object
        """
        if len(spectrum_sequence.shape) != 3:
            # If not 2D spectra, create a simple motion proxy
            motion_magnitudes = []
            for i in range(1, len(spectrum_sequence)):
                motion = np.linalg.norm(spectrum_sequence[i] - spectrum_sequence[i-1])
                motion_magnitudes.append(motion)
            
            motion_score = 1.0 - np.std(motion_magnitudes) / (np.mean(motion_magnitudes) + 1e-8)
            interpretation = f"Motion consistency score: {motion_score:.3f}"
            
        else:
            motion_magnitudes = []
            motion_directions = []
            
            for i in range(1, len(spectrum_sequence)):
                prev_frame = spectrum_sequence[i-1].astype(np.float32)
                curr_frame = spectrum_sequence[i].astype(np.float32)
                
                if method == 'optical_flow':
                    # Lucas-Kanade optical flow
                    flow = cv2.calcOpticalFlowPyrLK(
                        prev_frame, curr_frame, None, None
                    )
                    if flow[0] is not None:
                        motion_mag = np.mean(np.linalg.norm(flow[0], axis=1))
                        motion_magnitudes.append(motion_mag)
                    else:
                        motion_magnitudes.append(0.0)
                        
                elif method == 'phase_correlation':
                    # Phase correlation for motion estimation
                    f_prev = np.fft.fft2(prev_frame)
                    f_curr = np.fft.fft2(curr_frame)
                    
                    cross_power = (f_prev * np.conj(f_curr)) / (np.abs(f_prev * np.conj(f_curr)) + 1e-8)
                    correlation = np.fft.ifft2(cross_power)
                    
                    # Find peak
                    peak_idx = np.unravel_index(np.argmax(np.abs(correlation)), correlation.shape)
                    motion_mag = np.linalg.norm(peak_idx)
                    motion_magnitudes.append(motion_mag)
            
            motion_magnitudes = np.array(motion_magnitudes)
            
            # Calculate motion consistency
            mean_motion = np.mean(motion_magnitudes)
            std_motion = np.std(motion_magnitudes)
            motion_score = 1.0 - std_motion / (mean_motion + 1e-8)
            
            interpretation = f"Motion consistency: {motion_score:.3f}, Avg motion: {mean_motion:.3f}"
        
        result = VideoAnalysisResult(
            metric_name=f"Motion Analysis ({method})",
            score=motion_score,
            temporal_data=np.array(motion_magnitudes),
            interpretation=interpretation,
            visualization_data={
                'motion_magnitudes': motion_magnitudes,
                'method': method
            },
            metadata={
                'mean_motion': np.mean(motion_magnitudes),
                'std_motion': np.std(motion_magnitudes),
                'method': method
            }
        )
        
        self.results.append(result)
        return result
    
    def analyze_frequency_domain_stability(
        self,
        spectrum_sequence: np.ndarray,
        sampling_rate: float = 1.0
    ) -> VideoAnalysisResult:
        """
        Analyze stability in frequency domain
        
        Args:
            spectrum_sequence: Sequence of spectra
            sampling_rate: Temporal sampling rate
            
        Returns:
            VideoAnalysisResult object
        """
        # Flatten each spectrum for frequency analysis
        if len(spectrum_sequence.shape) > 2:
            flattened_sequence = spectrum_sequence.reshape(len(spectrum_sequence), -1)
        else:
            flattened_sequence = spectrum_sequence
        
        # Analyze each feature/pixel over time
        frequency_stabilities = []
        dominant_frequencies = []
        
        for feature_idx in range(flattened_sequence.shape[1]):
            time_series = flattened_sequence[:, feature_idx]
            
            # Remove DC component
            time_series = time_series - np.mean(time_series)
            
            # Compute power spectral density
            frequencies, psd = signal.welch(time_series, fs=sampling_rate, nperseg=min(len(time_series)//2, 256))
            
            # Find dominant frequency
            dominant_freq_idx = np.argmax(psd[1:]) + 1  # Exclude DC
            dominant_freq = frequencies[dominant_freq_idx]
            dominant_frequencies.append(dominant_freq)
            
            # Calculate stability as inverse of frequency spread
            freq_spread = np.std(psd)
            stability = 1.0 / (1.0 + freq_spread)
            frequency_stabilities.append(stability)
        
        frequency_stabilities = np.array(frequency_stabilities)
        dominant_frequencies = np.array(dominant_frequencies)
        
        # Overall stability score
        overall_stability = np.mean(frequency_stabilities)
        
        # Frequency consistency
        freq_consistency = 1.0 - np.std(dominant_frequencies) / (np.mean(dominant_frequencies) + 1e-8)
        
        interpretation = f"Frequency stability: {overall_stability:.3f}, Frequency consistency: {freq_consistency:.3f}"
        
        result = VideoAnalysisResult(
            metric_name="Frequency Domain Stability",
            score=overall_stability,
            temporal_data=frequency_stabilities,
            interpretation=interpretation,
            visualization_data={
                'frequency_stabilities': frequency_stabilities,
                'dominant_frequencies': dominant_frequencies,
                'sampling_rate': sampling_rate
            },
            metadata={
                'overall_stability': overall_stability,
                'freq_consistency': freq_consistency,
                'feature_count': len(frequency_stabilities)
            }
        )
        
        self.results.append(result)
        return result
    
    def detect_temporal_anomalies(
        self,
        spectrum_sequence: np.ndarray,
        anomaly_threshold: float = 2.0
    ) -> VideoAnalysisResult:
        """
        Detect temporal anomalies in spectrum sequences
        
        Args:
            spectrum_sequence: Sequence of spectra
            anomaly_threshold: Threshold for anomaly detection (in standard deviations)
            
        Returns:
            VideoAnalysisResult object
        """
        # Calculate temporal features
        if len(spectrum_sequence.shape) > 2:
            # For 2D spectra, calculate summary statistics over space
            temporal_features = []
            for spectrum in spectrum_sequence:
                features = [
                    np.mean(spectrum),
                    np.std(spectrum),
                    np.max(spectrum),
                    np.min(spectrum),
                    np.median(spectrum)
                ]
                temporal_features.append(features)
            temporal_features = np.array(temporal_features)
        else:
            # For 1D features, use directly
            temporal_features = spectrum_sequence
        
        # Detect anomalies using statistical methods
        anomaly_scores = []
        anomaly_flags = []
        
        for i, features in enumerate(temporal_features):
            if i == 0:
                anomaly_scores.append(0.0)
                anomaly_flags.append(False)
                continue
            
            # Calculate z-score relative to previous frames
            window_size = min(i, 10)  # Use up to 10 previous frames
            window_data = temporal_features[max(0, i-window_size):i]
            
            if len(window_data) > 1:
                window_mean = np.mean(window_data, axis=0)
                window_std = np.std(window_data, axis=0)
                
                # Calculate anomaly score
                z_scores = np.abs((features - window_mean) / (window_std + 1e-8))
                anomaly_score = np.max(z_scores)
                anomaly_scores.append(anomaly_score)
                
                # Flag as anomaly if above threshold
                is_anomaly = anomaly_score > anomaly_threshold
                anomaly_flags.append(is_anomaly)
            else:
                anomaly_scores.append(0.0)
                anomaly_flags.append(False)
        
        anomaly_scores = np.array(anomaly_scores)
        anomaly_flags = np.array(anomaly_flags)
        
        # Calculate metrics
        anomaly_count = np.sum(anomaly_flags)
        anomaly_percentage = anomaly_count / len(spectrum_sequence) * 100
        
        # Anomaly score (inverse of anomaly rate)
        anomaly_score = 1.0 - anomaly_percentage / 100.0
        
        interpretation = f"Detected {anomaly_count} anomalies ({anomaly_percentage:.1f}% of sequence)"
        
        result = VideoAnalysisResult(
            metric_name="Temporal Anomaly Detection",
            score=anomaly_score,
            temporal_data=anomaly_scores,
            interpretation=interpretation,
            visualization_data={
                'anomaly_scores': anomaly_scores,
                'anomaly_flags': anomaly_flags,
                'threshold': anomaly_threshold,
                'anomaly_indices': np.where(anomaly_flags)[0]
            },
            metadata={
                'anomaly_count': anomaly_count,
                'anomaly_percentage': anomaly_percentage,
                'threshold': anomaly_threshold
            }
        )
        
        self.results.append(result)
        return result
    
    def analyze_periodic_patterns(
        self,
        spectrum_sequence: np.ndarray,
        sampling_rate: float = 1.0
    ) -> VideoAnalysisResult:
        """
        Analyze periodic patterns in spectrum sequences
        
        Args:
            spectrum_sequence: Sequence of spectra
            sampling_rate: Temporal sampling rate
            
        Returns:
            VideoAnalysisResult object
        """
        # Calculate global intensity over time
        if len(spectrum_sequence.shape) > 2:
            global_intensity = np.array([np.mean(spectrum) for spectrum in spectrum_sequence])
        else:
            global_intensity = np.mean(spectrum_sequence, axis=1)
        
        # Remove trend
        detrended = signal.detrend(global_intensity)
        
        # Autocorrelation analysis
        autocorr = np.correlate(detrended, detrended, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find peaks in autocorrelation (potential periods)
        peaks, _ = signal.find_peaks(autocorr[1:], height=0.1, distance=2)
        peaks += 1  # Adjust for offset
        
        # Calculate periodicity score
        if len(peaks) > 0:
            # Strength of strongest periodic component
            periodicity_strength = np.max(autocorr[peaks])
            dominant_period = peaks[np.argmax(autocorr[peaks])]
        else:
            periodicity_strength = 0.0
            dominant_period = 0
        
        # Frequency domain analysis
        frequencies, psd = signal.welch(detrended, fs=sampling_rate)
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(psd[1:]) + 1  # Exclude DC
        dominant_frequency = frequencies[dominant_freq_idx]
        
        interpretation = f"Periodicity strength: {periodicity_strength:.3f}"
        if dominant_period > 0:
            interpretation += f", Dominant period: {dominant_period} frames"
        if dominant_frequency > 0:
            interpretation += f", Dominant frequency: {dominant_frequency:.3f} Hz"
        
        result = VideoAnalysisResult(
            metric_name="Periodic Pattern Analysis",
            score=periodicity_strength,
            temporal_data=autocorr,
            interpretation=interpretation,
            visualization_data={
                'global_intensity': global_intensity,
                'detrended': detrended,
                'autocorr': autocorr,
                'peaks': peaks,
                'frequencies': frequencies,
                'psd': psd,
                'dominant_period': dominant_period,
                'dominant_frequency': dominant_frequency
            },
            metadata={
                'periodicity_strength': periodicity_strength,
                'dominant_period': dominant_period,
                'dominant_frequency': dominant_frequency,
                'peak_count': len(peaks)
            }
        )
        
        self.results.append(result)
        return result
    
    def compare_temporal_pipelines(
        self,
        numerical_sequence: np.ndarray,
        visual_sequence: np.ndarray,
        time_points: Optional[np.ndarray] = None
    ) -> List[VideoAnalysisResult]:
        """
        Compare temporal behavior between numerical and visual pipelines
        
        Args:
            numerical_sequence: Numerical pipeline spectrum sequence
            visual_sequence: Visual pipeline spectrum sequence
            time_points: Optional time points
            
        Returns:
            List of VideoAnalysisResult objects
        """
        results = []
        
        # Ensure sequences have same length
        min_length = min(len(numerical_sequence), len(visual_sequence))
        numerical_sequence = numerical_sequence[:min_length]
        visual_sequence = visual_sequence[:min_length]
        
        if time_points is not None:
            time_points = time_points[:min_length]
        
        # 1. Temporal correlation analysis
        if len(numerical_sequence.shape) > 2:
            num_global = np.array([np.mean(s) for s in numerical_sequence])
            vis_global = np.array([np.mean(s) for s in visual_sequence])
        else:
            num_global = np.mean(numerical_sequence, axis=1)
            vis_global = np.mean(visual_sequence, axis=1)
        
        correlation, p_value = pearsonr(num_global, vis_global)
        
        correlation_result = VideoAnalysisResult(
            metric_name="Pipeline Temporal Correlation",
            score=abs(correlation),
            temporal_data=np.column_stack([num_global, vis_global]),
            interpretation=f"Temporal correlation: {correlation:.3f} (p={p_value:.3f})",
            visualization_data={
                'numerical_global': num_global,
                'visual_global': vis_global,
                'correlation': correlation,
                'p_value': p_value
            },
            metadata={
                'correlation': correlation,
                'p_value': p_value,
                'sequence_length': min_length
            }
        )
        results.append(correlation_result)
        
        # 2. Temporal consistency comparison
        num_consistency = self.analyze_temporal_consistency(numerical_sequence, time_points)
        vis_consistency = self.analyze_temporal_consistency(visual_sequence, time_points)
        
        consistency_diff = abs(num_consistency.score - vis_consistency.score)
        consistency_comparison = VideoAnalysisResult(
            metric_name="Pipeline Consistency Comparison",
            score=1.0 - consistency_diff,  # Higher score = more similar consistency
            temporal_data=np.column_stack([num_consistency.temporal_data, vis_consistency.temporal_data]),
            interpretation=f"Consistency difference: {consistency_diff:.3f} (Numerical: {num_consistency.score:.3f}, Visual: {vis_consistency.score:.3f})",
            visualization_data={
                'numerical_consistency': num_consistency.score,
                'visual_consistency': vis_consistency.score,
                'numerical_differences': num_consistency.temporal_data,
                'visual_differences': vis_consistency.temporal_data
            },
            metadata={
                'numerical_consistency': num_consistency.score,
                'visual_consistency': vis_consistency.score,
                'consistency_difference': consistency_diff
            }
        )
        results.append(consistency_comparison)
        
        # 3. Synchronization analysis
        # Cross-correlation to find temporal offset
        cross_corr = np.correlate(num_global - np.mean(num_global), 
                                 vis_global - np.mean(vis_global), mode='full')
        cross_corr = cross_corr / (np.std(num_global) * np.std(vis_global) * len(num_global))
        
        # Find peak correlation and offset
        peak_idx = np.argmax(np.abs(cross_corr))
        offset = peak_idx - len(num_global) + 1
        sync_score = np.abs(cross_corr[peak_idx])
        
        sync_result = VideoAnalysisResult(
            metric_name="Pipeline Synchronization",
            score=sync_score,
            temporal_data=cross_corr,
            interpretation=f"Synchronization score: {sync_score:.3f}, Temporal offset: {offset} frames",
            visualization_data={
                'cross_correlation': cross_corr,
                'peak_correlation': sync_score,
                'temporal_offset': offset
            },
            metadata={
                'sync_score': sync_score,
                'temporal_offset': offset,
                'peak_index': peak_idx
            }
        )
        results.append(sync_result)
        
        self.results.extend(results)
        return results
    
    def plot_video_analysis_results(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive video analysis visualization"""
        if not self.results:
            return None
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Video Analysis Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Overall scores
        metrics = [r.metric_name for r in self.results]
        scores = [r.score for r in self.results]
        
        colors = ['green' if score > 0.8 else 'orange' if score > 0.6 else 'red' for score in scores]
        axes[0, 0].bar(range(len(metrics)), scores, color=colors, alpha=0.7)
        axes[0, 0].set_xticks(range(len(metrics)))
        axes[0, 0].set_xticklabels([m.split()[0] for m in metrics], rotation=45, ha='right')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Video Analysis Scores')
        axes[0, 0].set_ylim(0, 1)
        
        # Plot 2: Temporal consistency (if available)
        consistency_result = next((r for r in self.results if 'Consistency' in r.metric_name and 'Comparison' not in r.metric_name), None)
        if consistency_result and consistency_result.temporal_data is not None:
            axes[0, 1].plot(consistency_result.temporal_data, 'b-', alpha=0.7)
            if consistency_result.visualization_data and 'threshold' in consistency_result.visualization_data:
                threshold = consistency_result.visualization_data['threshold']
                axes[0, 1].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
                axes[0, 1].legend()
            axes[0, 1].set_xlabel('Frame')
            axes[0, 1].set_ylabel('Frame Difference')
            axes[0, 1].set_title('Temporal Consistency')
        
        # Plot 3: Motion analysis (if available)
        motion_result = next((r for r in self.results if 'Motion' in r.metric_name), None)
        if motion_result and motion_result.temporal_data is not None:
            axes[1, 0].plot(motion_result.temporal_data, 'g-', alpha=0.7)
            axes[1, 0].set_xlabel('Frame')
            axes[1, 0].set_ylabel('Motion Magnitude')
            axes[1, 0].set_title('Motion Analysis')
        
        # Plot 4: Frequency stability (if available)
        freq_result = next((r for r in self.results if 'Frequency' in r.metric_name), None)
        if freq_result and freq_result.visualization_data:
            stabilities = freq_result.visualization_data.get('frequency_stabilities', [])
            if len(stabilities) > 0:
                axes[1, 1].hist(stabilities, bins=20, alpha=0.7, edgecolor='black')
                axes[1, 1].set_xlabel('Stability Score')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Frequency Stability Distribution')
        
        # Plot 5: Anomaly detection (if available)
        anomaly_result = next((r for r in self.results if 'Anomaly' in r.metric_name), None)
        if anomaly_result and anomaly_result.temporal_data is not None:
            axes[2, 0].plot(anomaly_result.temporal_data, 'r-', alpha=0.7)
            if anomaly_result.visualization_data:
                threshold = anomaly_result.visualization_data.get('threshold', 2.0)
                axes[2, 0].axhline(y=threshold, color='k', linestyle='--', label='Threshold')
                
                anomaly_indices = anomaly_result.visualization_data.get('anomaly_indices', [])
                if len(anomaly_indices) > 0:
                    axes[2, 0].scatter(anomaly_indices, anomaly_result.temporal_data[anomaly_indices], 
                                     color='red', s=50, label='Anomalies')
                axes[2, 0].legend()
            axes[2, 0].set_xlabel('Frame')
            axes[2, 0].set_ylabel('Anomaly Score')
            axes[2, 0].set_title('Temporal Anomaly Detection')
        
        # Plot 6: Periodic patterns (if available)
        periodic_result = next((r for r in self.results if 'Periodic' in r.metric_name), None)
        if periodic_result and periodic_result.visualization_data:
            autocorr = periodic_result.visualization_data.get('autocorr', [])
            if len(autocorr) > 0:
                lags = np.arange(len(autocorr))
                axes[2, 1].plot(lags, autocorr, 'purple', alpha=0.7)
                
                peaks = periodic_result.visualization_data.get('peaks', [])
                if len(peaks) > 0:
                    axes[2, 1].scatter(peaks, autocorr[peaks], color='red', s=50, label='Peaks')
                    axes[2, 1].legend()
                
                axes[2, 1].set_xlabel('Lag (frames)')
                axes[2, 1].set_ylabel('Autocorrelation')
                axes[2, 1].set_title('Periodic Pattern Analysis')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_video_analysis_report(self) -> pd.DataFrame:
        """Generate comprehensive video analysis report"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Metric': result.metric_name,
                'Score': result.score,
                'Interpretation': result.interpretation
            })
        
        return pd.DataFrame(data)
    
    def get_overall_video_score(self) -> float:
        """Calculate overall video analysis score"""
        if not self.results:
            return 0.0
        
        return np.mean([r.score for r in self.results])
    
    def clear_results(self) -> None:
        """Clear all stored results"""
        self.results = [] 
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path
import h5py
from dotenv import load_dotenv
import os

from lavoisier.models import (
    SpecTUSModel,
    CMSSPModel,
    create_spectus_model,
    create_cmssp_model
)

# Load environment variables
load_dotenv()

class MSVideoAnalyzer:
    def __init__(self, resolution: Tuple[int, int] = (1024, 1024),
                 rt_window: int = 30,
                 mz_range: Tuple[float, float] = (100, 1000)):
        """
        Initialize MS Video Analyzer

        Parameters:
        -----------
        resolution : Tuple[int, int]
            Resolution of each frame (mz_bins, intensity_bins)
        rt_window : int
            Number of frames to keep in memory (sliding window)
        mz_range : Tuple[float, float]
            Mass range to consider
        """
        self.resolution = resolution
        self.rt_window = rt_window
        self.mz_range = mz_range
        self.current_frame = np.zeros(resolution)
        self.frame_buffer = np.zeros((rt_window, *resolution))
        self.video_writer = None
        
        # Initialize ML models if enabled
        self.device = "cuda" if torch.cuda.is_available() and os.getenv('ENABLE_GPU', 'true').lower() == 'true' else "cpu"
        self.models = {}
        
        if os.getenv('ENABLE_VISUAL_MODELS', 'true').lower() == 'true':
            if os.getenv('ENABLE_SPECTUS', 'true').lower() == 'true':
                self.models['spectus'] = create_spectus_model(device=self.device)
            
            if os.getenv('ENABLE_CMSSP', 'true').lower() == 'true':
                self.models['cmssp'] = create_cmssp_model(device=self.device)

    def _initialize_video_writer(self, output_path: str):
        """Initialize OpenCV video writer"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            30.0,  # fps
            self.resolution,
            isColor=False
        )

    def process_spectrum_with_models(self, mzs: np.ndarray, intensities: np.ndarray) -> Dict[str, Any]:
        """Process spectrum with ML models for enhanced feature detection"""
        results = {}
        
        try:
            # Get embeddings from CMSSP model
            if 'cmssp' in self.models:
                embedding = self.models['cmssp'].encode_spectrum(mzs, intensities)
                results['embedding'] = embedding
                
                # Use embedding for enhanced feature detection
                feature_vector = embedding.reshape(-1, 1)  # Reshape for OpenCV
                feature_vector = (feature_vector * 255).astype(np.uint8)
                
                # Create feature mask
                feature_mask = cv2.adaptiveThreshold(
                    feature_vector,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11,
                    2
                )
                results['feature_mask'] = feature_mask
            
            # Get structure prediction from SpecTUS
            if 'spectus' in self.models:
                smiles = self.models['spectus'].process_spectrum(mzs, intensities)
                results['predicted_structure'] = smiles
                
        except Exception as e:
            print(f"Error in model processing: {str(e)}")
            
        return results

    def process_spectrum(self, mzs: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        """
        Convert a single spectrum to a 2D image frame

        Parameters:
        -----------
        mzs : np.ndarray
            Mass-to-charge ratios
        intensities : np.ndarray
            Intensity values

        Returns:
        --------
        np.ndarray
            2D frame representing the spectrum
        """
        frame = np.zeros(self.resolution)

        # Map m/z values to pixel coordinates
        mz_pixels = np.interp(
            mzs,
            [self.mz_range[0], self.mz_range[1]],
            [0, self.resolution[0] - 1]
        ).astype(int)

        # Map intensities to pixel values (log scale)
        intensity_pixels = np.interp(
            np.log1p(intensities),
            [0, np.log1p(intensities.max())],
            [0, 255]
        ).astype(np.uint8)

        # Update frame
        frame[mz_pixels, :] = intensity_pixels[:, np.newaxis]

        # Apply Gaussian blur to simulate liquid ripples
        frame = gaussian_filter(frame, sigma=1)
        
        # Enhance frame with ML model features if available
        if self.models:
            model_results = self.process_spectrum_with_models(mzs, intensities)
            
            if 'feature_mask' in model_results:
                # Apply feature mask to enhance important regions
                feature_mask = cv2.resize(model_results['feature_mask'], self.resolution)
                frame = cv2.addWeighted(frame, 0.7, feature_mask, 0.3, 0)

        return frame

    def update_frame_buffer(self, new_frame: np.ndarray):
        """Update sliding window of frames"""
        self.frame_buffer = np.roll(self.frame_buffer, -1, axis=0)
        self.frame_buffer[-1] = new_frame

    def detect_features(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect interesting features in the current frame

        Returns list of (x, y) coordinates of detected features
        """
        # Convert frame to uint8
        frame_uint8 = (frame * 255).astype(np.uint8)

        # Use SIFT detector
        sift = cv2.SIFT_create()
        keypoints = sift.detect(frame_uint8, None)
        
        # If ML models are available, use them to filter/enhance keypoints
        if self.models and hasattr(self, 'current_model_results'):
            if 'embedding' in self.current_model_results:
                embedding = self.current_model_results['embedding']
                # Use embedding to weight keypoint importance
                weighted_keypoints = []
                for kp in keypoints:
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    if x < len(embedding):
                        weight = embedding[x]
                        if weight > np.mean(embedding):
                            weighted_keypoints.append(kp)
                keypoints = weighted_keypoints

        return [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]

    def analyze_temporal_patterns(self) -> np.ndarray:
        """
        Analyze temporal patterns in the frame buffer

        Returns correlation matrix between frames
        """
        n_frames = self.frame_buffer.shape[0]
        correlation_matrix = np.zeros((n_frames, n_frames))

        for i in range(n_frames):
            for j in range(i, n_frames):
                correlation = np.corrcoef(
                    self.frame_buffer[i].ravel(),
                    self.frame_buffer[j].ravel()
                )[0, 1]
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation

        return correlation_matrix

    def extract_spectra_as_video(self, input_data: List[Tuple[np.ndarray, np.ndarray]], output_path: str):
        """
        Process input data and generate video representation
        """
        self._initialize_video_writer(output_path)
        
        # Store ML results for the entire sequence
        self.sequence_ml_results = []

        for mzs, intensities in input_data:
            # Process with ML models first
            if self.models:
                self.current_model_results = self.process_spectrum_with_models(mzs, intensities)
                self.sequence_ml_results.append(self.current_model_results)
            else:
                self.current_model_results = {}
            
            # Generate frame
            frame = self.process_spectrum(mzs, intensities)

            # Update buffer
            self.update_frame_buffer(frame)

            # Detect features
            features = self.detect_features(frame)

            # Draw features on frame
            frame_with_features = frame.copy()
            for x, y in features:
                cv2.circle(frame_with_features, (x, y), 3, (255, 0, 0), -1)
            
            # Add ML-based annotations if available
            if 'predicted_structure' in self.current_model_results:
                # Add structure prediction as text
                structure = self.current_model_results['predicted_structure']
                cv2.putText(
                    frame_with_features,
                    f"Structure: {structure}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

            # Write frame to video
            self.video_writer.write(frame_with_features.astype(np.uint8))

        self.video_writer.release()
        
        # Save ML results alongside video
        if self.sequence_ml_results:
            output_dir = str(Path(output_path).parent)
            results_path = os.path.join(output_dir, "ml_results.json")
            with open(results_path, 'w') as f:
                import json
                json.dump(self.sequence_ml_results, f, indent=2)

    def analyze_video(self, video_path: str):
        """
        Analyze generated MS video for patterns
        """
        cap = cv2.VideoCapture(video_path)

        prev_frame = None
        flow_history = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if prev_frame is not None:
                # Calculate optical flow using Farneback method
                flow = cv2.calcOpticalFlowFarneback(
                    cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                flow_history.append(flow)

                # Analyze flow patterns
                self._analyze_flow_patterns(flow)

            prev_frame = frame

        cap.release()
        return flow_history

    def _analyze_flow_patterns(self, flow: np.ndarray):
        """
        Analyze optical flow patterns to detect MS features
        """
        # Calculate flow magnitude and direction
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Detect significant movements
        threshold = np.mean(magnitude) + 2 * np.std(magnitude)
        significant_points = np.where(magnitude > threshold)

        return significant_points

    def visualize_3d(self, frame_buffer: np.ndarray):
        """
        Create 3D visualization of MS data
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        x, y = np.meshgrid(
            np.linspace(self.mz_range[0], self.mz_range[1], self.resolution[0]),
            np.arange(self.rt_window)
        )

        surf = ax.plot_surface(x, y, frame_buffer, cmap='viridis')

        ax.set_xlabel('m/z')
        ax.set_ylabel('Retention Time')
        ax.set_zlabel('Intensity')

        plt.colorbar(surf)
        plt.show()

    def save_analysis(self, output_path: str):
        """
        Save analysis results to HDF5 format
        """
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('frame_buffer', data=self.frame_buffer)
            f.create_dataset('correlation_matrix',
                             data=self.analyze_temporal_patterns())

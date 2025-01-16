import cv2
import numpy as np
from typing import Tuple, List

from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path
import h5py


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

        for mzs, intensities in input_data:
            frame = self.process_spectrum(mzs, intensities)

            # Update buffer
            self.update_frame_buffer(frame)

            # Detect features
            features = self.detect_features(frame)

            # Draw features on frame
            frame_with_features = frame.copy()
            for x, y in features:
                cv2.circle(frame_with_features, (x, y), 3, (255, 0, 0), -1)

            # Write frame to video
            self.video_writer.write(frame_with_features.astype(np.uint8))

        self.video_writer.release()

    def analyze_video(self, video_path: str):
        """
        Analyze generated MS video for patterns
        """
        cap = cv2.VideoCapture(video_path)
        optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()

        prev_frame = None
        flow_history = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if prev_frame is not None:
                # Calculate optical flow
                flow = optical_flow.calc(
                    cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    None
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

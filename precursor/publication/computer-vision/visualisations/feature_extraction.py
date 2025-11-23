"""
CV Feature Extraction Pipeline
===============================

Extracts multiple types of visual features from thermodynamic droplet images:
1. Classical CV features (SIFT, ORB, AKAZE)
2. Optical flow features
3. Texture features (Gabor, LBP, GLCM)
4. Frequency domain features
5. Morphological features
6. Statistical features

Demonstrates the rich information content of the CV method.

Author: Kundai Sachikonye
Date: 2025-10-27
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.gridspec import GridSpec
import cv2
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, fftshift
from scipy.signal import find_peaks
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.0)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']


class CVFeatureExtractor:
    """Comprehensive feature extraction from CV images"""

    def __init__(self, spectrum_id=None):
        self.data_dir = Path(__file__).parent.parent / 'data'

        # Auto-detect first available spectrum if not specified
        if spectrum_id is None:
            spectrum_id = self._detect_first_spectrum()

        self.spectrum_id = spectrum_id
        self.image = None
        self.droplet_data = None
        self.features = {}

        print(f"Data directory: {self.data_dir}")
        print(f"Selected spectrum: {self.spectrum_id}")

    def _detect_first_spectrum(self):
        """Auto-detect first available spectrum ID (prefer smaller spectra for memory efficiency)"""
        images_dir = self.data_dir / 'vision' / 'images'
        # Prefer smaller spectra (100-104) to avoid memory crashes
        preferred_ids = [100, 101, 102, 103, 104]

        if images_dir.exists():
            # First try preferred IDs
            for preferred_id in preferred_ids:
                file = images_dir / f'spectrum_{preferred_id}_droplet.png'
                if file.exists():
                    return preferred_id

            # If none of the preferred ones exist, take any available
            for file in sorted(images_dir.glob('spectrum_*_droplet.png')):
                spec_id = int(file.stem.split('_')[1])
                # Skip spectrum 105 (too large - 65870 droplets)
                if spec_id != 105:
                    return spec_id

        return 100  # Default fallback

    def load_data(self):
        """Load image and droplet data"""
        print(f"Loading data for spectrum {self.spectrum_id}...")

        # Try to load image
        try:
            image_file = self.data_dir / 'vision' / 'images' / f'spectrum_{self.spectrum_id}_droplet.png'
            self.image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
            if self.image is not None:
                print(f"  Loaded image: {self.image.shape}")
            else:
                print(f"  Warning: Could not load image from {self.data_dir / 'vision' / 'images'}")
                return False
        except:
            print(f"  Warning: Image file not found in {self.data_dir / 'vision' / 'images'}")
            return False

        # Load droplet data
        try:
            droplet_file = self.data_dir / 'vision' / 'droplets' / f'spectrum_{self.spectrum_id}_droplets.tsv'
            self.droplet_data = pd.read_csv(droplet_file, sep='\t')
            print(f"  Loaded droplets: {len(self.droplet_data)} droplets")
        except:
            print(f"  Warning: Droplet data not found in {self.data_dir / 'vision' / 'droplets'}")
            self.droplet_data = None

        return True

    def extract_sift_features(self):
        """Extract SIFT (Scale-Invariant Feature Transform) features"""
        print("\nExtracting SIFT features...")

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(self.image, None)

        print(f"  Found {len(keypoints)} SIFT keypoints")

        # Store features
        self.features['sift'] = {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'num_keypoints': len(keypoints),
            'descriptor_dim': descriptors.shape[1] if descriptors is not None else 0
        }

        # Statistical summary of descriptors
        if descriptors is not None:
            self.features['sift']['descriptor_mean'] = np.mean(descriptors, axis=0)
            self.features['sift']['descriptor_std'] = np.std(descriptors, axis=0)
            self.features['sift']['descriptor_global_mean'] = np.mean(descriptors)
            self.features['sift']['descriptor_global_std'] = np.std(descriptors)

        return keypoints, descriptors

    def extract_orb_features(self):
        """Extract ORB (Oriented FAST and Rotated BRIEF) features"""
        print("\nExtracting ORB features...")

        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=500)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(self.image, None)

        print(f"  Found {len(keypoints)} ORB keypoints")

        # Store features
        self.features['orb'] = {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'num_keypoints': len(keypoints),
            'descriptor_dim': descriptors.shape[1] if descriptors is not None else 0
        }

        return keypoints, descriptors

    def extract_akaze_features(self):
        """Extract AKAZE (Accelerated-KAZE) features"""
        print("\nExtracting AKAZE features...")

        # Initialize AKAZE detector
        akaze = cv2.AKAZE_create()

        # Detect keypoints and compute descriptors
        keypoints, descriptors = akaze.detectAndCompute(self.image, None)

        print(f"  Found {len(keypoints)} AKAZE keypoints")

        # Store features
        self.features['akaze'] = {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'num_keypoints': len(keypoints),
            'descriptor_dim': descriptors.shape[1] if descriptors is not None else 0
        }

        return keypoints, descriptors

    def extract_optical_flow_features(self):
        """Extract optical flow features (wave motion patterns)"""
        print("\nExtracting optical flow features...")

        # Create synthetic temporal sequence by shifting image
        # This simulates wave propagation over time
        flow_features = []

        # Shift in different directions
        shifts = [(5, 0), (0, 5), (-5, 0), (0, -5)]

        for dx, dy in shifts:
            # Create shifted version
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(self.image, M,
                                    (self.image.shape[1], self.image.shape[0]))

            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self.image, shifted, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            # Extract flow statistics
            flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_angle = np.arctan2(flow[..., 1], flow[..., 0])

            flow_features.append({
                'magnitude_mean': np.mean(flow_magnitude),
                'magnitude_std': np.std(flow_magnitude),
                'magnitude_max': np.max(flow_magnitude),
                'angle_mean': np.mean(flow_angle),
                'angle_std': np.std(flow_angle)
            })

        self.features['optical_flow'] = {
            'flow_features': flow_features,
            'num_directions': len(shifts)
        }

        print(f"  Computed optical flow for {len(shifts)} directions")

        return flow_features

    def extract_texture_features(self):
        """Extract texture features (Gabor, LBP, GLCM)"""
        print("\nExtracting texture features...")

        texture_features = {}

        # 1. Gabor filters (multi-scale, multi-orientation)
        print("  Computing Gabor features...")
        gabor_features = []

        frequencies = [0.1, 0.2, 0.3, 0.4]
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        for freq in frequencies:
            for theta in orientations:
                filt_real, filt_imag = gabor(self.image, frequency=freq, theta=theta)
                gabor_features.append({
                    'freq': freq,
                    'theta': theta,
                    'real_mean': np.mean(filt_real),
                    'real_std': np.std(filt_real),
                    'imag_mean': np.mean(filt_imag),
                    'imag_std': np.std(filt_imag),
                    'magnitude_mean': np.mean(np.sqrt(filt_real**2 + filt_imag**2))
                })

        texture_features['gabor'] = gabor_features
        print(f"    Computed {len(gabor_features)} Gabor filters")

        # 2. Local Binary Patterns (LBP)
        print("  Computing LBP features...")
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(self.image, n_points, radius, method='uniform')

        # LBP histogram
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points+2,
                                   range=(0, n_points+2), density=True)

        texture_features['lbp'] = {
            'histogram': lbp_hist,
            'mean': np.mean(lbp),
            'std': np.std(lbp),
            'entropy': -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
        }
        print(f"    LBP entropy: {texture_features['lbp']['entropy']:.3f} bits")

        # 3. Gray-Level Co-occurrence Matrix (GLCM)
        print("  Computing GLCM features...")

        # Normalize image to 0-255 range with fewer levels for GLCM
        image_normalized = ((self.image / self.image.max()) * 63).astype(np.uint8)

        # Compute GLCM for different angles
        distances = [1, 3, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        glcm_features = []
        for dist in distances:
            glcm = graycomatrix(image_normalized, [dist], angles,
                               levels=64, symmetric=True, normed=True)

            # Extract Haralick features
            contrast = graycoprops(glcm, 'contrast').flatten()
            dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
            homogeneity = graycoprops(glcm, 'homogeneity').flatten()
            energy = graycoprops(glcm, 'energy').flatten()
            correlation = graycoprops(glcm, 'correlation').flatten()

            glcm_features.append({
                'distance': dist,
                'contrast_mean': np.mean(contrast),
                'dissimilarity_mean': np.mean(dissimilarity),
                'homogeneity_mean': np.mean(homogeneity),
                'energy_mean': np.mean(energy),
                'correlation_mean': np.mean(correlation)
            })

        texture_features['glcm'] = glcm_features
        print(f"    Computed GLCM for {len(distances)} distances")

        self.features['texture'] = texture_features

        return texture_features

    def extract_frequency_features(self):
        """Extract frequency domain features"""
        print("\nExtracting frequency domain features...")

        # Compute 2D FFT
        fft = fft2(self.image)
        fft_shifted = fftshift(fft)
        magnitude_spectrum = np.abs(fft_shifted)
        phase_spectrum = np.angle(fft_shifted)
        power_spectrum = magnitude_spectrum**2

        # Radial profile
        center = (fft_shifted.shape[0]//2, fft_shifted.shape[1]//2)
        y, x = np.ogrid[:fft_shifted.shape[0], :fft_shifted.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)

        max_r = min(center)
        radial_bins = np.arange(0, max_r, 1)
        radial_profile = np.zeros(len(radial_bins))

        for i, r_val in enumerate(radial_bins):
            mask = (r >= r_val) & (r < r_val + 1)
            if np.any(mask):
                radial_profile[i] = np.mean(magnitude_spectrum[mask])

        # Find dominant frequencies
        peaks, properties = find_peaks(radial_profile,
                                      height=np.max(radial_profile)*0.1,
                                      distance=5)

        frequency_features = {
            'magnitude_mean': np.mean(magnitude_spectrum),
            'magnitude_std': np.std(magnitude_spectrum),
            'magnitude_max': np.max(magnitude_spectrum),
            'power_mean': np.mean(power_spectrum),
            'power_std': np.std(power_spectrum),
            'phase_mean': np.mean(phase_spectrum),
            'phase_std': np.std(phase_spectrum),
            'radial_profile': radial_profile,
            'dominant_frequencies': radial_bins[peaks] if len(peaks) > 0 else [],
            'num_dominant_frequencies': len(peaks),
            'spectral_centroid': np.sum(radial_bins * radial_profile) / np.sum(radial_profile),
            'spectral_spread': np.sqrt(np.sum(((radial_bins -
                                               np.sum(radial_bins * radial_profile) /
                                               np.sum(radial_profile))**2) * radial_profile) /
                                      np.sum(radial_profile))
        }

        self.features['frequency'] = frequency_features

        print(f"  Found {len(peaks)} dominant frequencies")
        print(f"  Spectral centroid: {frequency_features['spectral_centroid']:.2f}")

        return frequency_features

    def extract_morphological_features(self):
        """Extract morphological features (shapes, contours)"""
        print("\nExtracting morphological features...")

        # Threshold image to create binary mask
        _, binary = cv2.threshold(self.image, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)

        print(f"  Found {len(contours)} contours")

        # Analyze contours
        contour_features = []
        for contour in contours:
            if len(contour) >= 5:  # Need at least 5 points for ellipse fitting
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)

                if area > 10:  # Filter small contours
                    # Fit ellipse
                    try:
                        ellipse = cv2.fitEllipse(contour)
                        (x, y), (MA, ma), angle = ellipse

                        # Moments
                        moments = cv2.moments(contour)

                        contour_features.append({
                            'area': area,
                            'perimeter': perimeter,
                            'circularity': 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0,
                            'major_axis': MA,
                            'minor_axis': ma,
                            'eccentricity': np.sqrt(1 - (ma/MA)**2) if MA > 0 else 0,
                            'orientation': angle,
                            'hu_moments': cv2.HuMoments(moments).flatten()
                        })
                    except:
                        pass

        # Summary statistics
        if contour_features:
            morphological_features = {
                'num_contours': len(contour_features),
                'area_mean': np.mean([c['area'] for c in contour_features]),
                'area_std': np.std([c['area'] for c in contour_features]),
                'circularity_mean': np.mean([c['circularity'] for c in contour_features]),
                'circularity_std': np.std([c['circularity'] for c in contour_features]),
                'eccentricity_mean': np.mean([c['eccentricity'] for c in contour_features]),
                'eccentricity_std': np.std([c['eccentricity'] for c in contour_features]),
                'contours': contour_features
            }
        else:
            morphological_features = {
                'num_contours': 0,
                'contours': []
            }

        self.features['morphological'] = morphological_features

        print(f"  Analyzed {len(contour_features)} valid contours")

        return morphological_features

    def extract_statistical_features(self):
        """Extract statistical features from image"""
        print("\nExtracting statistical features...")

        statistical_features = {
            # First-order statistics
            'mean': np.mean(self.image),
            'std': np.std(self.image),
            'variance': np.var(self.image),
            'min': np.min(self.image),
            'max': np.max(self.image),
            'range': np.ptp(self.image),
            'median': np.median(self.image),
            'q25': np.percentile(self.image, 25),
            'q75': np.percentile(self.image, 75),
            'iqr': np.percentile(self.image, 75) - np.percentile(self.image, 25),
            'skewness': self._calculate_skewness(self.image),
            'kurtosis': self._calculate_kurtosis(self.image),

            # Entropy
            'entropy': self._calculate_entropy(self.image),

            # Energy
            'energy': np.sum(self.image**2),
            'rms': np.sqrt(np.mean(self.image**2)),

            # Coefficient of variation
            'cv': np.std(self.image) / np.mean(self.image) if np.mean(self.image) > 0 else 0
        }

        self.features['statistical'] = statistical_features

        print(f"  Mean: {statistical_features['mean']:.2f}")
        print(f"  Std: {statistical_features['std']:.2f}")
        print(f"  Entropy: {statistical_features['entropy']:.3f} bits")

        return statistical_features

    def _calculate_skewness(self, data):
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std)**3)

    def _calculate_kurtosis(self, data):
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std)**4) - 3

    def _calculate_entropy(self, data):
        """Calculate Shannon entropy"""
        hist, _ = np.histogram(data.ravel(), bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    def extract_droplet_features(self):
        """Extract features from droplet data"""
        print("\nExtracting droplet-based features...")

        if self.droplet_data is None:
            print("  Warning: No droplet data available")
            return None

        droplet_features = {
            'num_droplets': len(self.droplet_data),

            # S-Entropy coordinates
            's_knowledge_mean': self.droplet_data['s_knowledge'].mean(),
            's_knowledge_std': self.droplet_data['s_knowledge'].std(),
            's_time_mean': self.droplet_data['s_time'].mean(),
            's_time_std': self.droplet_data['s_time'].std(),
            's_entropy_mean': self.droplet_data['s_entropy'].mean(),
            's_entropy_std': self.droplet_data['s_entropy'].std(),

            # Thermodynamic parameters
            'velocity_mean': self.droplet_data['velocity'].mean(),
            'velocity_std': self.droplet_data['velocity'].std(),
            'radius_mean': self.droplet_data['radius'].mean(),
            'radius_std': self.droplet_data['radius'].std(),
            'phase_coherence_mean': self.droplet_data['phase_coherence'].mean(),
            'phase_coherence_std': self.droplet_data['phase_coherence'].std(),

            # Physics quality
            'physics_quality_mean': self.droplet_data['physics_quality'].mean(),
            'physics_quality_std': self.droplet_data['physics_quality'].std(),

            # Intensity statistics
            'intensity_mean': self.droplet_data['intensity'].mean(),
            'intensity_std': self.droplet_data['intensity'].std(),
            'intensity_max': self.droplet_data['intensity'].max(),
            'intensity_min': self.droplet_data['intensity'].min()
        }

        self.features['droplet'] = droplet_features

        print(f"  Extracted features from {droplet_features['num_droplets']} droplets")

        return droplet_features

    def extract_all_features(self):
        """Extract all feature types"""
        print("\n" + "="*80)
        print(f"EXTRACTING ALL FEATURES FOR SPECTRUM {self.spectrum_id}")
        print("="*80)

        if not self.load_data():
            print("\nError: Could not load data")
            return None

        # Extract all feature types
        self.extract_sift_features()
        self.extract_orb_features()
        self.extract_akaze_features()
        self.extract_optical_flow_features()
        self.extract_texture_features()
        self.extract_frequency_features()
        self.extract_morphological_features()
        self.extract_statistical_features()
        self.extract_droplet_features()

        print("\n" + "="*80)
        print("FEATURE EXTRACTION COMPLETE")
        print("="*80)

        return self.features

    def create_feature_summary_vector(self):
        """Create a single feature vector from all features"""
        print("\nCreating feature summary vector...")

        feature_vector = []
        feature_names = []

        # Statistical features (16 features)
        if 'statistical' in self.features:
            stat_features = ['mean', 'std', 'variance', 'min', 'max', 'range',
                           'median', 'q25', 'q75', 'iqr', 'skewness', 'kurtosis',
                           'entropy', 'energy', 'rms', 'cv']
            for feat in stat_features:
                feature_vector.append(self.features['statistical'][feat])
                feature_names.append(f'stat_{feat}')

        # Frequency features (8 features)
        if 'frequency' in self.features:
            freq_features = ['magnitude_mean', 'magnitude_std', 'magnitude_max',
                           'power_mean', 'power_std', 'num_dominant_frequencies',
                           'spectral_centroid', 'spectral_spread']
            for feat in freq_features:
                feature_vector.append(self.features['frequency'][feat])
                feature_names.append(f'freq_{feat}')

        # Texture features (summary - 10 features)
        if 'texture' in self.features:
            # Gabor summary (mean of all orientations/frequencies)
            if 'gabor' in self.features['texture']:
                gabor_mags = [g['magnitude_mean'] for g in self.features['texture']['gabor']]
                feature_vector.append(np.mean(gabor_mags))
                feature_vector.append(np.std(gabor_mags))
                feature_names.extend(['texture_gabor_mean', 'texture_gabor_std'])

            # LBP
            if 'lbp' in self.features['texture']:
                feature_vector.extend([
                    self.features['texture']['lbp']['mean'],
                    self.features['texture']['lbp']['std'],
                    self.features['texture']['lbp']['entropy']
                ])
                feature_names.extend(['texture_lbp_mean', 'texture_lbp_std',
                                    'texture_lbp_entropy'])

            # GLCM summary
            if 'glcm' in self.features['texture']:
                glcm_contrast = [g['contrast_mean'] for g in self.features['texture']['glcm']]
                glcm_homogeneity = [g['homogeneity_mean'] for g in self.features['texture']['glcm']]
                glcm_energy = [g['energy_mean'] for g in self.features['texture']['glcm']]
                glcm_correlation = [g['correlation_mean'] for g in self.features['texture']['glcm']]

                feature_vector.extend([
                    np.mean(glcm_contrast),
                    np.mean(glcm_homogeneity),
                    np.mean(glcm_energy),
                    np.mean(glcm_correlation)
                ])
                feature_names.extend(['texture_glcm_contrast', 'texture_glcm_homogeneity',
                                    'texture_glcm_energy', 'texture_glcm_correlation'])

        # Morphological features (6 features)
        if 'morphological' in self.features:
            morph_features = ['num_contours', 'area_mean', 'area_std',
                            'circularity_mean', 'eccentricity_mean', 'eccentricity_std']
            for feat in morph_features:
                if feat in self.features['morphological']:
                    feature_vector.append(self.features['morphological'][feat])
                    feature_names.append(f'morph_{feat}')

        # Keypoint counts (3 features)
        for method in ['sift', 'orb', 'akaze']:
            if method in self.features:
                feature_vector.append(self.features[method]['num_keypoints'])
                feature_names.append(f'{method}_num_keypoints')

        # Optical flow summary (5 features)
        if 'optical_flow' in self.features:
            flow_mags = [f['magnitude_mean'] for f in self.features['optical_flow']['flow_features']]
            flow_angles = [f['angle_mean'] for f in self.features['optical_flow']['flow_features']]

            feature_vector.extend([
                np.mean(flow_mags),
                np.std(flow_mags),
                np.max(flow_mags),
                np.mean(flow_angles),
                np.std(flow_angles)
            ])
            feature_names.extend(['flow_mag_mean', 'flow_mag_std', 'flow_mag_max',
                                'flow_angle_mean', 'flow_angle_std'])

        # Droplet features (18 features)
        if 'droplet' in self.features:
            droplet_features = ['num_droplets', 's_knowledge_mean', 's_knowledge_std',
                              's_time_mean', 's_time_std', 's_entropy_mean', 's_entropy_std',
                              'velocity_mean', 'velocity_std', 'radius_mean', 'radius_std',
                              'phase_coherence_mean', 'phase_coherence_std',
                              'physics_quality_mean', 'physics_quality_std',
                              'intensity_mean', 'intensity_std', 'intensity_max']
            for feat in droplet_features:
                feature_vector.append(self.features['droplet'][feat])
                feature_names.append(f'droplet_{feat}')

        print(f"  Created feature vector with {len(feature_vector)} features")

        return np.array(feature_vector), feature_names

    def visualize_features(self):
        """Create comprehensive visualization of extracted features"""

        fig = plt.figure(figsize=(24, 28))
        gs = GridSpec(5, 3, figure=fig, hspace=0.4, wspace=0.3)

        # Row 1: Keypoint visualizations
        self._plot_keypoints(fig, gs[0, :])

        # Row 2: Texture features
        self._plot_texture_features(fig, gs[1, :])

        # Row 3: Frequency features
        self._plot_frequency_features(fig, gs[2, :])

        # Row 4: Morphological features
        self._plot_morphological_features(fig, gs[3, :])

        # Row 5: Feature summary
        self._plot_feature_summary(fig, gs[4, :])

        plt.suptitle(f'CV Feature Extraction Summary (Spectrum {self.spectrum_id})',
                    fontsize=22, fontweight='bold', y=0.995)

        plt.savefig(f'cv_features_spectrum_{self.spectrum_id}.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'cv_features_spectrum_{self.spectrum_id}.pdf',
                   bbox_inches='tight', facecolor='white')
        print(f"\nSaved: cv_features_spectrum_{self.spectrum_id}.png/pdf")

        return fig

    def _plot_keypoints(self, fig, gs):
        """Plot keypoint detections"""

        # Panel A: SIFT keypoints
        ax1 = fig.add_subplot(gs[0])

        if 'sift' in self.features:
            img_sift = cv2.drawKeypoints(self.image, self.features['sift']['keypoints'],
                                        None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            ax1.imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
            ax1.set_title(f"A. SIFT Keypoints ({self.features['sift']['num_keypoints']} detected)",
                         fontsize=16, fontweight='bold', pad=20)
            ax1.axis('off')

        # Panel B: ORB keypoints
        ax2 = fig.add_subplot(gs[1])

        if 'orb' in self.features:
            img_orb = cv2.drawKeypoints(self.image, self.features['orb']['keypoints'],
                                       None, color=(0, 255, 0))
            ax2.imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB))
            ax2.set_title(f"B. ORB Keypoints ({self.features['orb']['num_keypoints']} detected)",
                         fontsize=16, fontweight='bold', pad=20)
            ax2.axis('off')

        # Panel C: AKAZE keypoints
        ax3 = fig.add_subplot(gs[2])

        if 'akaze' in self.features:
            img_akaze = cv2.drawKeypoints(self.image, self.features['akaze']['keypoints'],
                                         None, color=(255, 0, 0))
            ax3.imshow(cv2.cvtColor(img_akaze, cv2.COLOR_BGR2RGB))
            ax3.set_title(f"C. AKAZE Keypoints ({self.features['akaze']['num_keypoints']} detected)",
                         fontsize=16, fontweight='bold', pad=20)
            ax3.axis('off')

    def _plot_texture_features(self, fig, gs):
        """Plot texture features"""

        # Panel D: Gabor filter response
        ax1 = fig.add_subplot(gs[0])

        if 'texture' in self.features and 'gabor' in self.features['texture']:
            # Show mean magnitude for each frequency
            freqs = sorted(set([g['freq'] for g in self.features['texture']['gabor']]))
            mags = []
            for freq in freqs:
                freq_mags = [g['magnitude_mean'] for g in self.features['texture']['gabor']
                           if g['freq'] == freq]
                mags.append(np.mean(freq_mags))

            ax1.bar(range(len(freqs)), mags, color='skyblue', edgecolor='black', linewidth=1.5)
            ax1.set_xticks(range(len(freqs)))
            ax1.set_xticklabels([f'{f:.2f}' for f in freqs])
            ax1.set_xlabel('Frequency', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Mean Magnitude', fontsize=14, fontweight='bold')
            ax1.set_title('D. Gabor Filter Response', fontsize=16, fontweight='bold', pad=20)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # Panel E: LBP histogram
        ax2 = fig.add_subplot(gs[1])

        if 'texture' in self.features and 'lbp' in self.features['texture']:
            hist = self.features['texture']['lbp']['histogram']
            ax2.bar(range(len(hist)), hist, color='lightcoral', edgecolor='black', linewidth=1)
            ax2.set_xlabel('LBP Pattern', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Frequency', fontsize=14, fontweight='bold')
            ax2.set_title(f"E. LBP Distribution (Entropy: {self.features['texture']['lbp']['entropy']:.3f} bits)",
                         fontsize=16, fontweight='bold', pad=20)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # Panel F: GLCM properties
        ax3 = fig.add_subplot(gs[2])

        if 'texture' in self.features and 'glcm' in self.features['texture']:
            properties = ['contrast_mean', 'dissimilarity_mean', 'homogeneity_mean',
                         'energy_mean', 'correlation_mean']
            prop_labels = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation']

            # Average across distances
            values = []
            for prop in properties:
                prop_vals = [g[prop] for g in self.features['texture']['glcm']]
                values.append(np.mean(prop_vals))

            ax3.barh(range(len(values)), values, color='lightgreen',
                    edgecolor='black', linewidth=1.5)
            ax3.set_yticks(range(len(values)))
            ax3.set_yticklabels(prop_labels, fontsize=12)
            ax3.set_xlabel('Value', fontsize=14, fontweight='bold')
            ax3.set_title('F. GLCM Properties', fontsize=16, fontweight='bold', pad=20)
            ax3.grid(axis='x', alpha=0.3, linestyle='--')

    def _plot_frequency_features(self, fig, gs):
        """Plot frequency domain features"""

        # Panel G: FFT magnitude spectrum
        ax1 = fig.add_subplot(gs[0])

        if 'frequency' in self.features:
            fft = fft2(self.image)
            fft_shifted = fftshift(fft)
            magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)

            im1 = ax1.imshow(magnitude_spectrum, cmap='hot', interpolation='bilinear')
            ax1.set_title('G. Frequency Spectrum (2D FFT)',
                         fontsize=16, fontweight='bold', pad=20)
            ax1.axis('off')

            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('Log Magnitude', fontsize=10, fontweight='bold')

        # Panel H: Radial frequency profile
        ax2 = fig.add_subplot(gs[1])

        if 'frequency' in self.features:
            radial_profile = self.features['frequency']['radial_profile']
            radial_bins = np.arange(len(radial_profile))

            ax2.plot(radial_bins, radial_profile, linewidth=2, color='#0173B2')
            ax2.fill_between(radial_bins, 0, radial_profile, alpha=0.3, color='#0173B2')

            # Mark dominant frequencies
            dom_freqs = self.features['frequency']['dominant_frequencies']
            if len(dom_freqs) > 0:
                for freq in dom_freqs:
                    if freq < len(radial_profile):
                        ax2.axvline(freq, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

            ax2.set_xlabel('Spatial Frequency', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Magnitude', fontsize=14, fontweight='bold')
            ax2.set_title(f"H. Radial Profile ({len(dom_freqs)} dominant frequencies)",
                         fontsize=16, fontweight='bold', pad=20)
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3, linestyle='--')

        # Panel I: Frequency statistics
        ax3 = fig.add_subplot(gs[2])
        ax3.axis('off')

        if 'frequency' in self.features:
            freq_stats = self.features['frequency']

            stats_text = f"""
$\\mathbf{{Frequency\\ Domain\\ Statistics}}$

Magnitude:
  Mean: {freq_stats['magnitude_mean']:.2f}
  Std: {freq_stats['magnitude_std']:.2f}
  Max: {freq_stats['magnitude_max']:.2f}

Power:
  Mean: {freq_stats['power_mean']:.2e}
  Std: {freq_stats['power_std']:.2e}

Spectral Properties:
  Centroid: {freq_stats['spectral_centroid']:.2f}
  Spread: {freq_stats['spectral_spread']:.2f}

Dominant Frequencies: {freq_stats['num_dominant_frequencies']}
            """

            ax3.text(0.1, 0.95, stats_text,
                    transform=ax3.transAxes, fontsize=11,
                    verticalalignment='top', family='serif',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

            ax3.set_title('I. Frequency Statistics',
                         fontsize=16, fontweight='bold', pad=20)

    def _plot_morphological_features(self, fig, gs):
        """Plot morphological features"""

        # Panel J: Contours
        ax1 = fig.add_subplot(gs[0])

        if 'morphological' in self.features:
            # Threshold image
            _, binary = cv2.threshold(self.image, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours on original image
            img_contours = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)

            ax1.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
            ax1.set_title(f"J. Detected Contours ({self.features['morphological']['num_contours']})",
                         fontsize=16, fontweight='bold', pad=20)
            ax1.axis('off')

        # Panel K: Shape properties
        ax2 = fig.add_subplot(gs[1])

        if 'morphological' in self.features and len(self.features['morphological']['contours']) > 0:
            contours = self.features['morphological']['contours']

            circularities = [c['circularity'] for c in contours]
            eccentricities = [c['eccentricity'] for c in contours]

            ax2.scatter(circularities, eccentricities, s=100, alpha=0.6,
                       c=range(len(circularities)), cmap='viridis',
                       edgecolors='black', linewidth=1)

            ax2.set_xlabel('Circularity', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Eccentricity', fontsize=14, fontweight='bold')
            ax2.set_title('K. Shape Properties', fontsize=16, fontweight='bold', pad=20)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)

        # Panel L: Area distribution
        ax3 = fig.add_subplot(gs[2])

        if 'morphological' in self.features and len(self.features['morphological']['contours']) > 0:
            areas = [c['area'] for c in self.features['morphological']['contours']]

            ax3.hist(areas, bins=30, color='plum', edgecolor='black', alpha=0.7)
            ax3.set_xlabel('Contour Area (pixels²)', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Frequency', fontsize=14, fontweight='bold')
            ax3.set_title('L. Area Distribution', fontsize=16, fontweight='bold', pad=20)
            ax3.set_yscale('log')
            ax3.grid(axis='y', alpha=0.3, linestyle='--')

            # Add statistics
            ax3.axvline(np.mean(areas), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(areas):.1f}')
            ax3.legend(fontsize=11)

    def _plot_feature_summary(self, fig, gs):
        """Plot feature summary"""

        # Panel M: Feature vector
        ax1 = fig.add_subplot(gs[0])

        feature_vector, feature_names = self.create_feature_summary_vector()

        # Group features by type
        feature_groups = {
            'Statistical': [i for i, name in enumerate(feature_names) if name.startswith('stat_')],
            'Frequency': [i for i, name in enumerate(feature_names) if name.startswith('freq_')],
            'Texture': [i for i, name in enumerate(feature_names) if name.startswith('texture_')],
            'Morphological': [i for i, name in enumerate(feature_names) if name.startswith('morph_')],
            'Keypoints': [i for i, name in enumerate(feature_names) if 'keypoints' in name],
            'Optical Flow': [i for i, name in enumerate(feature_names) if name.startswith('flow_')],
            'Droplet': [i for i, name in enumerate(feature_names) if name.startswith('droplet_')]
        }

        # Normalize features for visualization
        feature_vector_norm = (feature_vector - feature_vector.min()) / (feature_vector.max() - feature_vector.min() + 1e-10)

        # Color by group
        colors = []
        color_map = {
            'Statistical': '#0173B2',
            'Frequency': '#DE8F05',
            'Texture': '#029E73',
            'Morphological': '#CC78BC',
            'Keypoints': '#CA9161',
            'Optical Flow': '#949494',
            'Droplet': '#EE3377'
        }

        for i in range(len(feature_vector)):
            for group, indices in feature_groups.items():
                if i in indices:
                    colors.append(color_map[group])
                    break

        ax1.bar(range(len(feature_vector_norm)), feature_vector_norm,
               color=colors, edgecolor='black', linewidth=0.5, alpha=0.7)
        ax1.set_xlabel('Feature Index', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Normalized Value', fontsize=14, fontweight='bold')
        ax1.set_title(f'M. Feature Vector ({len(feature_vector)} features)',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=group)
                          for group, color in color_map.items()]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=2)

        # Panel N: Feature group sizes
        ax2 = fig.add_subplot(gs[1])

        group_sizes = {group: len(indices) for group, indices in feature_groups.items()}

        wedges, texts, autotexts = ax2.pie(group_sizes.values(),
                                           labels=group_sizes.keys(),
                                           autopct='%1.1f%%',
                                           colors=[color_map[g] for g in group_sizes.keys()],
                                           startangle=90,
                                           textprops={'fontsize': 10, 'fontweight': 'bold'},
                                           wedgeprops={'edgecolor': 'black', 'linewidth': 2})

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)

        ax2.set_title('N. Feature Distribution by Type',
                     fontsize=16, fontweight='bold', pad=20)

        # Panel O: Feature summary table
        ax3 = fig.add_subplot(gs[2])
        ax3.axis('off')

        summary_data = [
            ['Feature Type', 'Count', 'Example'],
            ['Statistical', len(feature_groups['Statistical']), 'mean, std, entropy'],
            ['Frequency', len(feature_groups['Frequency']), 'FFT, spectral centroid'],
            ['Texture', len(feature_groups['Texture']), 'Gabor, LBP, GLCM'],
            ['Morphological', len(feature_groups['Morphological']), 'contours, shape'],
            ['Keypoints', len(feature_groups['Keypoints']), 'SIFT, ORB, AKAZE'],
            ['Optical Flow', len(feature_groups['Optical Flow']), 'wave motion'],
            ['Droplet', len(feature_groups['Droplet']), 'S-Entropy, physics'],
            ['', '', ''],
            ['TOTAL', len(feature_vector), f'{len(feature_vector)} features']
        ]

        table = ax3.table(cellText=summary_data[1:],
                         colLabels=summary_data[0],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.3, 0.2, 0.5])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        # Style header
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color rows by type
        for i in range(1, 8):
            group_name = summary_data[i][0]
            if group_name in color_map:
                for j in range(3):
                    table[(i, j)].set_facecolor(color_map[group_name])
                    table[(i, j)].set_alpha(0.3)

        # Highlight total row
        for j in range(3):
            table[(8, j)].set_facecolor('#FFD700')
            table[(8, j)].set_alpha(0.5)
            table[(8, j)].set_text_props(weight='bold')

        ax3.set_title('O. Feature Summary', fontsize=16, fontweight='bold', pad=20)


def main():
    """Main execution function"""

    print("="*80)
    print("CV FEATURE EXTRACTION PIPELINE")
    print("="*80)

    # Create extractor (will auto-detect first available spectrum)
    extractor = CVFeatureExtractor()

    # Extract all features
    features = extractor.extract_all_features()

    if features is None:
        print("\nError: Feature extraction failed")
        return

    # Create feature vector
    feature_vector, feature_names = extractor.create_feature_summary_vector()

    print(f"\nFeature vector shape: {feature_vector.shape}")
    print(f"Feature names: {len(feature_names)}")

    # Note: Pickle save skipped due to cv2.KeyPoint serialization issues
    # Feature data is preserved in the visualizations below

    # Visualize features
    print("\nGenerating feature visualization...")
    fig = extractor.visualize_features()
    plt.close(fig)

    print("\n" + "="*80)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

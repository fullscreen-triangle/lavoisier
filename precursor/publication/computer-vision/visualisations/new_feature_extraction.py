#!/usr/bin/env python3
"""
Computer Vision Feature Extraction and Analysis
================================================

Comprehensive analysis of CV features extracted from thermodynamic droplet images:
- SIFT, ORB, AKAZE keypoint detection
- Texture analysis (Gabor filters, LBP, GLCM)
- Frequency domain features
- Morphological analysis
- Statistical features
- Feature comparison and visualization

This creates publication-quality figures demonstrating the richness
of the CV representation.

Author: Lavoisier CV Team
Date: 2025-10-27
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import cv2
from scipy import ndimage, signal
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.figsize': (12, 8)
})


class FeatureExtractor:
    """Extract and analyze CV features from droplet images"""

    def __init__(self, spectrum_id=100):
        self.spectrum_id = spectrum_id
        self.data_dir = Path(__file__).parent.parent / 'data'
        self.output_dir = Path(__file__).parent

        # Data containers
        self.image_data = None
        self.droplet_data = None

        # Feature containers
        self.features = {}

    def load_data(self):
        """Load image and droplet data"""
        print(f"\nLoading data for spectrum {self.spectrum_id}...")

        try:
            # Load image
            image_file = self.data_dir / 'vision' / 'images' / f'spectrum_{self.spectrum_id}_droplet.png'
            self.image_data = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
            print(f"  ✓ Image: {self.image_data.shape}")

            # Load droplet data
            droplet_file = self.data_dir / 'vision' / 'droplets' / f'spectrum_{self.spectrum_id}_droplets.tsv'
            self.droplet_data = pd.read_csv(droplet_file, sep='\t')
            print(f"  ✓ Droplets: {len(self.droplet_data)} droplets")

            return True

        except Exception as e:
            print(f"  ✗ Error loading data: {e}")
            return False

    def extract_all_features(self):
        """Extract all CV features"""
        print("\n" + "="*80)
        print("EXTRACTING CV FEATURES")
        print("="*80)

        feature_funcs = [
            ('SIFT keypoints', self.extract_sift_features),
            ('ORB keypoints', self.extract_orb_features),
            ('AKAZE keypoints', self.extract_akaze_features),
            ('Texture features', self.extract_texture_features),
            ('Frequency features', self.extract_frequency_features),
            ('Morphological features', self.extract_morphological_features),
            ('Statistical features', self.extract_statistical_features),
        ]

        for name, func in feature_funcs:
            print(f"\nExtracting {name}...")
            try:
                func()
                print(f"  ✓ Success")
            except Exception as e:
                print(f"  ✗ Error: {e}")

    def extract_sift_features(self):
        """Extract SIFT keypoints and descriptors"""
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(self.image_data, None)

        self.features['sift'] = {
            'n_keypoints': len(keypoints),
            'keypoints': keypoints,
            'descriptors': descriptors,
            'scales': [kp.size for kp in keypoints],
            'responses': [kp.response for kp in keypoints],
            'angles': [kp.angle for kp in keypoints],
        }

        print(f"    Found {len(keypoints)} SIFT keypoints")

    def extract_orb_features(self):
        """Extract ORB keypoints and descriptors"""
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(self.image_data, None)

        self.features['orb'] = {
            'n_keypoints': len(keypoints),
            'keypoints': keypoints,
            'descriptors': descriptors,
            'scales': [kp.size for kp in keypoints],
            'responses': [kp.response for kp in keypoints],
        }

        print(f"    Found {len(keypoints)} ORB keypoints")

    def extract_akaze_features(self):
        """Extract AKAZE keypoints and descriptors"""
        akaze = cv2.AKAZE_create()
        keypoints, descriptors = akaze.detectAndCompute(self.image_data, None)

        self.features['akaze'] = {
            'n_keypoints': len(keypoints),
            'keypoints': keypoints,
            'descriptors': descriptors,
            'scales': [kp.size for kp in keypoints],
            'responses': [kp.response for kp in keypoints],
        }

        print(f"    Found {len(keypoints)} AKAZE keypoints")

    def extract_texture_features(self):
        """Extract texture features"""
        # Gabor filters
        gabor_features = []
        for theta in np.arange(0, np.pi, np.pi / 4):
            for sigma in [1, 3, 5]:
                for frequency in [0.05, 0.1, 0.2]:
                    kernel = cv2.getGaborKernel((21, 21), sigma, theta, frequency, 0.5, 0, ktype=cv2.CV_32F)
                    filtered = cv2.filter2D(self.image_data, cv2.CV_8UC3, kernel)
                    gabor_features.append({
                        'theta': theta,
                        'sigma': sigma,
                        'frequency': frequency,
                        'mean': np.mean(filtered),
                        'std': np.std(filtered),
                        'energy': np.sum(filtered ** 2),
                    })

        # Edge detection
        edges = cv2.Canny(self.image_data, 50, 150)

        self.features['texture'] = {
            'gabor_features': gabor_features,
            'n_gabor': len(gabor_features),
            'edge_density': np.sum(edges > 0) / edges.size,
            'edge_image': edges,
        }

        print(f"    Computed {len(gabor_features)} Gabor features")

    def extract_frequency_features(self):
        """Extract frequency domain features"""
        # FFT
        fft = np.fft.fft2(self.image_data)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        phase = np.angle(fft_shift)

        # Power spectrum
        power_spectrum = magnitude ** 2

        # Radial profile
        y, x = np.indices(magnitude.shape)
        center = np.array([(y.max()-y.min())/2.0, (x.max()-x.min())/2.0])
        r = np.sqrt((y - center[0])**2 + (x - center[1])**2)
        r = r.astype(int)

        tbin = np.bincount(r.ravel(), power_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / (nr + 1e-10)

        # Spectral features
        self.features['frequency'] = {
            'fft_magnitude': magnitude,
            'fft_phase': phase,
            'power_spectrum': power_spectrum,
            'radial_profile': radial_profile,
            'dc_component': magnitude[magnitude.shape[0]//2, magnitude.shape[1]//2],
            'spectral_centroid': np.sum(magnitude * r) / (np.sum(magnitude) + 1e-10),
            'spectral_spread': np.sqrt(np.sum(((r - np.sum(magnitude * r) / (np.sum(magnitude) + 1e-10))**2) * magnitude) / (np.sum(magnitude) + 1e-10)),
        }

        print(f"    Computed frequency domain features")

    def extract_morphological_features(self):
        """Extract morphological features"""
        # Threshold image
        _, binary = cv2.threshold(self.image_data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Contour properties
        contour_areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
        contour_perimeters = [cv2.arcLength(c, True) for c in contours if cv2.contourArea(c) > 10]

        # Moments
        moments_list = []
        for c in contours:
            if cv2.contourArea(c) > 10:
                M = cv2.moments(c)
                if M['m00'] != 0:
                    moments_list.append(M)

        # Hu moments
        hu_moments = []
        for M in moments_list[:100]:  # Limit to first 100
            hu = cv2.HuMoments(M).flatten()
            hu_moments.append(hu)

        self.features['morphology'] = {
            'n_contours': len(contour_areas),
            'contour_areas': contour_areas,
            'contour_perimeters': contour_perimeters,
            'mean_area': np.mean(contour_areas) if contour_areas else 0,
            'mean_perimeter': np.mean(contour_perimeters) if contour_perimeters else 0,
            'hu_moments': hu_moments,
            'binary_image': binary,
        }

        print(f"    Found {len(contour_areas)} valid contours")

    def extract_statistical_features(self):
        """Extract statistical features"""
        # Basic statistics
        mean_val = np.mean(self.image_data)
        std_val = np.std(self.image_data)
        skew = np.mean(((self.image_data - mean_val) / (std_val + 1e-10)) ** 3)
        kurtosis = np.mean(((self.image_data - mean_val) / (std_val + 1e-10)) ** 4)

        # Histogram
        hist, bins = np.histogram(self.image_data.ravel(), bins=256, density=True)

        # Entropy
        hist_prob = hist + 1e-10
        shannon_entropy = -np.sum(hist_prob * np.log2(hist_prob))

        # Gradient statistics
        gx = ndimage.sobel(self.image_data, axis=0)
        gy = ndimage.sobel(self.image_data, axis=1)
        gradient_mag = np.sqrt(gx**2 + gy**2)

        self.features['statistics'] = {
            'mean': mean_val,
            'std': std_val,
            'min': np.min(self.image_data),
            'max': np.max(self.image_data),
            'median': np.median(self.image_data),
            'skewness': skew,
            'kurtosis': kurtosis,
            'entropy': shannon_entropy,
            'gradient_mean': np.mean(gradient_mag),
            'gradient_std': np.std(gradient_mag),
            'histogram': hist,
            'bins': bins,
        }

        print(f"    Computed statistical features (entropy: {shannon_entropy:.3f} bits)")

    def generate_all_figures(self):
        """Generate all visualization figures"""
        print("\n" + "="*80)
        print("GENERATING FEATURE VISUALIZATIONS")
        print("="*80)

        figures = [
            ('Figure 1: Keypoint Detection', self.create_figure_1_keypoints),
            ('Figure 2: Texture Analysis', self.create_figure_2_texture),
            ('Figure 3: Frequency Domain', self.create_figure_3_frequency),
            ('Figure 4: Morphological Analysis', self.create_figure_4_morphology),
            ('Figure 5: Feature Comparison', self.create_figure_5_comparison),
        ]

        for i, (name, func) in enumerate(figures, 1):
            print(f"\n[{i}/{len(figures)}] {name}")
            try:
                func()
                print(f"  ✓ Success")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "="*80)
        print("FEATURE EXTRACTION COMPLETE")
        print("="*80)

    def create_figure_1_keypoints(self):
        """Figure 1: Keypoint detection comparison"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # SIFT keypoints
        ax1 = fig.add_subplot(gs[0, 0])
        img_sift = cv2.drawKeypoints(self.image_data, self.features['sift']['keypoints'],
                                     None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax1.imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"SIFT Keypoints (n={self.features['sift']['n_keypoints']})",
                     fontweight='bold', fontsize=14)
        ax1.axis('off')

        # ORB keypoints
        ax2 = fig.add_subplot(gs[0, 1])
        img_orb = cv2.drawKeypoints(self.image_data, self.features['orb']['keypoints'],
                                    None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax2.imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB))
        ax2.set_title(f"ORB Keypoints (n={self.features['orb']['n_keypoints']})",
                     fontweight='bold', fontsize=14)
        ax2.axis('off')

        # AKAZE keypoints
        ax3 = fig.add_subplot(gs[0, 2])
        img_akaze = cv2.drawKeypoints(self.image_data, self.features['akaze']['keypoints'],
                                      None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax3.imshow(cv2.cvtColor(img_akaze, cv2.COLOR_BGR2RGB))
        ax3.set_title(f"AKAZE Keypoints (n={self.features['akaze']['n_keypoints']})",
                     fontweight='bold', fontsize=14)
        ax3.axis('off')

        # SIFT scale distribution
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(self.features['sift']['scales'], bins=50, color='skyblue',
                alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Scale', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('SIFT Scale Distribution', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # ORB response distribution
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.hist(self.features['orb']['responses'], bins=50, color='lightcoral',
                alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Response', fontweight='bold')
        ax5.set_ylabel('Frequency', fontweight='bold')
        ax5.set_title('ORB Response Distribution', fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # Keypoint count comparison
        ax6 = fig.add_subplot(gs[1, 2])
        methods = ['SIFT', 'ORB', 'AKAZE']
        counts = [
            self.features['sift']['n_keypoints'],
            self.features['orb']['n_keypoints'],
            self.features['akaze']['n_keypoints']
        ]
        colors_bar = ['skyblue', 'lightcoral', 'lightgreen']

        bars = ax6.bar(methods, counts, color=colors_bar, alpha=0.7, edgecolor='black')
        ax6.set_ylabel('Number of Keypoints', fontweight='bold')
        ax6.set_title('Keypoint Detection Comparison', fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom', fontweight='bold')

        fig.suptitle(f'Keypoint Detection Analysis\nSpectrum {self.spectrum_id}',
                    fontsize=18, fontweight='bold')

        plt.savefig(self.output_dir / f'features_fig1_keypoints_{self.spectrum_id}.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def create_figure_2_texture(self):
        """Figure 2: Texture analysis"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.image_data, cmap='gray')
        ax1.set_title('Original Image', fontweight='bold', fontsize=14)
        ax1.axis('off')

        # Edge detection
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(self.features['texture']['edge_image'], cmap='hot')
        ax2.set_title(f"Canny Edges (density={self.features['texture']['edge_density']:.3f})",
                     fontweight='bold', fontsize=14)
        ax2.axis('off')

        # Sample Gabor filter response
        ax3 = fig.add_subplot(gs[0, 2])
        gabor_kernel = cv2.getGaborKernel((21, 21), 3, np.pi/4, 0.1, 0.5, 0, ktype=cv2.CV_32F)
        gabor_response = cv2.filter2D(self.image_data, cv2.CV_8UC3, gabor_kernel)
        ax3.imshow(gabor_response, cmap='viridis')
        ax3.set_title('Gabor Filter Response (θ=45°)', fontweight='bold', fontsize=14)
        ax3.axis('off')

        # Gabor feature distribution (energy)
        ax4 = fig.add_subplot(gs[1, 0])
        energies = [f['energy'] for f in self.features['texture']['gabor_features']]
        ax4.hist(energies, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Gabor Energy', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('Gabor Feature Energy Distribution', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # Gabor features by orientation
        ax5 = fig.add_subplot(gs[1, 1])
        thetas = [f['theta'] for f in self.features['texture']['gabor_features']]
        theta_unique = sorted(set(thetas))
        theta_means = [np.mean([f['mean'] for f in self.features['texture']['gabor_features'] if f['theta'] == t])
                      for t in theta_unique]

        ax5.plot([t * 180/np.pi for t in theta_unique], theta_means, 'o-', linewidth=2, markersize=8)
        ax5.set_xlabel('Orientation (degrees)', fontweight='bold')
        ax5.set_ylabel('Mean Response', fontweight='bold')
        ax5.set_title('Gabor Response by Orientation', fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # Texture statistics table
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')

        table_data = [
            ['Metric', 'Value'],
            ['', ''],
            ['Gabor Filters', f"{self.features['texture']['n_gabor']}"],
            ['Edge Density', f"{self.features['texture']['edge_density']:.4f}"],
            ['', ''],
            ['Energy Stats', ''],
            ['  Mean', f"{np.mean(energies):.2e}"],
            ['  Std', f"{np.std(energies):.2e}"],
            ['  Min', f"{np.min(energies):.2e}"],
            ['  Max', f"{np.max(energies):.2e}"],
        ]

        table = ax6.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        table[(0, 0)].set_facecolor('#4CAF50')
        table[(0, 1)].set_facecolor('#4CAF50')
        table[(0, 0)].set_text_props(weight='bold', color='white')
        table[(0, 1)].set_text_props(weight='bold', color='white')

        fig.suptitle(f'Texture Analysis\nSpectrum {self.spectrum_id}',
                    fontsize=18, fontweight='bold')

        plt.savefig(self.output_dir / f'features_fig2_texture_{self.spectrum_id}.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def create_figure_3_frequency(self):
        """Figure 3: Frequency domain analysis"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.image_data, cmap='gray')
        ax1.set_title('Spatial Domain', fontweight='bold', fontsize=14)
        ax1.axis('off')

        # FFT magnitude
        ax2 = fig.add_subplot(gs[0, 1])
        magnitude_log = np.log(self.features['frequency']['fft_magnitude'] + 1)
        im2 = ax2.imshow(magnitude_log, cmap='jet')
        ax2.set_title('FFT Magnitude (log scale)', fontweight='bold', fontsize=14)
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        # FFT phase
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(self.features['frequency']['fft_phase'], cmap='hsv')
        ax3.set_title('FFT Phase', fontweight='bold', fontsize=14)
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)

        # Power spectrum
        ax4 = fig.add_subplot(gs[1, 0])
        power_log = np.log(self.features['frequency']['power_spectrum'] + 1)
        im4 = ax4.imshow(power_log, cmap='hot')
        ax4.set_title('Power Spectrum (log scale)', fontweight='bold', fontsize=14)
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046)

        # Radial profile
        ax5 = fig.add_subplot(gs[1, 1])
        radial = self.features['frequency']['radial_profile']
        ax5.semilogy(radial[:len(radial)//2], linewidth=2, color='darkblue')
        ax5.set_xlabel('Radial Frequency', fontweight='bold')
        ax5.set_ylabel('Power (log scale)', fontweight='bold')
        ax5.set_title('Radial Power Spectrum', fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # Frequency statistics
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')

        table_data = [
            ['Metric', 'Value'],
            ['', ''],
            ['DC Component', f"{self.features['frequency']['dc_component']:.2e}"],
            ['', ''],
            ['Spectral Centroid', f"{self.features['frequency']['spectral_centroid']:.2f}"],
            ['Spectral Spread', f"{self.features['frequency']['spectral_spread']:.2f}"],
            ['', ''],
            ['Total Power', f"{np.sum(self.features['frequency']['power_spectrum']):.2e}"],
        ]

        table = ax6.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)

        # Style header
        table[(0, 0)].set_facecolor('#4CAF50')
        table[(0, 1)].set_facecolor('#4CAF50')
        table[(0, 0)].set_text_props(weight='bold', color='white')
        table[(0, 1)].set_text_props(weight='bold', color='white')

        fig.suptitle(f'Frequency Domain Analysis\nSpectrum {self.spectrum_id}',
                    fontsize=18, fontweight='bold')

        plt.savefig(self.output_dir / f'features_fig3_frequency_{self.spectrum_id}.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def create_figure_4_morphology(self):
        """Figure 4: Morphological analysis"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.image_data, cmap='gray')
        ax1.set_title('Original Image', fontweight='bold', fontsize=14)
        ax1.axis('off')

        # Binary image
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(self.features['morphology']['binary_image'], cmap='gray')
        ax2.set_title('Binary (Otsu threshold)', fontweight='bold', fontsize=14)
        ax2.axis('off')

        # Distance transform
        ax3 = fig.add_subplot(gs[0, 2])
        dist_transform = ndimage.distance_transform_edt(self.features['morphology']['binary_image'])
        im3 = ax3.imshow(dist_transform, cmap='jet')
        ax3.set_title('Distance Transform', fontweight='bold', fontsize=14)
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)

        # Contour area distribution
        ax4 = fig.add_subplot(gs[1, 0])
        areas = self.features['morphology']['contour_areas']
        if areas:
            ax4.hist(areas, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Contour Area (pixels)', fontweight='bold')
            ax4.set_ylabel('Frequency', fontweight='bold')
            ax4.set_title(f'Contour Area Distribution (n={len(areas)})', fontweight='bold')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)

        # Contour perimeter distribution
        ax5 = fig.add_subplot(gs[1, 1])
        perimeters = self.features['morphology']['contour_perimeters']
        if perimeters:
            ax5.hist(perimeters, bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
            ax5.set_xlabel('Contour Perimeter (pixels)', fontweight='bold')
            ax5.set_ylabel('Frequency', fontweight='bold')
            ax5.set_title('Contour Perimeter Distribution', fontweight='bold')
            ax5.set_yscale('log')
            ax5.grid(True, alpha=0.3)

        # Morphological statistics
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')

        table_data = [
            ['Metric', 'Value'],
            ['', ''],
            ['Number of Contours', f"{self.features['morphology']['n_contours']}"],
            ['', ''],
            ['Area Statistics', ''],
            ['  Mean', f"{self.features['morphology']['mean_area']:.2f} px²"],
            ['  Std', f"{np.std(areas):.2f} px²" if areas else 'N/A'],
            ['', ''],
            ['Perimeter Statistics', ''],
            ['  Mean', f"{self.features['morphology']['mean_perimeter']:.2f} px"],
            ['  Std', f"{np.std(perimeters):.2f} px" if perimeters else 'N/A'],
        ]

        table = ax6.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        table[(0, 0)].set_facecolor('#4CAF50')
        table[(0, 1)].set_facecolor('#4CAF50')
        table[(0, 0)].set_text_props(weight='bold', color='white')
        table[(0, 1)].set_text_props(weight='bold', color='white')

        fig.suptitle(f'Morphological Analysis\nSpectrum {self.spectrum_id}',
                    fontsize=18, fontweight='bold')

        plt.savefig(self.output_dir / f'features_fig4_morphology_{self.spectrum_id}.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def create_figure_5_comparison(self):
        """Figure 5: Feature comparison and summary"""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

        # Feature richness comparison
        ax1 = fig.add_subplot(gs[0, 0])
        feature_types = ['SIFT', 'ORB', 'AKAZE', 'Gabor', 'Contours']
        feature_counts = [
            self.features['sift']['n_keypoints'],
            self.features['orb']['n_keypoints'],
            self.features['akaze']['n_keypoints'],
            self.features['texture']['n_gabor'],
            self.features['morphology']['n_contours']
        ]
        colors_bar = ['#3498DB', '#E74C3C', '#2ECC71', '#9B59B6', '#F39C12']

        bars = ax1.barh(feature_types, feature_counts, color=colors_bar, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Feature Count', fontweight='bold')
        ax1.set_title('Feature Richness Comparison', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for bar, count in zip(bars, feature_counts):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(count)}', ha='left', va='center', fontweight='bold')

        # Statistical feature summary
        ax2 = fig.add_subplot(gs[0, 1])
        stats_names = ['Mean', 'Std', 'Entropy', 'Gradient\nMean']
        stats_values = [
            self.features['statistics']['mean'],
            self.features['statistics']['std'],
            self.features['statistics']['entropy'] * 10,  # Scale for visibility
            self.features['statistics']['gradient_mean']
        ]

        ax2.bar(stats_names, stats_values, color='lightseagreen', alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Value (normalized)', fontweight='bold')
        ax2.set_title('Statistical Feature Summary', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')

        # Intensity histogram
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(self.features['statistics']['bins'][:-1],
                self.features['statistics']['histogram'],
                color='navy', linewidth=2)
        ax3.fill_between(self.features['statistics']['bins'][:-1],
                        self.features['statistics']['histogram'],
                        alpha=0.3, color='lightblue')
        ax3.set_xlabel('Pixel Value', fontweight='bold')
        ax3.set_ylabel('Probability Density', fontweight='bold')
        ax3.set_title('Pixel Intensity Distribution', fontweight='bold', fontsize=14)
        ax3.grid(True, alpha=0.3)

        # Feature dimensionality
        ax4 = fig.add_subplot(gs[1, 1])
        dims = {
            'SIFT': 128 * self.features['sift']['n_keypoints'],
            'ORB': 32 * self.features['orb']['n_keypoints'],
            'AKAZE': len(self.features['akaze']['keypoints']) * 61 if self.features['akaze']['keypoints'] else 0,
            'Gabor': self.features['texture']['n_gabor'] * 3,
            'Stats': 10,
        }

        ax4.pie(dims.values(), labels=dims.keys(), autopct='%1.1f%%',
               colors=colors_bar, startangle=90)
        ax4.set_title('Feature Space Dimensionality', fontweight='bold', fontsize=14)

        # Comprehensive feature table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')

        table_data = [
            ['Feature Type', 'Count', 'Dimensionality', 'Key Metric'],
            ['', '', '', ''],
            ['SIFT Keypoints', f"{self.features['sift']['n_keypoints']}", '128-D per keypoint',
             f"Mean scale: {np.mean(self.features['sift']['scales']):.2f}"],
            ['ORB Keypoints', f"{self.features['orb']['n_keypoints']}", '32-D per keypoint',
             f"Mean response: {np.mean(self.features['orb']['responses']):.2f}"],
            ['AKAZE Keypoints', f"{self.features['akaze']['n_keypoints']}", '61-D per keypoint',
             f"Mean scale: {np.mean(self.features['akaze']['scales']):.2f}"],
            ['', '', '', ''],
            ['Gabor Filters', f"{self.features['texture']['n_gabor']}", '3-D per filter',
             f"Edge density: {self.features['texture']['edge_density']:.4f}"],
            ['Frequency Features', '5', 'Various',
             f"Spectral centroid: {self.features['frequency']['spectral_centroid']:.2f}"],
            ['Morphology', f"{self.features['morphology']['n_contours']}", 'Various',
             f"Mean area: {self.features['morphology']['mean_area']:.2f} px²"],
            ['Statistics', '10', 'Scalar',
             f"Entropy: {self.features['statistics']['entropy']:.3f} bits"],
            ['', '', '', ''],
            ['TOTAL', f"{sum([self.features['sift']['n_keypoints'], self.features['orb']['n_keypoints'], self.features['akaze']['n_keypoints'], self.features['texture']['n_gabor'], self.features['morphology']['n_contours']])}",
             f"{sum(dims.values())}-D", 'Multi-modal representation'],
        ]

        table = ax5.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.25, 0.15, 0.25, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style total row
        for i in range(4):
            table[(len(table_data)-1, i)].set_facecolor('#FFC107')
            table[(len(table_data)-1, i)].set_text_props(weight='bold')

        fig.suptitle(f'Comprehensive Feature Summary\nSpectrum {self.spectrum_id}',
                    fontsize=18, fontweight='bold')

        plt.savefig(self.output_dir / f'features_fig5_summary_{self.spectrum_id}.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


def main():
    """Main execution"""
    print("="*80)
    print("COMPUTER VISION FEATURE EXTRACTION")
    print("Comprehensive analysis of CV features from droplet images")
    print("="*80)

    extractor = FeatureExtractor(spectrum_id=100)

    if not extractor.load_data():
        print("\nERROR: Failed to load data")
        return 1

    extractor.extract_all_features()
    extractor.generate_all_figures()

    print(f"\n✓ All figures saved to: {extractor.output_dir}")
    print("  Files: features_fig1_*.png through features_fig5_*.png")

    return 0


if __name__ == "__main__":
    exit(main())

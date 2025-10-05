#!/usr/bin/env python3
"""
Visual Pipeline Orchestrator - Standalone Implementation
Computer vision-based processing pipeline for mass spectrometry data validation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import warnings
from collections import defaultdict

# Import our standalone components
from .mzml_reader import StandaloneMzMLReader, Spectrum, load_mzml_file

warnings.filterwarnings("ignore")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("OpenCV not available, using fallback image processing")
    CV2_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    print("PIL not available, using basic image processing")
    PIL_AVAILABLE = False


@dataclass
class Ion:
    """Ion representation for drip pathway"""
    mz: float
    intensity: float
    charge: int
    ion_type: str  # 'precursor', 'fragment', 'adduct'
    formula: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DripSpectrum:
    """Drip spectrum representation after ion-to-drip conversion"""
    spectrum_id: str
    original_spectrum: Spectrum
    drip_coordinates: np.ndarray  # 2D coordinates for drip visualization
    drip_intensities: np.ndarray
    drip_image: Optional[np.ndarray] = None
    drip_metadata: Optional[Dict[str, Any]] = None


@dataclass
class VisualAnnotation:
    """Visual annotation result from LipidMaps overlay"""
    compound_id: str
    compound_name: str
    visual_similarity_score: float
    mathematical_similarity_score: float
    overlay_image: Optional[np.ndarray] = None
    similarity_metrics: Optional[Dict[str, float]] = None


class IonDripConverter:
    """Converts spectra to ions and then to drip spectra"""

    def __init__(self):
        self.conversion_params = {
            'drip_resolution': (512, 512),  # Drip image resolution
            'intensity_scaling': 'log',      # 'linear', 'log', 'sqrt'
            'coordinate_mapping': 'spiral',  # 'spiral', 'grid', 'radial'
            'drip_radius_scaling': True      # Scale drip size by intensity
        }

    def spectrum_to_ions(self, spectrum: Spectrum) -> List[Ion]:
        """Convert spectrum peaks to ion representations"""
        ions = []

        for mz, intensity in zip(spectrum.mz_array, spectrum.intensity_array):
            # Determine ion type based on m/z and intensity
            ion_type = self._classify_ion_type(mz, intensity, spectrum)

            # Estimate charge (simplified)
            charge = self._estimate_charge(mz, spectrum.polarity)

            # Create ion
            ion = Ion(
                mz=mz,
                intensity=intensity,
                charge=charge,
                ion_type=ion_type,
                metadata={
                    'spectrum_id': spectrum.scan_id,
                    'retention_time': spectrum.retention_time,
                    'polarity': spectrum.polarity
                }
            )
            ions.append(ion)

        return ions

    def _classify_ion_type(self, mz: float, intensity: float, spectrum: Spectrum) -> str:
        """Classify ion type based on m/z and intensity"""
        base_peak_mz, base_peak_intensity = spectrum.base_peak

        # Simple classification rules
        if abs(mz - base_peak_mz) < 1.0:
            return 'precursor'
        elif intensity > base_peak_intensity * 0.1:
            return 'fragment'
        else:
            return 'adduct'

    def _estimate_charge(self, mz: float, polarity: str) -> int:
        """Estimate charge state (simplified)"""
        # Very simplified charge estimation
        if polarity == 'positive':
            return 1 if mz < 600 else 2
        else:
            return -1 if mz < 600 else -2

    def ions_to_drip_spectrum(self, ions: List[Ion], spectrum_id: str) -> DripSpectrum:
        """Convert ions to drip spectrum representation"""
        if not ions:
            return self._create_empty_drip_spectrum(spectrum_id)

        # Generate drip coordinates
        drip_coordinates = self._generate_drip_coordinates(ions)

        # Extract intensities for drip representation
        drip_intensities = np.array([ion.intensity for ion in ions])

        # Normalize intensities
        if len(drip_intensities) > 0:
            drip_intensities = drip_intensities / np.max(drip_intensities)

        # Create drip image
        drip_image = self._create_drip_image(drip_coordinates, drip_intensities)

        # Create metadata
        drip_metadata = {
            'n_ions': len(ions),
            'ion_types': {ion_type: sum(1 for ion in ions if ion.ion_type == ion_type)
                         for ion_type in ['precursor', 'fragment', 'adduct']},
            'coordinate_method': self.conversion_params['coordinate_mapping'],
            'image_resolution': self.conversion_params['drip_resolution']
        }

        return DripSpectrum(
            spectrum_id=spectrum_id,
            original_spectrum=ions[0].metadata.get('spectrum') if ions else None,
            drip_coordinates=drip_coordinates,
            drip_intensities=drip_intensities,
            drip_image=drip_image,
            drip_metadata=drip_metadata
        )

    def _generate_drip_coordinates(self, ions: List[Ion]) -> np.ndarray:
        """Generate 2D coordinates for drip visualization"""
        n_ions = len(ions)
        coordinates = np.zeros((n_ions, 2))

        method = self.conversion_params['coordinate_mapping']

        if method == 'spiral':
            coordinates = self._spiral_coordinates(ions)
        elif method == 'grid':
            coordinates = self._grid_coordinates(ions)
        elif method == 'radial':
            coordinates = self._radial_coordinates(ions)
        else:
            # Default to spiral
            coordinates = self._spiral_coordinates(ions)

        return coordinates

    def _spiral_coordinates(self, ions: List[Ion]) -> np.ndarray:
        """Generate spiral coordinates based on m/z values"""
        coordinates = []

        # Sort ions by m/z for spiral arrangement
        sorted_ions = sorted(ions, key=lambda ion: ion.mz)

        for i, ion in enumerate(sorted_ions):
            # Spiral parameters
            angle = i * 0.5  # Angular step
            radius = np.sqrt(i) * 10  # Increasing radius

            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            coordinates.append([x, y])

        return np.array(coordinates)

    def _grid_coordinates(self, ions: List[Ion]) -> np.ndarray:
        """Generate grid coordinates"""
        n_ions = len(ions)
        grid_size = int(np.ceil(np.sqrt(n_ions)))

        coordinates = []
        for i in range(n_ions):
            row = i // grid_size
            col = i % grid_size

            x = col * 50  # Grid spacing
            y = row * 50

            coordinates.append([x, y])

        return np.array(coordinates)

    def _radial_coordinates(self, ions: List[Ion]) -> np.ndarray:
        """Generate radial coordinates based on intensity"""
        coordinates = []

        # Sort by intensity for radial arrangement
        sorted_ions = sorted(ions, key=lambda ion: ion.intensity, reverse=True)

        for i, ion in enumerate(sorted_ions):
            # Place high-intensity ions closer to center
            radius = 200 - (ion.intensity / np.max([ion.intensity for ion in ions])) * 150
            angle = i * (2 * np.pi / len(ions))

            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            coordinates.append([x, y])

        return np.array(coordinates)

    def _create_drip_image(self, coordinates: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        """Create visual drip image from coordinates and intensities"""
        height, width = self.conversion_params['drip_resolution']
        drip_image = np.zeros((height, width), dtype=np.float32)

        if len(coordinates) == 0:
            return drip_image

        # Normalize coordinates to image dimensions
        if coordinates.shape[0] > 0:
            # Scale coordinates to fit image
            x_coords = coordinates[:, 0]
            y_coords = coordinates[:, 1]

            # Normalize to [0, 1] range
            if np.max(x_coords) != np.min(x_coords):
                x_norm = (x_coords - np.min(x_coords)) / (np.max(x_coords) - np.min(x_coords))
            else:
                x_norm = np.ones_like(x_coords) * 0.5

            if np.max(y_coords) != np.min(y_coords):
                y_norm = (y_coords - np.min(y_coords)) / (np.max(y_coords) - np.min(y_coords))
            else:
                y_norm = np.ones_like(y_coords) * 0.5

            # Scale to image dimensions with margin
            margin = 0.1
            x_pixels = ((x_norm * (1 - 2*margin) + margin) * (width - 1)).astype(int)
            y_pixels = ((y_norm * (1 - 2*margin) + margin) * (height - 1)).astype(int)

            # Place drips in image
            for i, (x, y, intensity) in enumerate(zip(x_pixels, y_pixels, intensities)):
                # Determine drip radius based on intensity
                if self.conversion_params['drip_radius_scaling']:
                    radius = max(2, int(intensity * 20))  # Scale radius with intensity
                else:
                    radius = 5  # Fixed radius

                # Draw circle (drip) at position
                y_min, y_max = max(0, y - radius), min(height, y + radius + 1)
                x_min, x_max = max(0, x - radius), min(width, x + radius + 1)

                for dy in range(y_min, y_max):
                    for dx in range(x_min, x_max):
                        distance = np.sqrt((dx - x)**2 + (dy - y)**2)
                        if distance <= radius:
                            # Gaussian-like intensity falloff
                            value = intensity * np.exp(-(distance / radius)**2)
                            drip_image[dy, dx] = max(drip_image[dy, dx], value)

        return drip_image

    def _create_empty_drip_spectrum(self, spectrum_id: str) -> DripSpectrum:
        """Create empty drip spectrum for empty input"""
        height, width = self.conversion_params['drip_resolution']

        return DripSpectrum(
            spectrum_id=spectrum_id,
            original_spectrum=None,
            drip_coordinates=np.array([]),
            drip_intensities=np.array([]),
            drip_image=np.zeros((height, width)),
            drip_metadata={'n_ions': 0, 'empty': True}
        )


class LipidMapsAnnotator:
    """Standalone LipidMaps database search and annotation"""

    def __init__(self):
        self.lipid_database = self._initialize_lipidmaps_db()

    def _initialize_lipidmaps_db(self) -> Dict[str, Any]:
        """Initialize simplified LipidMaps database"""
        # Generate synthetic LipidMaps database
        lipid_classes = ['PC', 'PE', 'PS', 'PA', 'PG', 'PI', 'TG', 'DG', 'MG', 'CE', 'SM', 'LPC', 'LPE']

        compounds = []
        for i, lipid_class in enumerate(lipid_classes):
            for j in range(100):  # 100 compounds per class
                compound_id = f"LMGP{i:02d}{j:06d}"

                # Generate realistic lipid masses
                if lipid_class in ['PC', 'PE', 'PS', 'PA', 'PG', 'PI']:
                    base_mass = 700 + i * 50 + j * 2
                elif lipid_class in ['TG']:
                    base_mass = 800 + j * 3
                elif lipid_class in ['DG']:
                    base_mass = 600 + j * 2
                else:
                    base_mass = 500 + i * 30 + j * 1.5

                exact_mass = base_mass + np.random.normal(0, 10)

                compound = {
                    'id': compound_id,
                    'name': f"{lipid_class}({16 + j//20}:{j%4}/0:0)" if '/' in f"{lipid_class}_example" else f"{lipid_class}_{j}",
                    'lipid_class': lipid_class,
                    'exact_mass': max(200, exact_mass),
                    'formula': f"C{30+j}H{60+j*2}NO{4+i}P" if lipid_class in ['PC', 'PE'] else f"C{25+j}H{50+j}O{3+i}",
                    'retention_time_predicted': 8.0 + j * 0.05,
                    # Generate synthetic drip pattern for each compound
                    'drip_pattern': self._generate_compound_drip_pattern(compound_id, exact_mass)
                }
                compounds.append(compound)

        return {'compounds': compounds, 'search_tolerance': 0.01}

    def _generate_compound_drip_pattern(self, compound_id: str, mass: float) -> Dict[str, Any]:
        """Generate synthetic drip pattern for a compound"""
        # Create characteristic drip pattern based on compound properties
        n_drips = np.random.randint(5, 25)

        # Generate drip coordinates (normalized)
        drip_coords = []
        for i in range(n_drips):
            # Use mass to influence pattern
            angle = (hash(compound_id + str(i)) % 1000) / 1000.0 * 2 * np.pi
            radius = (mass % 100) / 100.0 * 0.8 + 0.1

            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            drip_coords.append([x, y])

        intensities = np.random.exponential(0.5, n_drips)
        intensities = intensities / np.max(intensities)  # Normalize

        return {
            'coordinates': drip_coords,
            'intensities': intensities.tolist(),
            'pattern_hash': hash(compound_id) % 10000
        }

    def search_by_mass(self, query_mass: float, tolerance: float = None) -> List[Dict[str, Any]]:
        """Search LipidMaps by mass"""
        if tolerance is None:
            tolerance = self.lipid_database['search_tolerance']

        matches = []
        for compound in self.lipid_database['compounds']:
            mass_diff = abs(compound['exact_mass'] - query_mass)
            if mass_diff <= tolerance:
                confidence = max(0.1, 1.0 - (mass_diff / tolerance))

                match = compound.copy()
                match['confidence_score'] = confidence
                match['mass_error_da'] = mass_diff
                match['mass_error_ppm'] = (mass_diff / query_mass) * 1e6
                matches.append(match)

        return sorted(matches, key=lambda x: x['confidence_score'], reverse=True)[:10]

    def annotate_drip_spectrum(self, drip_spectrum: DripSpectrum) -> List[VisualAnnotation]:
        """Annotate drip spectrum using LipidMaps database"""
        annotations = []

        # Get precursor ions from original spectrum
        if drip_spectrum.original_spectrum is not None:
            base_peak_mz, _ = drip_spectrum.original_spectrum.base_peak

            # Search database
            matches = self.search_by_mass(base_peak_mz)

            for match in matches[:5]:  # Top 5 matches
                # Convert compound drip pattern to drip spectrum
                compound_drip = self._compound_to_drip_spectrum(match)

                # Calculate similarities
                visual_similarity = self._calculate_visual_similarity(drip_spectrum, compound_drip)
                mathematical_similarity = self._calculate_mathematical_similarity(drip_spectrum, compound_drip)

                # Create overlay image
                overlay_image = self._create_overlay_image(drip_spectrum, compound_drip)

                annotation = VisualAnnotation(
                    compound_id=match['id'],
                    compound_name=match['name'],
                    visual_similarity_score=visual_similarity,
                    mathematical_similarity_score=mathematical_similarity,
                    overlay_image=overlay_image,
                    similarity_metrics={
                        'mass_confidence': match['confidence_score'],
                        'pattern_similarity': visual_similarity,
                        'coordinate_correlation': mathematical_similarity
                    }
                )
                annotations.append(annotation)

        return annotations

    def _compound_to_drip_spectrum(self, compound: Dict[str, Any]) -> DripSpectrum:
        """Convert compound drip pattern to DripSpectrum for comparison"""
        pattern = compound['drip_pattern']

        # Convert coordinates and intensities to arrays
        coordinates = np.array(pattern['coordinates'])
        intensities = np.array(pattern['intensities'])

        # Create drip image
        converter = IonDripConverter()
        drip_image = converter._create_drip_image(coordinates * 100, intensities)  # Scale coordinates

        return DripSpectrum(
            spectrum_id=f"compound_{compound['id']}",
            original_spectrum=None,
            drip_coordinates=coordinates,
            drip_intensities=intensities,
            drip_image=drip_image,
            drip_metadata={
                'compound_id': compound['id'],
                'compound_name': compound['name'],
                'is_reference': True
            }
        )

    def _calculate_visual_similarity(self, query_drip: DripSpectrum, reference_drip: DripSpectrum) -> float:
        """Calculate visual similarity between drip images"""
        if query_drip.drip_image is None or reference_drip.drip_image is None:
            return 0.0

        # Ensure same dimensions
        query_img = query_drip.drip_image
        ref_img = reference_drip.drip_image

        if query_img.shape != ref_img.shape:
            # Resize reference to match query
            if CV2_AVAILABLE:
                ref_img = cv2.resize(ref_img, (query_img.shape[1], query_img.shape[0]))
            else:
                # Simple nearest neighbor resize
                ref_img = self._simple_resize(ref_img, query_img.shape)

        # Calculate structural similarity (simplified SSIM)
        ssim_score = self._calculate_ssim(query_img, ref_img)

        return ssim_score

    def _calculate_mathematical_similarity(self, query_drip: DripSpectrum, reference_drip: DripSpectrum) -> float:
        """Calculate mathematical similarity between coordinate patterns"""
        if len(query_drip.drip_coordinates) == 0 or len(reference_drip.drip_coordinates) == 0:
            return 0.0

        # Calculate coordinate correlation
        query_coords = query_drip.drip_coordinates.flatten()
        ref_coords = reference_drip.drip_coordinates.flatten()

        # Pad shorter array to match lengths
        max_len = max(len(query_coords), len(ref_coords))
        query_padded = np.pad(query_coords, (0, max_len - len(query_coords)), 'constant')
        ref_padded = np.pad(ref_coords, (0, max_len - len(ref_coords)), 'constant')

        # Calculate Pearson correlation
        correlation = np.corrcoef(query_padded, ref_padded)[0, 1]

        # Handle NaN case
        if np.isnan(correlation):
            correlation = 0.0

        return abs(correlation)  # Return absolute correlation

    def _create_overlay_image(self, query_drip: DripSpectrum, reference_drip: DripSpectrum) -> np.ndarray:
        """Create overlay visualization of query and reference drip patterns"""
        if query_drip.drip_image is None:
            return reference_drip.drip_image if reference_drip.drip_image is not None else np.zeros((512, 512))

        if reference_drip.drip_image is None:
            return query_drip.drip_image

        # Ensure same dimensions
        query_img = query_drip.drip_image
        ref_img = reference_drip.drip_image

        if query_img.shape != ref_img.shape:
            if CV2_AVAILABLE:
                ref_img = cv2.resize(ref_img, (query_img.shape[1], query_img.shape[0]))
            else:
                ref_img = self._simple_resize(ref_img, query_img.shape)

        # Create RGB overlay (query=red, reference=green, overlap=yellow)
        overlay = np.zeros((*query_img.shape, 3))
        overlay[:, :, 0] = query_img  # Red channel for query
        overlay[:, :, 1] = ref_img    # Green channel for reference

        # Normalize to [0, 1]
        if np.max(overlay) > 0:
            overlay = overlay / np.max(overlay)

        return overlay

    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Simplified Structural Similarity Index calculation"""
        # Constants for SSIM
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Calculate means
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)

        # Calculate variances and covariance
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

        # SSIM formula
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim = numerator / max(denominator, 1e-10)  # Avoid division by zero

        return max(0, min(1, ssim))  # Clamp to [0, 1]

    def _simple_resize(self, img: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Simple nearest-neighbor resize when OpenCV is not available"""
        target_height, target_width = target_shape
        current_height, current_width = img.shape

        resized = np.zeros((target_height, target_width))

        for i in range(target_height):
            for j in range(target_width):
                # Map to original image coordinates
                orig_i = int(i * current_height / target_height)
                orig_j = int(j * current_width / target_width)

                orig_i = min(orig_i, current_height - 1)
                orig_j = min(orig_j, current_width - 1)

                resized[i, j] = img[orig_i, orig_j]

        return resized


class VisualPipelineOrchestrator:
    """Main visual pipeline orchestrator"""

    def __init__(self):
        self.mzml_reader = StandaloneMzMLReader()
        self.ion_drip_converter = IonDripConverter()
        self.lipidmaps_annotator = LipidMapsAnnotator()

    def process_dataset(self, mzml_filepath: str,
                       create_visualizations: bool = True,
                       save_drip_images: bool = False) -> Dict[str, Any]:
        """
        Complete visual processing pipeline

        Args:
            mzml_filepath: Path to mzML file
            create_visualizations: Whether to create drip visualizations
            save_drip_images: Whether to save drip images to files

        Returns:
            Complete processing results
        """
        start_time = time.time()

        print(f"Starting visual processing of {mzml_filepath}")

        # Step 1: Load mzML file (extracts polarity from filename)
        print("Step 1: Loading mzML file...")
        spectra = self.mzml_reader.load_mzml(mzml_filepath)
        dataset_summary = self.mzml_reader.get_dataset_summary(spectra)

        # Step 2: Convert spectra to ions and then to drip spectra
        print("Step 2: Converting spectra to ions and drip spectra...")
        drip_spectra = []
        ion_conversion_stats = {
            'total_spectra_processed': 0,
            'total_ions_extracted': 0,
            'drip_spectra_created': 0,
            'ion_type_distribution': defaultdict(int)
        }

        # Process subset for demonstration (limit for performance)
        processing_limit = min(50, len(spectra))

        for i, spectrum in enumerate(spectra[:processing_limit]):
            # Convert spectrum to ions
            ions = self.ion_drip_converter.spectrum_to_ions(spectrum)

            # Convert ions to drip spectrum
            drip_spectrum = self.ion_drip_converter.ions_to_drip_spectrum(ions, spectrum.scan_id)

            drip_spectra.append(drip_spectrum)

            # Update statistics
            ion_conversion_stats['total_spectra_processed'] += 1
            ion_conversion_stats['total_ions_extracted'] += len(ions)
            ion_conversion_stats['drip_spectra_created'] += 1

            for ion in ions:
                ion_conversion_stats['ion_type_distribution'][ion.ion_type] += 1

        # Step 3: Perform LipidMaps annotation with drip overlay
        print("Step 3: LipidMaps annotation and drip overlay...")
        annotation_results = {}

        # Annotate subset of drip spectra
        annotation_limit = min(10, len(drip_spectra))

        for drip_spectrum in drip_spectra[:annotation_limit]:
            annotations = self.lipidmaps_annotator.annotate_drip_spectrum(drip_spectrum)
            if annotations:
                annotation_results[drip_spectrum.spectrum_id] = annotations

        # Calculate similarity metrics summary
        similarity_stats = self._calculate_similarity_statistics(annotation_results)

        processing_time = time.time() - start_time

        # Compile results
        results = {
            'pipeline_info': {
                'input_file': mzml_filepath,
                'processing_time': processing_time,
                'create_visualizations': create_visualizations,
                'save_drip_images': save_drip_images
            },
            'dataset_summary': dataset_summary,
            'ion_conversion': {
                'statistics': dict(ion_conversion_stats),
                'conversion_parameters': self.ion_drip_converter.conversion_params,
                'drip_spectra_count': len(drip_spectra)
            },
            'lipidmaps_annotation': {
                'annotated_spectra': len(annotation_results),
                'total_annotations': sum(len(annotations) for annotations in annotation_results.values()),
                'similarity_statistics': similarity_stats,
                'sample_annotations': self._get_sample_annotations(annotation_results)
            },
            'visual_processing_summary': {
                'spectra_processed': ion_conversion_stats['total_spectra_processed'],
                'ions_extracted': ion_conversion_stats['total_ions_extracted'],
                'drip_images_created': len(drip_spectra),
                'annotations_generated': len(annotation_results)
            }
        }

        print(f"Visual processing completed in {processing_time:.2f} seconds")

        return results

    def _calculate_similarity_statistics(self, annotation_results: Dict[str, List[VisualAnnotation]]) -> Dict[str, Any]:
        """Calculate summary statistics for similarity scores"""
        visual_scores = []
        mathematical_scores = []

        for annotations in annotation_results.values():
            for annotation in annotations:
                visual_scores.append(annotation.visual_similarity_score)
                mathematical_scores.append(annotation.mathematical_similarity_score)

        if not visual_scores:
            return {'error': 'No similarity scores available'}

        stats = {
            'visual_similarity': {
                'mean': np.mean(visual_scores),
                'std': np.std(visual_scores),
                'max': np.max(visual_scores),
                'min': np.min(visual_scores),
                'high_similarity_count': sum(1 for score in visual_scores if score > 0.8)
            },
            'mathematical_similarity': {
                'mean': np.mean(mathematical_scores),
                'std': np.std(mathematical_scores),
                'max': np.max(mathematical_scores),
                'min': np.min(mathematical_scores),
                'high_similarity_count': sum(1 for score in mathematical_scores if score > 0.8)
            },
            'overall_metrics': {
                'total_comparisons': len(visual_scores),
                'avg_combined_similarity': np.mean([(v + m) / 2 for v, m in zip(visual_scores, mathematical_scores)])
            }
        }

        return stats

    def _get_sample_annotations(self, annotation_results: Dict[str, List[VisualAnnotation]]) -> Dict[str, Any]:
        """Get sample annotations for results preview"""
        sample_annotations = {}

        for spectrum_id, annotations in list(annotation_results.items())[:3]:  # First 3 spectra
            sample_annotations[spectrum_id] = [
                {
                    'compound_id': ann.compound_id,
                    'compound_name': ann.compound_name,
                    'visual_similarity': ann.visual_similarity_score,
                    'mathematical_similarity': ann.mathematical_similarity_score
                }
                for ann in annotations[:2]  # Top 2 annotations per spectrum
            ]

        return sample_annotations


# Convenience functions for validation framework
def create_visual_validator() -> VisualPipelineOrchestrator:
    """Create a visual pipeline validator"""
    return VisualPipelineOrchestrator()


def process_mzml_visual(filepath: str, create_visualizations: bool = True) -> Dict[str, Any]:
    """
    Process mzML file through visual pipeline

    Args:
        filepath: Path to mzML file
        create_visualizations: Whether to create visual outputs

    Returns:
        Processing results dictionary
    """
    orchestrator = VisualPipelineOrchestrator()
    return orchestrator.process_dataset(filepath, create_visualizations=create_visualizations)


if __name__ == "__main__":
    # Test the visual pipeline
    test_files = ["PL_Neg_Waters_qTOF.mzML", "TG_Pos_Thermo_Orbi.mzML"]

    orchestrator = VisualPipelineOrchestrator()

    for test_file in test_files:
        print(f"\n{'='*60}")
        print(f"Testing visual pipeline with: {test_file}")
        print('='*60)

        results = orchestrator.process_dataset(test_file, create_visualizations=True)

        # Print summary
        print(f"\nProcessing Summary:")
        print(f"Processing time: {results['pipeline_info']['processing_time']:.2f} seconds")
        print(f"Input spectra: {results['dataset_summary']['total_spectra']}")
        print(f"Spectra processed: {results['visual_processing_summary']['spectra_processed']}")
        print(f"Ions extracted: {results['visual_processing_summary']['ions_extracted']}")
        print(f"Drip images created: {results['visual_processing_summary']['drip_images_created']}")
        print(f"Annotated spectra: {results['visual_processing_summary']['annotations_generated']}")

        # Ion conversion stats
        ion_stats = results['ion_conversion']['statistics']
        print(f"\nIon Type Distribution:")
        for ion_type, count in ion_stats['ion_type_distribution'].items():
            print(f"  {ion_type}: {count}")

        # Similarity statistics
        sim_stats = results['lipidmaps_annotation']['similarity_statistics']
        if 'error' not in sim_stats:
            print(f"\nSimilarity Statistics:")
            print(f"Visual similarity (mean): {sim_stats['visual_similarity']['mean']:.3f}")
            print(f"Mathematical similarity (mean): {sim_stats['mathematical_similarity']['mean']:.3f}")
            print(f"High visual similarity matches: {sim_stats['visual_similarity']['high_similarity_count']}")
            print(f"High mathematical similarity matches: {sim_stats['mathematical_similarity']['high_similarity_count']}")

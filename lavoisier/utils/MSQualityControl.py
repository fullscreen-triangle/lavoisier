from typing import Tuple, Dict, List, Any

import cv2
import numpy as np



class MSQualityControl:
    def __init__(self, resolution: Tuple[int, int] = (1024, 1024)):
        self.resolution = resolution
        self.reference_patterns = {}
        self.mass_balance_thresholds = {
            'shift_tolerance': 0.002,  # Da
            'intensity_variance': 0.1,  # 10%
            'isotope_ratio_tolerance': 0.05  # 5%
        }

    def detect_mass_shifts(self,
                           reference_image: np.ndarray,
                           test_image: np.ndarray) -> Dict[str, float]:
        """
        Detect mass shifts using image registration techniques
        Returns shift metrics in both x (m/z) and y (intensity) directions
        """
        # Calculate optical flow between reference and test
        flow = cv2.calcOpticalFlowFarneback(
            reference_image.astype(np.uint8),
            test_image.astype(np.uint8),
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Analyze flow patterns
        x_shift = np.median(flow[..., 0])  # m/z shift
        y_shift = np.median(flow[..., 1])  # intensity shift

        # Convert pixel shifts to mass units
        mz_per_pixel = (1000 - 100) / self.resolution[0]  # example range
        mass_shift = x_shift * mz_per_pixel

        return {
            'mass_shift_da': mass_shift,
            'intensity_shift_percent': (y_shift / self.resolution[1]) * 100,
            'flow_consistency': np.std(flow[..., 0])  # measure of shift consistency
        }

    def detect_contaminants(self,
                            clean_reference: np.ndarray,
                            test_image: np.ndarray) -> Dict[str, Any]:
        """
        Detect potential contaminants by image difference analysis
        """
        # Normalize images
        ref_norm = cv2.normalize(clean_reference, None, 0, 255, cv2.NORM_MINMAX)
        test_norm = cv2.normalize(test_image, None, 0, 255, cv2.NORM_MINMAX)

        # Calculate difference image
        diff = cv2.absdiff(ref_norm, test_norm)

        # Threshold to find significant differences
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Find contaminant regions
        contours, _ = cv2.findContours(thresh.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        contaminant_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Convert pixel coordinates to m/z ranges
            mz_start = (x / self.resolution[0]) * 900 + 100  # example range
            mz_end = ((x + w) / self.resolution[0]) * 900 + 100

            contaminant_regions.append({
                'mz_range': (mz_start, mz_end),
                'intensity': np.mean(test_image[y:y + h, x:x + w]),
                'area': cv2.contourArea(contour)
            })

        return {
            'num_contaminants': len(contaminant_regions),
            'contaminant_regions': contaminant_regions,
            'total_contamination_area': sum(c['area'] for c in contaminant_regions),
            'difference_map': diff
        }

    def mass_balance_test(self,
                          spectrum_image: np.ndarray,
                          expected_mass: float,
                          tolerance: float = 0.01) -> Dict[str, float]:
        """
        Perform mass balance test using image moments
        """
        # Calculate image moments
        moments = cv2.moments(spectrum_image)

        # Center of mass in x-direction (m/z)
        if moments['m00'] != 0:
            center_x = moments['m10'] / moments['m00']
        else:
            center_x = 0

        # Convert to mass units
        measured_mass = (center_x / self.resolution[0]) * 900 + 100  # example range

        # Calculate mass balance metrics
        mass_error = abs(measured_mass - expected_mass)
        mass_error_ppm = (mass_error / expected_mass) * 1e6

        # Analyze peak shape
        symmetry = moments['mu11'] / moments['mu02'] if moments['mu02'] != 0 else 0

        return {
            'mass_error_da': mass_error,
            'mass_error_ppm': mass_error_ppm,
            'peak_symmetry': symmetry,
            'is_within_tolerance': mass_error <= tolerance
        }

    def isotope_pattern_analysis(self,
                                 spectrum_image: np.ndarray,
                                 theoretical_pattern: Dict[float, float]) -> Dict[str, float]:
        """
        Analyze isotope patterns using image processing
        """
        # Extract peaks from image
        peaks = self._extract_peaks_from_image(spectrum_image)

        # Compare with theoretical pattern
        pattern_similarity = self._compare_isotope_patterns(
            peaks,
            theoretical_pattern
        )

        return {
            'pattern_similarity': pattern_similarity,
            'detected_peaks': peaks,
            'isotope_ratio_errors': self._calculate_ratio_errors(
                peaks,
                theoretical_pattern
            )
        }

    def _extract_peaks_from_image(self,
                                  spectrum_image: np.ndarray) -> Dict[float, float]:
        """Extract peaks from spectrum image"""
        # Use image processing to find local maxima
        local_max = cv2.dilate(spectrum_image, None) == spectrum_image

        peaks = {}
        for y, x in zip(*np.where(local_max)):
            mz = (x / self.resolution[0]) * 900 + 100  # example range
            intensity = spectrum_image[y, x]
            peaks[mz] = intensity

        return peaks

    def batch_quality_control(self,
                              reference_image: np.ndarray,
                              test_images: List[np.ndarray],
                              expected_mass: float) -> List[Dict]:
        """
        Perform comprehensive quality control on a batch of spectra
        """
        results = []
        for test_image in test_images:
            # Mass shifts
            shifts = self.detect_mass_shifts(reference_image, test_image)

            # Contaminants
            contaminants = self.detect_contaminants(reference_image, test_image)

            # Mass balance
            balance = self.mass_balance_test(test_image, expected_mass)

            results.append({
                'shifts': shifts,
                'contaminants': contaminants,
                'mass_balance': balance,
                'overall_quality_score': self._calculate_quality_score(
                    shifts, contaminants, balance
                )
            })

        return results

    def _calculate_quality_score(self,
                                 shifts: Dict,
                                 contaminants: Dict,
                                 balance: Dict) -> float:
        """Calculate overall quality score"""
        score = 100.0

        # Penalize mass shifts
        score -= abs(shifts['mass_shift_da']) * 50

        # Penalize contaminants
        score -= contaminants['num_contaminants'] * 10

        # Penalize mass balance errors
        score -= balance['mass_error_ppm'] * 0.1

        return max(0, score)

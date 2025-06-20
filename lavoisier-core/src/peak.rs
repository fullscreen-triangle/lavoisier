use serde::{Deserialize, Serialize};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;

/// Peak data structure for mass spectrometry peaks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Peak {
    pub mz: f64,
    pub intensity: f64,
    pub retention_time: f64,
    pub peak_width: Option<f64>,
    pub area: Option<f64>,
    pub signal_to_noise: Option<f64>,
}

impl Peak {
    pub fn new(mz: f64, intensity: f64, retention_time: f64) -> Self {
        Self {
            mz,
            intensity,
            retention_time,
            peak_width: None,
            area: None,
            signal_to_noise: None,
        }
    }

    /// Calculate peak area using trapezoidal integration
    pub fn calculate_area(&mut self, mz_values: &[f64], intensity_values: &[f64], tolerance: f64) {
        // Find peak boundaries within tolerance
        let start_idx = mz_values
            .iter()
            .position(|&mz| (mz - self.mz).abs() <= tolerance)
            .unwrap_or(0);
        
        let end_idx = mz_values
            .iter()
            .rposition(|&mz| (mz - self.mz).abs() <= tolerance)
            .unwrap_or(mz_values.len() - 1);

        if start_idx < end_idx {
            let mut area = 0.0;
            for i in start_idx..end_idx {
                let width = mz_values[i + 1] - mz_values[i];
                let avg_height = (intensity_values[i] + intensity_values[i + 1]) / 2.0;
                area += width * avg_height;
            }
            self.area = Some(area);
            self.peak_width = Some(mz_values[end_idx] - mz_values[start_idx]);
        }
    }

    /// Calculate signal-to-noise ratio
    pub fn calculate_signal_to_noise(&mut self, noise_level: f64) {
        if noise_level > 0.0 {
            self.signal_to_noise = Some(self.intensity / noise_level);
        }
    }
}

impl PartialEq for Peak {
    fn eq(&self, other: &Self) -> bool {
        (self.mz - other.mz).abs() < f64::EPSILON
            && (self.intensity - other.intensity).abs() < f64::EPSILON
            && (self.retention_time - other.retention_time).abs() < f64::EPSILON
    }
}

impl PartialOrd for Peak {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.intensity.partial_cmp(&other.intensity)
    }
}

/// Peak detection algorithms
pub struct PeakDetector {
    min_intensity: f64,
    min_signal_to_noise: f64,
    window_size: usize,
}

impl PeakDetector {
    pub fn new(min_intensity: f64, min_signal_to_noise: f64, window_size: usize) -> Self {
        Self {
            min_intensity,
            min_signal_to_noise,
            window_size,
        }
    }

    /// Detect peaks using local maxima algorithm with noise estimation
    pub fn detect_peaks(&self, mz: &[f64], intensity: &[f64], retention_time: f64) -> Vec<Peak> {
        if mz.len() != intensity.len() || mz.len() < self.window_size {
            return Vec::new();
        }

        // Estimate noise level using median absolute deviation
        let noise_level = self.estimate_noise_level(intensity);
        let effective_threshold = self.min_intensity.max(noise_level * self.min_signal_to_noise);

        let mut peaks = Vec::new();
        let half_window = self.window_size / 2;

        for i in half_window..intensity.len() - half_window {
            if intensity[i] < effective_threshold {
                continue;
            }

            // Check if this is a local maximum
            let is_peak = (i.saturating_sub(half_window)..i)
                .chain((i + 1)..=(i + half_window).min(intensity.len() - 1))
                .all(|j| intensity[i] >= intensity[j]);

            if is_peak {
                let mut peak = Peak::new(mz[i], intensity[i], retention_time);
                peak.calculate_signal_to_noise(noise_level);
                
                if let Some(snr) = peak.signal_to_noise {
                    if snr >= self.min_signal_to_noise {
                        peak.calculate_area(mz, intensity, 0.01); // 0.01 Da tolerance
                        peaks.push(peak);
                    }
                }
            }
        }

        // Sort peaks by intensity (descending)
        peaks.sort_by(|a, b| b.intensity.partial_cmp(&a.intensity).unwrap());
        peaks
    }

    /// Estimate noise level using median absolute deviation
    fn estimate_noise_level(&self, intensity: &[f64]) -> f64 {
        let mut sorted_intensity = intensity.to_vec();
        sorted_intensity.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let median_idx = sorted_intensity.len() / 2;
        let median = if sorted_intensity.len() % 2 == 0 {
            (sorted_intensity[median_idx - 1] + sorted_intensity[median_idx]) / 2.0
        } else {
            sorted_intensity[median_idx]
        };

        // Calculate median absolute deviation
        let mut deviations: Vec<f64> = intensity
            .iter()
            .map(|&x| (x - median).abs())
            .collect();
        
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mad_idx = deviations.len() / 2;
        let mad = if deviations.len() % 2 == 0 {
            (deviations[mad_idx - 1] + deviations[mad_idx]) / 2.0
        } else {
            deviations[mad_idx]
        };

        // Convert MAD to standard deviation estimate
        mad * 1.4826
    }

    /// Detect peaks with centroiding for improved mass accuracy
    pub fn detect_peaks_with_centroiding(&self, mz: &[f64], intensity: &[f64], retention_time: f64) -> Vec<Peak> {
        let mut raw_peaks = self.detect_peaks(mz, intensity, retention_time);
        
        // Apply centroiding to improve mass accuracy
        for peak in &mut raw_peaks {
            if let Some(centroid) = self.calculate_centroid(mz, intensity, peak.mz, 0.01) {
                peak.mz = centroid.0;
                peak.intensity = centroid.1;
            }
        }
        
        raw_peaks
    }

    /// Calculate centroid for improved mass accuracy
    fn calculate_centroid(&self, mz: &[f64], intensity: &[f64], peak_mz: f64, tolerance: f64) -> Option<(f64, f64)> {
        let mut weighted_mz_sum = 0.0;
        let mut intensity_sum = 0.0;
        let mut max_intensity = 0.0;

        for (i, &current_mz) in mz.iter().enumerate() {
            if (current_mz - peak_mz).abs() <= tolerance {
                weighted_mz_sum += current_mz * intensity[i];
                intensity_sum += intensity[i];
                max_intensity = max_intensity.max(intensity[i]);
            }
        }

        if intensity_sum > 0.0 {
            Some((weighted_mz_sum / intensity_sum, max_intensity))
        } else {
            None
        }
    }
}

/// Parallel peak detection for multiple spectra
pub fn detect_peaks_parallel(
    spectra_data: &[(Vec<f64>, Vec<f64>, f64)], // (mz, intensity, rt)
    detector: &PeakDetector,
) -> Vec<Vec<Peak>> {
    spectra_data
        .par_iter()
        .map(|(mz, intensity, rt)| detector.detect_peaks_with_centroiding(mz, intensity, *rt))
        .collect()
}

/// Peak matching between different spectra
pub struct PeakMatcher {
    mz_tolerance: f64,
    rt_tolerance: f64,
}

impl PeakMatcher {
    pub fn new(mz_tolerance: f64, rt_tolerance: f64) -> Self {
        Self {
            mz_tolerance,
            rt_tolerance,
        }
    }

    /// Match peaks between two peak lists
    pub fn match_peaks(&self, peaks1: &[Peak], peaks2: &[Peak]) -> Vec<(usize, usize, f64)> {
        let mut matches = Vec::new();

        for (i, peak1) in peaks1.iter().enumerate() {
            for (j, peak2) in peaks2.iter().enumerate() {
                let mz_diff = (peak1.mz - peak2.mz).abs();
                let rt_diff = (peak1.retention_time - peak2.retention_time).abs();

                if mz_diff <= self.mz_tolerance && rt_diff <= self.rt_tolerance {
                    let score = 1.0 / (1.0 + mz_diff + rt_diff);
                    matches.push((i, j, score));
                }
            }
        }

        // Sort matches by score (descending)
        matches.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        matches
    }
}

// Python bindings
#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct PyPeak {
    pub inner: Peak,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyPeak {
    #[new]
    fn new(mz: f64, intensity: f64, retention_time: f64) -> Self {
        Self {
            inner: Peak::new(mz, intensity, retention_time),
        }
    }

    #[getter]
    fn mz(&self) -> f64 {
        self.inner.mz
    }

    #[getter]
    fn intensity(&self) -> f64 {
        self.inner.intensity
    }

    #[getter]
    fn retention_time(&self) -> f64 {
        self.inner.retention_time
    }

    #[getter]
    fn peak_width(&self) -> Option<f64> {
        self.inner.peak_width
    }

    #[getter]
    fn area(&self) -> Option<f64> {
        self.inner.area
    }

    #[getter]
    fn signal_to_noise(&self) -> Option<f64> {
        self.inner.signal_to_noise
    }

    fn __repr__(&self) -> String {
        format!(
            "Peak(mz={:.4}, intensity={:.2}, rt={:.2})",
            self.inner.mz, self.inner.intensity, self.inner.retention_time
        )
    }
}

#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct PyPeakDetector {
    inner: PeakDetector,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyPeakDetector {
    #[new]
    fn new(min_intensity: f64, min_signal_to_noise: f64, window_size: usize) -> Self {
        Self {
            inner: PeakDetector::new(min_intensity, min_signal_to_noise, window_size),
        }
    }

    fn detect_peaks(
        &self,
        mz: Vec<f64>,
        intensity: Vec<f64>,
        retention_time: f64,
    ) -> Vec<PyPeak> {
        self.inner
            .detect_peaks_with_centroiding(&mz, &intensity, retention_time)
            .into_iter()
            .map(|peak| PyPeak { inner: peak })
            .collect()
    }
} 
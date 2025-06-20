// Additional spectrum processing functions
// The main Spectrum struct is defined in lib.rs

use crate::{Spectrum, Peak, NormalizationMethod};
use rayon::prelude::*;
use std::collections::HashMap;

/// Spectrum processing utilities
pub struct SpectrumProcessor;

impl SpectrumProcessor {
    /// Smooth spectrum using moving average
    pub fn smooth_moving_average(spectrum: &mut Spectrum, window_size: usize) {
        if spectrum.intensity.len() < window_size {
            return;
        }

        let half_window = window_size / 2;
        let mut smoothed = Vec::with_capacity(spectrum.intensity.len());

        for i in 0..spectrum.intensity.len() {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(spectrum.intensity.len());
            
            let sum: f64 = spectrum.intensity[start..end].iter().sum();
            let count = end - start;
            smoothed.push(sum / count as f64);
        }

        spectrum.intensity = smoothed;
    }

    /// Smooth spectrum using Gaussian filter
    pub fn smooth_gaussian(spectrum: &mut Spectrum, sigma: f64) {
        let kernel_size = (6.0 * sigma).ceil() as usize;
        if kernel_size >= spectrum.intensity.len() {
            return;
        }

        let kernel = Self::gaussian_kernel(kernel_size, sigma);
        let mut smoothed = vec![0.0; spectrum.intensity.len()];

        for i in 0..spectrum.intensity.len() {
            let mut sum = 0.0;
            let mut weight_sum = 0.0;

            for (j, &weight) in kernel.iter().enumerate() {
                let idx = i as i32 + j as i32 - kernel_size as i32 / 2;
                if idx >= 0 && idx < spectrum.intensity.len() as i32 {
                    sum += spectrum.intensity[idx as usize] * weight;
                    weight_sum += weight;
                }
            }

            if weight_sum > 0.0 {
                smoothed[i] = sum / weight_sum;
            }
        }

        spectrum.intensity = smoothed;
    }

    /// Generate Gaussian kernel
    fn gaussian_kernel(size: usize, sigma: f64) -> Vec<f64> {
        let mut kernel = Vec::with_capacity(size);
        let center = size as f64 / 2.0;
        let factor = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());

        for i in 0..size {
            let x = i as f64 - center;
            let value = factor * (-0.5 * (x / sigma).powi(2)).exp();
            kernel.push(value);
        }

        kernel
    }

    /// Remove baseline using rolling ball algorithm
    pub fn remove_baseline_rolling_ball(spectrum: &mut Spectrum, radius: usize) {
        if spectrum.intensity.len() < radius * 2 {
            return;
        }

        let baseline = Self::calculate_rolling_ball_baseline(&spectrum.intensity, radius);
        
        for (i, &baseline_val) in baseline.iter().enumerate() {
            spectrum.intensity[i] = (spectrum.intensity[i] - baseline_val).max(0.0);
        }
    }

    /// Calculate rolling ball baseline
    fn calculate_rolling_ball_baseline(intensity: &[f64], radius: usize) -> Vec<f64> {
        let mut baseline = Vec::with_capacity(intensity.len());
        
        for i in 0..intensity.len() {
            let start = i.saturating_sub(radius);
            let end = (i + radius + 1).min(intensity.len());
            
            let min_val = intensity[start..end].iter().cloned().fold(f64::INFINITY, f64::min);
            baseline.push(min_val);
        }
        
        baseline
    }

    /// Resample spectrum to uniform m/z grid
    pub fn resample_uniform(spectrum: &mut Spectrum, min_mz: f64, max_mz: f64, num_points: usize) {
        if num_points == 0 || spectrum.mz.is_empty() {
            return;
        }

        let step = (max_mz - min_mz) / (num_points - 1) as f64;
        let mut new_mz = Vec::with_capacity(num_points);
        let mut new_intensity = Vec::with_capacity(num_points);

        for i in 0..num_points {
            let target_mz = min_mz + i as f64 * step;
            new_mz.push(target_mz);
            
            // Linear interpolation
            let intensity = Self::interpolate_linear(&spectrum.mz, &spectrum.intensity, target_mz);
            new_intensity.push(intensity);
        }

        spectrum.mz = new_mz;
        spectrum.intensity = new_intensity;
    }

    /// Linear interpolation
    fn interpolate_linear(x: &[f64], y: &[f64], target: f64) -> f64 {
        if x.is_empty() || y.is_empty() {
            return 0.0;
        }

        // Find surrounding points
        let mut left_idx = 0;
        let mut right_idx = x.len() - 1;

        for i in 0..x.len() - 1 {
            if x[i] <= target && target <= x[i + 1] {
                left_idx = i;
                right_idx = i + 1;
                break;
            }
        }

        if left_idx == right_idx {
            return y[left_idx];
        }

        let x1 = x[left_idx];
        let x2 = x[right_idx];
        let y1 = y[left_idx];
        let y2 = y[right_idx];

        if (x2 - x1).abs() < f64::EPSILON {
            return y1;
        }

        y1 + (y2 - y1) * (target - x1) / (x2 - x1)
    }

    /// Calculate spectrum similarity using cosine similarity
    pub fn cosine_similarity(spec1: &Spectrum, spec2: &Spectrum, tolerance: f64) -> f64 {
        let aligned = Self::align_spectra(spec1, spec2, tolerance);
        
        let mut dot_product = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        for (int1, int2) in aligned {
            dot_product += int1 * int2;
            norm1 += int1 * int1;
            norm2 += int2 * int2;
        }

        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }

        dot_product / (norm1.sqrt() * norm2.sqrt())
    }

    /// Align two spectra for comparison
    fn align_spectra(spec1: &Spectrum, spec2: &Spectrum, tolerance: f64) -> Vec<(f64, f64)> {
        let mut aligned = Vec::new();
        let mut used_indices = vec![false; spec2.mz.len()];

        for (i, &mz1) in spec1.mz.iter().enumerate() {
            let mut best_match = None;
            let mut best_distance = tolerance;

            for (j, &mz2) in spec2.mz.iter().enumerate() {
                if used_indices[j] {
                    continue;
                }

                let distance = (mz1 - mz2).abs();
                if distance <= best_distance {
                    best_distance = distance;
                    best_match = Some(j);
                }
            }

            if let Some(j) = best_match {
                aligned.push((spec1.intensity[i], spec2.intensity[j]));
                used_indices[j] = true;
            } else {
                aligned.push((spec1.intensity[i], 0.0));
            }
        }

        aligned
    }

    /// Calculate spectrum quality metrics
    pub fn calculate_quality_metrics(spectrum: &Spectrum) -> SpectrumQualityMetrics {
        let mut metrics = SpectrumQualityMetrics::default();

        if spectrum.intensity.is_empty() {
            return metrics;
        }

        // Basic statistics
        metrics.total_ion_current = spectrum.intensity.iter().sum();
        metrics.base_peak_intensity = spectrum.intensity.iter().cloned().fold(0.0, f64::max);
        metrics.mean_intensity = metrics.total_ion_current / spectrum.intensity.len() as f64;
        
        // Find base peak m/z
        if let Some(max_idx) = spectrum.intensity.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
        {
            metrics.base_peak_mz = spectrum.mz[max_idx];
        }

        // Signal-to-noise estimation
        let sorted_intensities: Vec<f64> = {
            let mut intensities = spectrum.intensity.clone();
            intensities.sort_by(|a, b| a.partial_cmp(b).unwrap());
            intensities
        };

        let median_idx = sorted_intensities.len() / 2;
        let noise_level = if sorted_intensities.len() % 2 == 0 {
            (sorted_intensities[median_idx - 1] + sorted_intensities[median_idx]) / 2.0
        } else {
            sorted_intensities[median_idx]
        };

        metrics.signal_to_noise = if noise_level > 0.0 {
            metrics.base_peak_intensity / noise_level
        } else {
            f64::INFINITY
        };

        // Peak count estimation
        metrics.estimated_peak_count = spectrum.intensity.iter()
            .filter(|&&intensity| intensity > noise_level * 3.0)
            .count();

        // Mass range
        if !spectrum.mz.is_empty() {
            metrics.mz_range_start = spectrum.mz.iter().cloned().fold(f64::INFINITY, f64::min);
            metrics.mz_range_end = spectrum.mz.iter().cloned().fold(0.0, f64::max);
        }

        metrics
    }
}

/// Spectrum quality metrics
#[derive(Debug, Clone, Default)]
pub struct SpectrumQualityMetrics {
    pub total_ion_current: f64,
    pub base_peak_intensity: f64,
    pub base_peak_mz: f64,
    pub mean_intensity: f64,
    pub signal_to_noise: f64,
    pub estimated_peak_count: usize,
    pub mz_range_start: f64,
    pub mz_range_end: f64,
}

/// Batch spectrum processing
pub fn process_spectra_parallel<F>(spectra: &mut [Spectrum], mut processor: F) 
where
    F: Fn(&mut Spectrum) + Send + Sync,
{
    spectra.par_iter_mut().for_each(|spectrum| {
        processor(spectrum);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moving_average_smoothing() {
        let mut spectrum = Spectrum::new(
            vec![100.0, 200.0, 300.0, 400.0, 500.0],
            vec![10.0, 20.0, 30.0, 20.0, 10.0],
            1.0,
            1,
            "test_scan".to_string(),
        );

        SpectrumProcessor::smooth_moving_average(&mut spectrum, 3);
        
        // Check that smoothing was applied
        assert_ne!(spectrum.intensity[2], 30.0); // Should be smoothed
    }

    #[test]
    fn test_baseline_removal() {
        let mut spectrum = Spectrum::new(
            vec![100.0, 200.0, 300.0, 400.0, 500.0],
            vec![15.0, 25.0, 35.0, 25.0, 15.0],
            1.0,
            1,
            "test_scan".to_string(),
        );

        SpectrumProcessor::remove_baseline_rolling_ball(&mut spectrum, 2);
        
        // Check that baseline was removed
        assert!(spectrum.intensity.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_quality_metrics() {
        let spectrum = Spectrum::new(
            vec![100.0, 200.0, 300.0],
            vec![1000.0, 2000.0, 1500.0],
            1.0,
            1,
            "test_scan".to_string(),
        );

        let metrics = SpectrumProcessor::calculate_quality_metrics(&spectrum);
        
        assert_eq!(metrics.total_ion_current, 4500.0);
        assert_eq!(metrics.base_peak_intensity, 2000.0);
        assert_eq!(metrics.base_peak_mz, 200.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let spec1 = Spectrum::new(
            vec![100.0, 200.0, 300.0],
            vec![10.0, 20.0, 30.0],
            1.0,
            1,
            "spec1".to_string(),
        );

        let spec2 = Spectrum::new(
            vec![100.0, 200.0, 300.0],
            vec![10.0, 20.0, 30.0],
            1.0,
            1,
            "spec2".to_string(),
        );

        let similarity = SpectrumProcessor::cosine_similarity(&spec1, &spec2, 0.01);
        assert!((similarity - 1.0).abs() < 1e-10); // Should be identical
    }
} 
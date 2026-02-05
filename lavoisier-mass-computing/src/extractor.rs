//! Spectrum extraction from S-entropy coordinates
//!
//! This module implements the observable extraction functions that map
//! S-coordinates to mass spectrometric observables:
//! - Mass-to-charge ratio (m/z)
//! - Retention time
//! - Fragment ions
//! - Isotope patterns

use crate::errors::Result;
use crate::scoord::SEntropyCoord;
use crate::ternary::TernaryAddress;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A fragment ion with m/z and relative intensity
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Fragment {
    /// Fragment m/z
    pub mz: f64,
    /// Relative intensity (0-1)
    pub intensity: f64,
}

impl Fragment {
    pub fn new(mz: f64, intensity: f64) -> Self {
        Self { mz, intensity }
    }
}

/// An isotope peak with m/z offset and relative intensity
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IsotopePeak {
    /// m/z offset from monoisotopic mass (0, 1.003, 2.006, ...)
    pub mz_offset: f64,
    /// Relative intensity (normalized, monoisotopic = 1.0)
    pub intensity: f64,
}

/// A synthesized spectrum containing all extracted observables
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SynthesizedSpectrum {
    /// The ternary address (as string)
    pub address: String,
    /// S-entropy coordinates
    pub s_k: f64,
    pub s_t: f64,
    pub s_e: f64,
    /// Mass-to-charge ratio
    pub mz: f64,
    /// Retention time (minutes)
    pub retention_time: f64,
    /// Base intensity (arbitrary units)
    pub intensity: f64,
    /// Fragment ions
    pub fragments: Vec<Fragment>,
    /// Isotope pattern
    pub isotope_pattern: Vec<IsotopePeak>,
}

impl SynthesizedSpectrum {
    /// Get all peaks (parent + fragments) as (mz, intensity) pairs
    pub fn all_peaks(&self) -> Vec<(f64, f64)> {
        let mut peaks = Vec::with_capacity(1 + self.fragments.len());
        peaks.push((self.mz, self.intensity));
        for frag in &self.fragments {
            peaks.push((frag.mz, frag.intensity * self.intensity));
        }
        peaks
    }

    /// Get the full isotope envelope as (mz, intensity) pairs
    pub fn isotope_envelope(&self) -> Vec<(f64, f64)> {
        self.isotope_pattern
            .iter()
            .map(|iso| (self.mz + iso.mz_offset, iso.intensity * self.intensity))
            .collect()
    }
}

/// Spectrum extractor with calibrated parameters
///
/// Maps S-entropy coordinates to mass spectrometric observables.
/// The extraction functions are calibrated for typical LC-MS conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumExtractor {
    /// Minimum m/z in the calibrated range
    pub mass_min: f64,
    /// Maximum m/z in the calibrated range
    pub mass_max: f64,
    /// Void time (t₀) for chromatography (minutes)
    pub t0: f64,
    /// Maximum gradient time for chromatography (minutes)
    pub t_max: f64,
    /// Log ratio for mass scaling
    log_mass_ratio: f64,
    /// Log of minimum mass
    log_mass_min: f64,
}

impl SpectrumExtractor {
    /// Create a new extractor with specified calibration
    pub fn new(mass_min: f64, mass_max: f64, t0: f64, t_max: f64) -> Self {
        let log_mass_min = mass_min.log10();
        let log_mass_max = mass_max.log10();
        Self {
            mass_min,
            mass_max,
            t0,
            t_max,
            log_mass_ratio: log_mass_max - log_mass_min,
            log_mass_min,
        }
    }

    /// Extract m/z directly from S_k coordinate
    ///
    /// Uses logarithmic scaling: S_k=0 → mass_max, S_k=1 → mass_min
    #[inline]
    pub fn mass_from_sk(&self, s_k: f64) -> f64 {
        let log_mass = self.log_mass_min + self.log_mass_ratio * (1.0 - s_k);
        10.0_f64.powf(log_mass)
    }

    /// Extract retention time from S_t coordinate
    ///
    /// Linear mapping: S_t=0 → t0, S_t=1 → t_max
    #[inline]
    pub fn retention_time_from_st(&self, s_t: f64) -> f64 {
        self.t0 + s_t * (self.t_max - self.t0)
    }

    /// Extract fragment ions from S_k and S_e coordinates
    ///
    /// Higher S_e means more extensive fragmentation
    pub fn fragments_from_scoord(&self, s_k: f64, s_e: f64) -> Vec<Fragment> {
        let parent_mass = self.mass_from_sk(s_k);

        // Number of fragments depends on S_e
        let n_frags = (s_e * 5.0).floor() as usize;
        if n_frags == 0 {
            return Vec::new();
        }

        let mut fragments = Vec::with_capacity(n_frags);

        for i in 1..=n_frags {
            // Fragments are progressively smaller fractions of parent
            let frag_ratio = 0.8 - 0.15 * (i as f64);
            if frag_ratio > 0.1 {
                let frag_mass = parent_mass * frag_ratio;
                // Intensity follows power law decay
                let intensity = 1.0 / (i as f64).powf(1.2);
                fragments.push(Fragment::new(frag_mass, intensity));
            }
        }

        fragments
    }

    /// Extract isotope pattern from S_k coordinate
    ///
    /// Uses simplified binomial model based on estimated carbon count
    pub fn isotope_pattern_from_sk(&self, s_k: f64) -> Vec<IsotopePeak> {
        let mass = self.mass_from_sk(s_k);

        // Estimate carbon count (average ~14 Da per carbon in organic molecules)
        let n_carbons = (mass / 14.0).round() as usize;
        let n_carbons = n_carbons.max(1).min(100);

        // 13C natural abundance
        let p_c13 = 0.0107;

        // Calculate isotope pattern using binomial approximation
        let max_isotopes = 4.min(n_carbons);
        let mut pattern = Vec::with_capacity(max_isotopes);

        for k in 0..max_isotopes {
            // Binomial coefficient approximation for large n
            let prob: f64 = if k == 0 {
                (1.0_f64 - p_c13).powi(n_carbons as i32)
            } else {
                // Use Poisson approximation: (n*p)^k * exp(-n*p) / k!
                let lambda = n_carbons as f64 * p_c13;
                let mut factorial = 1.0_f64;
                for j in 1..=k {
                    factorial *= j as f64;
                }
                lambda.powi(k as i32) * (-lambda).exp() / factorial
            };

            pattern.push(IsotopePeak {
                mz_offset: k as f64 * 1.003355,
                intensity: prob,
            });
        }

        // Normalize to monoisotopic = 1.0
        if let Some(mono) = pattern.first() {
            if mono.intensity > 0.0 {
                let norm = mono.intensity;
                for peak in &mut pattern {
                    peak.intensity /= norm;
                }
            }
        }

        pattern
    }

    /// Extract all observables from S-entropy coordinates
    pub fn extract_from_scoord(&self, coord: &SEntropyCoord) -> SynthesizedSpectrum {
        let mz = self.mass_from_sk(coord.s_k);
        let rt = self.retention_time_from_st(coord.s_t);
        let fragments = self.fragments_from_scoord(coord.s_k, coord.s_e);
        let isotopes = self.isotope_pattern_from_sk(coord.s_k);

        SynthesizedSpectrum {
            address: String::new(),
            s_k: coord.s_k,
            s_t: coord.s_t,
            s_e: coord.s_e,
            mz,
            retention_time: rt,
            intensity: 1.0,
            fragments,
            isotope_pattern: isotopes,
        }
    }

    /// Extract all observables from a ternary address
    pub fn extract(&self, addr: &TernaryAddress) -> SynthesizedSpectrum {
        let coord = addr.to_scoord();
        let mut spectrum = self.extract_from_scoord(&coord);
        spectrum.address = addr.to_string();
        spectrum
    }

    /// Batch extraction with parallel processing
    pub fn extract_batch(&self, addresses: &[TernaryAddress]) -> Vec<SynthesizedSpectrum> {
        addresses
            .par_iter()
            .map(|addr| self.extract(addr))
            .collect()
    }

    /// Batch extraction from address strings
    pub fn extract_batch_str(&self, addresses: &[&str]) -> Result<Vec<SynthesizedSpectrum>> {
        addresses
            .par_iter()
            .map(|s| {
                let addr = TernaryAddress::from_str(s)?;
                Ok(self.extract(&addr))
            })
            .collect()
    }
}

impl Default for SpectrumExtractor {
    /// Default calibration for typical small molecule LC-MS
    fn default() -> Self {
        Self::new(100.0, 1000.0, 0.5, 20.0)
    }
}

/// Molecule encoder: inverse of SpectrumExtractor
///
/// Maps molecular properties to ternary addresses
#[derive(Debug, Clone)]
pub struct MoleculeEncoder {
    /// Extractor for calibration parameters
    extractor: SpectrumExtractor,
}

impl MoleculeEncoder {
    /// Create with default calibration
    pub fn new() -> Self {
        Self {
            extractor: SpectrumExtractor::default(),
        }
    }

    /// Create with custom calibration
    pub fn with_extractor(extractor: SpectrumExtractor) -> Self {
        Self { extractor }
    }

    /// Encode molecular properties to ternary address
    pub fn encode(
        &self,
        exact_mass: f64,
        retention_time: f64,
        fragmentation: f64,
        depth: usize,
    ) -> Result<TernaryAddress> {
        // S_k from mass (inverse log relationship)
        let mass_clamped = exact_mass.clamp(self.extractor.mass_min, self.extractor.mass_max);
        let log_mass = mass_clamped.log10();
        let s_k = 1.0 - (log_mass - self.extractor.log_mass_min) / self.extractor.log_mass_ratio;
        let s_k = s_k.clamp(0.01, 0.99);

        // S_t from retention time
        let rt_clamped = retention_time.clamp(self.extractor.t0, self.extractor.t_max);
        let s_t = (rt_clamped - self.extractor.t0) / (self.extractor.t_max - self.extractor.t0);
        let s_t = s_t.clamp(0.01, 0.99);

        // S_e from fragmentation
        let s_e = fragmentation.clamp(0.01, 0.99);

        let coord = SEntropyCoord::new_unchecked(s_k, s_t, s_e);
        TernaryAddress::from_scoord(&coord, depth)
    }
}

impl Default for MoleculeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mass_extraction() {
        let extractor = SpectrumExtractor::default();

        // S_k = 0 -> max mass (1000)
        assert_abs_diff_eq!(extractor.mass_from_sk(0.0), 1000.0, epsilon = 0.1);

        // S_k = 1 -> min mass (100)
        assert_abs_diff_eq!(extractor.mass_from_sk(1.0), 100.0, epsilon = 0.1);

        // S_k = 0.5 -> geometric mean (sqrt(100*1000) ≈ 316)
        assert_abs_diff_eq!(extractor.mass_from_sk(0.5), 316.23, epsilon = 1.0);
    }

    #[test]
    fn test_retention_time_extraction() {
        let extractor = SpectrumExtractor::default();

        // S_t = 0 -> t0 (0.5 min)
        assert_abs_diff_eq!(extractor.retention_time_from_st(0.0), 0.5, epsilon = 0.01);

        // S_t = 1 -> t_max (20 min)
        assert_abs_diff_eq!(extractor.retention_time_from_st(1.0), 20.0, epsilon = 0.01);

        // S_t = 0.5 -> midpoint (10.25 min)
        assert_abs_diff_eq!(extractor.retention_time_from_st(0.5), 10.25, epsilon = 0.01);
    }

    #[test]
    fn test_fragment_extraction() {
        let extractor = SpectrumExtractor::default();

        // Low S_e -> no fragments
        let frags_low = extractor.fragments_from_scoord(0.5, 0.1);
        assert!(frags_low.is_empty());

        // High S_e -> multiple fragments
        let frags_high = extractor.fragments_from_scoord(0.5, 0.8);
        assert!(!frags_high.is_empty());
        assert!(frags_high.len() <= 4);

        // Fragments should be smaller than parent
        let parent = extractor.mass_from_sk(0.5);
        for frag in &frags_high {
            assert!(frag.mz < parent);
        }
    }

    #[test]
    fn test_isotope_pattern() {
        let extractor = SpectrumExtractor::default();
        let pattern = extractor.isotope_pattern_from_sk(0.5);

        // Should have at least monoisotopic peak
        assert!(!pattern.is_empty());

        // Monoisotopic should be normalized to 1.0
        assert_abs_diff_eq!(pattern[0].intensity, 1.0, epsilon = 0.01);

        // M+1 should be present and smaller
        if pattern.len() > 1 {
            assert!(pattern[1].intensity < 1.0);
            assert_abs_diff_eq!(pattern[1].mz_offset, 1.003355, epsilon = 0.001);
        }
    }

    #[test]
    fn test_full_extraction() {
        let addr = TernaryAddress::from_str("012102012102012102").unwrap();
        let extractor = SpectrumExtractor::default();
        let spectrum = extractor.extract(&addr);

        assert!(!spectrum.address.is_empty());
        assert!(spectrum.mz > 100.0 && spectrum.mz < 1000.0);
        assert!(spectrum.retention_time >= 0.5 && spectrum.retention_time <= 20.0);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let encoder = MoleculeEncoder::default();
        let extractor = SpectrumExtractor::default();

        let original_mass = 500.0;
        let original_rt = 10.0;
        let original_frag = 0.5;

        let addr = encoder.encode(original_mass, original_rt, original_frag, 18).unwrap();
        let spectrum = extractor.extract(&addr);

        // Should be close within resolution
        let mass_error_ppm = (spectrum.mz - original_mass).abs() / original_mass * 1e6;
        let rt_error = (spectrum.retention_time - original_rt).abs();

        assert!(mass_error_ppm < 5000.0); // Within 0.5% for 18-trit
        assert!(rt_error < 0.5); // Within 0.5 min
    }

    #[test]
    fn test_batch_extraction() {
        let addresses = vec![
            TernaryAddress::from_str("012102").unwrap(),
            TernaryAddress::from_str("210012").unwrap(),
            TernaryAddress::from_str("121212").unwrap(),
        ];

        let extractor = SpectrumExtractor::default();
        let spectra = extractor.extract_batch(&addresses);

        assert_eq!(spectra.len(), 3);
        for spectrum in &spectra {
            assert!(spectrum.mz > 0.0);
        }
    }
}

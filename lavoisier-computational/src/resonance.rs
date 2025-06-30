//! Resonance Detection and Internal Spectrometer Module
//!
//! Advanced resonance detection between virtual molecular simulations and
//! hardware oscillations, implementing the computer as an internal spectrometer.

use std::collections::HashMap;
use std::f64::consts::PI;
use nalgebra::{DVector, DMatrix};
use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use crate::hardware::{HardwareOscillation, HardwareHarvester};
use crate::simulation::VirtualMolecule;

/// Resonance detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceMatch {
    pub molecule_id: String,
    pub resonance_strength: f64,
    pub frequency_matches: Vec<FrequencyMatch>,
    pub confidence_score: f64,
    pub spectral_correlation: f64,
}

/// Individual frequency match between virtual and hardware oscillations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyMatch {
    pub virtual_frequency: f64,
    pub hardware_frequency: f64,
    pub match_strength: f64,
    pub phase_correlation: f64,
    pub amplitude_ratio: f64,
}

/// Internal computer spectrometer using hardware oscillations
pub struct ResonanceSpectrometer {
    hardware_harvester: HardwareHarvester,
    virtual_molecules: Vec<VirtualMolecule>,
    resonance_threshold: f64,
    correlation_window: usize,
    fft_planner: FftPlanner<f64>,
    spectral_library: HashMap<String, SpectralReference>,
}

/// Reference spectral data for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralReference {
    pub compound_name: String,
    pub frequencies: Vec<f64>,
    pub amplitudes: Vec<f64>,
    pub phase_pattern: Vec<f64>,
    pub characteristic_peaks: Vec<f64>,
}

impl ResonanceSpectrometer {
    /// Create a new resonance spectrometer
    pub fn new(sample_rate: f64, buffer_size: usize, resonance_threshold: f64) -> Self {
        Self {
            hardware_harvester: HardwareHarvester::new(sample_rate, buffer_size),
            virtual_molecules: Vec::new(),
            resonance_threshold,
            correlation_window: 1024,
            fft_planner: FftPlanner::new(),
            spectral_library: HashMap::new(),
        }
    }

    /// Add a virtual molecule to the detection system
    pub fn add_virtual_molecule(&mut self, molecule: VirtualMolecule) {
        self.virtual_molecules.push(molecule);
    }

    /// Start the internal spectrometer
    pub async fn start_spectrometer(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.hardware_harvester.start_harvesting().await?;
        self.build_spectral_library();
        Ok(())
    }

    /// Stop the internal spectrometer
    pub fn stop_spectrometer(&self) {
        self.hardware_harvester.stop_harvesting();
    }

    /// Build spectral library from virtual molecules
    fn build_spectral_library(&mut self) {
        for molecule in &self.virtual_molecules {
            let reference = SpectralReference {
                compound_name: molecule.formula.clone(),
                frequencies: molecule.vibrational_frequencies.clone(),
                amplitudes: molecule.spectral_signature.intensities.clone(),
                phase_pattern: self.generate_phase_pattern(&molecule.vibrational_frequencies),
                characteristic_peaks: molecule.spectral_signature.mz_peaks.clone(),
            };

            self.spectral_library.insert(molecule.id.clone(), reference);
        }
    }

    /// Generate phase pattern from vibrational frequencies
    fn generate_phase_pattern(&self, frequencies: &[f64]) -> Vec<f64> {
        frequencies.iter().enumerate().map(|(i, freq)| {
            (freq * i as f64 * PI / 180.0) % (2.0 * PI)
        }).collect()
    }

    /// Perform resonance detection and molecular identification
    pub fn detect_molecular_resonances(&self, duration: f64) -> Vec<ResonanceMatch> {
        // Get hardware oscillation spectrum
        let hardware_spectrum = self.hardware_harvester.get_oscillation_spectrum(duration);
        let hardware_oscillations = self.get_recent_oscillations(duration);

        if hardware_oscillations.is_empty() {
            return Vec::new();
        }

        // Extract frequency and amplitude data from hardware
        let hardware_frequencies: Vec<f64> = hardware_oscillations.iter().map(|osc| osc.frequency).collect();
        let hardware_amplitudes: Vec<f64> = hardware_oscillations.iter().map(|osc| osc.amplitude).collect();
        let hardware_phases: Vec<f64> = hardware_oscillations.iter().map(|osc| osc.phase).collect();

        // Compare against each virtual molecule
        let mut resonance_matches = Vec::new();

        for molecule in &self.virtual_molecules {
            let resonance_match = self.calculate_molecular_resonance(
                molecule,
                &hardware_frequencies,
                &hardware_amplitudes,
                &hardware_phases,
                &hardware_spectrum,
            );

            if resonance_match.resonance_strength > self.resonance_threshold {
                resonance_matches.push(resonance_match);
            }
        }

        // Sort by resonance strength
        resonance_matches.sort_by(|a, b| b.resonance_strength.partial_cmp(&a.resonance_strength).unwrap());
        resonance_matches
    }

    /// Calculate resonance between a virtual molecule and hardware oscillations
    fn calculate_molecular_resonance(
        &self,
        molecule: &VirtualMolecule,
        hardware_frequencies: &[f64],
        hardware_amplitudes: &[f64],
        hardware_phases: &[f64],
        hardware_spectrum: &[f64],
    ) -> ResonanceMatch {
        let mut frequency_matches = Vec::new();
        let mut total_resonance = 0.0;
        let mut match_count = 0;

        // Compare vibrational frequencies
        for (i, &vib_freq) in molecule.vibrational_frequencies.iter().enumerate() {
            let mut best_match_strength = 0.0;
            let mut best_hw_freq = 0.0;
            let mut best_phase_corr = 0.0;
            let mut best_amp_ratio = 0.0;

            for (j, &hw_freq) in hardware_frequencies.iter().enumerate() {
                // Calculate frequency match strength
                let freq_diff = (vib_freq - hw_freq).abs();
                let freq_match_strength = (-freq_diff / (vib_freq * 0.1)).exp(); // Gaussian match

                if freq_match_strength > best_match_strength && freq_match_strength > 0.5 {
                    best_match_strength = freq_match_strength;
                    best_hw_freq = hw_freq;

                    // Calculate phase correlation
                    let expected_phase = (vib_freq * i as f64 * PI / 180.0) % (2.0 * PI);
                    let hw_phase = hardware_phases.get(j).unwrap_or(&0.0);
                    best_phase_corr = (expected_phase - hw_phase).cos().abs();

                    // Calculate amplitude ratio
                    let vib_amplitude = molecule.spectral_signature.intensities.get(i).unwrap_or(&1.0) / 100.0;
                    let hw_amplitude = hardware_amplitudes.get(j).unwrap_or(&0.0);
                    best_amp_ratio = if hw_amplitude > &0.0 {
                        (vib_amplitude / hw_amplitude).min(hw_amplitude / vib_amplitude)
                    } else {
                        0.0
                    };
                }
            }

            if best_match_strength > 0.5 {
                frequency_matches.push(FrequencyMatch {
                    virtual_frequency: vib_freq,
                    hardware_frequency: best_hw_freq,
                    match_strength: best_match_strength,
                    phase_correlation: best_phase_corr,
                    amplitude_ratio: best_amp_ratio,
                });

                total_resonance += best_match_strength * best_phase_corr * best_amp_ratio;
                match_count += 1;
            }
        }

        // Calculate overall resonance strength
        let resonance_strength = if match_count > 0 {
            total_resonance / match_count as f64
        } else {
            0.0
        };

        // Calculate spectral correlation using FFT
        let spectral_correlation = self.calculate_spectral_correlation(
            &molecule.spectral_signature.intensities,
            hardware_spectrum,
        );

        // Calculate confidence score
        let confidence_score = self.calculate_confidence_score(
            resonance_strength,
            spectral_correlation,
            match_count,
            molecule.vibrational_frequencies.len(),
        );

        ResonanceMatch {
            molecule_id: molecule.id.clone(),
            resonance_strength,
            frequency_matches,
            confidence_score,
            spectral_correlation,
        }
    }

    /// Calculate spectral correlation using cross-correlation
    fn calculate_spectral_correlation(&self, virtual_spectrum: &[f64], hardware_spectrum: &[f64]) -> f64 {
        if virtual_spectrum.is_empty() || hardware_spectrum.is_empty() {
            return 0.0;
        }

        // Normalize spectra
        let virtual_normalized = self.normalize_spectrum(virtual_spectrum);
        let hardware_normalized = self.normalize_spectrum(hardware_spectrum);

        // Calculate cross-correlation
        let min_len = virtual_normalized.len().min(hardware_normalized.len());
        let mut correlation = 0.0;

        for i in 0..min_len {
            correlation += virtual_normalized[i] * hardware_normalized[i];
        }

        correlation / min_len as f64
    }

    /// Normalize spectrum to unit vector
    fn normalize_spectrum(&self, spectrum: &[f64]) -> Vec<f64> {
        let magnitude: f64 = spectrum.iter().map(|x| x * x).sum::<f64>().sqrt();
        if magnitude > 0.0 {
            spectrum.iter().map(|x| x / magnitude).collect()
        } else {
            vec![0.0; spectrum.len()]
        }
    }

    /// Calculate confidence score for molecular identification
    fn calculate_confidence_score(
        &self,
        resonance_strength: f64,
        spectral_correlation: f64,
        match_count: usize,
        total_frequencies: usize,
    ) -> f64 {
        let match_ratio = match_count as f64 / total_frequencies.max(1) as f64;
        let frequency_coverage = match_ratio.sqrt();

        // Weighted combination of factors
        let weights = [0.4, 0.3, 0.3]; // resonance, correlation, coverage
        let scores = [resonance_strength, spectral_correlation, frequency_coverage];

        weights.iter().zip(scores.iter()).map(|(w, s)| w * s).sum()
    }

    /// Get recent hardware oscillations
    fn get_recent_oscillations(&self, duration: f64) -> Vec<HardwareOscillation> {
        // This would access the hardware harvester's oscillation buffer
        // For now, return empty vector as the actual implementation would require
        // proper synchronization with the hardware harvester
        Vec::new()
    }

    /// Perform advanced spectral analysis using FFT
    pub fn perform_fft_analysis(&self, signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut input: Vec<Complex<f64>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Pad to power of 2 for efficient FFT
        let n = input.len().next_power_of_two();
        input.resize(n, Complex::new(0.0, 0.0));

        let mut fft = self.fft_planner.plan_fft_forward(n);
        fft.process(&mut input);

        let frequencies: Vec<f64> = (0..n/2).map(|i| i as f64 / n as f64).collect();
        let magnitudes: Vec<f64> = input[0..n/2].iter().map(|c| c.norm()).collect();

        (frequencies, magnitudes)
    }

    /// Generate synthetic spectral signature for virtual molecule validation
    pub fn generate_validation_signature(&self, molecule: &VirtualMolecule) -> Vec<f64> {
        let mut signature = vec![0.0; 1024];

        for (i, &freq) in molecule.vibrational_frequencies.iter().enumerate() {
            let intensity = molecule.spectral_signature.intensities.get(i).unwrap_or(&1.0);
            let bin = ((freq / 4000.0) * 1023.0) as usize; // Map to bin

            if bin < signature.len() {
                signature[bin] += intensity / 100.0;
            }
        }

        signature
    }
}

/// Hardware-molecular validator for cross-validation
pub struct HardwareMolecularValidator {
    spectrometer: ResonanceSpectrometer,
    validation_threshold: f64,
    cross_validation_enabled: bool,
}

impl HardwareMolecularValidator {
    pub fn new(sample_rate: f64, buffer_size: usize, validation_threshold: f64) -> Self {
        Self {
            spectrometer: ResonanceSpectrometer::new(sample_rate, buffer_size, 0.3),
            validation_threshold,
            cross_validation_enabled: true,
        }
    }

    /// Validate molecular identification through hardware-virtual cross-correlation
    pub async fn validate_molecular_identification(
        &mut self,
        candidate_molecules: Vec<VirtualMolecule>,
        analysis_duration: f64,
    ) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error + Send + Sync>> {
        // Add molecules to spectrometer
        for molecule in candidate_molecules {
            self.spectrometer.add_virtual_molecule(molecule);
        }

        // Start hardware analysis
        self.spectrometer.start_spectrometer().await?;

        // Allow time for data collection
        tokio::time::sleep(tokio::time::Duration::from_secs_f64(analysis_duration)).await;

        // Perform resonance detection
        let resonance_matches = self.spectrometer.detect_molecular_resonances(analysis_duration);

        // Convert to validation results
        let validation_results = resonance_matches.into_iter().map(|match_result| {
            ValidationResult {
                molecule_id: match_result.molecule_id,
                validation_score: match_result.confidence_score,
                resonance_evidence: match_result.resonance_strength,
                spectral_evidence: match_result.spectral_correlation,
                is_validated: match_result.confidence_score > self.validation_threshold,
                cross_validation_passed: self.cross_validation_enabled &&
                    match_result.frequency_matches.len() >= 3,
            }
        }).collect();

        self.spectrometer.stop_spectrometer();
        Ok(validation_results)
    }
}

/// Validation result for molecular identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub molecule_id: String,
    pub validation_score: f64,
    pub resonance_evidence: f64,
    pub spectral_evidence: f64,
    pub is_validated: bool,
    pub cross_validation_passed: bool,
}

// Python bindings
#[pyclass]
pub struct PyResonanceSpectrometer {
    inner: ResonanceSpectrometer,
    runtime: tokio::runtime::Runtime,
}

#[pymethods]
impl PyResonanceSpectrometer {
    #[new]
    fn new(sample_rate: f64, buffer_size: usize, resonance_threshold: f64) -> PyResult<Self> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        Ok(Self {
            inner: ResonanceSpectrometer::new(sample_rate, buffer_size, resonance_threshold),
            runtime,
        })
    }

    fn start_spectrometer(&mut self) -> PyResult<()> {
        self.runtime.block_on(async {
            self.inner.start_spectrometer().await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to start spectrometer: {}", e)))
        })
    }

    fn stop_spectrometer(&self) {
        self.inner.stop_spectrometer();
    }

    fn detect_molecular_resonances(&self, duration: f64) -> PyResult<PyObject> {
        let matches = self.inner.detect_molecular_resonances(duration);

        Python::with_gil(|py| {
            let list = pyo3::types::PyList::empty(py);
            for resonance_match in matches {
                let dict = PyDict::new(py);
                dict.set_item("molecule_id", &resonance_match.molecule_id)?;
                dict.set_item("resonance_strength", resonance_match.resonance_strength)?;
                dict.set_item("confidence_score", resonance_match.confidence_score)?;
                dict.set_item("spectral_correlation", resonance_match.spectral_correlation)?;
                list.append(dict)?;
            }
            Ok(list.into())
        })
    }

    fn perform_fft_analysis(&self, signal: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
        self.inner.perform_fft_analysis(&signal)
    }
}

#[pyclass]
pub struct PyHardwareMolecularValidator {
    inner: HardwareMolecularValidator,
    runtime: tokio::runtime::Runtime,
}

#[pymethods]
impl PyHardwareMolecularValidator {
    #[new]
    fn new(sample_rate: f64, buffer_size: usize, validation_threshold: f64) -> PyResult<Self> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        Ok(Self {
            inner: HardwareMolecularValidator::new(sample_rate, buffer_size, validation_threshold),
            runtime,
        })
    }
}

#[pyfunction]
pub fn py_detect_resonance(
    virtual_frequencies: Vec<f64>,
    hardware_frequencies: Vec<f64>,
    hardware_amplitudes: Vec<f64>,
    threshold: f64,
) -> Vec<(f64, f64, f64)> {
    let mut matches = Vec::new();

    for &vf in &virtual_frequencies {
        for (i, &hf) in hardware_frequencies.iter().enumerate() {
            let freq_diff = (vf - hf).abs();
            let match_strength = (-freq_diff / (vf * 0.1)).exp();

            if match_strength > threshold {
                let amplitude = hardware_amplitudes.get(i).unwrap_or(&0.0);
                matches.push((vf, hf, match_strength * amplitude));
            }
        }
    }

    matches.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    matches
}

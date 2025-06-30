//! High-Performance Computational Module for Lavoisier
//!
//! This crate provides hardware harvesting, molecular simulation, and resonance
//! detection capabilities optimized for real-time analytical applications.

use pyo3::prelude::*;

pub mod bayesian_optimization;
pub mod evidence_network;
pub mod hardware;
pub mod hardware_validation;
pub mod molecular_simulation;
pub mod noise_modeling;
pub mod optimization;
pub mod oscillation;
pub mod resonance;
pub mod simulation;
pub mod trajectory_optimization;

// Re-export main types
pub use hardware::{HardwareHarvester, SystemOscillationProfiler};
pub use optimization::{ConvergenceAccelerator, TrajectoryOptimizer};
pub use oscillation::{OscillationAnalyzer, OscillationPattern};
pub use resonance::{HardwareMolecularValidator, ResonanceSpectrometer};
pub use simulation::{MolecularResonanceEngine, VirtualMolecularSimulator};

/// Python module definition
#[pymodule]
fn lavoisier_computational(_py: Python, m: &PyModule) -> PyResult<()> {
    // Hardware harvesting classes
    m.add_class::<hardware::PyHardwareHarvester>()?;
    m.add_class::<hardware::PySystemOscillationProfiler>()?;

    // Molecular simulation classes
    m.add_class::<simulation::PyVirtualMolecularSimulator>()?;
    m.add_class::<simulation::PyMolecularResonanceEngine>()?;

    // Resonance detection classes
    m.add_class::<resonance::PyResonanceSpectrometer>()?;
    m.add_class::<resonance::PyHardwareMolecularValidator>()?;

    // Optimization classes
    m.add_class::<optimization::PyTrajectoryOptimizer>()?;
    m.add_class::<optimization::PyConvergenceAccelerator>()?;

    // Utility functions
    m.add_function(wrap_pyfunction!(hardware::py_get_system_oscillations, m)?)?;
    m.add_function(wrap_pyfunction!(simulation::py_simulate_molecule, m)?)?;
    m.add_function(wrap_pyfunction!(resonance::py_detect_resonance, m)?)?;

    Ok(())
}

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

/// Core error type for computational modules
#[derive(Debug, Clone)]
pub enum ComputationalError {
    OptimizationFailed(String),
    InvalidInput(String),
    HardwareError(String),
    SimulationError(String),
}

impl fmt::Display for ComputationalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ComputationalError::OptimizationFailed(msg) => {
                write!(f, "Optimization failed: {}", msg)
            }
            ComputationalError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            ComputationalError::HardwareError(msg) => write!(f, "Hardware error: {}", msg),
            ComputationalError::SimulationError(msg) => write!(f, "Simulation error: {}", msg),
        }
    }
}

impl Error for ComputationalError {}

/// Result type for computational operations
pub type ComputationalResult<T> = Result<T, ComputationalError>;

/// Configuration for computational modules
#[derive(Debug, Clone)]
pub struct ComputationalConfig {
    pub num_threads: usize,
    pub memory_limit_gb: f64,
    pub optimization_iterations: usize,
    pub noise_levels: Vec<f64>,
    pub hardware_validation: bool,
}

impl Default for ComputationalConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            memory_limit_gb: 16.0,
            optimization_iterations: 100,
            noise_levels: (1..=9).map(|i| i as f64 * 0.1).collect(),
            hardware_validation: true,
        }
    }
}

/// Main computational engine for Lavoisier
pub struct ComputationalEngine {
    config: ComputationalConfig,
    bayesian_optimizer: bayesian_optimization::BayesianOptimizer,
    noise_modeler: noise_modeling::NoiseModeler,
    evidence_network: evidence_network::EvidenceNetwork,
    hardware_validator: Option<hardware_validation::HardwareValidator>,
}

impl ComputationalEngine {
    /// Create new computational engine
    pub fn new(config: ComputationalConfig) -> ComputationalResult<Self> {
        let bayesian_optimizer = bayesian_optimization::BayesianOptimizer::new(&config)?;
        let noise_modeler = noise_modeling::NoiseModeler::new(&config)?;
        let evidence_network = evidence_network::EvidenceNetwork::new(&config)?;

        let hardware_validator = if config.hardware_validation {
            Some(hardware_validation::HardwareValidator::new(&config)?)
        } else {
            None
        };

        Ok(Self {
            config,
            bayesian_optimizer,
            noise_modeler,
            evidence_network,
            hardware_validator,
        })
    }

    /// Process large-scale mass spectrometry data with global optimization
    pub fn process_large_dataset(
        &mut self,
        mz_data: &[f64],
        intensity_data: &[f64],
        spectrum_metadata: &HashMap<String, String>,
    ) -> ComputationalResult<ProcessingResult> {
        // Validate input data
        if mz_data.len() != intensity_data.len() {
            return Err(ComputationalError::InvalidInput(
                "M/Z and intensity arrays must have same length".to_string(),
            ));
        }

        // Global Bayesian optimization for noise-modulated analysis
        let optimization_result = self.bayesian_optimizer.optimize_noise_level(
            mz_data,
            intensity_data,
            &self.config.noise_levels,
        )?;

        // Build evidence network with optimal parameters
        let evidence_result = self.evidence_network.build_evidence_network(
            mz_data,
            intensity_data,
            optimization_result.optimal_noise_level,
        )?;

        // Hardware-assisted validation if enabled
        let hardware_validation = if let Some(ref mut validator) = self.hardware_validator {
            Some(validator.validate_results(&evidence_result)?)
        } else {
            None
        };

        Ok(ProcessingResult {
            optimization_result,
            evidence_result,
            hardware_validation,
            processing_time_ms: 0, // Will be measured by caller
        })
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> PerformanceStats {
        PerformanceStats {
            num_threads: self.config.num_threads,
            memory_usage_gb: self.get_memory_usage(),
            optimization_iterations: self.config.optimization_iterations,
            hardware_validation_enabled: self.hardware_validator.is_some(),
        }
    }

    fn get_memory_usage(&self) -> f64 {
        // Placeholder - would use actual memory monitoring
        0.0
    }
}

/// Result of computational processing
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    pub optimization_result: bayesian_optimization::OptimizationResult,
    pub evidence_result: evidence_network::EvidenceResult,
    pub hardware_validation: Option<hardware_validation::ValidationResult>,
    pub processing_time_ms: u64,
}

/// Performance statistics
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub num_threads: usize,
    pub memory_usage_gb: f64,
    pub optimization_iterations: usize,
    pub hardware_validation_enabled: bool,
}

//! # Mass Computing
//!
//! Ternary partition synthesis framework for mass spectrometry.
//!
//! This crate implements the Mass Computing paradigm where mass spectrometric
//! observables emerge from ternary partition synthesis rather than physical simulation.
//!
//! ## Core Concepts
//!
//! - **S-Entropy Space**: A unit cube `[0,1]³` with coordinates `(S_k, S_t, S_e)`
//! - **Ternary Address**: A k-trit string addressing one of `3^k` cells in S-space
//! - **Partition Determinism**: Addresses uniquely determine spectra without dynamics
//! - **MassScript**: A DSL for virtual experiments via ternary operations
//!
//! ## Example
//!
//! ```rust
//! use lavoisier_mass_computing::prelude::*;
//!
//! // Create address from string
//! let addr = TernaryAddress::from_str("012102012102012102").unwrap();
//!
//! // Extract S-coordinates
//! let coords = addr.to_scoord();
//! println!("S_k={:.3}, S_t={:.3}, S_e={:.3}", coords.s_k, coords.s_t, coords.s_e);
//!
//! // Synthesize spectrum
//! let extractor = SpectrumExtractor::default();
//! let spectrum = extractor.extract(&addr);
//! println!("m/z={:.2}, RT={:.2} min", spectrum.mz, spectrum.retention_time);
//! ```

pub mod ternary;
pub mod scoord;
pub mod extractor;
pub mod massscript;
pub mod partition;
pub mod errors;
pub mod lookup;

#[cfg(feature = "python-bindings")]
pub mod python;

// Re-exports
pub use ternary::{TernaryAddress, Trit, Tryte};
pub use scoord::SEntropyCoord;
pub use extractor::{SpectrumExtractor, SynthesizedSpectrum, Fragment};
pub use partition::PartitionState;
pub use massscript::{MassScript, MassScriptInterpreter, MassScriptCommand};
pub use errors::{MassComputingError, Result};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::ternary::{TernaryAddress, Trit, Tryte};
    pub use crate::scoord::SEntropyCoord;
    pub use crate::extractor::{SpectrumExtractor, SynthesizedSpectrum};
    pub use crate::partition::PartitionState;
    pub use crate::massscript::{MassScript, MassScriptInterpreter};
    pub use crate::errors::Result;
}

/// Convenience function to synthesize a spectrum from an address string
pub fn synthesize(address: &str) -> Result<SynthesizedSpectrum> {
    let addr = TernaryAddress::from_str(address)?;
    let extractor = SpectrumExtractor::default();
    Ok(extractor.extract(&addr))
}

/// Convenience function to observe (extract all observables) from an address
pub fn observe(address: &str) -> Result<serde_json::Value> {
    let spectrum = synthesize(address)?;
    Ok(serde_json::to_value(&spectrum)?)
}

// Python module registration
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

#[cfg(feature = "python-bindings")]
#[pymodule]
fn lavoisier_mass_computing(py: Python, m: &PyModule) -> PyResult<()> {
    python::register_module(py, m)
}

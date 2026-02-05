//! Python bindings for Mass Computing
//!
//! Exposes the Rust implementation to Python via pyo3

use crate::errors::MassComputingError;
use crate::extractor::{MoleculeEncoder, SpectrumExtractor, SynthesizedSpectrum};
use crate::massscript::{ExecutionResult, MassScript, MassScriptInterpreter};
use crate::partition::PartitionState;
use crate::scoord::SEntropyCoord;
use crate::ternary::TernaryAddress;

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use numpy::{PyArray1, PyReadonlyArray1};
use std::collections::HashMap;

/// Convert MassComputingError to PyErr
impl From<MassComputingError> for PyErr {
    fn from(e: MassComputingError) -> PyErr {
        PyValueError::new_err(e.to_string())
    }
}

/// Python wrapper for TernaryAddress
#[pyclass(name = "TernaryAddress")]
#[derive(Clone)]
pub struct PyTernaryAddress {
    inner: TernaryAddress,
}

#[pymethods]
impl PyTernaryAddress {
    /// Create from string of '0', '1', '2' characters
    #[new]
    fn new(address: &str) -> PyResult<Self> {
        let inner = TernaryAddress::from_str(address)?;
        Ok(Self { inner })
    }

    /// Create from S-entropy coordinates
    #[staticmethod]
    fn from_scoord(s_k: f64, s_t: f64, s_e: f64, depth: usize) -> PyResult<Self> {
        let coord = SEntropyCoord::new(s_k, s_t, s_e)?;
        let inner = TernaryAddress::from_scoord(&coord, depth)?;
        Ok(Self { inner })
    }

    /// Convert to S-entropy coordinates
    fn to_scoord(&self) -> (f64, f64, f64) {
        let coord = self.inner.to_scoord();
        (coord.s_k, coord.s_t, coord.s_e)
    }

    /// Get the depth (number of trits)
    #[getter]
    fn depth(&self) -> usize {
        self.inner.depth()
    }

    /// Get the cell volume
    #[getter]
    fn cell_volume(&self) -> f64 {
        self.inner.cell_volume()
    }

    /// Get resolution for each axis
    fn resolution(&self) -> [f64; 3] {
        self.inner.resolution()
    }

    /// Get the trits as a list
    fn trits(&self) -> Vec<u8> {
        self.inner.trits().to_vec()
    }

    /// Extend with another address
    fn extend(&self, other: &PyTernaryAddress) -> Self {
        Self {
            inner: self.inner.extend(&other.inner),
        }
    }

    /// Fragment at position k
    fn fragment_at(&self, k: usize) -> PyResult<(Self, Self)> {
        let (prefix, suffix) = self.inner.fragment_at(k)?;
        Ok((Self { inner: prefix }, Self { inner: suffix }))
    }

    /// Get prefix of length k
    fn prefix(&self, k: usize) -> PyResult<Self> {
        let prefix = self.inner.prefix(k)?;
        Ok(Self { inner: prefix })
    }

    /// Check if this is a prefix of another address
    fn is_prefix_of(&self, other: &PyTernaryAddress) -> bool {
        self.inner.is_prefix_of(&other.inner)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("TernaryAddress('{}')", self.inner)
    }

    fn __len__(&self) -> usize {
        self.inner.depth()
    }
}

/// Python wrapper for SEntropyCoord
#[pyclass(name = "SEntropyCoord")]
#[derive(Clone)]
pub struct PySEntropyCoord {
    inner: SEntropyCoord,
}

#[pymethods]
impl PySEntropyCoord {
    #[new]
    fn new(s_k: f64, s_t: f64, s_e: f64) -> PyResult<Self> {
        let inner = SEntropyCoord::new(s_k, s_t, s_e)?;
        Ok(Self { inner })
    }

    #[getter]
    fn s_k(&self) -> f64 {
        self.inner.s_k
    }

    #[getter]
    fn s_t(&self) -> f64 {
        self.inner.s_t
    }

    #[getter]
    fn s_e(&self) -> f64 {
        self.inner.s_e
    }

    /// Convert to ternary address
    fn to_address(&self, depth: usize) -> PyResult<PyTernaryAddress> {
        let inner = TernaryAddress::from_scoord(&self.inner, depth)?;
        Ok(PyTernaryAddress { inner })
    }

    /// Distance to another coordinate
    fn distance(&self, other: &PySEntropyCoord) -> f64 {
        self.inner.distance(&other.inner)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "SEntropyCoord(s_k={:.4}, s_t={:.4}, s_e={:.4})",
            self.inner.s_k, self.inner.s_t, self.inner.s_e
        )
    }
}

/// Python wrapper for PartitionState
#[pyclass(name = "PartitionState")]
#[derive(Clone)]
pub struct PyPartitionState {
    inner: PartitionState,
}

#[pymethods]
impl PyPartitionState {
    #[new]
    fn new(n: u32, l: u32, m: i32, s: f64) -> Self {
        Self {
            inner: PartitionState::new(n, l, m, s),
        }
    }

    /// Create from S-entropy coordinates
    #[staticmethod]
    fn from_scoord(coord: &PySEntropyCoord) -> Self {
        Self {
            inner: PartitionState::from_scoord(&coord.inner),
        }
    }

    /// Create from ternary address
    #[staticmethod]
    fn from_address(addr: &PyTernaryAddress) -> Self {
        Self {
            inner: PartitionState::from_address(&addr.inner),
        }
    }

    #[getter]
    fn n(&self) -> u32 {
        self.inner.n
    }

    #[getter]
    fn l(&self) -> u32 {
        self.inner.l
    }

    #[getter]
    fn m(&self) -> i32 {
        self.inner.m
    }

    #[getter]
    fn s(&self) -> f64 {
        self.inner.s
    }

    /// Get capacity C(n) = 2n²
    fn capacity(&self) -> u64 {
        self.inner.capacity()
    }

    /// Get degeneracy 2l + 1
    fn degeneracy(&self) -> u32 {
        self.inner.degeneracy()
    }

    fn is_valid(&self) -> bool {
        self.inner.is_valid()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "PartitionState(n={}, l={}, m={}, s={:+.1})",
            self.inner.n, self.inner.l, self.inner.m, self.inner.s
        )
    }
}

/// Python wrapper for SpectrumExtractor
#[pyclass(name = "SpectrumExtractor")]
pub struct PySpectrumExtractor {
    inner: SpectrumExtractor,
}

#[pymethods]
impl PySpectrumExtractor {
    #[new]
    #[pyo3(signature = (mass_min=100.0, mass_max=1000.0, t0=0.5, t_max=20.0))]
    fn new(mass_min: f64, mass_max: f64, t0: f64, t_max: f64) -> Self {
        Self {
            inner: SpectrumExtractor::new(mass_min, mass_max, t0, t_max),
        }
    }

    /// Extract spectrum from ternary address
    fn extract(&self, addr: &PyTernaryAddress) -> PyResult<PyDict> {
        let spectrum = self.inner.extract(&addr.inner);
        spectrum_to_dict(spectrum)
    }

    /// Extract spectrum from address string
    fn extract_str(&self, address: &str) -> PyResult<PyDict> {
        let addr = TernaryAddress::from_str(address)?;
        let spectrum = self.inner.extract(&addr);
        spectrum_to_dict(spectrum)
    }

    /// Extract m/z from S_k
    fn mass_from_sk(&self, s_k: f64) -> f64 {
        self.inner.mass_from_sk(s_k)
    }

    /// Extract retention time from S_t
    fn retention_time_from_st(&self, s_t: f64) -> f64 {
        self.inner.retention_time_from_st(s_t)
    }

    /// Batch extraction from address strings
    fn extract_batch(&self, addresses: Vec<String>) -> PyResult<Vec<PyDict>> {
        let addrs: Result<Vec<TernaryAddress>, _> = addresses
            .iter()
            .map(|s| TernaryAddress::from_str(s))
            .collect();

        let spectra = self.inner.extract_batch(&addrs?);
        spectra.into_iter().map(spectrum_to_dict).collect()
    }
}

/// Helper to convert spectrum to Python dict
fn spectrum_to_dict(spectrum: SynthesizedSpectrum) -> PyResult<PyDict> {
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("address", spectrum.address)?;
        dict.set_item("s_k", spectrum.s_k)?;
        dict.set_item("s_t", spectrum.s_t)?;
        dict.set_item("s_e", spectrum.s_e)?;
        dict.set_item("mz", spectrum.mz)?;
        dict.set_item("retention_time", spectrum.retention_time)?;
        dict.set_item("intensity", spectrum.intensity)?;

        let fragments: Vec<(f64, f64)> = spectrum
            .fragments
            .into_iter()
            .map(|f| (f.mz, f.intensity))
            .collect();
        dict.set_item("fragments", fragments)?;

        let isotopes: Vec<(f64, f64)> = spectrum
            .isotope_pattern
            .into_iter()
            .map(|i| (i.mz_offset, i.intensity))
            .collect();
        dict.set_item("isotope_pattern", isotopes)?;

        Ok(dict.into())
    })
}

/// PyDict type alias
type PyDict = pyo3::types::PyDict;

/// Python wrapper for MoleculeEncoder
#[pyclass(name = "MoleculeEncoder")]
pub struct PyMoleculeEncoder {
    inner: MoleculeEncoder,
}

#[pymethods]
impl PyMoleculeEncoder {
    #[new]
    fn new() -> Self {
        Self {
            inner: MoleculeEncoder::default(),
        }
    }

    /// Encode molecular properties to ternary address
    #[pyo3(signature = (exact_mass, retention_time=10.0, fragmentation=0.5, depth=18))]
    fn encode(
        &self,
        exact_mass: f64,
        retention_time: f64,
        fragmentation: f64,
        depth: usize,
    ) -> PyResult<PyTernaryAddress> {
        let inner = self.inner.encode(exact_mass, retention_time, fragmentation, depth)?;
        Ok(PyTernaryAddress { inner })
    }
}

/// Python wrapper for MassScript interpreter
#[pyclass(name = "MassScriptInterpreter")]
pub struct PyMassScriptInterpreter {
    inner: MassScriptInterpreter,
}

#[pymethods]
impl PyMassScriptInterpreter {
    #[new]
    fn new() -> Self {
        Self {
            inner: MassScriptInterpreter::new(),
        }
    }

    /// Execute MassScript source code
    fn execute(&mut self, source: &str, py: Python) -> PyResult<Py<PyDict>> {
        let result = self.inner.execute_str(source)?;
        execution_result_to_dict(py, result)
    }

    /// Reset interpreter state
    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Convert ExecutionResult to Python dict
fn execution_result_to_dict(py: Python, result: ExecutionResult) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);

    // Convert observations
    let obs_list: Vec<_> = result.observations.into_iter().map(|obs| {
        let obs_dict = PyDict::new(py);
        obs_dict.set_item("name", obs.name).ok();
        obs_dict.set_item("address", obs.address).ok();
        obs_dict.set_item("mz", obs.spectrum.mz).ok();
        obs_dict.set_item("retention_time", obs.spectrum.retention_time).ok();
        obs_dict.set_item("s_k", obs.spectrum.s_k).ok();
        obs_dict.set_item("s_t", obs.spectrum.s_t).ok();
        obs_dict.set_item("s_e", obs.spectrum.s_e).ok();

        let frags: Vec<(f64, f64)> = obs.spectrum.fragments.iter()
            .map(|f| (f.mz, f.intensity))
            .collect();
        obs_dict.set_item("fragments", frags).ok();

        obs_dict
    }).collect();
    dict.set_item("observations", obs_list)?;

    // Convert fragments
    let frag_list: Vec<_> = result.fragments.into_iter().map(|frag| {
        let frag_dict = PyDict::new(py);
        frag_dict.set_item("name", frag.name).ok();
        frag_dict.set_item("prefix", frag.prefix).ok();
        frag_dict.set_item("suffix", frag.suffix).ok();
        frag_dict.set_item("position", frag.position).ok();
        frag_dict
    }).collect();
    dict.set_item("fragments", frag_list)?;

    dict.set_item("final_address", result.final_address)?;
    dict.set_item("variables", result.variables)?;

    Ok(dict.into())
}

/// Convenience function: synthesize spectrum from address string
#[pyfunction]
fn synthesize(address: &str) -> PyResult<PyDict> {
    let addr = TernaryAddress::from_str(address)?;
    let extractor = SpectrumExtractor::default();
    let spectrum = extractor.extract(&addr);
    spectrum_to_dict(spectrum)
}

/// Convenience function: observe all properties from address
#[pyfunction]
fn observe(address: &str) -> PyResult<PyDict> {
    synthesize(address)
}

/// Register Python module
pub fn register_module(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTernaryAddress>()?;
    m.add_class::<PySEntropyCoord>()?;
    m.add_class::<PyPartitionState>()?;
    m.add_class::<PySpectrumExtractor>()?;
    m.add_class::<PyMoleculeEncoder>()?;
    m.add_class::<PyMassScriptInterpreter>()?;
    m.add_function(wrap_pyfunction!(synthesize, m)?)?;
    m.add_function(wrap_pyfunction!(observe, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_address_creation() {
        let addr = PyTernaryAddress::new("012102").unwrap();
        assert_eq!(addr.depth(), 6);
    }
}

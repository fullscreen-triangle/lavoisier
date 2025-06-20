use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use std::collections::HashMap;
use dashmap::DashMap;
use parking_lot::RwLock;
use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod spectrum;
pub mod peak;
pub mod processing;
pub mod memory;
pub mod errors;

pub use spectrum::*;
pub use peak::*;
pub use processing::*;
pub use memory::*;
pub use errors::*;

/// Core spectrum data structure optimized for high-performance processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spectrum {
    pub mz: Vec<f64>,
    pub intensity: Vec<f64>,
    pub retention_time: f64,
    pub ms_level: u8,
    pub precursor_mz: Option<f64>,
    pub scan_id: String,
    pub metadata: HashMap<String, String>,
}

impl Spectrum {
    pub fn new(
        mz: Vec<f64>,
        intensity: Vec<f64>,
        retention_time: f64,
        ms_level: u8,
        scan_id: String,
    ) -> Self {
        Self {
            mz,
            intensity,
            retention_time,
            ms_level,
            precursor_mz: None,
            scan_id,
            metadata: HashMap::new(),
        }
    }

    /// Apply intensity threshold filtering
    pub fn filter_intensity(&mut self, threshold: f64) {
        let indices: Vec<usize> = self
            .intensity
            .iter()
            .enumerate()
            .filter_map(|(i, &intensity)| {
                if intensity >= threshold {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        self.mz = indices.iter().map(|&i| self.mz[i]).collect();
        self.intensity = indices.iter().map(|&i| self.intensity[i]).collect();
    }

    /// Apply m/z tolerance filtering
    pub fn filter_mz_range(&mut self, min_mz: f64, max_mz: f64) {
        let indices: Vec<usize> = self
            .mz
            .iter()
            .enumerate()
            .filter_map(|(i, &mz)| {
                if mz >= min_mz && mz <= max_mz {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        self.mz = indices.iter().map(|&i| self.mz[i]).collect();
        self.intensity = indices.iter().map(|&i| self.intensity[i]).collect();
    }

    /// Normalize intensity values
    pub fn normalize_intensity(&mut self, method: NormalizationMethod) {
        match method {
            NormalizationMethod::MaxIntensity => {
                if let Some(&max_intensity) = self.intensity.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
                    if max_intensity > 0.0 {
                        self.intensity.iter_mut().for_each(|x| *x /= max_intensity);
                    }
                }
            }
            NormalizationMethod::TotalIonCurrent => {
                let tic: f64 = self.intensity.iter().sum();
                if tic > 0.0 {
                    self.intensity.iter_mut().for_each(|x| *x /= tic);
                }
            }
            NormalizationMethod::ZScore => {
                let mean: f64 = self.intensity.iter().sum::<f64>() / self.intensity.len() as f64;
                let variance: f64 = self.intensity.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / self.intensity.len() as f64;
                let std_dev = variance.sqrt();
                
                if std_dev > 0.0 {
                    self.intensity.iter_mut().for_each(|x| *x = (*x - mean) / std_dev);
                }
            }
        }
    }

    /// Find peaks using a simple local maxima algorithm
    pub fn find_peaks(&self, min_intensity: f64, window_size: usize) -> Vec<Peak> {
        let mut peaks = Vec::new();
        let half_window = window_size / 2;

        for i in half_window..self.intensity.len() - half_window {
            if self.intensity[i] < min_intensity {
                continue;
            }

            let is_peak = (i.saturating_sub(half_window)..i)
                .chain((i + 1)..=(i + half_window).min(self.intensity.len() - 1))
                .all(|j| self.intensity[i] >= self.intensity[j]);

            if is_peak {
                peaks.push(Peak {
                    mz: self.mz[i],
                    intensity: self.intensity[i],
                    retention_time: self.retention_time,
                    peak_width: None,
                    area: None,
                    signal_to_noise: None,
                });
            }
        }

        peaks
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NormalizationMethod {
    MaxIntensity,
    TotalIonCurrent,
    ZScore,
}

/// High-performance spectrum collection with concurrent access
pub struct SpectrumCollection {
    spectra: DashMap<String, Spectrum>,
    rt_index: RwLock<Vec<(f64, String)>>, // (retention_time, scan_id)
}

impl SpectrumCollection {
    pub fn new() -> Self {
        Self {
            spectra: DashMap::new(),
            rt_index: RwLock::new(Vec::new()),
        }
    }

    pub fn add_spectrum(&self, spectrum: Spectrum) {
        let scan_id = spectrum.scan_id.clone();
        let rt = spectrum.retention_time;
        
        self.spectra.insert(scan_id.clone(), spectrum);
        
        let mut rt_index = self.rt_index.write();
        rt_index.push((rt, scan_id));
        rt_index.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }

    pub fn get_spectrum(&self, scan_id: &str) -> Option<Spectrum> {
        self.spectra.get(scan_id).map(|s| s.clone())
    }

    pub fn get_spectra_in_rt_range(&self, min_rt: f64, max_rt: f64) -> Vec<Spectrum> {
        let rt_index = self.rt_index.read();
        rt_index
            .iter()
            .filter(|(rt, _)| *rt >= min_rt && *rt <= max_rt)
            .filter_map(|(_, scan_id)| self.get_spectrum(scan_id))
            .collect()
    }

    pub fn process_parallel<F>(&self, mut processor: F) -> Result<()>
    where
        F: Fn(&mut Spectrum) -> Result<()> + Send + Sync,
    {
        self.spectra
            .par_iter_mut()
            .try_for_each(|mut entry| processor(entry.value_mut()))?;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.spectra.len()
    }
}

// Python bindings
#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct PySpectrum {
    inner: Spectrum,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PySpectrum {
    #[new]
    fn new(
        mz: PyReadonlyArray1<f64>,
        intensity: PyReadonlyArray1<f64>,
        retention_time: f64,
        ms_level: u8,
        scan_id: String,
    ) -> Self {
        let mz_vec = mz.as_array().to_vec();
        let intensity_vec = intensity.as_array().to_vec();
        
        Self {
            inner: Spectrum::new(mz_vec, intensity_vec, retention_time, ms_level, scan_id),
        }
    }

    #[getter]
    fn mz<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_slice(py, &self.inner.mz)
    }

    #[getter]
    fn intensity<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_slice(py, &self.inner.intensity)
    }

    #[getter]
    fn retention_time(&self) -> f64 {
        self.inner.retention_time
    }

    #[getter]
    fn ms_level(&self) -> u8 {
        self.inner.ms_level
    }

    #[getter]
    fn scan_id(&self) -> String {
        self.inner.scan_id.clone()
    }

    fn filter_intensity(&mut self, threshold: f64) {
        self.inner.filter_intensity(threshold);
    }

    fn filter_mz_range(&mut self, min_mz: f64, max_mz: f64) {
        self.inner.filter_mz_range(min_mz, max_mz);
    }

    fn normalize_intensity(&mut self, method: String) -> PyResult<()> {
        let norm_method = match method.as_str() {
            "max" => NormalizationMethod::MaxIntensity,
            "tic" => NormalizationMethod::TotalIonCurrent,
            "zscore" => NormalizationMethod::ZScore,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid normalization method. Use 'max', 'tic', or 'zscore'"
            )),
        };
        
        self.inner.normalize_intensity(norm_method);
        Ok(())
    }

    fn find_peaks(&self, min_intensity: f64, window_size: usize) -> Vec<PyPeak> {
        self.inner
            .find_peaks(min_intensity, window_size)
            .into_iter()
            .map(|peak| PyPeak { inner: peak })
            .collect()
    }
}

#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct PySpectrumCollection {
    inner: SpectrumCollection,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PySpectrumCollection {
    #[new]
    fn new() -> Self {
        Self {
            inner: SpectrumCollection::new(),
        }
    }

    fn add_spectrum(&self, spectrum: PySpectrum) {
        self.inner.add_spectrum(spectrum.inner);
    }

    fn get_spectrum(&self, scan_id: String) -> Option<PySpectrum> {
        self.inner.get_spectrum(&scan_id).map(|s| PySpectrum { inner: s })
    }

    fn get_spectra_in_rt_range(&self, min_rt: f64, max_rt: f64) -> Vec<PySpectrum> {
        self.inner
            .get_spectra_in_rt_range(min_rt, max_rt)
            .into_iter()
            .map(|s| PySpectrum { inner: s })
            .collect()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// High-performance batch processing functions
#[cfg(feature = "python-bindings")]
#[pyfunction]
fn batch_filter_intensity(
    spectra: Vec<PySpectrum>,
    threshold: f64,
) -> PyResult<Vec<PySpectrum>> {
    Ok(spectra
        .into_par_iter()
        .map(|mut spectrum| {
            spectrum.filter_intensity(threshold);
            spectrum
        })
        .collect())
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
fn batch_normalize_spectra(
    spectra: Vec<PySpectrum>,
    method: String,
) -> PyResult<Vec<PySpectrum>> {
    let norm_method = match method.as_str() {
        "max" => NormalizationMethod::MaxIntensity,
        "tic" => NormalizationMethod::TotalIonCurrent,
        "zscore" => NormalizationMethod::ZScore,
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid normalization method"
        )),
    };

    Ok(spectra
        .into_par_iter()
        .map(|mut spectrum| {
            spectrum.inner.normalize_intensity(norm_method);
            spectrum
        })
        .collect())
}

#[cfg(feature = "python-bindings")]
#[pymodule]
fn lavoisier_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySpectrum>()?;
    m.add_class::<PySpectrumCollection>()?;
    m.add_class::<PyPeak>()?;
    m.add_function(wrap_pyfunction!(batch_filter_intensity, m)?)?;
    m.add_function(wrap_pyfunction!(batch_normalize_spectra, m)?)?;
    Ok(())
} 
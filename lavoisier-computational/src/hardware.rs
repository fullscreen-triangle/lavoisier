//! Hardware Oscillation Harvesting Module
//!
//! High-performance real-time capture of computational hardware oscillations
//! for use in molecular validation and resonance detection systems.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sysinfo::{System, SystemExt, CpuExt, NetworkExt, DiskExt, ComponentExt};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyReadonlyArray1};
use tokio::time::interval;
use crossbeam::channel::{bounded, Receiver, Sender};
use rayon::prelude::*;

/// Represents a captured hardware oscillation pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareOscillation {
    pub timestamp: f64,
    pub source: String,
    pub frequency: f64,
    pub amplitude: f64,
    pub phase: f64,
    pub metadata: std::collections::HashMap<String, f64>,
}

/// High-performance hardware oscillation harvester
pub struct HardwareHarvester {
    oscillation_buffer: Arc<RwLock<VecDeque<HardwareOscillation>>>,
    sample_rate: f64,
    buffer_size: usize,
    is_harvesting: Arc<parking_lot::Mutex<bool>>,
    system: Arc<RwLock<System>>,
    sender: Option<Sender<HardwareOscillation>>,
    receiver: Option<Receiver<HardwareOscillation>>,
}

impl HardwareHarvester {
    /// Create a new hardware harvester
    pub fn new(sample_rate: f64, buffer_size: usize) -> Self {
        let (sender, receiver) = bounded(buffer_size);

        Self {
            oscillation_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(buffer_size))),
            sample_rate,
            buffer_size,
            is_harvesting: Arc::new(parking_lot::Mutex::new(false)),
            system: Arc::new(RwLock::new(System::new_all())),
            sender: Some(sender),
            receiver: Some(receiver),
        }
    }

    /// Start harvesting hardware oscillations
    pub async fn start_harvesting(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut is_harvesting = self.is_harvesting.lock();
        if *is_harvesting {
            return Ok(());
        }
        *is_harvesting = true;
        drop(is_harvesting);

        let sender = self.sender.as_ref().unwrap().clone();
        let receiver = self.receiver.take().unwrap();
        let buffer = Arc::clone(&self.oscillation_buffer);
        let sample_rate = self.sample_rate;
        let system = Arc::clone(&self.system);
        let is_harvesting_flag = Arc::clone(&self.is_harvesting);

        // Spawn harvesting task
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs_f64(1.0 / sample_rate));

            while *is_harvesting_flag.lock() {
                interval.tick().await;

                // Capture oscillations in parallel
                let oscillations = Self::capture_system_oscillations(&system).await;

                // Send oscillations to buffer
                for osc in oscillations {
                    if sender.send(osc).is_err() {
                        break;
                    }
                }
            }
        });

        // Spawn buffer management task
        tokio::spawn(async move {
            while let Ok(oscillation) = receiver.recv() {
                let mut buffer = buffer.write();
                buffer.push_back(oscillation);

                // Maintain buffer size
                while buffer.len() > buffer.capacity() {
                    buffer.pop_front();
                }
            }
        });

        Ok(())
    }

    /// Stop harvesting
    pub fn stop_harvesting(&self) {
        let mut is_harvesting = self.is_harvesting.lock();
        *is_harvesting = false;
    }

    /// Capture system oscillations in parallel
    async fn capture_system_oscillations(system: &Arc<RwLock<System>>) -> Vec<HardwareOscillation> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let mut oscillations = Vec::new();

        // Update system information
        {
            let mut sys = system.write();
            sys.refresh_all();
        }

        let sys = system.read();

        // CPU oscillations
        let cpu_usage = sys.global_cpu_info().cpu_usage();
        oscillations.push(HardwareOscillation {
            timestamp,
            source: "cpu".to_string(),
            frequency: cpu_usage as f64,
            amplitude: cpu_usage as f64 / 100.0,
            phase: (timestamp * cpu_usage as f64) % 1.0,
            metadata: {
                let mut map = std::collections::HashMap::new();
                map.insert("usage".to_string(), cpu_usage as f64);
                map.insert("core_count".to_string(), sys.cpus().len() as f64);
                map
            },
        });

        // Memory oscillations
        let memory_usage = sys.used_memory() as f64 / sys.total_memory() as f64 * 100.0;
        oscillations.push(HardwareOscillation {
            timestamp,
            source: "memory".to_string(),
            frequency: memory_usage,
            amplitude: memory_usage / 100.0,
            phase: sys.available_memory() as f64 / sys.total_memory() as f64,
            metadata: {
                let mut map = std::collections::HashMap::new();
                map.insert("used_mb".to_string(), sys.used_memory() as f64 / 1024.0 / 1024.0);
                map.insert("total_mb".to_string(), sys.total_memory() as f64 / 1024.0 / 1024.0);
                map
            },
        });

        // Thermal oscillations
        if let Some(component) = sys.components().first() {
            let temperature = component.temperature() as f64;
            oscillations.push(HardwareOscillation {
                timestamp,
                source: "thermal".to_string(),
                frequency: temperature,
                amplitude: (temperature - 20.0) / 80.0, // Normalized to 0-1 range
                phase: temperature % 10.0 / 10.0,
                metadata: {
                    let mut map = std::collections::HashMap::new();
                    map.insert("temperature".to_string(), temperature);
                    map.insert("critical".to_string(), component.critical_temperature().unwrap_or(100.0) as f64);
                    map
                },
            });
        }

        // Network oscillations
        let network_data: Vec<_> = sys.networks().collect();
        if !network_data.is_empty() {
            let total_received: u64 = network_data.iter().map(|(_, data)| data.received()).sum();
            let total_transmitted: u64 = network_data.iter().map(|(_, data)| data.transmitted()).sum();
            let total_activity = total_received + total_transmitted;

            oscillations.push(HardwareOscillation {
                timestamp,
                source: "network".to_string(),
                frequency: (total_activity % 1000) as f64 / 10.0,
                amplitude: (total_activity as f64 / 1024.0 / 1024.0).min(1.0),
                phase: timestamp % 1.0,
                metadata: {
                    let mut map = std::collections::HashMap::new();
                    map.insert("received_mb".to_string(), total_received as f64 / 1024.0 / 1024.0);
                    map.insert("transmitted_mb".to_string(), total_transmitted as f64 / 1024.0 / 1024.0);
                    map
                },
            });
        }

        // Disk oscillations
        let disk_data: Vec<_> = sys.disks().collect();
        if !disk_data.is_empty() {
            let total_read: u64 = disk_data.iter().map(|disk| disk.total_read_bytes()).sum();
            let total_written: u64 = disk_data.iter().map(|disk| disk.total_written_bytes()).sum();
            let total_activity = total_read + total_written;

            oscillations.push(HardwareOscillation {
                timestamp,
                source: "disk".to_string(),
                frequency: (total_activity % 1000) as f64 / 10.0,
                amplitude: (total_activity as f64 / 1024.0 / 1024.0 / 1024.0).min(1.0),
                phase: (timestamp * 2.0) % 1.0,
                metadata: {
                    let mut map = std::collections::HashMap::new();
                    map.insert("read_gb".to_string(), total_read as f64 / 1024.0 / 1024.0 / 1024.0);
                    map.insert("written_gb".to_string(), total_written as f64 / 1024.0 / 1024.0 / 1024.0);
                    map
                },
            });
        }

        oscillations
    }

    /// Get oscillation spectrum for a given duration
    pub fn get_oscillation_spectrum(&self, duration: f64) -> Vec<f64> {
        let buffer = self.oscillation_buffer.read();
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let recent_oscillations: Vec<_> = buffer
            .iter()
            .filter(|osc| current_time - osc.timestamp <= duration)
            .collect();

        if recent_oscillations.is_empty() {
            return vec![0.0; 100];
        }

        // Generate frequency spectrum using parallel processing
        let frequencies: Vec<f64> = recent_oscillations.par_iter().map(|osc| osc.frequency).collect();
        let amplitudes: Vec<f64> = recent_oscillations.par_iter().map(|osc| osc.amplitude).collect();

        // Create histogram
        let mut spectrum = vec![0.0; 100];
        let max_freq = frequencies.iter().fold(0.0, |a, &b| a.max(b)).max(1.0);

        for (freq, amp) in frequencies.iter().zip(amplitudes.iter()) {
            let bin = ((freq / max_freq) * 99.0) as usize;
            spectrum[bin] += amp;
        }

        spectrum
    }

    /// Get resonance signature at target frequency
    pub fn get_resonance_signature(&self, target_frequency: f64, tolerance: f64) -> f64 {
        let buffer = self.oscillation_buffer.read();

        let resonant_oscillations: Vec<_> = buffer
            .iter()
            .filter(|osc| (osc.frequency - target_frequency).abs() <= tolerance)
            .collect();

        if resonant_oscillations.is_empty() {
            return 0.0;
        }

        // Calculate average resonance strength
        let total_amplitude: f64 = resonant_oscillations.par_iter().map(|osc| osc.amplitude).sum();
        total_amplitude / resonant_oscillations.len() as f64
    }
}

/// System oscillation profiler for coordinated analysis
pub struct SystemOscillationProfiler {
    harvester: HardwareHarvester,
    analysis_window: f64,
}

impl SystemOscillationProfiler {
    pub fn new(sample_rate: f64, buffer_size: usize) -> Self {
        Self {
            harvester: HardwareHarvester::new(sample_rate, buffer_size),
            analysis_window: 10.0,
        }
    }

    pub async fn start_profiling(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.harvester.start_harvesting().await
    }

    pub fn stop_profiling(&self) {
        self.harvester.stop_harvesting();
    }

    pub fn get_system_resonance_map(&self) -> std::collections::HashMap<String, Vec<f64>> {
        let buffer = self.harvester.oscillation_buffer.read();
        let mut resonance_map = std::collections::HashMap::new();

        let sources = ["cpu", "memory", "thermal", "network", "disk"];

        for source in &sources {
            let source_oscillations: Vec<_> = buffer
                .iter()
                .filter(|osc| osc.source == *source)
                .collect();

            if !source_oscillations.is_empty() {
                let frequencies: Vec<f64> = source_oscillations.iter().map(|osc| osc.frequency).collect();
                let amplitudes: Vec<f64> = source_oscillations.iter().map(|osc| osc.amplitude).collect();

                // Generate spectrum
                let mut spectrum = vec![0.0; 50];
                let max_freq = frequencies.iter().fold(0.0, |a, &b| a.max(b)).max(1.0);

                for (freq, amp) in frequencies.iter().zip(amplitudes.iter()) {
                    let bin = ((freq / max_freq) * 49.0) as usize;
                    spectrum[bin] += amp;
                }

                resonance_map.insert(source.to_string(), spectrum);
            }
        }

        resonance_map
    }
}

// Python bindings
#[pyclass]
pub struct PyHardwareHarvester {
    inner: HardwareHarvester,
    runtime: tokio::runtime::Runtime,
}

#[pymethods]
impl PyHardwareHarvester {
    #[new]
    fn new(sample_rate: f64, buffer_size: usize) -> PyResult<Self> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        Ok(Self {
            inner: HardwareHarvester::new(sample_rate, buffer_size),
            runtime,
        })
    }

    fn start_harvesting(&mut self) -> PyResult<()> {
        self.runtime.block_on(async {
            self.inner.start_harvesting().await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to start harvesting: {}", e)))
        })
    }

    fn stop_harvesting(&self) {
        self.inner.stop_harvesting();
    }

    fn get_oscillation_spectrum(&self, duration: f64) -> Vec<f64> {
        self.inner.get_oscillation_spectrum(duration)
    }

    fn get_resonance_signature(&self, target_frequency: f64, tolerance: f64) -> f64 {
        self.inner.get_resonance_signature(target_frequency, tolerance)
    }
}

#[pyclass]
pub struct PySystemOscillationProfiler {
    inner: SystemOscillationProfiler,
    runtime: tokio::runtime::Runtime,
}

#[pymethods]
impl PySystemOscillationProfiler {
    #[new]
    fn new(sample_rate: f64, buffer_size: usize) -> PyResult<Self> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        Ok(Self {
            inner: SystemOscillationProfiler::new(sample_rate, buffer_size),
            runtime,
        })
    }

    fn start_profiling(&mut self) -> PyResult<()> {
        self.runtime.block_on(async {
            self.inner.start_profiling().await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to start profiling: {}", e)))
        })
    }

    fn stop_profiling(&self) {
        self.inner.stop_profiling();
    }

    fn get_system_resonance_map(&self) -> PyResult<PyObject> {
        let map = self.inner.get_system_resonance_map();

        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            for (key, value) in map {
                dict.set_item(key, value)?;
            }
            Ok(dict.into())
        })
    }
}

/// Utility function for Python interface
#[pyfunction]
pub fn py_get_system_oscillations(duration: f64) -> PyResult<Vec<(String, f64, f64, f64)>> {
    let mut harvester = HardwareHarvester::new(100.0, 1000);

    // Capture a few oscillations
    let oscillations = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async {
            harvester.start_harvesting().await.unwrap();
            tokio::time::sleep(Duration::from_secs_f64(duration)).await;
            harvester.stop_harvesting();

            let buffer = harvester.oscillation_buffer.read();
            buffer.iter().map(|osc| {
                (osc.source.clone(), osc.frequency, osc.amplitude, osc.phase)
            }).collect::<Vec<_>>()
        });

    Ok(oscillations)
}

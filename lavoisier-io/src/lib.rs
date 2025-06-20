use pyo3::prelude::*;
use lavoisier_core::{Spectrum, SpectrumCollection};
use memmap2::Mmap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use anyhow::{Result, Context};
use thiserror::Error;
use quick_xml::events::Event;
use quick_xml::Reader;
use base64::prelude::*;
use byteorder::{LittleEndian, ReadBytesExt};
use rayon::prelude::*;
use crossbeam::channel;
use dashmap::DashMap;
use parking_lot::Mutex;
use std::sync::Arc;
use std::collections::HashMap;
use regex::Regex;

pub mod mzml;
pub mod compression;
pub mod indexing;
pub mod streaming;

pub use mzml::*;
pub use compression::*;
pub use indexing::*;
pub use streaming::*;

#[derive(Error, Debug)]
pub enum IoError {
    #[error("File not found: {0}")]
    FileNotFound(String),
    #[error("Invalid mzML format: {0}")]
    InvalidFormat(String),
    #[error("Compression error: {0}")]
    CompressionError(String),
    #[error("XML parsing error: {0}")]
    XmlError(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// High-performance mzML reader with memory mapping and parallel processing
pub struct MzMLReader {
    file_path: String,
    mmap: Option<Arc<Mmap>>,
    index: Option<MzMLIndex>,
    spectrum_cache: Arc<DashMap<String, Spectrum>>,
    compression_cache: Arc<Mutex<lru::LruCache<String, Vec<u8>>>>,
}

impl MzMLReader {
    pub fn new<P: AsRef<Path>>(file_path: P) -> Result<Self> {
        let path_str = file_path.as_ref().to_string_lossy().to_string();
        
        // Memory map the file for efficient random access
        let file = File::open(&file_path)
            .with_context(|| format!("Failed to open file: {}", path_str))?;
        
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("Failed to memory map file: {}", path_str))?;

        Ok(Self {
            file_path: path_str,
            mmap: Some(Arc::new(mmap)),
            index: None,
            spectrum_cache: Arc::new(DashMap::new()),
            compression_cache: Arc::new(Mutex::new(lru::LruCache::new(std::num::NonZeroUsize::new(1000).unwrap()))),
        })
    }

    /// Build index for fast random access to spectra
    pub fn build_index(&mut self) -> Result<()> {
        let mmap = self.mmap.as_ref().ok_or(IoError::InvalidFormat("No memory map available".to_string()))?;
        
        let mut index_builder = MzMLIndexBuilder::new();
        let index = index_builder.build_index(&mmap)?;
        
        self.index = Some(index);
        Ok(())
    }

    /// Read all spectra from the file
    pub fn read_spectra(&self) -> Result<SpectrumCollection> {
        let collection = SpectrumCollection::new();
        
        if let Some(index) = &self.index {
            // Use index for efficient access
            self.read_spectra_indexed(&collection)?;
        } else {
            // Sequential reading
            self.read_spectra_sequential(&collection)?;
        }
        
        Ok(collection)
    }

    /// Read spectra using index for random access
    fn read_spectra_indexed(&self, collection: &SpectrumCollection) -> Result<()> {
        let index = self.index.as_ref().unwrap();
        let mmap = self.mmap.as_ref().unwrap();
        
        // Process spectra in parallel
        let spectrum_entries: Vec<_> = index.spectrum_offsets.iter().collect();
        let (sender, receiver) = channel::unbounded();
        
        // Parallel spectrum parsing
        spectrum_entries.par_iter().try_for_each(|(scan_id, &offset)| -> Result<()> {
            let spectrum = self.parse_spectrum_at_offset(mmap, offset, scan_id)?;
            sender.send(spectrum).map_err(|_| IoError::InvalidFormat("Channel send error".to_string()))?;
            Ok(())
        })?;
        
        // Close sender
        drop(sender);
        
        // Collect results
        while let Ok(spectrum) = receiver.recv() {
            collection.add_spectrum(spectrum);
        }
        
        Ok(())
    }

    /// Read spectra sequentially
    fn read_spectra_sequential(&self, collection: &SpectrumCollection) -> Result<()> {
        let mmap = self.mmap.as_ref().unwrap();
        let mut reader = Reader::from_reader(&**mmap);
        reader.trim_text(true);
        
        let mut buf = Vec::new();
        let mut in_spectrum = false;
        let mut current_spectrum_data = Vec::new();
        
        loop {
            match reader.read_event(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    if e.name() == b"spectrum" {
                        in_spectrum = true;
                        current_spectrum_data.clear();
                    }
                    if in_spectrum {
                        current_spectrum_data.extend_from_slice(&buf);
                    }
                }
                Ok(Event::End(ref e)) => {
                    if in_spectrum {
                        current_spectrum_data.extend_from_slice(&buf);
                    }
                    if e.name() == b"spectrum" {
                        in_spectrum = false;
                        if let Ok(spectrum) = self.parse_spectrum_xml(&current_spectrum_data) {
                            collection.add_spectrum(spectrum);
                        }
                    }
                }
                Ok(Event::Text(_)) | Ok(Event::Empty(_)) => {
                    if in_spectrum {
                        current_spectrum_data.extend_from_slice(&buf);
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(IoError::XmlError(format!("XML parsing error: {}", e)).into()),
                _ => {
                    if in_spectrum {
                        current_spectrum_data.extend_from_slice(&buf);
                    }
                }
            }
            buf.clear();
        }
        
        Ok(())
    }

    /// Parse spectrum at specific offset
    fn parse_spectrum_at_offset(&self, mmap: &Mmap, offset: u64, scan_id: &str) -> Result<Spectrum> {
        // Check cache first
        if let Some(cached) = self.spectrum_cache.get(scan_id) {
            return Ok(cached.clone());
        }
        
        // Find spectrum XML bounds
        let start_pos = offset as usize;
        let data = &mmap[start_pos..];
        
        // Find end of spectrum element
        let spectrum_start = b"<spectrum";
        let spectrum_end = b"</spectrum>";
        
        if let Some(end_pos) = find_pattern(data, spectrum_end) {
            let spectrum_xml = &data[..end_pos + spectrum_end.len()];
            let spectrum = self.parse_spectrum_xml(spectrum_xml)?;
            
            // Cache the result
            self.spectrum_cache.insert(scan_id.to_string(), spectrum.clone());
            
            Ok(spectrum)
        } else {
            Err(IoError::InvalidFormat("Could not find spectrum end tag".to_string()).into())
        }
    }

    /// Parse spectrum from XML data
    fn parse_spectrum_xml(&self, xml_data: &[u8]) -> Result<Spectrum> {
        let mut reader = Reader::from_reader(xml_data);
        reader.trim_text(true);
        
        let mut buf = Vec::new();
        let mut scan_id = String::new();
        let mut ms_level = 1u8;
        let mut retention_time = 0.0;
        let mut precursor_mz = None;
        let mut mz_data = Vec::new();
        let mut intensity_data = Vec::new();
        let mut metadata = HashMap::new();
        
        let mut in_binary_data = false;
        let mut is_mz_array = false;
        let mut is_intensity_array = false;
        let mut compression_type = CompressionType::None;
        let mut precision = DataPrecision::Float64;
        
        loop {
            match reader.read_event(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    match e.name() {
                        b"spectrum" => {
                            // Extract spectrum attributes
                            for attr in e.attributes() {
                                if let Ok(attr) = attr {
                                    match attr.key {
                                        b"id" => scan_id = String::from_utf8_lossy(&attr.value).to_string(),
                                        _ => {}
                                    }
                                }
                            }
                        }
                        b"cvParam" => {
                            let mut name = String::new();
                            let mut value = String::new();
                            
                            for attr in e.attributes() {
                                if let Ok(attr) = attr {
                                    match attr.key {
                                        b"name" => name = String::from_utf8_lossy(&attr.value).to_string(),
                                        b"value" => value = String::from_utf8_lossy(&attr.value).to_string(),
                                        _ => {}
                                    }
                                }
                            }
                            
                            // Process CV parameters
                            match name.as_str() {
                                "ms level" => {
                                    if let Ok(level) = value.parse::<u8>() {
                                        ms_level = level;
                                    }
                                }
                                "scan start time" => {
                                    if let Ok(rt) = value.parse::<f64>() {
                                        retention_time = rt;
                                    }
                                }
                                "selected ion m/z" => {
                                    if let Ok(mz) = value.parse::<f64>() {
                                        precursor_mz = Some(mz);
                                    }
                                }
                                "zlib compression" => compression_type = CompressionType::Zlib,
                                "no compression" => compression_type = CompressionType::None,
                                "64-bit float" => precision = DataPrecision::Float64,
                                "32-bit float" => precision = DataPrecision::Float32,
                                "m/z array" => is_mz_array = true,
                                "intensity array" => is_intensity_array = true,
                                _ => {
                                    metadata.insert(name, value);
                                }
                            }
                        }
                        b"binary" => {
                            in_binary_data = true;
                        }
                        _ => {}
                    }
                }
                Ok(Event::Text(e)) => {
                    if in_binary_data {
                        let base64_data = e.unescape_and_decode(&reader)?;
                        let binary_data = BASE64_STANDARD.decode(base64_data.as_bytes())
                            .map_err(|e| IoError::InvalidFormat(format!("Base64 decode error: {}", e)))?;
                        
                        let decompressed = self.decompress_data(&binary_data, compression_type)?;
                        let values = self.parse_binary_data(&decompressed, precision)?;
                        
                        if is_mz_array {
                            mz_data = values;
                            is_mz_array = false;
                        } else if is_intensity_array {
                            intensity_data = values;
                            is_intensity_array = false;
                        }
                    }
                }
                Ok(Event::End(ref e)) => {
                    if e.name() == b"binary" {
                        in_binary_data = false;
                    } else if e.name() == b"spectrum" {
                        break;
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(IoError::XmlError(format!("XML parsing error: {}", e)).into()),
                _ => {}
            }
            buf.clear();
        }
        
        let mut spectrum = Spectrum::new(mz_data, intensity_data, retention_time, ms_level, scan_id);
        spectrum.precursor_mz = precursor_mz;
        spectrum.metadata = metadata;
        
        Ok(spectrum)
    }

    /// Decompress binary data
    fn decompress_data(&self, data: &[u8], compression: CompressionType) -> Result<Vec<u8>> {
        let cache_key = format!("{:?}_{}", compression, data.len());
        
        // Check cache
        {
            let mut cache = self.compression_cache.lock();
            if let Some(cached) = cache.get(&cache_key) {
                return Ok(cached.clone());
            }
        }
        
        let decompressed = match compression {
            CompressionType::None => data.to_vec(),
            CompressionType::Zlib => {
                use flate2::read::ZlibDecoder;
                let mut decoder = ZlibDecoder::new(data);
                let mut result = Vec::new();
                decoder.read_to_end(&mut result)?;
                result
            }
            CompressionType::Gzip => {
                use flate2::read::GzDecoder;
                let mut decoder = GzDecoder::new(data);
                let mut result = Vec::new();
                decoder.read_to_end(&mut result)?;
                result
            }
        };
        
        // Cache result
        {
            let mut cache = self.compression_cache.lock();
            cache.put(cache_key, decompressed.clone());
        }
        
        Ok(decompressed)
    }

    /// Parse binary data to f64 values
    fn parse_binary_data(&self, data: &[u8], precision: DataPrecision) -> Result<Vec<f64>> {
        let mut reader = std::io::Cursor::new(data);
        let mut values = Vec::new();
        
        match precision {
            DataPrecision::Float64 => {
                while reader.position() < data.len() as u64 {
                    match reader.read_f64::<LittleEndian>() {
                        Ok(value) => values.push(value),
                        Err(_) => break,
                    }
                }
            }
            DataPrecision::Float32 => {
                while reader.position() < data.len() as u64 {
                    match reader.read_f32::<LittleEndian>() {
                        Ok(value) => values.push(value as f64),
                        Err(_) => break,
                    }
                }
            }
        }
        
        Ok(values)
    }

    /// Get spectrum by scan ID
    pub fn get_spectrum(&self, scan_id: &str) -> Result<Option<Spectrum>> {
        // Check cache first
        if let Some(cached) = self.spectrum_cache.get(scan_id) {
            return Ok(Some(cached.clone()));
        }
        
        if let Some(index) = &self.index {
            if let Some(&offset) = index.spectrum_offsets.get(scan_id) {
                let mmap = self.mmap.as_ref().unwrap();
                let spectrum = self.parse_spectrum_at_offset(mmap, offset, scan_id)?;
                return Ok(Some(spectrum));
            }
        }
        
        Ok(None)
    }

    /// Get file metadata
    pub fn get_metadata(&self) -> Result<HashMap<String, String>> {
        let mut metadata = HashMap::new();
        
        if let Some(mmap) = &self.mmap {
            // Extract basic file information
            metadata.insert("file_size".to_string(), mmap.len().to_string());
            metadata.insert("file_path".to_string(), self.file_path.clone());
            
            // Parse mzML header for additional metadata
            if let Ok(header_metadata) = self.parse_mzml_header(mmap) {
                metadata.extend(header_metadata);
            }
        }
        
        Ok(metadata)
    }

    /// Parse mzML header for metadata
    fn parse_mzml_header(&self, mmap: &Mmap) -> Result<HashMap<String, String>> {
        let mut metadata = HashMap::new();
        let mut reader = Reader::from_reader(&**mmap);
        reader.trim_text(true);
        
        let mut buf = Vec::new();
        let mut header_done = false;
        
        while !header_done {
            match reader.read_event(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    match e.name() {
                        b"mzML" => {
                            for attr in e.attributes() {
                                if let Ok(attr) = attr {
                                    let key = String::from_utf8_lossy(attr.key);
                                    let value = String::from_utf8_lossy(&attr.value);
                                    metadata.insert(format!("mzML_{}", key), value.to_string());
                                }
                            }
                        }
                        b"spectrumList" => {
                            header_done = true;
                        }
                        _ => {}
                    }
                }
                Ok(Event::Eof) => break,
                Err(_) => break,
                _ => {}
            }
            buf.clear();
        }
        
        Ok(metadata)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CompressionType {
    None,
    Zlib,
    Gzip,
}

#[derive(Debug, Clone, Copy)]
pub enum DataPrecision {
    Float32,
    Float64,
}

/// Utility function to find pattern in byte slice
fn find_pattern(data: &[u8], pattern: &[u8]) -> Option<usize> {
    data.windows(pattern.len()).position(|window| window == pattern)
}

// Python bindings
#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct PyMzMLReader {
    inner: MzMLReader,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyMzMLReader {
    #[new]
    fn new(file_path: String) -> PyResult<Self> {
        let reader = MzMLReader::new(&file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to create reader: {}", e)))?;
        
        Ok(Self { inner: reader })
    }

    fn build_index(&mut self) -> PyResult<()> {
        self.inner.build_index()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to build index: {}", e)))
    }

    fn read_spectra(&self) -> PyResult<lavoisier_core::PySpectrumCollection> {
        let collection = self.inner.read_spectra()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read spectra: {}", e)))?;
        
        Ok(lavoisier_core::PySpectrumCollection { inner: collection })
    }

    fn get_spectrum(&self, scan_id: String) -> PyResult<Option<lavoisier_core::PySpectrum>> {
        let spectrum = self.inner.get_spectrum(&scan_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to get spectrum: {}", e)))?;
        
        Ok(spectrum.map(|s| lavoisier_core::PySpectrum { inner: s }))
    }

    fn get_metadata(&self) -> PyResult<HashMap<String, String>> {
        self.inner.get_metadata()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to get metadata: {}", e)))
    }
}

#[cfg(feature = "python-bindings")]
#[pymodule]
fn lavoisier_io(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyMzMLReader>()?;
    Ok(())
} 
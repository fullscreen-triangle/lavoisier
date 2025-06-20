use std::collections::HashMap;
use anyhow::Result;
use crate::{IoError, CompressionType};

/// mzML file format constants and utilities
pub struct MzMLConstants;

impl MzMLConstants {
    // Common CV parameter accessions
    pub const MS_LEVEL: &'static str = "MS:1000511"; // ms level
    pub const SCAN_START_TIME: &'static str = "MS:1000016"; // scan start time
    pub const SELECTED_ION_MZ: &'static str = "MS:1000744"; // selected ion m/z
    pub const TOTAL_ION_CURRENT: &'static str = "MS:1000285"; // total ion current
    pub const BASE_PEAK_MZ: &'static str = "MS:1000504"; // base peak m/z
    pub const BASE_PEAK_INTENSITY: &'static str = "MS:1000505"; // base peak intensity
    
    // Binary data array types
    pub const MZ_ARRAY: &'static str = "MS:1000514"; // m/z array
    pub const INTENSITY_ARRAY: &'static str = "MS:1000515"; // intensity array
    
    // Compression types
    pub const NO_COMPRESSION: &'static str = "MS:1000576"; // no compression
    pub const ZLIB_COMPRESSION: &'static str = "MS:1000574"; // zlib compression
    pub const GZIP_COMPRESSION: &'static str = "MS:1000572"; // gzip compression
    
    // Data precision
    pub const FLOAT_32: &'static str = "MS:1000521"; // 32-bit float
    pub const FLOAT_64: &'static str = "MS:1000523"; // 64-bit float
    pub const INT_32: &'static str = "MS:1000519"; // 32-bit integer
    pub const INT_64: &'static str = "MS:1000522"; // 64-bit integer
}

/// CV Parameter representation
#[derive(Debug, Clone)]
pub struct CVParam {
    pub accession: String,
    pub name: String,
    pub value: Option<String>,
    pub unit_accession: Option<String>,
    pub unit_name: Option<String>,
}

impl CVParam {
    pub fn new(accession: String, name: String) -> Self {
        Self {
            accession,
            name,
            value: None,
            unit_accession: None,
            unit_name: None,
        }
    }

    pub fn with_value(mut self, value: String) -> Self {
        self.value = Some(value);
        self
    }

    pub fn with_unit(mut self, unit_accession: String, unit_name: String) -> Self {
        self.unit_accession = Some(unit_accession);
        self.unit_name = Some(unit_name);
        self
    }

    /// Parse CV parameter from XML attributes
    pub fn from_attributes(attributes: &HashMap<String, String>) -> Result<Self> {
        let accession = attributes.get("accession")
            .ok_or_else(|| IoError::InvalidFormat("Missing accession in cvParam".to_string()))?
            .clone();
        
        let name = attributes.get("name")
            .ok_or_else(|| IoError::InvalidFormat("Missing name in cvParam".to_string()))?
            .clone();
        
        let mut param = CVParam::new(accession, name);
        
        if let Some(value) = attributes.get("value") {
            param.value = Some(value.clone());
        }
        
        if let Some(unit_accession) = attributes.get("unitAccession") {
            if let Some(unit_name) = attributes.get("unitName") {
                param.unit_accession = Some(unit_accession.clone());
                param.unit_name = Some(unit_name.clone());
            }
        }
        
        Ok(param)
    }

    /// Get value as f64
    pub fn value_as_f64(&self) -> Option<f64> {
        self.value.as_ref().and_then(|v| v.parse().ok())
    }

    /// Get value as i32
    pub fn value_as_i32(&self) -> Option<i32> {
        self.value.as_ref().and_then(|v| v.parse().ok())
    }

    /// Get value as string
    pub fn value_as_string(&self) -> Option<&str> {
        self.value.as_deref()
    }
}

/// Binary data array metadata
#[derive(Debug, Clone)]
pub struct BinaryDataArray {
    pub encoded_length: usize,
    pub data_processing_ref: Option<String>,
    pub array_length: Option<usize>,
    pub cv_params: Vec<CVParam>,
    pub binary_data: Vec<u8>,
}

impl BinaryDataArray {
    pub fn new() -> Self {
        Self {
            encoded_length: 0,
            data_processing_ref: None,
            array_length: None,
            cv_params: Vec::new(),
            binary_data: Vec::new(),
        }
    }

    /// Get compression type from CV parameters
    pub fn get_compression_type(&self) -> CompressionType {
        for param in &self.cv_params {
            match param.accession.as_str() {
                "MS:1000574" => return CompressionType::Zlib,
                "MS:1000572" => return CompressionType::Gzip,
                "MS:1000576" => return CompressionType::None,
                _ => continue,
            }
        }
        CompressionType::None
    }

    /// Get data precision from CV parameters
    pub fn get_data_precision(&self) -> DataPrecision {
        for param in &self.cv_params {
            match param.accession.as_str() {
                "MS:1000521" => return DataPrecision::Float32,
                "MS:1000523" => return DataPrecision::Float64,
                "MS:1000519" => return DataPrecision::Int32,
                "MS:1000522" => return DataPrecision::Int64,
                _ => continue,
            }
        }
        DataPrecision::Float64 // Default
    }

    /// Check if this is an m/z array
    pub fn is_mz_array(&self) -> bool {
        self.cv_params.iter().any(|p| p.accession == "MS:1000514")
    }

    /// Check if this is an intensity array
    pub fn is_intensity_array(&self) -> bool {
        self.cv_params.iter().any(|p| p.accession == "MS:1000515")
    }
}

/// Data precision enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataPrecision {
    Float32,
    Float64,
    Int32,
    Int64,
}

impl DataPrecision {
    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            DataPrecision::Float32 | DataPrecision::Int32 => 4,
            DataPrecision::Float64 | DataPrecision::Int64 => 8,
        }
    }
}

/// Spectrum metadata extracted from mzML
#[derive(Debug, Clone)]
pub struct SpectrumMetadata {
    pub id: String,
    pub index: Option<usize>,
    pub default_array_length: Option<usize>,
    pub data_processing_ref: Option<String>,
    pub source_file_ref: Option<String>,
    pub scan_list: Vec<ScanMetadata>,
    pub precursor_list: Vec<PrecursorMetadata>,
    pub product_list: Vec<ProductMetadata>,
    pub cv_params: Vec<CVParam>,
    pub user_params: HashMap<String, String>,
}

impl SpectrumMetadata {
    pub fn new(id: String) -> Self {
        Self {
            id,
            index: None,
            default_array_length: None,
            data_processing_ref: None,
            source_file_ref: None,
            scan_list: Vec::new(),
            precursor_list: Vec::new(),
            product_list: Vec::new(),
            cv_params: Vec::new(),
            user_params: HashMap::new(),
        }
    }

    /// Get MS level
    pub fn get_ms_level(&self) -> u8 {
        for param in &self.cv_params {
            if param.accession == MzMLConstants::MS_LEVEL {
                if let Some(level) = param.value_as_i32() {
                    return level as u8;
                }
            }
        }
        1 // Default to MS1
    }

    /// Get scan start time
    pub fn get_scan_start_time(&self) -> Option<f64> {
        for param in &self.cv_params {
            if param.accession == MzMLConstants::SCAN_START_TIME {
                return param.value_as_f64();
            }
        }
        
        // Check scan list
        for scan in &self.scan_list {
            for param in &scan.cv_params {
                if param.accession == MzMLConstants::SCAN_START_TIME {
                    return param.value_as_f64();
                }
            }
        }
        
        None
    }

    /// Get total ion current
    pub fn get_total_ion_current(&self) -> Option<f64> {
        for param in &self.cv_params {
            if param.accession == MzMLConstants::TOTAL_ION_CURRENT {
                return param.value_as_f64();
            }
        }
        None
    }

    /// Get base peak m/z
    pub fn get_base_peak_mz(&self) -> Option<f64> {
        for param in &self.cv_params {
            if param.accession == MzMLConstants::BASE_PEAK_MZ {
                return param.value_as_f64();
            }
        }
        None
    }

    /// Get base peak intensity
    pub fn get_base_peak_intensity(&self) -> Option<f64> {
        for param in &self.cv_params {
            if param.accession == MzMLConstants::BASE_PEAK_INTENSITY {
                return param.value_as_f64();
            }
        }
        None
    }
}

/// Scan metadata
#[derive(Debug, Clone)]
pub struct ScanMetadata {
    pub cv_params: Vec<CVParam>,
    pub user_params: HashMap<String, String>,
    pub scan_windows: Vec<ScanWindow>,
}

/// Scan window metadata
#[derive(Debug, Clone)]
pub struct ScanWindow {
    pub cv_params: Vec<CVParam>,
    pub user_params: HashMap<String, String>,
}

/// Precursor metadata
#[derive(Debug, Clone)]
pub struct PrecursorMetadata {
    pub spectrum_ref: Option<String>,
    pub source_file_ref: Option<String>,
    pub external_spectrum_id: Option<String>,
    pub isolation_window: Option<IsolationWindow>,
    pub selected_ion_list: Vec<SelectedIon>,
    pub activation: Option<Activation>,
}

/// Isolation window
#[derive(Debug, Clone)]
pub struct IsolationWindow {
    pub cv_params: Vec<CVParam>,
    pub user_params: HashMap<String, String>,
}

/// Selected ion
#[derive(Debug, Clone)]
pub struct SelectedIon {
    pub cv_params: Vec<CVParam>,
    pub user_params: HashMap<String, String>,
}

impl SelectedIon {
    /// Get selected ion m/z
    pub fn get_mz(&self) -> Option<f64> {
        for param in &self.cv_params {
            if param.accession == MzMLConstants::SELECTED_ION_MZ {
                return param.value_as_f64();
            }
        }
        None
    }
}

/// Activation metadata
#[derive(Debug, Clone)]
pub struct Activation {
    pub cv_params: Vec<CVParam>,
    pub user_params: HashMap<String, String>,
}

/// Product metadata
#[derive(Debug, Clone)]
pub struct ProductMetadata {
    pub isolation_window: Option<IsolationWindow>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cv_param_creation() {
        let param = CVParam::new("MS:1000511".to_string(), "ms level".to_string())
            .with_value("1".to_string());
        
        assert_eq!(param.accession, "MS:1000511");
        assert_eq!(param.name, "ms level");
        assert_eq!(param.value, Some("1".to_string()));
        assert_eq!(param.value_as_i32(), Some(1));
    }

    #[test]
    fn test_binary_data_array() {
        let mut array = BinaryDataArray::new();
        array.cv_params.push(CVParam::new("MS:1000514".to_string(), "m/z array".to_string()));
        array.cv_params.push(CVParam::new("MS:1000574".to_string(), "zlib compression".to_string()));
        
        assert!(array.is_mz_array());
        assert!(!array.is_intensity_array());
        assert_eq!(array.get_compression_type(), CompressionType::Zlib);
    }

    #[test]
    fn test_spectrum_metadata() {
        let mut metadata = SpectrumMetadata::new("scan=1".to_string());
        metadata.cv_params.push(
            CVParam::new("MS:1000511".to_string(), "ms level".to_string())
                .with_value("2".to_string())
        );
        
        assert_eq!(metadata.get_ms_level(), 2);
    }
} 
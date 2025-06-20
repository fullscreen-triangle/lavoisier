use std::collections::HashMap;
use anyhow::Result;
use crate::IoError;

/// Index for fast random access to mzML spectra
#[derive(Debug, Clone)]
pub struct MzMLIndex {
    pub spectrum_offsets: HashMap<String, u64>,
    pub chromatogram_offsets: HashMap<String, u64>,
    pub spectrum_count: usize,
    pub chromatogram_count: usize,
}

impl MzMLIndex {
    pub fn new() -> Self {
        Self {
            spectrum_offsets: HashMap::new(),
            chromatogram_offsets: HashMap::new(),
            spectrum_count: 0,
            chromatogram_count: 0,
        }
    }
}

/// Builder for creating mzML indices
pub struct MzMLIndexBuilder {
    index: MzMLIndex,
}

impl MzMLIndexBuilder {
    pub fn new() -> Self {
        Self {
            index: MzMLIndex::new(),
        }
    }

    /// Build index from memory-mapped mzML data
    pub fn build_index(&mut self, data: &[u8]) -> Result<MzMLIndex> {
        // Simple implementation - scan for spectrum tags
        let spectrum_tag = b"<spectrum";
        let chromatogram_tag = b"<chromatogram";
        
        let mut pos = 0;
        let mut spectrum_counter = 0;
        let mut chromatogram_counter = 0;
        
        while pos < data.len() {
            if let Some(spectrum_pos) = find_next_pattern(&data[pos..], spectrum_tag) {
                let actual_pos = pos + spectrum_pos;
                
                // Extract scan ID from spectrum tag
                if let Some(scan_id) = self.extract_scan_id(&data[actual_pos..]) {
                    self.index.spectrum_offsets.insert(scan_id, actual_pos as u64);
                    spectrum_counter += 1;
                }
                
                pos = actual_pos + spectrum_tag.len();
            } else if let Some(chrom_pos) = find_next_pattern(&data[pos..], chromatogram_tag) {
                let actual_pos = pos + chrom_pos;
                
                // Extract chromatogram ID
                if let Some(chrom_id) = self.extract_chromatogram_id(&data[actual_pos..]) {
                    self.index.chromatogram_offsets.insert(chrom_id, actual_pos as u64);
                    chromatogram_counter += 1;
                }
                
                pos = actual_pos + chromatogram_tag.len();
            } else {
                break;
            }
        }
        
        self.index.spectrum_count = spectrum_counter;
        self.index.chromatogram_count = chromatogram_counter;
        
        Ok(self.index.clone())
    }

    /// Extract scan ID from spectrum tag
    fn extract_scan_id(&self, data: &[u8]) -> Option<String> {
        // Look for id="..." pattern
        let id_pattern = b"id=\"";
        if let Some(id_start) = find_next_pattern(data, id_pattern) {
            let id_value_start = id_start + id_pattern.len();
            if let Some(quote_end) = find_next_pattern(&data[id_value_start..], b"\"") {
                let id_bytes = &data[id_value_start..id_value_start + quote_end];
                return String::from_utf8(id_bytes.to_vec()).ok();
            }
        }
        None
    }

    /// Extract chromatogram ID from chromatogram tag
    fn extract_chromatogram_id(&self, data: &[u8]) -> Option<String> {
        // Look for id="..." pattern
        let id_pattern = b"id=\"";
        if let Some(id_start) = find_next_pattern(data, id_pattern) {
            let id_value_start = id_start + id_pattern.len();
            if let Some(quote_end) = find_next_pattern(&data[id_value_start..], b"\"") {
                let id_bytes = &data[id_value_start..id_value_start + quote_end];
                return String::from_utf8(id_bytes.to_vec()).ok();
            }
        }
        None
    }
}

/// Find next occurrence of pattern in data
fn find_next_pattern(data: &[u8], pattern: &[u8]) -> Option<usize> {
    data.windows(pattern.len()).position(|window| window == pattern)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_building() {
        let test_data = b"<spectrum id=\"scan=1\"><spectrum id=\"scan=2\">";
        let mut builder = MzMLIndexBuilder::new();
        let index = builder.build_index(test_data).unwrap();
        
        assert_eq!(index.spectrum_count, 2);
        assert!(index.spectrum_offsets.contains_key("scan=1"));
        assert!(index.spectrum_offsets.contains_key("scan=2"));
    }

    #[test]
    fn test_scan_id_extraction() {
        let test_data = b"<spectrum id=\"scan=123\" ms_level=\"1\">";
        let builder = MzMLIndexBuilder::new();
        let scan_id = builder.extract_scan_id(test_data);
        
        assert_eq!(scan_id, Some("scan=123".to_string()));
    }
} 
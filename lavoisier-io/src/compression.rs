use std::io::Read;
use anyhow::Result;
use crate::IoError;

/// Compression algorithms supported for mzML binary data
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionType {
    None,
    Zlib,
    Gzip,
    Lz4,
    Zstd,
}

impl CompressionType {
    /// Detect compression type from CV parameter name
    pub fn from_cv_param(param_name: &str) -> Self {
        match param_name {
            "zlib compression" => CompressionType::Zlib,
            "gzip compression" => CompressionType::Gzip,
            "lz4 compression" => CompressionType::Lz4,
            "zstd compression" => CompressionType::Zstd,
            "no compression" | _ => CompressionType::None,
        }
    }
}

/// High-performance decompression functions
pub struct Decompressor;

impl Decompressor {
    /// Decompress data based on compression type
    pub fn decompress(data: &[u8], compression: CompressionType) -> Result<Vec<u8>> {
        match compression {
            CompressionType::None => Ok(data.to_vec()),
            CompressionType::Zlib => Self::decompress_zlib(data),
            CompressionType::Gzip => Self::decompress_gzip(data),
            CompressionType::Lz4 => Self::decompress_lz4(data),
            CompressionType::Zstd => Self::decompress_zstd(data),
        }
    }

    /// Decompress zlib data
    fn decompress_zlib(data: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::ZlibDecoder;
        let mut decoder = ZlibDecoder::new(data);
        let mut result = Vec::new();
        decoder.read_to_end(&mut result)
            .map_err(|e| IoError::CompressionError(format!("Zlib decompression failed: {}", e)))?;
        Ok(result)
    }

    /// Decompress gzip data
    fn decompress_gzip(data: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::GzDecoder;
        let mut decoder = GzDecoder::new(data);
        let mut result = Vec::new();
        decoder.read_to_end(&mut result)
            .map_err(|e| IoError::CompressionError(format!("Gzip decompression failed: {}", e)))?;
        Ok(result)
    }

    /// Decompress LZ4 data
    fn decompress_lz4(data: &[u8]) -> Result<Vec<u8>> {
        lz4::block::decompress(data, None)
            .map_err(|e| IoError::CompressionError(format!("LZ4 decompression failed: {}", e)).into())
    }

    /// Decompress Zstd data
    fn decompress_zstd(data: &[u8]) -> Result<Vec<u8>> {
        zstd::bulk::decompress(data, 1024 * 1024) // 1MB max
            .map_err(|e| IoError::CompressionError(format!("Zstd decompression failed: {}", e)).into())
    }

    /// Estimate decompressed size for buffer allocation
    pub fn estimate_decompressed_size(compressed_size: usize, compression: CompressionType) -> usize {
        match compression {
            CompressionType::None => compressed_size,
            CompressionType::Zlib | CompressionType::Gzip => compressed_size * 4, // Conservative estimate
            CompressionType::Lz4 => compressed_size * 3, // LZ4 typically has lower compression ratio
            CompressionType::Zstd => compressed_size * 5, // Zstd can achieve high compression ratios
        }
    }
}

/// Parallel decompression for multiple data chunks
pub struct ParallelDecompressor;

impl ParallelDecompressor {
    /// Decompress multiple chunks in parallel
    pub fn decompress_parallel(
        chunks: Vec<(Vec<u8>, CompressionType)>,
    ) -> Result<Vec<Vec<u8>>> {
        use rayon::prelude::*;
        
        chunks
            .into_par_iter()
            .map(|(data, compression)| Decompressor::decompress(&data, compression))
            .collect()
    }
}

/// Streaming decompressor for large files
pub struct StreamingDecompressor {
    compression: CompressionType,
    buffer_size: usize,
}

impl StreamingDecompressor {
    pub fn new(compression: CompressionType, buffer_size: usize) -> Self {
        Self {
            compression,
            buffer_size,
        }
    }

    /// Decompress data in streaming fashion
    pub fn decompress_stream<R: Read>(&self, mut reader: R) -> Result<Vec<u8>> {
        let mut buffer = vec![0u8; self.buffer_size];
        let mut compressed_data = Vec::new();
        
        // Read all compressed data first
        loop {
            match reader.read(&mut buffer) {
                Ok(0) => break, // EOF
                Ok(n) => compressed_data.extend_from_slice(&buffer[..n]),
                Err(e) => return Err(IoError::Io(e).into()),
            }
        }
        
        // Decompress all at once
        Decompressor::decompress(&compressed_data, self.compression)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_type_detection() {
        assert_eq!(CompressionType::from_cv_param("zlib compression"), CompressionType::Zlib);
        assert_eq!(CompressionType::from_cv_param("no compression"), CompressionType::None);
        assert_eq!(CompressionType::from_cv_param("unknown"), CompressionType::None);
    }

    #[test]
    fn test_no_compression() {
        let data = b"test data";
        let result = Decompressor::decompress(data, CompressionType::None).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_zlib_compression() {
        use flate2::write::ZlibEncoder;
        use flate2::Compression;
        use std::io::Write;
        
        let original_data = b"test data for compression";
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original_data).unwrap();
        let compressed = encoder.finish().unwrap();
        
        let decompressed = Decompressor::decompress(&compressed, CompressionType::Zlib).unwrap();
        assert_eq!(decompressed, original_data);
    }

    #[test]
    fn test_size_estimation() {
        let size = Decompressor::estimate_decompressed_size(100, CompressionType::Zlib);
        assert_eq!(size, 400);
        
        let size = Decompressor::estimate_decompressed_size(100, CompressionType::None);
        assert_eq!(size, 100);
    }
} 
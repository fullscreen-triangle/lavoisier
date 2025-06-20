use crate::{Spectrum, SpectrumCollection, Peak, NormalizationMethod};
use rayon::prelude::*;
use std::sync::Arc;
use anyhow::Result;
use dashmap::DashMap;

/// High-performance batch processing pipeline
pub struct BatchProcessor {
    thread_pool_size: usize,
    chunk_size: usize,
}

impl BatchProcessor {
    pub fn new(thread_pool_size: usize, chunk_size: usize) -> Self {
        Self {
            thread_pool_size,
            chunk_size,
        }
    }

    /// Process multiple spectra in parallel with a given function
    pub fn process_parallel<F>(&self, spectra: &mut [Spectrum], processor: F) -> Result<()>
    where
        F: Fn(&mut Spectrum) -> Result<()> + Send + Sync,
    {
        spectra
            .par_chunks_mut(self.chunk_size)
            .try_for_each(|chunk| {
                chunk.iter_mut().try_for_each(|spectrum| processor(spectrum))
            })?;
        Ok(())
    }

    /// Batch normalize spectra
    pub fn batch_normalize(&self, spectra: &mut [Spectrum], method: NormalizationMethod) -> Result<()> {
        self.process_parallel(spectra, |spectrum| {
            spectrum.normalize_intensity(method);
            Ok(())
        })
    }

    /// Batch filter by intensity
    pub fn batch_filter_intensity(&self, spectra: &mut [Spectrum], threshold: f64) -> Result<()> {
        self.process_parallel(spectra, |spectrum| {
            spectrum.filter_intensity(threshold);
            Ok(())
        })
    }

    /// Batch filter by m/z range
    pub fn batch_filter_mz_range(&self, spectra: &mut [Spectrum], min_mz: f64, max_mz: f64) -> Result<()> {
        self.process_parallel(spectra, |spectrum| {
            spectrum.filter_mz_range(min_mz, max_mz);
            Ok(())
        })
    }

    /// Batch peak detection
    pub fn batch_detect_peaks(&self, spectra: &[Spectrum], min_intensity: f64, window_size: usize) -> Result<Vec<Vec<Peak>>> {
        let results: Result<Vec<_>, _> = spectra
            .par_chunks(self.chunk_size)
            .map(|chunk| {
                chunk.iter()
                    .map(|spectrum| spectrum.find_peaks(min_intensity, window_size))
                    .collect::<Vec<_>>()
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|chunks| chunks.into_iter().flatten().collect());
        
        results.map_err(|e| anyhow::anyhow!("Peak detection failed: {}", e))
    }
}

/// Processing pipeline for complete workflows
pub struct ProcessingPipeline {
    steps: Vec<Box<dyn ProcessingStep + Send + Sync>>,
}

impl ProcessingPipeline {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
        }
    }

    /// Add a processing step to the pipeline
    pub fn add_step<T: ProcessingStep + Send + Sync + 'static>(mut self, step: T) -> Self {
        self.steps.push(Box::new(step));
        self
    }

    /// Execute the pipeline on a spectrum collection
    pub fn execute(&self, collection: &mut SpectrumCollection) -> Result<ProcessingReport> {
        let mut report = ProcessingReport::new();
        
        for (i, step) in self.steps.iter().enumerate() {
            let step_name = step.name();
            let start_time = std::time::Instant::now();
            
            step.process(collection)?;
            
            let duration = start_time.elapsed();
            report.add_step_result(step_name, duration, true, None);
        }
        
        Ok(report)
    }
}

/// Trait for processing steps
pub trait ProcessingStep {
    fn name(&self) -> String;
    fn process(&self, collection: &mut SpectrumCollection) -> Result<()>;
}

/// Intensity filtering step
pub struct IntensityFilterStep {
    threshold: f64,
}

impl IntensityFilterStep {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl ProcessingStep for IntensityFilterStep {
    fn name(&self) -> String {
        format!("Intensity Filter (threshold: {})", self.threshold)
    }

    fn process(&self, collection: &mut SpectrumCollection) -> Result<()> {
        collection.process_parallel(|spectrum| {
            spectrum.filter_intensity(self.threshold);
            Ok(())
        })
    }
}

/// Normalization step
pub struct NormalizationStep {
    method: NormalizationMethod,
}

impl NormalizationStep {
    pub fn new(method: NormalizationMethod) -> Self {
        Self { method }
    }
}

impl ProcessingStep for NormalizationStep {
    fn name(&self) -> String {
        format!("Normalization ({:?})", self.method)
    }

    fn process(&self, collection: &mut SpectrumCollection) -> Result<()> {
        collection.process_parallel(|spectrum| {
            spectrum.normalize_intensity(self.method);
            Ok(())
        })
    }
}

/// M/Z range filtering step
pub struct MzRangeFilterStep {
    min_mz: f64,
    max_mz: f64,
}

impl MzRangeFilterStep {
    pub fn new(min_mz: f64, max_mz: f64) -> Self {
        Self { min_mz, max_mz }
    }
}

impl ProcessingStep for MzRangeFilterStep {
    fn name(&self) -> String {
        format!("M/Z Range Filter ({:.2} - {:.2})", self.min_mz, self.max_mz)
    }

    fn process(&self, collection: &mut SpectrumCollection) -> Result<()> {
        collection.process_parallel(|spectrum| {
            spectrum.filter_mz_range(self.min_mz, self.max_mz);
            Ok(())
        })
    }
}

/// Processing report
#[derive(Debug)]
pub struct ProcessingReport {
    steps: Vec<StepResult>,
    total_duration: std::time::Duration,
}

impl ProcessingReport {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            total_duration: std::time::Duration::from_secs(0),
        }
    }

    pub fn add_step_result(&mut self, name: String, duration: std::time::Duration, success: bool, error: Option<String>) {
        self.steps.push(StepResult {
            name,
            duration,
            success,
            error,
        });
        self.total_duration += duration;
    }

    pub fn total_duration(&self) -> std::time::Duration {
        self.total_duration
    }

    pub fn success_count(&self) -> usize {
        self.steps.iter().filter(|s| s.success).count()
    }

    pub fn failure_count(&self) -> usize {
        self.steps.iter().filter(|s| !s.success).count()
    }

    pub fn steps(&self) -> &[StepResult] {
        &self.steps
    }
}

#[derive(Debug)]
pub struct StepResult {
    pub name: String,
    pub duration: std::time::Duration,
    pub success: bool,
    pub error: Option<String>,
}

/// Memory-efficient streaming processor for large datasets
pub struct StreamingProcessor {
    buffer_size: usize,
    processing_pipeline: ProcessingPipeline,
}

impl StreamingProcessor {
    pub fn new(buffer_size: usize, pipeline: ProcessingPipeline) -> Self {
        Self {
            buffer_size,
            processing_pipeline: pipeline,
        }
    }

    /// Process spectra in streaming fashion
    pub fn process_stream<I>(&self, spectra_iter: I) -> Result<Vec<ProcessingReport>>
    where
        I: Iterator<Item = Spectrum>,
    {
        let mut reports = Vec::new();
        let mut buffer = Vec::with_capacity(self.buffer_size);

        for spectrum in spectra_iter {
            buffer.push(spectrum);

            if buffer.len() >= self.buffer_size {
                let mut collection = SpectrumCollection::new();
                for spec in buffer.drain(..) {
                    collection.add_spectrum(spec);
                }

                let report = self.processing_pipeline.execute(&mut collection)?;
                reports.push(report);
            }
        }

        // Process remaining spectra
        if !buffer.is_empty() {
            let mut collection = SpectrumCollection::new();
            for spec in buffer {
                collection.add_spectrum(spec);
            }

            let report = self.processing_pipeline.execute(&mut collection)?;
            reports.push(report);
        }

        Ok(reports)
    }
}

/// Parallel processing utilities
pub struct ParallelUtils;

impl ParallelUtils {
    /// Get optimal thread count for current system
    pub fn optimal_thread_count() -> usize {
        num_cpus::get().max(1)
    }

    /// Get optimal chunk size for given data size
    pub fn optimal_chunk_size(data_size: usize, thread_count: usize) -> usize {
        (data_size / thread_count).max(1).min(1000)
    }

    /// Initialize thread pool with optimal settings
    pub fn init_thread_pool() -> Result<()> {
        let thread_count = Self::optimal_thread_count();
        rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .thread_name(|i| format!("lavoisier-worker-{}", i))
            .build_global()
            .map_err(|e| anyhow::anyhow!("Failed to initialize thread pool: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_processor() {
        let processor = BatchProcessor::new(4, 10);
        let mut spectra = vec![
            Spectrum::new(vec![100.0, 200.0], vec![1000.0, 2000.0], 1.0, 1, "scan1".to_string()),
            Spectrum::new(vec![100.0, 200.0], vec![500.0, 1500.0], 2.0, 1, "scan2".to_string()),
        ];

        processor.batch_normalize(&mut spectra, NormalizationMethod::MaxIntensity).unwrap();
        
        // Check that normalization was applied
        assert_eq!(spectra[0].intensity[1], 1.0); // Max intensity should be 1.0
        assert_eq!(spectra[1].intensity[1], 1.0); // Max intensity should be 1.0
    }

    #[test]
    fn test_processing_pipeline() {
        let pipeline = ProcessingPipeline::new()
            .add_step(IntensityFilterStep::new(100.0))
            .add_step(NormalizationStep::new(NormalizationMethod::MaxIntensity));

        let mut collection = SpectrumCollection::new();
        collection.add_spectrum(Spectrum::new(
            vec![100.0, 200.0, 300.0],
            vec![50.0, 1000.0, 2000.0],
            1.0,
            1,
            "test_scan".to_string(),
        ));

        let report = pipeline.execute(&mut collection).unwrap();
        assert_eq!(report.success_count(), 2);
        assert_eq!(report.failure_count(), 0);
    }

    #[test]
    fn test_parallel_utils() {
        let thread_count = ParallelUtils::optimal_thread_count();
        assert!(thread_count >= 1);

        let chunk_size = ParallelUtils::optimal_chunk_size(1000, thread_count);
        assert!(chunk_size >= 1);
        assert!(chunk_size <= 1000);
    }
} 
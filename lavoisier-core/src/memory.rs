use std::alloc::{alloc, dealloc, Layout};
use std::ptr;
use std::sync::Arc;
use parking_lot::Mutex;

/// Memory pool for efficient allocation of spectrum data
pub struct MemoryPool {
    pools: Vec<Pool>,
    chunk_size: usize,
}

struct Pool {
    data: Arc<Mutex<Vec<u8>>>,
    allocated: Arc<Mutex<Vec<bool>>>,
    chunk_size: usize,
}

impl MemoryPool {
    pub fn new(chunk_size: usize, initial_pools: usize) -> Self {
        let mut pools = Vec::with_capacity(initial_pools);
        
        for _ in 0..initial_pools {
            pools.push(Pool::new(chunk_size));
        }
        
        Self {
            pools,
            chunk_size,
        }
    }

    /// Allocate a chunk of memory
    pub fn allocate(&mut self, size: usize) -> Option<MemoryChunk> {
        if size > self.chunk_size {
            // For large allocations, allocate directly
            return Some(MemoryChunk::new_direct(size));
        }

        // Try to find a free chunk in existing pools
        for (pool_idx, pool) in self.pools.iter().enumerate() {
            if let Some(chunk_idx) = pool.allocate() {
                return Some(MemoryChunk::new_pooled(pool_idx, chunk_idx, self.chunk_size));
            }
        }

        // Create a new pool if needed
        let new_pool = Pool::new(self.chunk_size);
        let chunk_idx = new_pool.allocate().expect("New pool should have free chunks");
        let pool_idx = self.pools.len();
        self.pools.push(new_pool);
        
        Some(MemoryChunk::new_pooled(pool_idx, chunk_idx, self.chunk_size))
    }

    /// Deallocate a memory chunk
    pub fn deallocate(&mut self, chunk: MemoryChunk) {
        match chunk.allocation_type {
            AllocationType::Pooled { pool_idx, chunk_idx } => {
                if let Some(pool) = self.pools.get(pool_idx) {
                    pool.deallocate(chunk_idx);
                }
            }
            AllocationType::Direct { .. } => {
                // Direct allocations are handled by the chunk's Drop implementation
            }
        }
    }
}

impl Pool {
    fn new(chunk_size: usize) -> Self {
        let pool_size = 1024 * 1024; // 1MB pool
        let num_chunks = pool_size / chunk_size;
        
        Self {
            data: Arc::new(Mutex::new(vec![0u8; pool_size])),
            allocated: Arc::new(Mutex::new(vec![false; num_chunks])),
            chunk_size,
        }
    }

    fn allocate(&self) -> Option<usize> {
        let mut allocated = self.allocated.lock();
        
        for (i, &is_allocated) in allocated.iter().enumerate() {
            if !is_allocated {
                allocated[i] = true;
                return Some(i);
            }
        }
        
        None
    }

    fn deallocate(&self, chunk_idx: usize) {
        let mut allocated = self.allocated.lock();
        if chunk_idx < allocated.len() {
            allocated[chunk_idx] = false;
        }
    }
}

/// Memory chunk representation
pub struct MemoryChunk {
    allocation_type: AllocationType,
    size: usize,
}

enum AllocationType {
    Pooled { pool_idx: usize, chunk_idx: usize },
    Direct { ptr: *mut u8, layout: Layout },
}

impl MemoryChunk {
    fn new_pooled(pool_idx: usize, chunk_idx: usize, size: usize) -> Self {
        Self {
            allocation_type: AllocationType::Pooled { pool_idx, chunk_idx },
            size,
        }
    }

    fn new_direct(size: usize) -> Self {
        let layout = Layout::from_size_align(size, 8).expect("Invalid layout");
        let ptr = unsafe { alloc(layout) };
        
        if ptr.is_null() {
            panic!("Failed to allocate memory");
        }
        
        Self {
            allocation_type: AllocationType::Direct { ptr, layout },
            size,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for MemoryChunk {
    fn drop(&mut self) {
        if let AllocationType::Direct { ptr, layout } = self.allocation_type {
            unsafe {
                dealloc(ptr, layout);
            }
        }
    }
}

unsafe impl Send for MemoryChunk {}
unsafe impl Sync for MemoryChunk {}

/// Memory-mapped buffer for large datasets
pub struct MappedBuffer {
    data: memmap2::Mmap,
    size: usize,
}

impl MappedBuffer {
    pub fn new(file: std::fs::File) -> Result<Self, std::io::Error> {
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let size = mmap.len();
        
        Ok(Self {
            data: mmap,
            size,
        })
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Get a slice of the mapped data
    pub fn slice(&self, start: usize, len: usize) -> Option<&[u8]> {
        if start + len <= self.size {
            Some(&self.data[start..start + len])
        } else {
            None
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub peak_allocated: usize,
    pub current_allocated: usize,
    pub allocation_count: usize,
    pub deallocation_count: usize,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            total_allocated: 0,
            peak_allocated: 0,
            current_allocated: 0,
            allocation_count: 0,
            deallocation_count: 0,
        }
    }
}

/// Global memory tracker
static mut MEMORY_STATS: MemoryStats = MemoryStats {
    total_allocated: 0,
    peak_allocated: 0,
    current_allocated: 0,
    allocation_count: 0,
    deallocation_count: 0,
};

static MEMORY_STATS_MUTEX: Mutex<()> = Mutex::new(());

pub fn track_allocation(size: usize) {
    let _lock = MEMORY_STATS_MUTEX.lock();
    unsafe {
        MEMORY_STATS.total_allocated += size;
        MEMORY_STATS.current_allocated += size;
        MEMORY_STATS.allocation_count += 1;
        
        if MEMORY_STATS.current_allocated > MEMORY_STATS.peak_allocated {
            MEMORY_STATS.peak_allocated = MEMORY_STATS.current_allocated;
        }
    }
}

pub fn track_deallocation(size: usize) {
    let _lock = MEMORY_STATS_MUTEX.lock();
    unsafe {
        MEMORY_STATS.current_allocated = MEMORY_STATS.current_allocated.saturating_sub(size);
        MEMORY_STATS.deallocation_count += 1;
    }
}

pub fn get_memory_stats() -> MemoryStats {
    let _lock = MEMORY_STATS_MUTEX.lock();
    unsafe { MEMORY_STATS.clone() }
}

pub fn reset_memory_stats() {
    let _lock = MEMORY_STATS_MUTEX.lock();
    unsafe {
        MEMORY_STATS = MemoryStats::default();
    }
}

/// Zero-copy data structures for efficient processing
pub struct ZeroCopySlice<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> ZeroCopySlice<'a> {
    pub fn new(data: &'a [u8], offset: usize) -> Self {
        Self { data, offset }
    }

    pub fn read_f64(&mut self) -> Option<f64> {
        if self.offset + 8 <= self.data.len() {
            let bytes = &self.data[self.offset..self.offset + 8];
            self.offset += 8;
            Some(f64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7],
            ]))
        } else {
            None
        }
    }

    pub fn read_f32(&mut self) -> Option<f32> {
        if self.offset + 4 <= self.data.len() {
            let bytes = &self.data[self.offset..self.offset + 4];
            self.offset += 4;
            Some(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        } else {
            None
        }
    }

    pub fn read_u32(&mut self) -> Option<u32> {
        if self.offset + 4 <= self.data.len() {
            let bytes = &self.data[self.offset..self.offset + 4];
            self.offset += 4;
            Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        } else {
            None
        }
    }

    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(1024, 1);
        
        let chunk1 = pool.allocate(512).unwrap();
        let chunk2 = pool.allocate(512).unwrap();
        
        assert_eq!(chunk1.size(), 1024); // Pool chunk size
        assert_eq!(chunk2.size(), 1024); // Pool chunk size
        
        pool.deallocate(chunk1);
        pool.deallocate(chunk2);
    }

    #[test]
    fn test_zero_copy_slice() {
        let data = vec![
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // 1.0 as f64
            0x00, 0x00, 0x80, 0x3F, // 1.0 as f32
        ];
        
        let mut slice = ZeroCopySlice::new(&data, 0);
        
        let f64_val = slice.read_f64().unwrap();
        let f32_val = slice.read_f32().unwrap();
        
        assert!((f64_val - 1.0).abs() < f64::EPSILON);
        assert!((f32_val - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_memory_tracking() {
        reset_memory_stats();
        
        track_allocation(1024);
        track_allocation(512);
        
        let stats = get_memory_stats();
        assert_eq!(stats.total_allocated, 1536);
        assert_eq!(stats.current_allocated, 1536);
        assert_eq!(stats.allocation_count, 2);
        
        track_deallocation(512);
        
        let stats = get_memory_stats();
        assert_eq!(stats.current_allocated, 1024);
        assert_eq!(stats.deallocation_count, 1);
    }
} 
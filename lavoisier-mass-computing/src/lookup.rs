//! Precomputed lookup tables for high-performance ternary operations

use lazy_static::lazy_static;

/// Maximum supported address depth
pub const MAX_DEPTH: usize = 64;

/// Tryte size (6 trits)
pub const TRYTE_SIZE: usize = 6;

/// Number of cells in a tryte (3^6 = 729)
pub const TRYTE_CELLS: usize = 729;

lazy_static! {
    /// Precomputed powers of 3: POWERS_OF_3[k] = 3^k
    pub static ref POWERS_OF_3: [u64; MAX_DEPTH + 1] = {
        let mut powers = [0u64; MAX_DEPTH + 1];
        powers[0] = 1;
        for k in 1..=MAX_DEPTH {
            powers[k] = powers[k - 1].saturating_mul(3);
        }
        powers
    };

    /// Precomputed inverse powers of 3: INV_POWERS_OF_3[k] = 1/3^k
    pub static ref INV_POWERS_OF_3: [f64; MAX_DEPTH + 1] = {
        let mut inv_powers = [0.0f64; MAX_DEPTH + 1];
        inv_powers[0] = 1.0;
        for k in 1..=MAX_DEPTH {
            inv_powers[k] = inv_powers[k - 1] / 3.0;
        }
        inv_powers
    };

    /// Precomputed cell widths at each depth: CELL_WIDTHS[k] = (1/3)^k
    pub static ref CELL_WIDTHS: [f64; MAX_DEPTH + 1] = *INV_POWERS_OF_3;

    /// Precomputed trit-to-offset multipliers for coordinate conversion
    /// TRIT_OFFSETS[trit] = trit / 3.0
    pub static ref TRIT_OFFSETS: [f64; 3] = [0.0, 1.0 / 3.0, 2.0 / 3.0];

    /// Precomputed tryte-to-cell index lookup (for all 729 possible trytes)
    pub static ref TRYTE_TO_INDEX: [u16; 729] = {
        let mut lookup = [0u16; 729];
        for i in 0..729 {
            lookup[i] = i as u16;
        }
        lookup
    };

    /// Precomputed index-to-tryte lookup (for all 729 cells)
    pub static ref INDEX_TO_TRYTE: [[u8; 6]; 729] = {
        let mut lookup = [[0u8; 6]; 729];
        for i in 0..729 {
            let mut val = i;
            for j in (0..6).rev() {
                lookup[i][j] = (val % 3) as u8;
                val /= 3;
            }
        }
        lookup
    };
}

/// Get the power of 3 for a given exponent
#[inline]
pub fn pow3(k: usize) -> u64 {
    if k <= MAX_DEPTH {
        POWERS_OF_3[k]
    } else {
        3u64.saturating_pow(k as u32)
    }
}

/// Get the inverse power of 3 for a given exponent
#[inline]
pub fn inv_pow3(k: usize) -> f64 {
    if k <= MAX_DEPTH {
        INV_POWERS_OF_3[k]
    } else {
        (1.0 / 3.0_f64).powi(k as i32)
    }
}

/// Get the cell width at a given depth
#[inline]
pub fn cell_width(depth: usize) -> f64 {
    inv_pow3(depth)
}

/// Convert trit value to offset in [0, 1) interval
#[inline]
pub fn trit_to_offset(trit: u8) -> f64 {
    debug_assert!(trit < 3, "Invalid trit value: {}", trit);
    TRIT_OFFSETS[trit as usize]
}

/// Convert a 6-trit tryte to cell index (0-728)
#[inline]
pub fn tryte_to_index(tryte: &[u8; 6]) -> u16 {
    let mut index = 0u16;
    for &t in tryte {
        index = index * 3 + t as u16;
    }
    index
}

/// Convert cell index (0-728) to 6-trit tryte
#[inline]
pub fn index_to_tryte(index: u16) -> [u8; 6] {
    debug_assert!(index < 729, "Invalid tryte index: {}", index);
    INDEX_TO_TRYTE[index as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_powers_of_3() {
        assert_eq!(pow3(0), 1);
        assert_eq!(pow3(1), 3);
        assert_eq!(pow3(2), 9);
        assert_eq!(pow3(6), 729);
        assert_eq!(pow3(10), 59049);
    }

    #[test]
    fn test_inv_powers_of_3() {
        assert!((inv_pow3(0) - 1.0).abs() < 1e-10);
        assert!((inv_pow3(1) - 1.0 / 3.0).abs() < 1e-10);
        assert!((inv_pow3(6) - 1.0 / 729.0).abs() < 1e-10);
    }

    #[test]
    fn test_tryte_conversion() {
        let tryte = [0, 1, 2, 0, 1, 2];
        let index = tryte_to_index(&tryte);
        let back = index_to_tryte(index);
        assert_eq!(tryte, back);
    }

    #[test]
    fn test_all_tryte_indices() {
        for i in 0..729 {
            let tryte = index_to_tryte(i);
            let back = tryte_to_index(&tryte);
            assert_eq!(i, back);
        }
    }
}

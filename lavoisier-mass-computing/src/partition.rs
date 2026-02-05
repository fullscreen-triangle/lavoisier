//! Partition state representation
//!
//! The partition state (n, ℓ, m, s) provides quantum-number-like indices
//! derived from S-entropy coordinates, analogous to atomic orbital quantum numbers.

use crate::scoord::SEntropyCoord;
use crate::ternary::TernaryAddress;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Partition state with quantum-number-like indices
///
/// These coordinates arise from the categorical structure of S-space:
/// - **n** (principal): Partition shell index, related to mass scale
/// - **ℓ** (angular): Partition momentum, related to chromatography
/// - **m** (magnetic): Partition projection, related to isotope state
/// - **s** (spin): Partition chirality, related to polarity
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PartitionState {
    /// Principal partition number (n ≥ 1), inversely related to S_k
    pub n: u32,
    /// Angular partition number (0 ≤ ℓ < n), from S_t
    pub l: u32,
    /// Magnetic partition number (-ℓ ≤ m ≤ ℓ), from S_e
    pub m: i32,
    /// Spin partition number (±0.5), from S_e polarity
    pub s: f64,
}

impl PartitionState {
    /// Create a new partition state
    pub fn new(n: u32, l: u32, m: i32, s: f64) -> Self {
        Self { n, l, m, s }
    }

    /// Derive partition state from S-entropy coordinates
    pub fn from_scoord(coord: &SEntropyCoord) -> Self {
        // Principal quantum number: inverse of S_k
        // n = floor(1 / S_k) + 1, with S_k clamped to avoid division by zero
        let s_k_clamped = coord.s_k.max(0.01);
        let n = (1.0 / s_k_clamped).floor() as u32 + 1;
        let n = n.max(1); // Ensure n >= 1

        // Angular quantum number: from S_t, 0 <= l < n
        let l = ((n as f64 * coord.s_t).floor() as u32).min(n - 1);

        // Magnetic quantum number: from S_e, -l <= m <= l
        let m_range = 2 * l + 1;
        let m = (m_range as f64 * coord.s_e).floor() as i32 - l as i32;

        // Spin: from S_e polarity
        let s = if coord.s_e >= 0.5 { 0.5 } else { -0.5 };

        Self { n, l, m, s }
    }

    /// Derive partition state from a ternary address
    pub fn from_address(addr: &TernaryAddress) -> Self {
        let coord = addr.to_scoord();
        Self::from_scoord(&coord)
    }

    /// Calculate the capacity C(n) = 2n²
    /// This represents the number of accessible microstates at partition level n
    #[inline]
    pub fn capacity(&self) -> u64 {
        2 * (self.n as u64) * (self.n as u64)
    }

    /// Calculate the degeneracy at this state: 2l + 1
    #[inline]
    pub fn degeneracy(&self) -> u32 {
        2 * self.l + 1
    }

    /// Check if this state satisfies quantum number constraints:
    /// n >= 1, 0 <= l < n, -l <= m <= l, s = ±0.5
    pub fn is_valid(&self) -> bool {
        self.n >= 1
            && self.l < self.n
            && self.m >= -(self.l as i32)
            && self.m <= self.l as i32
            && (self.s == 0.5 || self.s == -0.5)
    }

    /// Convert to tuple (n, l, m, s)
    pub fn to_tuple(&self) -> (u32, u32, i32, f64) {
        (self.n, self.l, self.m, self.s)
    }

    /// Create from tuple (n, l, m, s)
    pub fn from_tuple((n, l, m, s): (u32, u32, i32, f64)) -> Self {
        Self { n, l, m, s }
    }

    /// Estimate approximate S-coordinates from partition state
    /// Note: This is an inverse mapping and loses precision
    pub fn to_approx_scoord(&self) -> SEntropyCoord {
        let s_k = 1.0 / (self.n as f64);
        let s_t = (self.l as f64 + 0.5) / (self.n as f64);
        let s_e = ((self.m + self.l as i32) as f64 + 0.5) / (2 * self.l + 1) as f64;

        SEntropyCoord::new_unchecked(
            s_k.clamp(0.0, 1.0),
            s_t.clamp(0.0, 1.0),
            s_e.clamp(0.0, 1.0),
        )
    }
}

impl Default for PartitionState {
    fn default() -> Self {
        Self {
            n: 1,
            l: 0,
            m: 0,
            s: 0.5,
        }
    }
}

impl fmt::Display for PartitionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "|n={}, ℓ={}, m={}, s={:+.1}⟩",
            self.n, self.l, self.m, self.s
        )
    }
}

/// Compact notation for partition state: |n, l, m, s⟩
impl From<(u32, u32, i32, f64)> for PartitionState {
    fn from(tuple: (u32, u32, i32, f64)) -> Self {
        Self::from_tuple(tuple)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_from_scoord() {
        // S_k = 0.5 -> n ≈ 3 (floor(2) + 1)
        let coord = SEntropyCoord::new(0.5, 0.5, 0.5).unwrap();
        let state = PartitionState::from_scoord(&coord);

        assert_eq!(state.n, 3);
        assert!(state.l < state.n);
        assert!(state.m >= -(state.l as i32));
        assert!(state.m <= state.l as i32);
    }

    #[test]
    fn test_capacity() {
        let state = PartitionState::new(3, 1, 0, 0.5);
        assert_eq!(state.capacity(), 18); // 2 * 3^2 = 18
    }

    #[test]
    fn test_degeneracy() {
        let state = PartitionState::new(3, 2, 0, 0.5);
        assert_eq!(state.degeneracy(), 5); // 2*2 + 1 = 5
    }

    #[test]
    fn test_is_valid() {
        let valid = PartitionState::new(3, 2, -1, 0.5);
        assert!(valid.is_valid());

        let invalid_l = PartitionState::new(3, 3, 0, 0.5);
        assert!(!invalid_l.is_valid());

        let invalid_m = PartitionState::new(3, 1, 5, 0.5);
        assert!(!invalid_m.is_valid());
    }

    #[test]
    fn test_display() {
        let state = PartitionState::new(3, 1, -1, -0.5);
        let s = format!("{}", state);
        assert!(s.contains("n=3"));
        assert!(s.contains("m=-1"));
    }

    #[test]
    fn test_roundtrip_approx() {
        let original = SEntropyCoord::new(0.4, 0.6, 0.7).unwrap();
        let state = PartitionState::from_scoord(&original);
        let recovered = state.to_approx_scoord();

        // The approximation should be in the right ballpark
        assert!(recovered.is_valid());
        // Rough check - exact roundtrip not expected due to discretization
        assert_abs_diff_eq!(original.s_k, recovered.s_k, epsilon = 0.3);
    }
}

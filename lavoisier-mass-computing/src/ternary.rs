//! Ternary address representation for S-entropy space
//!
//! A ternary address is a sequence of trits (ternary digits) that specifies
//! a unique cell in the S-entropy coordinate space. The address simultaneously
//! encodes position and trajectory - the "address IS the trajectory" principle.

use crate::errors::{MassComputingError, Result};
use crate::lookup::{inv_pow3, TRYTE_SIZE};
use crate::scoord::SEntropyCoord;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::fmt;
use std::str::FromStr;

/// A ternary digit (trit) with values 0, 1, or 2
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Trit {
    Zero = 0,
    One = 1,
    Two = 2,
}

impl Trit {
    /// Create a Trit from a u8 value
    #[inline]
    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            0 => Ok(Trit::Zero),
            1 => Ok(Trit::One),
            2 => Ok(Trit::Two),
            _ => Err(MassComputingError::InvalidTrit(val)),
        }
    }

    /// Convert to u8
    #[inline]
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Get the axis this trit refines (0=S_k, 1=S_t, 2=S_e)
    /// when at position `pos` in an interleaved address
    #[inline]
    pub fn axis_at_position(pos: usize) -> usize {
        pos % 3
    }
}

impl From<Trit> for u8 {
    fn from(t: Trit) -> u8 {
        t as u8
    }
}

impl TryFrom<u8> for Trit {
    type Error = MassComputingError;

    fn try_from(val: u8) -> Result<Self> {
        Trit::from_u8(val)
    }
}

impl TryFrom<char> for Trit {
    type Error = MassComputingError;

    fn try_from(c: char) -> Result<Self> {
        match c {
            '0' => Ok(Trit::Zero),
            '1' => Ok(Trit::One),
            '2' => Ok(Trit::Two),
            _ => Err(MassComputingError::InvalidTrit(c as u8)),
        }
    }
}

impl fmt::Display for Trit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_u8())
    }
}

/// A tryte is a 6-trit unit (analogous to a byte in binary)
/// Represents 3^6 = 729 distinct values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Tryte([Trit; TRYTE_SIZE]);

impl Tryte {
    /// Create a new tryte from 6 trits
    pub fn new(trits: [Trit; 6]) -> Self {
        Self(trits)
    }

    /// Create from array of u8 values
    pub fn from_u8_array(vals: [u8; 6]) -> Result<Self> {
        Ok(Self([
            Trit::from_u8(vals[0])?,
            Trit::from_u8(vals[1])?,
            Trit::from_u8(vals[2])?,
            Trit::from_u8(vals[3])?,
            Trit::from_u8(vals[4])?,
            Trit::from_u8(vals[5])?,
        ]))
    }

    /// Convert to cell index (0-728)
    pub fn to_index(&self) -> u16 {
        let mut index = 0u16;
        for t in &self.0 {
            index = index * 3 + t.as_u8() as u16;
        }
        index
    }

    /// Create from cell index (0-728)
    pub fn from_index(mut index: u16) -> Self {
        debug_assert!(index < 729);
        let mut trits = [Trit::Zero; 6];
        for i in (0..6).rev() {
            trits[i] = Trit::from_u8((index % 3) as u8).unwrap();
            index /= 3;
        }
        Self(trits)
    }

    /// Get the underlying trits
    pub fn trits(&self) -> &[Trit; 6] {
        &self.0
    }
}

impl fmt::Display for Tryte {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for t in &self.0 {
            write!(f, "{}", t)?;
        }
        Ok(())
    }
}

/// Ternary address in S-entropy space
///
/// A k-trit address specifies one of 3^k cells in the unit cube [0,1]^3.
/// Uses interleaved encoding where position i refines axis (i mod 3):
/// - i ≡ 0 (mod 3): refines S_k (knowledge entropy / mass)
/// - i ≡ 1 (mod 3): refines S_t (temporal entropy / retention)
/// - i ≡ 2 (mod 3): refines S_e (evolution entropy / fragmentation)
///
/// The address encodes BOTH position and trajectory - they are identical.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TernaryAddress {
    /// The trit sequence, stored compactly
    /// SmallVec uses inline storage for addresses up to 24 trits (8 trytes)
    trits: SmallVec<[u8; 24]>,
}

impl TernaryAddress {
    /// Create an empty address (addresses the entire S-space)
    pub fn empty() -> Self {
        Self {
            trits: SmallVec::new(),
        }
    }

    /// Create from a slice of trit values (0, 1, or 2)
    pub fn from_trits(trits: &[u8]) -> Result<Self> {
        for &t in trits {
            if t > 2 {
                return Err(MassComputingError::InvalidTrit(t));
            }
        }
        Ok(Self {
            trits: SmallVec::from_slice(trits),
        })
    }

    /// Create from a string of '0', '1', '2' characters
    pub fn from_str(s: &str) -> Result<Self> {
        let trits: Result<SmallVec<[u8; 24]>> = s
            .chars()
            .filter(|c| *c == '0' || *c == '1' || *c == '2')
            .map(|c| {
                Trit::try_from(c).map(|t| t.as_u8())
            })
            .collect();
        Ok(Self { trits: trits? })
    }

    /// Create from S-entropy coordinates with specified depth
    ///
    /// Uses interleaved encoding: each trit refines one axis, cycling through S_k, S_t, S_e
    pub fn from_scoord(coord: &SEntropyCoord, depth: usize) -> Result<Self> {
        coord.validate()?;

        let mut coords = [coord.s_k, coord.s_t, coord.s_e];
        let mut trits = SmallVec::with_capacity(depth);

        for i in 0..depth {
            let axis = i % 3;
            let c = coords[axis];

            let (trit, new_c) = if c < 1.0 / 3.0 {
                (0u8, c * 3.0)
            } else if c < 2.0 / 3.0 {
                (1u8, (c - 1.0 / 3.0) * 3.0)
            } else {
                (2u8, (c - 2.0 / 3.0) * 3.0)
            };

            trits.push(trit);
            coords[axis] = new_c;
        }

        Ok(Self { trits })
    }

    /// Convert to S-entropy coordinates (returns the cell center)
    pub fn to_scoord(&self) -> SEntropyCoord {
        // Track bounds for each axis [low, high]
        let mut bounds = [[0.0f64, 1.0f64]; 3];

        for (i, &trit) in self.trits.iter().enumerate() {
            let axis = i % 3;
            let [low, high] = bounds[axis];
            let width = (high - low) / 3.0;

            let new_low = low + (trit as f64) * width;
            bounds[axis] = [new_low, new_low + width];
        }

        // Return cell centers
        SEntropyCoord {
            s_k: (bounds[0][0] + bounds[0][1]) / 2.0,
            s_t: (bounds[1][0] + bounds[1][1]) / 2.0,
            s_e: (bounds[2][0] + bounds[2][1]) / 2.0,
        }
    }

    /// Get the cell bounds [min, max] for each axis
    pub fn cell_bounds(&self) -> [[f64; 2]; 3] {
        let mut bounds = [[0.0f64, 1.0f64]; 3];

        for (i, &trit) in self.trits.iter().enumerate() {
            let axis = i % 3;
            let [low, high] = bounds[axis];
            let width = (high - low) / 3.0;

            let new_low = low + (trit as f64) * width;
            bounds[axis] = [new_low, new_low + width];
        }

        bounds
    }

    /// Get the depth (number of trits)
    #[inline]
    pub fn depth(&self) -> usize {
        self.trits.len()
    }

    /// Get the resolution along each axis
    pub fn resolution(&self) -> [f64; 3] {
        let depth = self.depth();
        let trits_per_axis = [
            (depth + 2) / 3,  // S_k: positions 0, 3, 6, ...
            (depth + 1) / 3,  // S_t: positions 1, 4, 7, ...
            depth / 3,        // S_e: positions 2, 5, 8, ...
        ];

        [
            inv_pow3(trits_per_axis[0]),
            inv_pow3(trits_per_axis[1]),
            inv_pow3(trits_per_axis[2]),
        ]
    }

    /// Get the cell volume (always (1/3)^depth)
    #[inline]
    pub fn cell_volume(&self) -> f64 {
        inv_pow3(self.depth())
    }

    /// Get the underlying trits as a slice
    #[inline]
    pub fn trits(&self) -> &[u8] {
        &self.trits
    }

    /// Extend this address with additional trits
    pub fn extend(&self, extension: &TernaryAddress) -> Self {
        let mut new_trits = self.trits.clone();
        new_trits.extend_from_slice(&extension.trits);
        Self { trits: new_trits }
    }

    /// Extend with a single trit
    pub fn push(&self, trit: u8) -> Result<Self> {
        if trit > 2 {
            return Err(MassComputingError::InvalidTrit(trit));
        }
        let mut new_trits = self.trits.clone();
        new_trits.push(trit);
        Ok(Self { trits: new_trits })
    }

    /// Fragment address at position k, returning (prefix, suffix)
    pub fn fragment_at(&self, k: usize) -> Result<(Self, Self)> {
        if k > self.depth() {
            return Err(MassComputingError::FragmentOutOfBounds(k, self.depth()));
        }

        let prefix = Self {
            trits: SmallVec::from_slice(&self.trits[..k]),
        };
        let suffix = Self {
            trits: SmallVec::from_slice(&self.trits[k..]),
        };

        Ok((prefix, suffix))
    }

    /// Get prefix of length k
    pub fn prefix(&self, k: usize) -> Result<Self> {
        if k > self.depth() {
            return Err(MassComputingError::FragmentOutOfBounds(k, self.depth()));
        }
        Ok(Self {
            trits: SmallVec::from_slice(&self.trits[..k]),
        })
    }

    /// Get suffix starting at position k
    pub fn suffix(&self, k: usize) -> Result<Self> {
        if k > self.depth() {
            return Err(MassComputingError::FragmentOutOfBounds(k, self.depth()));
        }
        Ok(Self {
            trits: SmallVec::from_slice(&self.trits[k..]),
        })
    }

    /// Check if this address is a prefix of another
    pub fn is_prefix_of(&self, other: &TernaryAddress) -> bool {
        if self.depth() > other.depth() {
            return false;
        }
        self.trits.as_slice() == &other.trits[..self.depth()]
    }

    /// Get trytes (6-trit chunks) from this address
    pub fn trytes(&self) -> impl Iterator<Item = Tryte> + '_ {
        self.trits.chunks(6).map(|chunk| {
            let mut trits = [Trit::Zero; 6];
            for (i, &t) in chunk.iter().enumerate() {
                trits[i] = Trit::from_u8(t).unwrap_or(Trit::Zero);
            }
            Tryte::new(trits)
        })
    }
}

impl Default for TernaryAddress {
    fn default() -> Self {
        Self::empty()
    }
}

impl FromStr for TernaryAddress {
    type Err = MassComputingError;

    fn from_str(s: &str) -> Result<Self> {
        TernaryAddress::from_str(s)
    }
}

impl fmt::Display for TernaryAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for &t in &self.trits {
            write!(f, "{}", t)?;
        }
        Ok(())
    }
}

impl AsRef<[u8]> for TernaryAddress {
    fn as_ref(&self) -> &[u8] {
        &self.trits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_from_string() {
        let addr = TernaryAddress::from_str("012102").unwrap();
        assert_eq!(addr.depth(), 6);
        assert_eq!(addr.trits(), &[0, 1, 2, 1, 0, 2]);
    }

    #[test]
    fn test_to_string() {
        let addr = TernaryAddress::from_trits(&[0, 1, 2, 1, 0, 2]).unwrap();
        assert_eq!(addr.to_string(), "012102");
    }

    #[test]
    fn test_scoord_roundtrip() {
        let original = SEntropyCoord::new(0.25, 0.5, 0.75).unwrap();
        let addr = TernaryAddress::from_scoord(&original, 18).unwrap();
        let recovered = addr.to_scoord();

        // Should be close within the 18-trit resolution
        let res = addr.resolution();
        assert_abs_diff_eq!(original.s_k, recovered.s_k, epsilon = res[0]);
        assert_abs_diff_eq!(original.s_t, recovered.s_t, epsilon = res[1]);
        assert_abs_diff_eq!(original.s_e, recovered.s_e, epsilon = res[2]);
    }

    #[test]
    fn test_extend() {
        let addr1 = TernaryAddress::from_str("012").unwrap();
        let addr2 = TernaryAddress::from_str("210").unwrap();
        let combined = addr1.extend(&addr2);
        assert_eq!(combined.to_string(), "012210");
    }

    #[test]
    fn test_fragment() {
        let addr = TernaryAddress::from_str("012102").unwrap();
        let (prefix, suffix) = addr.fragment_at(3).unwrap();
        assert_eq!(prefix.to_string(), "012");
        assert_eq!(suffix.to_string(), "102");
    }

    #[test]
    fn test_cell_volume() {
        let addr = TernaryAddress::from_str("012102").unwrap();
        let expected = 1.0 / 729.0; // (1/3)^6
        assert_abs_diff_eq!(addr.cell_volume(), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_resolution() {
        // 6 trits: positions 0,3 for S_k (2 trits), 1,4 for S_t (2 trits), 2,5 for S_e (2 trits)
        let addr = TernaryAddress::from_str("012102").unwrap();
        let res = addr.resolution();
        let expected = 1.0 / 9.0; // (1/3)^2

        assert_abs_diff_eq!(res[0], expected, epsilon = 1e-10);
        assert_abs_diff_eq!(res[1], expected, epsilon = 1e-10);
        assert_abs_diff_eq!(res[2], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_is_prefix() {
        let short = TernaryAddress::from_str("012").unwrap();
        let long = TernaryAddress::from_str("012102").unwrap();
        let other = TernaryAddress::from_str("021102").unwrap();

        assert!(short.is_prefix_of(&long));
        assert!(!short.is_prefix_of(&other));
        assert!(!long.is_prefix_of(&short));
    }

    #[test]
    fn test_tryte_conversion() {
        let tryte = Tryte::from_u8_array([0, 1, 2, 0, 1, 2]).unwrap();
        let index = tryte.to_index();
        let back = Tryte::from_index(index);
        assert_eq!(tryte, back);
    }

    #[test]
    fn test_invalid_trit() {
        // from_str filters out invalid characters (matching Python behavior)
        let addr = TernaryAddress::from_str("0123").unwrap();
        assert_eq!(addr.to_string(), "012"); // '3' is filtered out

        // from_trits rejects invalid values
        assert!(TernaryAddress::from_trits(&[0, 1, 3, 2]).is_err());
    }
}

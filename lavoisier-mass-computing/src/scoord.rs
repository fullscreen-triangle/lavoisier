//! S-entropy coordinates in the unit cube [0,1]³
//!
//! The S-entropy coordinate space represents the complete state of a molecular
//! species in mass spectrometric analysis:
//! - S_k: Knowledge entropy (encodes mass/molecular identity)
//! - S_t: Temporal entropy (encodes chromatographic retention)
//! - S_e: Evolution entropy (encodes fragmentation state)

use crate::errors::{MassComputingError, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// S-entropy coordinates in the unit cube [0,1]³
///
/// Each molecule maps to a unique point in S-space. The coordinates have
/// physical interpretations:
///
/// - **S_k** (Knowledge entropy): Inverse relationship with mass.
///   - S_k ≈ 0 → High mass, complex molecules
///   - S_k ≈ 1 → Low mass, simple molecules
///
/// - **S_t** (Temporal entropy): Linear relationship with retention time.
///   - S_t ≈ 0 → Early elution (polar, small)
///   - S_t ≈ 1 → Late elution (nonpolar, large)
///
/// - **S_e** (Evolution entropy): Fragmentation/reaction state.
///   - S_e ≈ 0 → Intact molecular ion
///   - S_e ≈ 1 → Extensively fragmented
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SEntropyCoord {
    /// Knowledge entropy: encodes molecular identity/mass
    pub s_k: f64,
    /// Temporal entropy: encodes chromatographic time
    pub s_t: f64,
    /// Evolution entropy: encodes fragmentation state
    pub s_e: f64,
}

impl SEntropyCoord {
    /// Create new S-entropy coordinates, validating the range
    pub fn new(s_k: f64, s_t: f64, s_e: f64) -> Result<Self> {
        let coord = Self { s_k, s_t, s_e };
        coord.validate()?;
        Ok(coord)
    }

    /// Create without validation (use when values are known to be valid)
    #[inline]
    pub fn new_unchecked(s_k: f64, s_t: f64, s_e: f64) -> Self {
        Self { s_k, s_t, s_e }
    }

    /// Create at the center of S-space (0.5, 0.5, 0.5)
    pub fn center() -> Self {
        Self {
            s_k: 0.5,
            s_t: 0.5,
            s_e: 0.5,
        }
    }

    /// Create at the origin (0, 0, 0)
    pub fn origin() -> Self {
        Self {
            s_k: 0.0,
            s_t: 0.0,
            s_e: 0.0,
        }
    }

    /// Validate that all coordinates are in [0, 1]
    pub fn validate(&self) -> Result<()> {
        if self.s_k < 0.0 || self.s_k > 1.0 {
            return Err(MassComputingError::InvalidSCoord {
                coord: "S_k",
                value: self.s_k,
            });
        }
        if self.s_t < 0.0 || self.s_t > 1.0 {
            return Err(MassComputingError::InvalidSCoord {
                coord: "S_t",
                value: self.s_t,
            });
        }
        if self.s_e < 0.0 || self.s_e > 1.0 {
            return Err(MassComputingError::InvalidSCoord {
                coord: "S_e",
                value: self.s_e,
            });
        }
        Ok(())
    }

    /// Check if coordinates are valid (in [0, 1])
    pub fn is_valid(&self) -> bool {
        self.s_k >= 0.0
            && self.s_k <= 1.0
            && self.s_t >= 0.0
            && self.s_t <= 1.0
            && self.s_e >= 0.0
            && self.s_e <= 1.0
    }

    /// Clamp coordinates to [0, 1]
    pub fn clamp(&self) -> Self {
        Self {
            s_k: self.s_k.clamp(0.0, 1.0),
            s_t: self.s_t.clamp(0.0, 1.0),
            s_e: self.s_e.clamp(0.0, 1.0),
        }
    }

    /// Compute Euclidean distance to another point in S-space
    pub fn distance(&self, other: &Self) -> f64 {
        let dk = self.s_k - other.s_k;
        let dt = self.s_t - other.s_t;
        let de = self.s_e - other.s_e;
        (dk * dk + dt * dt + de * de).sqrt()
    }

    /// Compute squared distance (faster when comparing distances)
    #[inline]
    pub fn distance_squared(&self, other: &Self) -> f64 {
        let dk = self.s_k - other.s_k;
        let dt = self.s_t - other.s_t;
        let de = self.s_e - other.s_e;
        dk * dk + dt * dt + de * de
    }

    /// Convert to array [S_k, S_t, S_e]
    #[inline]
    pub fn to_array(&self) -> [f64; 3] {
        [self.s_k, self.s_t, self.s_e]
    }

    /// Create from array [S_k, S_t, S_e]
    pub fn from_array(arr: [f64; 3]) -> Result<Self> {
        Self::new(arr[0], arr[1], arr[2])
    }

    /// Linear interpolation between two coordinates
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        Self {
            s_k: self.s_k + (other.s_k - self.s_k) * t,
            s_t: self.s_t + (other.s_t - self.s_t) * t,
            s_e: self.s_e + (other.s_e - self.s_e) * t,
        }
    }
}

impl Default for SEntropyCoord {
    fn default() -> Self {
        Self::center()
    }
}

impl fmt::Display for SEntropyCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(S_k={:.4}, S_t={:.4}, S_e={:.4})",
            self.s_k, self.s_t, self.s_e
        )
    }
}

impl From<[f64; 3]> for SEntropyCoord {
    fn from(arr: [f64; 3]) -> Self {
        Self::new_unchecked(arr[0], arr[1], arr[2])
    }
}

impl From<(f64, f64, f64)> for SEntropyCoord {
    fn from((s_k, s_t, s_e): (f64, f64, f64)) -> Self {
        Self::new_unchecked(s_k, s_t, s_e)
    }
}

impl From<SEntropyCoord> for [f64; 3] {
    fn from(coord: SEntropyCoord) -> Self {
        coord.to_array()
    }
}

impl From<SEntropyCoord> for (f64, f64, f64) {
    fn from(coord: SEntropyCoord) -> Self {
        (coord.s_k, coord.s_t, coord.s_e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_new_valid() {
        let coord = SEntropyCoord::new(0.25, 0.5, 0.75).unwrap();
        assert_abs_diff_eq!(coord.s_k, 0.25);
        assert_abs_diff_eq!(coord.s_t, 0.5);
        assert_abs_diff_eq!(coord.s_e, 0.75);
    }

    #[test]
    fn test_new_invalid() {
        assert!(SEntropyCoord::new(1.5, 0.5, 0.5).is_err());
        assert!(SEntropyCoord::new(0.5, -0.1, 0.5).is_err());
        assert!(SEntropyCoord::new(0.5, 0.5, 1.1).is_err());
    }

    #[test]
    fn test_distance() {
        let a = SEntropyCoord::origin();
        let b = SEntropyCoord::new(1.0, 0.0, 0.0).unwrap();
        assert_abs_diff_eq!(a.distance(&b), 1.0);

        let c = SEntropyCoord::new(1.0, 1.0, 1.0).unwrap();
        assert_abs_diff_eq!(a.distance(&c), 3.0_f64.sqrt());
    }

    #[test]
    fn test_lerp() {
        let a = SEntropyCoord::origin();
        let b = SEntropyCoord::new(1.0, 1.0, 1.0).unwrap();

        let mid = a.lerp(&b, 0.5);
        assert_abs_diff_eq!(mid.s_k, 0.5);
        assert_abs_diff_eq!(mid.s_t, 0.5);
        assert_abs_diff_eq!(mid.s_e, 0.5);
    }

    #[test]
    fn test_clamp() {
        let invalid = SEntropyCoord::new_unchecked(1.5, -0.5, 0.5);
        let clamped = invalid.clamp();

        assert_abs_diff_eq!(clamped.s_k, 1.0);
        assert_abs_diff_eq!(clamped.s_t, 0.0);
        assert_abs_diff_eq!(clamped.s_e, 0.5);
    }

    #[test]
    fn test_display() {
        let coord = SEntropyCoord::new(0.1234, 0.5678, 0.9012).unwrap();
        let s = format!("{}", coord);
        assert!(s.contains("S_k=0.1234"));
    }
}

/**
 * Partition Lagrangian and analyser field topologies.
 *
 * From the paper:
 *   L_M = ½μ|ẋ|² + μẋ·A_M − M(x,t)
 *
 * with μ = α(m/z). All four analyser types emerge as specialisations
 * with different M and A_M. This module provides:
 *   - the analyser-specific field generators (for the GPU uniforms)
 *   - the observable extractors (T_TOF, ω_orbi, ω_c, Mathieu a/q)
 *
 * No forces are invoked: each "observable" is a property of the ion's
 * descent through the partition depth landscape.
 */

// Physical constants (SI)
export const AMU = 1.66053907e-27;      // kg
export const E_CHARGE = 1.602176634e-19; // C
export const HBAR = 1.054571817e-34;     // J·s
export const KB = 1.380649e-23;          // J/K

/**
 * Partition inertia μ = α · (m/z).
 * α = e (charge units) when m is in amu and z is integer.
 * Returns μ in units of kg per charge.
 *
 * @param {number} mz   mass-to-charge ratio (Da)
 * @param {number} [z=1]
 * @returns {number}
 */
export function partitionInertia(mz, z = 1) {
  return mz * AMU; // per unit charge: the 1/z is absorbed into the field
}

/* ---- TOF ---- */

/**
 * TOF: linear depth gradient M_TOF(z) = -κz.
 * The ion descends the gradient; flight time T ∝ √(m/z).
 *
 * @param {number} mz
 * @param {Object} cfg
 * @param {number} cfg.accelV   accelerating voltage (V)
 * @param {number} cfg.flightLength  flight tube length (m)
 * @returns {{T: number, observable: "flightTime", unit: "s"}}
 */
export function tofObservable(mz, { accelV = 5000, flightLength = 1.0 } = {}) {
  const mKg = mz * AMU;
  const T = flightLength * Math.sqrt(mKg / (2 * E_CHARGE * accelV));
  return { T, observable: "flightTime", unit: "s" };
}

/* ---- Quadrupole ---- */

/**
 * Quadrupole: saddle potential M = (κ₀/2)(x²−y²)[U + V cos(Ωt)].
 * Stability in the Mathieu parameters:
 *   a = 8 e U / (m r₀² Ω²)
 *   q = 4 e V / (m r₀² Ω²)
 *
 * @param {number} mz
 * @param {Object} cfg
 * @returns {{a: number, q: number, stable: boolean, observable: "mathieu"}}
 */
export function quadrupoleObservable(mz, {
  dcVoltage = 100,
  rfVoltage = 500,
  rfFrequency = 1e6,   // rad/s  (≈ 1 MHz)
  r0 = 5e-3,
} = {}) {
  const mKg = mz * AMU;
  const denom = mKg * r0 * r0 * rfFrequency * rfFrequency;
  const a = (8 * E_CHARGE * dcVoltage) / denom;
  const q = (4 * E_CHARGE * rfVoltage) / denom;
  const stable = Math.abs(a) < 0.237 && Math.abs(q) < 0.908;
  return { a, q, stable, observable: "mathieu" };
}

/* ---- Orbitrap ---- */

/**
 * Orbitrap: quadro-logarithmic depth field.
 * Axial ω = √(κ/μ) ∝ √(z/m).
 *
 * @param {number} mz
 * @param {Object} cfg
 * @param {number} cfg.kField      field curvature (N/m per charge)
 * @returns {{omega: number, frequencyHz: number, observable: "axialFrequency"}}
 */
export function orbitrapObservable(mz, { kField = 1e12 } = {}) {
  const mKg = mz * AMU;
  const omega = Math.sqrt((E_CHARGE * kField) / mKg);
  return {
    omega,
    frequencyHz: omega / (2 * Math.PI),
    observable: "axialFrequency",
  };
}

/* ---- FT-ICR ---- */

/**
 * FT-ICR: circular partition depth from A_M = (B/2)(-y, x, 0).
 * Cyclotron ω_c = zeB/m ∝ z/m.
 *
 * @param {number} mz
 * @param {Object} cfg
 * @param {number} cfg.B   magnetic field (T)
 * @returns {{omegaC: number, frequencyHz: number, observable: "cyclotronFrequency"}}
 */
export function fticrObservable(mz, { B = 7.0 } = {}) {
  const mKg = mz * AMU;
  const omegaC = (E_CHARGE * B) / mKg;
  return {
    omegaC,
    frequencyHz: omegaC / (2 * Math.PI),
    observable: "cyclotronFrequency",
  };
}

/* ---- Unified analyser dispatch ---- */

export const ANALYSERS = {
  tof: tofObservable,
  quadrupole: quadrupoleObservable,
  orbitrap: orbitrapObservable,
  fticr: fticrObservable,
};

/**
 * Compute the observable for a given analyser.
 * @param {"tof"|"quadrupole"|"orbitrap"|"fticr"} analyser
 * @param {number} mz
 * @param {Object} [cfg]
 */
export function observe(analyser, mz, cfg = {}) {
  const fn = ANALYSERS[analyser];
  if (!fn) throw new Error(`Unknown analyser: ${analyser}`);
  return fn(mz, cfg);
}

/* ---- Partition depth field (for GPU uniforms) ---- */

/**
 * Return the partition-depth-field parameters needed to render this
 * analyser's M(x,t) topology on the GPU. This is the bridge from the
 * physical Lagrangian to shader uniforms.
 *
 * @param {string} analyser
 * @param {Object} cfg
 * @returns {{type: string, uniforms: Object}}
 */
export function partitionField(analyser, cfg = {}) {
  switch (analyser) {
    case "tof":
      return {
        type: "linear",
        uniforms: {
          kappa: (E_CHARGE * (cfg.accelV || 5000)) / (cfg.flightLength || 1.0),
          axis: [0, 0, 1], // z-axis
        },
      };
    case "quadrupole":
      return {
        type: "saddle_rf",
        uniforms: {
          kappa0: (4 * E_CHARGE * (cfg.rfVoltage || 500)) /
                  Math.pow(cfg.r0 || 5e-3, 2),
          U: cfg.dcVoltage || 100,
          V: cfg.rfVoltage || 500,
          Omega: cfg.rfFrequency || 1e6,
        },
      };
    case "orbitrap":
      return {
        type: "quadrologarithmic",
        uniforms: {
          kappa: cfg.kField || 1e12,
          Rm: cfg.Rm || 1e-2,
        },
      };
    case "fticr":
      return {
        type: "vector_circular",
        uniforms: {
          B: cfg.B || 7.0,
        },
      };
    default:
      throw new Error(`Unknown analyser: ${analyser}`);
  }
}

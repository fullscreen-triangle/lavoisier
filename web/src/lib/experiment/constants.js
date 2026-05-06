/**
 * Atomic and physical constants used by the virtual instrument.
 * All masses are monoisotopic in Daltons (atomic mass units).
 */

export const ATOMIC_MASS = {
  H:  1.0078250319,
  C: 12.0,
  N: 14.0030740052,
  O: 15.9949146221,
  P: 30.97376151,
  S: 31.97207069,
  Na: 22.98976928,
  K:  38.96370668,
  Cl: 34.96885268,
  F:  18.99840316,
  electron: 0.00054858,
};

// Mass of common chemical groups (mono-isotopic)
export const GROUP_MASS = {
  H2O:    2 * ATOMIC_MASS.H + ATOMIC_MASS.O,
  NH3:    3 * ATOMIC_MASS.H + ATOMIC_MASS.N,
  CO2:    ATOMIC_MASS.C + 2 * ATOMIC_MASS.O,
  HPO4:   ATOMIC_MASS.H + ATOMIC_MASS.P + 4 * ATOMIC_MASS.O,
  CH2:    ATOMIC_MASS.C + 2 * ATOMIC_MASS.H,
  CH3OH:  ATOMIC_MASS.C + 4 * ATOMIC_MASS.H + ATOMIC_MASS.O,
  HCOOH:  ATOMIC_MASS.C + 2 * ATOMIC_MASS.H + 2 * ATOMIC_MASS.O,
};

export const PROTON_MASS = ATOMIC_MASS.H - ATOMIC_MASS.electron;

/**
 * Compute monoisotopic mass from an atomic composition.
 * @param {Object<string, number>} composition  e.g. {C: 42, H: 82, N: 1, O: 8, P: 1}
 */
export function monoisotopicMass(composition) {
  let m = 0;
  for (const [el, n] of Object.entries(composition)) {
    if (!ATOMIC_MASS[el]) throw new Error(`Unknown element ${el}`);
    m += n * ATOMIC_MASS[el];
  }
  return m;
}

/**
 * Format an atomic composition as a Hill-system formula string.
 */
export function formulaString(composition) {
  const order = ["C", "H", "N", "O", "P", "S", "Na", "K", "Cl", "F"];
  const parts = [];
  for (const el of order) {
    const n = composition[el] || 0;
    if (n > 0) parts.push(n === 1 ? el : `${el}${n}`);
  }
  return parts.join("");
}

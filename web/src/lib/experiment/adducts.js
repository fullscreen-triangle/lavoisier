/**
 * Ionisation adducts for ESI+ and ESI-.
 * mass = analyte M plus the adduct delta divided by charge.
 */

import { ATOMIC_MASS, PROTON_MASS } from "./constants";

export const ADDUCTS = {
  // Positive
  "[M+H]+":   { delta: PROTON_MASS,                              z: +1, polarity: "+", abbr: "H+" },
  "[M+Na]+":  { delta: ATOMIC_MASS.Na - ATOMIC_MASS.electron,    z: +1, polarity: "+", abbr: "Na+" },
  "[M+K]+":   { delta: ATOMIC_MASS.K - ATOMIC_MASS.electron,     z: +1, polarity: "+", abbr: "K+" },
  "[M+NH4]+": { delta: ATOMIC_MASS.N + 4 * ATOMIC_MASS.H - ATOMIC_MASS.electron,
                z: +1, polarity: "+", abbr: "NH4+" },
  "[M+H-H2O]+": { delta: PROTON_MASS - 2 * ATOMIC_MASS.H - ATOMIC_MASS.O,
                  z: +1, polarity: "+", abbr: "H+,-H2O" },

  // Negative
  "[M-H]-":   { delta: -PROTON_MASS,                             z: -1, polarity: "-", abbr: "-H" },
  "[M+Cl]-":  { delta: ATOMIC_MASS.Cl + ATOMIC_MASS.electron,    z: -1, polarity: "-", abbr: "Cl-" },
  "[M+HCOO]-": { delta: ATOMIC_MASS.H + ATOMIC_MASS.C + 2 * ATOMIC_MASS.O + ATOMIC_MASS.electron,
                 z: -1, polarity: "-", abbr: "HCOO-" },
  "[M+CH3COO]-": { delta: 2 * ATOMIC_MASS.C + 3 * ATOMIC_MASS.H + 2 * ATOMIC_MASS.O + ATOMIC_MASS.electron,
                   z: -1, polarity: "-", abbr: "OAc-" },

  // Multiply charged (important for proteomics)
  "[M+2H]2+": { delta: 2 * PROTON_MASS,                          z: +2, polarity: "+", abbr: "2H+" },
  "[M+3H]3+": { delta: 3 * PROTON_MASS,                          z: +3, polarity: "+", abbr: "3H+" },
  "[M+4H]4+": { delta: 4 * PROTON_MASS,                          z: +4, polarity: "+", abbr: "4H+" },
  "[M-2H]2-": { delta: -2 * PROTON_MASS,                         z: -2, polarity: "-", abbr: "-2H" },
};

export const ADDUCTS_POSITIVE = Object.keys(ADDUCTS).filter((k) => ADDUCTS[k].polarity === "+");
export const ADDUCTS_NEGATIVE = Object.keys(ADDUCTS).filter((k) => ADDUCTS[k].polarity === "-");

// Adduct subsets for proteomics (multiply-charged ESI)
export const ADDUCTS_PROTEOMICS_POSITIVE = ["[M+H]+", "[M+2H]2+", "[M+3H]3+", "[M+4H]4+"];
export const ADDUCTS_PROTEOMICS_NEGATIVE = ["[M-H]-", "[M-2H]2-"];

/**
 * Compute m/z for an analyte M with a given adduct.
 */
export function applyAdduct(M, adductKey) {
  const a = ADDUCTS[adductKey];
  if (!a) throw new Error(`Unknown adduct ${adductKey}`);
  return (M + a.delta) / Math.abs(a.z);
}

/**
 * Class-specific preferred adduct ranking (by typical observed intensity).
 * Returns adduct keys in descending preference.
 */
export function preferredAdducts(classKey, polarity) {
  if (polarity === "+") {
    switch (classKey) {
      case "PC":
      case "SM":
      case "LPC":
        return ["[M+H]+", "[M+Na]+", "[M+K]+"];
      case "PE":
        return ["[M+H]+", "[M+Na]+"];
      case "TAG":
      case "DAG":
        return ["[M+NH4]+", "[M+Na]+", "[M+H]+"];
      case "CE":
        return ["[M+NH4]+", "[M+Na]+", "[M+H-H2O]+"];
      case "Cer":
        return ["[M+H]+", "[M+H-H2O]+"];
      default:
        return ["[M+H]+", "[M+Na]+"];
    }
  } else {
    switch (classKey) {
      case "PE":
      case "PS":
      case "PG":
      case "PI":
      case "PA":
        return ["[M-H]-"];
      case "PC":
      case "SM":
        return ["[M+HCOO]-", "[M+Cl]-", "[M+CH3COO]-"];
      case "FA":
      case "Cer":
        return ["[M-H]-"];
      default:
        return ["[M-H]-"];
    }
  }
}

/**
 * Rough relative intensity scaling for an adduct given the class.
 * Used to weight predicted intensities. Returns a value in (0, 1].
 */
export function adductRelativeIntensity(classKey, adductKey) {
  const pref = preferredAdducts(classKey, ADDUCTS[adductKey].polarity);
  const idx = pref.indexOf(adductKey);
  if (idx < 0) return 0.05;
  // Decreasing geometric weights
  const weights = [1.0, 0.4, 0.15, 0.05];
  return weights[Math.min(idx, weights.length - 1)];
}

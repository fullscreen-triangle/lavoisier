/**
 * Fragmentation rules for HCD-style activation in positive and negative ESI.
 *
 * For each lipid class we encode the canonical product-ion patterns
 * documented in lipidomics literature. The result is a list of fragments
 * with predicted relative intensities.
 *
 * Output per analyte:
 *   { mz, intensity, label, type }   where type is one of:
 *     "precursor" | "head_loss" | "fa_loss" | "fa_anion" | "head_charged"
 *     | "neutral_loss" | "isotope"
 */

import { LIPID_CLASSES } from "./lipidomics.js";
import { RESIDUES } from "./proteomics.js";
import { ADDUCTS } from "./adducts.js";
import { ATOMIC_MASS, GROUP_MASS, PROTON_MASS } from "./constants.js";

const H2O_FRAG = 18.01056468;
const CO_MASS  = 27.99491462;
const NH3_MASS = 17.02654910;

/**
 * HCD b/y-ion series for a peptide analyte.
 * Generates b, a (b−CO), y ions plus doubly-charged variants for large
 * fragments, and selective neutral losses (−H2O, −NH3).
 */
function fragmentsPeptide(analyte, adductKey, ctx, CE = 25) {
  const seq   = analyte.sequence;
  const n     = seq.length;
  const precMz = ctx.precursor_mz;
  const precZ  = Math.abs((ADDUCTS[adductKey] || { z: 1 }).z);
  const out   = [];

  out.push({
    mz: precMz,
    intensity: 0.2 + 0.8 * Math.exp(-CE / 50),
    label: seq + " " + adductKey,
    type: "precursor",
  });

  // Prefix cumulative masses: prefix[k] = sum of first k residue masses
  const prefix = [0];
  for (let i = 0; i < n; i++) {
    prefix.push(prefix[i] + (RESIDUES[seq[i]]?.mass ?? 0));
  }

  // Suffix cumulative masses: suf[k] = sum of last k residue masses
  const suf = new Array(n + 1).fill(0);
  for (let k = 1; k <= n; k++) {
    suf[k] = suf[k - 1] + (RESIDUES[seq[n - k]]?.mass ?? 0);
  }

  const ceFactor = 0.4 + 0.6 * CE / 50;

  for (let k = 2; k <= n - 1; k++) {
    const bMass = prefix[k];
    const yMass = suf[k];

    // Gaussian intensity envelope centred at the middle of the sequence
    const pos      = (k - 1) / (n - 2);
    const envelope = Math.exp(-Math.pow((pos - 0.5) * 2.5, 2));

    const bI = 0.55 * envelope * ceFactor;
    const yI = 0.80 * envelope * ceFactor;

    if (bI > 0.03) {
      out.push({ mz: bMass + PROTON_MASS,        intensity: bI,        label: `b${k}`,   type: "b_ion" });
      out.push({ mz: bMass + PROTON_MASS - CO_MASS, intensity: bI * 0.22, label: `a${k}`,   type: "a_ion" });
      if (bMass > 700 && precZ >= 2) {
        out.push({ mz: (bMass + 2 * PROTON_MASS) / 2, intensity: bI * 0.28, label: `b${k}²⁺`, type: "b_ion" });
      }
    }

    if (yI > 0.03) {
      const yMz = yMass + H2O_FRAG + PROTON_MASS;
      out.push({ mz: yMz, intensity: yI, label: `y${k}`, type: "y_ion" });
      if (yMass > 700 && precZ >= 2) {
        out.push({ mz: (yMass + H2O_FRAG + 2 * PROTON_MASS) / 2, intensity: yI * 0.28, label: `y${k}²⁺`, type: "y_ion" });
      }
      const nTermRes = seq[n - k];
      if ("STED".includes(nTermRes)) {
        out.push({ mz: yMz - H2O_FRAG, intensity: yI * 0.18, label: `y${k}-H2O`, type: "neutral_loss" });
      }
      if ("KNQR".includes(nTermRes)) {
        out.push({ mz: yMz - NH3_MASS, intensity: yI * 0.14, label: `y${k}-NH3`, type: "neutral_loss" });
      }
    }
  }

  // Diagnostic immonium ions for aromatic / charged residues
  const immonium = { Y: 136.0757, W: 159.0922, F: 120.0808, H: 110.0718, R: 129.1135 };
  for (const [aa, mz] of Object.entries(immonium)) {
    if (seq.includes(aa)) {
      out.push({ mz, intensity: 0.12, label: `Im(${aa})`, type: "immonium" });
    }
  }

  return out.filter((f) => f.mz > 50 && Number.isFinite(f.mz) && f.intensity > 0.01);
}

/**
 * Helper: compute possible single-FA chain compositions for a class with X:Y total.
 * Returns coarse (n, db) pairs that sum to (X, Y).
 *
 * For two-chain lipids, we generate the most physiologically relevant
 * pairs: 16:0/X-16:Y, 18:0/X-18:Y, 18:1/X-18:Y-1 etc.
 */
function commonChainPairs(X, Y, faChains) {
  if (faChains === 1) return [[X, Y]];
  const pairs = [];
  if (faChains === 2) {
    const candidates = [
      [16, 0], [16, 1], [18, 0], [18, 1], [18, 2], [18, 3],
      [20, 0], [20, 4], [22, 0], [22, 6], [14, 0], [12, 0],
    ];
    for (const [c1, db1] of candidates) {
      const c2 = X - c1, db2 = Y - db1;
      if (c2 >= 12 && c2 <= 26 && db2 >= 0 && db2 <= 6) {
        pairs.push([c1, db1, c2, db2]);
      }
    }
    if (pairs.length === 0) {
      // Fallback: split evenly
      const c1 = Math.floor(X / 2);
      const db1 = Math.floor(Y / 2);
      pairs.push([c1, db1, X - c1, Y - db1]);
    }
  } else if (faChains === 3) {
    // TAG: simplify to common 16:0 + 18:1 + (X-34:Y-1)
    const fixed = [[16, 0], [18, 1]];
    for (const [c1, db1] of fixed) {
      for (const [c2, db2] of fixed) {
        const c3 = X - c1 - c2, db3 = Y - db1 - db2;
        if (c3 >= 12 && c3 <= 24 && db3 >= 0 && db3 <= 6) {
          pairs.push([c1, db1, c2, db2, c3, db3]);
        }
      }
    }
  }
  return pairs.slice(0, 4); // limit combinations
}

/**
 * Mass of a free fatty acid Cn H2n-2db O2 (as FA or as carboxylate -COO).
 */
function faMass(n, db) {
  return 14.0156501 * n - 2.0156501 * db - 0.000064;
}

/**
 * Compute MS2 fragments for an analyte at a given precursor adduct.
 *
 * @param {Object} analyte  { class, X, Y, mass, name }
 * @param {string} adductKey
 * @param {{precursor_mz: number, polarity: "+"|"-"}} ctx
 * @param {number} [collisionEnergy_eV=25]
 * @returns {Array<{mz: number, intensity: number, label: string, type: string}>}
 */
export function fragmentsFor(analyte, adductKey, ctx, collisionEnergy_eV = 25) {
  if (analyte.sequence) {
    return fragmentsPeptide(analyte, adductKey, ctx, collisionEnergy_eV);
  }

  const cls = LIPID_CLASSES[analyte.class];
  const polarity = ctx.polarity;
  const precMz = ctx.precursor_mz;
  const out = [];

  // Always include the precursor with high intensity
  out.push({
    mz: precMz,
    intensity: 0.4 + 0.6 * Math.exp(-collisionEnergy_eV / 50),
    label: analyte.name + adductKey,
    type: "precursor",
  });

  // ---- Head-group dissociation (positive mode) ----
  if (cls.headGroup && polarity === "+") {
    // Charged head-group ion (e.g. PC -> 184.0733)
    out.push({
      mz: cls.headGroup.charged,
      intensity: 0.95,
      label: cls.headGroup.name + "+",
      type: "head_charged",
    });
    // Neutral head-group loss
    out.push({
      mz: precMz - cls.headGroup.mass,
      intensity: 0.45,
      label: `[M-${cls.headGroup.name}]+`,
      type: "head_loss",
    });
  }

  // ---- Head-group anionic ions (negative mode) ----
  if (polarity === "-") {
    if (analyte.class === "PI") {
      out.push({ mz: 241.0119, intensity: 0.7,  label: "Inositol-1,2-cyclic phosphate-",   type: "head_charged" });
      out.push({ mz: 259.0224, intensity: 0.6,  label: "Inositol phosphate-",               type: "head_charged" });
    } else if (analyte.class === "PS") {
      out.push({ mz: precMz - 87.0320, intensity: 0.85, label: "[M-Ser]-", type: "head_loss" });
    } else if (analyte.class === "PG") {
      out.push({ mz: 152.9958, intensity: 0.45, label: "Glycerophosphate-", type: "head_charged" });
    }
  }

  // ---- Fatty-acid losses ----
  const pairs = commonChainPairs(analyte.X, analyte.Y, cls.faChains);
  for (const pair of pairs) {
    const stride = pair.length / 2;
    for (let i = 0; i < stride; i++) {
      const n = pair[2 * i];
      const db = pair[2 * i + 1];
      const fa = faMass(n, db);
      // Negative mode: free FA carboxylate
      if (polarity === "-") {
        out.push({
          mz: fa - PROTON_MASS,
          intensity: 0.7 * (collisionEnergy_eV / 50),
          label: `FA(${n}:${db})-`,
          type: "fa_anion",
        });
        // [M-FA-H]-
        out.push({
          mz: precMz - fa,
          intensity: 0.35,
          label: `[M-FA(${n}:${db})]-`,
          type: "fa_loss",
        });
      } else {
        // Positive mode: neutral FA loss
        out.push({
          mz: precMz - fa,
          intensity: 0.4,
          label: `[M-FA(${n}:${db})]+`,
          type: "fa_loss",
        });
        // Or ketene (FA - H2O) loss
        out.push({
          mz: precMz - fa + GROUP_MASS.H2O,
          intensity: 0.25,
          label: `[M-RCO(${n}:${db})]+`,
          type: "fa_loss",
        });
      }
    }
  }

  // ---- Class-specific extras ----
  if (analyte.class === "Cer") {
    // Sphingosine dehydration ions
    out.push({ mz: precMz - GROUP_MASS.H2O,     intensity: 0.6, label: "[M-H2O+H]+", type: "neutral_loss" });
    out.push({ mz: precMz - 2 * GROUP_MASS.H2O, intensity: 0.4, label: "[M-2H2O+H]+", type: "neutral_loss" });
  }
  if (analyte.class === "TAG") {
    // [DAG+H-H2O]+ fragments (loss of one FA)
    // Already produced above as fa_loss
  }
  if (analyte.class === "CE") {
    // Cholestene fragment
    out.push({ mz: 369.3515, intensity: 0.95, label: "Cholestene+", type: "head_charged" });
  }

  // Filter out spurious negative or zero m/z
  return out.filter((f) => f.mz > 30 && Number.isFinite(f.mz) && f.intensity > 0.001);
}

/**
 * Combine all fragments and renormalise intensities to [0, 1] by max.
 */
export function normaliseFragments(fragments) {
  if (fragments.length === 0) return [];
  const maxI = fragments.reduce((m, f) => Math.max(m, f.intensity), 0);
  if (maxI === 0) return fragments;
  return fragments.map((f) => ({ ...f, intensity: f.intensity / maxI }));
}

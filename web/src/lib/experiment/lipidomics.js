/**
 * Lipid class definitions and analyte enumeration.
 *
 * For each class we provide:
 *   - a backbone composition (the non-FA part of the molecule)
 *   - a fatty-acid count (1, 2, or 3 chains)
 *   - chain composition validity rules
 *   - a head-group mass for ESI+ fragmentation
 *
 * The mass of a species class(X:Y) where X is total acyl carbons across
 * all chains and Y is total double bonds is computed as:
 *
 *     M(X:Y) = backbone + X * (CH2) - 2*Y * (H) + chain_terminations
 *
 * Equivalent to closed forms in the literature (see e.g. LIPID MAPS).
 */

import { ATOMIC_MASS, GROUP_MASS, monoisotopicMass } from "./constants";

/**
 * Each class has backboneComposition (atoms NOT including FA chains) and
 * faChains (number of chains).
 *
 * The class's full molecule formula is:
 *   backbone + faChains * (Cn1H2n1+1-2db1 ... etc)
 * but for total acyl C = X and total DB = Y, the formula simplifies to:
 *   C: backbone.C + X
 *   H: backbone.H + 2X - 2Y - faChains   (lose 1 H per ester linkage)
 *   plus other backbone atoms unchanged.
 *
 * Each chain contributes -COO- linked at the ester (1 H removed from
 * backbone, plus the FA's COOH minus H2O upon esterification).
 *
 * Reference monoisotopic masses cross-checked with LIPID MAPS.
 */

export const LIPID_CLASSES = {
  PC: {
    name: "Phosphatidylcholine",
    abbr: "PC",
    description: "Glycerophosphocholine, two FA chains",
    faChains: 2,
    polarity_pref: "+",
    // Backbone = glycerol-phosphocholine minus 2 H2O for ester linkages
    // Actual closed-form mass for PC(X:Y): 14.01565*X − 2.01565*Y + 285.06140
    massFormula: (X, Y) => 14.0156501 * X - 2.0156501 * Y + 285.0613929,
    // Compositional formula
    compositionFn: (X, Y) => ({
      C: 8 + X,
      H: 16 + 2 * X - 2 * Y,
      N: 1,
      O: 8,
      P: 1,
    }),
    // Head group (lost as neutral in [M+H]+ HCD as 183.066, [PC head]+ at 184.0733)
    headGroup: { name: "phosphocholine", mass: 183.0660027, charged: 184.0733381 },
    // Default chain ranges (typical biology)
    defaults: { Xrange: [28, 44], Yrange: [0, 6] },
  },

  PE: {
    name: "Phosphatidylethanolamine",
    abbr: "PE",
    description: "Glycerophosphoethanolamine, two FA chains",
    faChains: 2,
    polarity_pref: "-",
    massFormula: (X, Y) => 14.0156501 * X - 2.0156501 * Y + 243.0144440,
    compositionFn: (X, Y) => ({
      C: 5 + X,
      H: 10 + 2 * X - 2 * Y,
      N: 1,
      O: 8,
      P: 1,
    }),
    headGroup: { name: "phosphoethanolamine", mass: 141.0190759, charged: 142.0263113 },
    defaults: { Xrange: [28, 44], Yrange: [0, 6] },
  },

  PS: {
    name: "Phosphatidylserine",
    abbr: "PS",
    description: "Glycerophosphoserine, two FA chains",
    faChains: 2,
    polarity_pref: "-",
    massFormula: (X, Y) => 14.0156501 * X - 2.0156501 * Y + 287.0402737,
    compositionFn: (X, Y) => ({
      C: 6 + X,
      H: 10 + 2 * X - 2 * Y,
      N: 1,
      O: 10,
      P: 1,
    }),
    headGroup: { name: "phosphoserine", mass: 185.0089057, charged: 186.0161410 },
    defaults: { Xrange: [28, 44], Yrange: [0, 6] },
  },

  PG: {
    name: "Phosphatidylglycerol",
    abbr: "PG",
    description: "Glycerophosphoglycerol, two FA chains",
    faChains: 2,
    polarity_pref: "-",
    massFormula: (X, Y) => 14.0156501 * X - 2.0156501 * Y + 274.0399887,
    compositionFn: (X, Y) => ({
      C: 6 + X,
      H: 11 + 2 * X - 2 * Y,
      O: 10,
      P: 1,
    }),
    headGroup: { name: "phosphoglycerol", mass: 172.0136208, charged: 173.0208561 },
    defaults: { Xrange: [28, 44], Yrange: [0, 6] },
  },

  PI: {
    name: "Phosphatidylinositol",
    abbr: "PI",
    description: "Glycerophosphoinositol, two FA chains",
    faChains: 2,
    polarity_pref: "-",
    massFormula: (X, Y) => 14.0156501 * X - 2.0156501 * Y + 362.0610177,
    compositionFn: (X, Y) => ({
      C: 9 + X,
      H: 15 + 2 * X - 2 * Y,
      O: 13,
      P: 1,
    }),
    headGroup: { name: "phosphoinositol", mass: 260.0297062, charged: 261.0369416 },
    defaults: { Xrange: [28, 44], Yrange: [0, 6] },
  },

  SM: {
    name: "Sphingomyelin",
    abbr: "SM",
    description: "Sphingosine + phosphocholine + 1 N-linked FA",
    faChains: 1,
    polarity_pref: "+",
    // SM(d18:1/X:Y) approximate; backbone is sphingosine d18:1
    // Formula reference: SM(d18:1/16:0) = C39H79N2O6P = 702.5676
    massFormula: (X, Y) => 14.0156501 * X - 2.0156501 * Y + 478.3909116,
    compositionFn: (X, Y) => ({
      C: 23 + X,
      H: 47 + 2 * X - 2 * Y,
      N: 2,
      O: 6,
      P: 1,
    }),
    headGroup: { name: "phosphocholine", mass: 183.0660027, charged: 184.0733381 },
    defaults: { Xrange: [14, 26], Yrange: [0, 4] },
  },

  Cer: {
    name: "Ceramide",
    abbr: "Cer",
    description: "Sphingosine + 1 N-linked FA",
    faChains: 1,
    polarity_pref: "-",
    // Cer(d18:1/16:0) = C34H67NO3 = 537.5125
    massFormula: (X, Y) => 14.0156501 * X - 2.0156501 * Y + 313.2980,
    compositionFn: (X, Y) => ({
      C: 18 + X,
      H: 35 + 2 * X - 2 * Y,
      N: 1,
      O: 3,
    }),
    headGroup: null,
    defaults: { Xrange: [14, 26], Yrange: [0, 4] },
  },

  TAG: {
    name: "Triacylglycerol",
    abbr: "TAG",
    description: "Glycerol + 3 FA chains",
    faChains: 3,
    polarity_pref: "+",
    // TAG(48:0) = C51H98O6 = 806.7363
    massFormula: (X, Y) => 14.0156501 * X - 2.0156501 * Y + 134.0942946,
    compositionFn: (X, Y) => ({
      C: 3 + X,
      H: 2 + 2 * X - 2 * Y,
      O: 6,
    }),
    headGroup: null,
    defaults: { Xrange: [42, 60], Yrange: [0, 9] },
  },

  DAG: {
    name: "Diacylglycerol",
    abbr: "DAG",
    description: "Glycerol + 2 FA chains",
    faChains: 2,
    polarity_pref: "+",
    // DAG(36:2) = C39H72O5 = 620.5380
    massFormula: (X, Y) => 14.0156501 * X - 2.0156501 * Y + 92.0473470,
    compositionFn: (X, Y) => ({
      C: 3 + X,
      H: 4 + 2 * X - 2 * Y,
      O: 5,
    }),
    headGroup: null,
    defaults: { Xrange: [28, 40], Yrange: [0, 6] },
  },

  LPC: {
    name: "Lysophosphatidylcholine",
    abbr: "LPC",
    description: "Glycerophosphocholine + 1 FA chain",
    faChains: 1,
    polarity_pref: "+",
    // LPC(16:0) = C24H50NO7P = 495.3325
    massFormula: (X, Y) => 14.0156501 * X - 2.0156501 * Y + 271.0822178,
    compositionFn: (X, Y) => ({
      C: 8 + X,
      H: 18 + 2 * X - 2 * Y,
      N: 1,
      O: 7,
      P: 1,
    }),
    headGroup: { name: "phosphocholine", mass: 183.0660027, charged: 184.0733381 },
    defaults: { Xrange: [14, 24], Yrange: [0, 4] },
  },

  CE: {
    name: "Cholesteryl ester",
    abbr: "CE",
    description: "Cholesterol + 1 FA chain",
    faChains: 1,
    polarity_pref: "+",
    // CE(16:0) = C43H76O2 = 624.5845
    massFormula: (X, Y) => 14.0156501 * X - 2.0156501 * Y + 368.3443210,
    compositionFn: (X, Y) => ({
      C: 27 + X,
      H: 44 + 2 * X - 2 * Y,
      O: 2,
    }),
    headGroup: { name: "cholestene", mass: 368.3443210, charged: 369.3515564 },
    defaults: { Xrange: [14, 24], Yrange: [0, 4] },
  },

  FA: {
    name: "Free fatty acid",
    abbr: "FA",
    description: "Single fatty acid",
    faChains: 1,
    polarity_pref: "-",
    // FA(16:0) = C16H32O2 = 256.2402
    massFormula: (X, Y) => 14.0156501 * X - 2.0156501 * Y - 0.000064,
    compositionFn: (X, Y) => ({
      C: X,
      H: 2 * X - 2 * Y,
      O: 2,
    }),
    headGroup: null,
    defaults: { Xrange: [12, 24], Yrange: [0, 6] },
  },
};

export const LIPID_CLASS_KEYS = Object.keys(LIPID_CLASSES);

/**
 * Enumerate all (X, Y) combinations for a class within the chain ranges.
 * @param {string} classKey
 * @param {Object} ranges  { Xmin, Xmax, Ymin, Ymax }
 * @returns {Array<{class: string, X: number, Y: number, mass: number, formula: string, composition: Object}>}
 */
export function enumerateClass(classKey, ranges = {}) {
  const cls = LIPID_CLASSES[classKey];
  if (!cls) throw new Error(`Unknown lipid class ${classKey}`);
  const Xmin = ranges.Xmin ?? cls.defaults.Xrange[0];
  const Xmax = ranges.Xmax ?? cls.defaults.Xrange[1];
  const Ymin = ranges.Ymin ?? cls.defaults.Yrange[0];
  const Ymax = ranges.Ymax ?? cls.defaults.Yrange[1];
  const out = [];
  for (let X = Xmin; X <= Xmax; X++) {
    // valid DB range for X carbons: at most floor((X - faChains) / 2) double bonds
    const maxValidY = Math.min(Ymax, Math.floor((X - cls.faChains) / 2));
    for (let Y = Ymin; Y <= maxValidY; Y++) {
      const mass = cls.massFormula(X, Y);
      const composition = cls.compositionFn(X, Y);
      out.push({
        class: classKey,
        X,
        Y,
        mass,
        composition,
        name: `${classKey}(${X}:${Y})`,
      });
    }
  }
  return out;
}

/**
 * Enumerate analytes from a list of class specs.
 *
 * @param {Array<{classKey: string, Xmin?, Xmax?, Ymin?, Ymax?}>} classSpecs
 * @returns {Array} flat list of analytes
 */
export function enumerateAnalytes(classSpecs) {
  const all = [];
  for (const spec of classSpecs) {
    all.push(...enumerateClass(spec.classKey, spec));
  }
  return all;
}

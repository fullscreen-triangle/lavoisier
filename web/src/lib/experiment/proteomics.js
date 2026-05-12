/**
 * Proteomics analyte definitions: amino acid residue masses, peptide mass
 * computation, and reference tryptic peptide libraries for common standards.
 *
 * Design-space analogy to lipidomics:
 *   X  → peptide length (number of residues)
 *   Y  → missed trypsin cleavages
 */

const H2O_MASS = 18.01056468;

/**
 * Monoisotopic residue masses and atomic compositions.
 * A residue = the amino acid condensation product in the peptide backbone
 * (i.e. amino acid minus one H2O molecule).
 */
export const RESIDUES = {
  G: { mass: 57.02146372,  C: 2,  H: 3,  N: 1, O: 1, S: 0 },
  A: { mass: 71.03711379,  C: 3,  H: 5,  N: 1, O: 1, S: 0 },
  V: { mass: 99.06841392,  C: 5,  H: 9,  N: 1, O: 1, S: 0 },
  L: { mass: 113.0840640,  C: 6,  H: 11, N: 1, O: 1, S: 0 },
  I: { mass: 113.0840640,  C: 6,  H: 11, N: 1, O: 1, S: 0 },
  P: { mass: 97.05276386,  C: 5,  H: 7,  N: 1, O: 1, S: 0 },
  F: { mass: 147.0684139,  C: 9,  H: 9,  N: 1, O: 1, S: 0 },
  W: { mass: 186.0793129,  C: 11, H: 10, N: 2, O: 1, S: 0 },
  M: { mass: 131.0404846,  C: 5,  H: 9,  N: 1, O: 1, S: 1 },
  S: { mass: 87.03202840,  C: 3,  H: 5,  N: 1, O: 2, S: 0 },
  T: { mass: 101.0476785,  C: 4,  H: 7,  N: 1, O: 2, S: 0 },
  C: { mass: 103.0091845,  C: 3,  H: 5,  N: 1, O: 1, S: 1 },
  Y: { mass: 163.0633286,  C: 9,  H: 9,  N: 1, O: 2, S: 0 },
  H: { mass: 137.0589119,  C: 6,  H: 7,  N: 3, O: 1, S: 0 },
  D: { mass: 115.0269430,  C: 4,  H: 5,  N: 1, O: 3, S: 0 },
  E: { mass: 129.0425931,  C: 5,  H: 7,  N: 1, O: 3, S: 0 },
  N: { mass: 114.0429274,  C: 4,  H: 6,  N: 2, O: 2, S: 0 },
  Q: { mass: 128.0585775,  C: 5,  H: 8,  N: 2, O: 2, S: 0 },
  K: { mass: 128.0949630,  C: 6,  H: 12, N: 2, O: 1, S: 0 },
  R: { mass: 156.1011110,  C: 6,  H: 12, N: 4, O: 1, S: 0 },
};

/** Neutral monoisotopic mass of a peptide sequence. */
export function peptideMass(sequence) {
  let m = H2O_MASS;
  for (const aa of sequence) {
    const r = RESIDUES[aa];
    if (r) m += r.mass;
  }
  return m;
}

/** Elemental composition of a peptide (residues + H2O for the termini). */
export function peptideComposition(sequence) {
  const comp = { C: 0, H: 2, N: 0, O: 1, S: 0 };
  for (const aa of sequence) {
    const r = RESIDUES[aa];
    if (!r) continue;
    comp.C += r.C;
    comp.H += r.H;
    comp.N += r.N;
    comp.O += r.O;
    comp.S += r.S;
  }
  return comp;
}

/**
 * Number of internal K/R not followed by P (missed tryptic cleavages).
 * Exported for use in the UI to estimate active peptide counts.
 */
export function countMissedCleavages(seq) {
  let mc = 0;
  for (let i = 0; i < seq.length - 1; i++) {
    if ((seq[i] === "K" || seq[i] === "R") && seq[i + 1] !== "P") mc++;
  }
  return mc;
}

/**
 * Reference tryptic peptide libraries for well-characterised proteomics
 * standards. Sequences are from UniProt-verified tryptic digests.
 */
export const PROTEIN_CLASSES = {
  HSA: {
    name: "Human Serum Albumin",
    abbr: "HSA",
    description: "Most abundant plasma protein — tryptic digest (UniProt P02768)",
    defaults: { lengthMin: 7, lengthMax: 20, mcMin: 0, mcMax: 1 },
    peptides: [
      "LVNEVTEFAK",
      "AEFAEVSK",
      "HLVDEPQNLIK",
      "YLYEIAR",
      "LQQCPFEDHVK",
      "SLHTLFGDK",
      "VPQVSTPTLVEVSR",
      "EYEATLEECCAK",
      "DVFLGMFLYEYAR",
      "NECFLSHK",
      "KVPQVSTPTLVEVSR",
      "FKDLGEEHFK",
      "RHPYFYAPELLYYANK",
    ],
  },

  HBB: {
    name: "Hemoglobin α/β",
    abbr: "Hb",
    description: "Abundant erythrocyte proteins — tryptic digest (P69905 / P68871)",
    defaults: { lengthMin: 7, lengthMax: 25, mcMin: 0, mcMax: 1 },
    peptides: [
      "VHLTPEEK",
      "SAVTALWGK",
      "LLVVYPWTQR",
      "FFESFGDLSTPDAVMGNPK",
      "EFTPPVQAAYQK",
      "TYFPHFDLSHGSAQVK",
      "MFLSFPTTK",
      "VNVDEVGGEALGRLLVVYPWTQR",
      "VGAHAGEYGAEALERMFLSFPTTK",
    ],
  },

  ENO1: {
    name: "Yeast Enolase",
    abbr: "ENO1",
    description: "Classic LC-MS/MS digest standard — yeast enolase (UniProt P00924)",
    defaults: { lengthMin: 7, lengthMax: 20, mcMin: 0, mcMax: 0 },
    peptides: [
      "GNPTVEVDLFTSK",
      "AAVPSGASTGIYEALELR",
      "VNQIGTLSESIK",
      "SGETEDTFIADLVVGLR",
      "YITPDQLADLYK",
      "IDQLIESGR",
      "IHVSTQNTIDDLYK",
      "QIGSVTESLQELAK",
      "DGVVLHSIEK",
      "ELAAFAR",
    ],
  },

  CYCS: {
    name: "Cytochrome C",
    abbr: "CytC",
    description: "Classic proteomics calibrant — horse/human CytC (UniProt P99999)",
    defaults: { lengthMin: 6, lengthMax: 20, mcMin: 0, mcMax: 1 },
    peptides: [
      "TGPNLHGLFGR",
      "GITWGEETLMEYLENPK",
      "MIFAGIK",
      "EDLIAYLK",
      "YIPGTK",
      "CAQCHTVEK",
      "KYIPGTK",
      "HKTGPNLHGLFGR",
    ],
  },

  CASE: {
    name: "β-Casein",
    abbr: "β-Cas",
    description: "Common digest standard — bovine β-casein (UniProt P02666)",
    defaults: { lengthMin: 7, lengthMax: 20, mcMin: 0, mcMax: 1 },
    peptides: [
      "VLPVPQK",
      "FALPQYLK",
      "IPIQYVLSR",
      "AVPYPQR",
      "FFVAPFPEVFGK",
      "ALNEINQFYQK",
      "DMPIQAFLLYQEPVLGPVR",
      "HIQKEDVPSER",
    ],
  },
};

export const PROTEIN_CLASS_KEYS = Object.keys(PROTEIN_CLASSES);

/**
 * Enumerate peptide analytes for a single protein class spec after
 * applying length and missed-cleavage filters.
 */
export function enumerateProteinClass(spec) {
  const cls = PROTEIN_CLASSES[spec.classKey];
  if (!cls) throw new Error(`Unknown protein class ${spec.classKey}`);
  const lengthMin = spec.lengthMin ?? cls.defaults.lengthMin;
  const lengthMax = spec.lengthMax ?? cls.defaults.lengthMax;
  const mcMin    = spec.mcMin    ?? cls.defaults.mcMin;
  const mcMax    = spec.mcMax    ?? cls.defaults.mcMax;

  return cls.peptides
    .filter((seq) => {
      const mc = countMissedCleavages(seq);
      return seq.length >= lengthMin && seq.length <= lengthMax
          && mc >= mcMin && mc <= mcMax;
    })
    .map((seq) => ({
      class: spec.classKey,
      sequence: seq,
      length: seq.length,
      missedCleavages: countMissedCleavages(seq),
      mass: peptideMass(seq),
      composition: peptideComposition(seq),
      name: seq,
    }));
}

/**
 * Enumerate all peptide analytes from a list of protein specs.
 */
export function enumerateProteinAnalytes(proteinSpecs) {
  const all = [];
  for (const spec of proteinSpecs) {
    all.push(...enumerateProteinClass(spec));
  }
  return all;
}

/**
 * Preferred ESI adducts for a peptide ordered by expected signal intensity.
 * Proteomics analytes form multiply-charged ions; the preferred charge state
 * scales with peptide mass.
 */
export function preferredProteinAdducts(mass, polarity) {
  if (polarity === "+") {
    if (mass < 900)  return ["[M+H]+",   "[M+2H]2+"];
    if (mass < 1800) return ["[M+2H]2+", "[M+H]+",   "[M+3H]3+"];
    if (mass < 3000) return ["[M+3H]3+", "[M+2H]2+", "[M+4H]4+"];
    return                  ["[M+3H]3+", "[M+4H]4+", "[M+2H]2+"];
  }
  return ["[M-2H]2-", "[M-H]-"];
}

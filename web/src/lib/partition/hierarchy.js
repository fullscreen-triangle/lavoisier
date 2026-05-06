/**
 * Hierarchical oscillator decomposition.
 *
 * Each peak in a spectrum IS an oscillator with its own partition
 * coordinates (n, l, m, s) — exactly as the partition Lagrangian
 * predicts. Instead of collapsing the spectrum into three summary
 * statistics, we map every peak to a partition cell and let the
 * molecule's identity emerge as the SET of occupied cells.
 *
 * The mapping (mz, intensity, polarity, charge) → (n, l, m, s) is
 * derived from the framework:
 *
 *   n  ∈ ℕ⁺          principal partition depth — mass shell
 *   l  ∈ {0..n-1}    angular complexity within shell
 *   m  ∈ {-l..+l}    orientation within (n, l)
 *   s  ∈ {-½, +½}    chirality (polarity / parity)
 *
 * Shell capacity C(n) = 2n² (the capacity formula). Cumulative through
 * shell n_max is n_max(n_max+1)(2n_max+1)/3 cells, so:
 *   n_max =  3 →  18 cells
 *   n_max =  5 →  55 cells
 *   n_max = 10 → 385 cells
 */

/**
 * @typedef {Object} OscillatorCoord
 * @property {number} n        principal partition depth ≥ 1
 * @property {number} l        angular complexity ∈ [0, n−1]
 * @property {number} m        orientation ∈ [−l, +l]
 * @property {number} s        chirality ∈ {−½, +½}
 * @property {number} mz       original m/z of the peak
 * @property {number} intensity absolute intensity
 * @property {number} weight   normalised intensity (occupancy weight)
 * @property {number} cellIndex flat index into the cumulative cell space
 */

/**
 * @typedef {Object} HierarchicalDecomposition
 * @property {OscillatorCoord[]} oscillators  one per peak (sorted by intensity)
 * @property {number} nMax                    deepest occupied shell
 * @property {Map<number, number>} occupancy  cellIndex → cumulative weight
 * @property {number} totalCells              capacity through nMax
 * @property {number} occupiedCells           number of distinct cells
 */

/* -------------------------------------------------------------------- */
/* Shell mapping: m/z → n                                               */
/* -------------------------------------------------------------------- */

/**
 * Reference mass for the smallest meaningful mass shell.
 * 14 Da ≈ CH₂ — the minimal "tick" of organic chemistry.
 * Hydrogen and protons therefore occupy n = 1.
 */
const M_REF = 14.0;

/**
 * Map m/z to a principal partition depth n ≥ 1.
 *
 * Derivation: the partition Lagrangian has potential wells at depths
 * proportional to the ion's mass. Using a square-root mapping
 *   n = ⌈√(m/z / m_ref)⌉
 * keeps the cell density roughly uniform per Da, since C(n) = 2n² ⇒
 * Δm ∝ n. Heavier ions occupy higher shells.
 *
 * @param {number} mz
 * @returns {number}  n ≥ 1
 */
export function mzToShell(mz) {
  if (!Number.isFinite(mz) || mz <= 0) return 1;
  const n = Math.ceil(Math.sqrt(mz / M_REF));
  return Math.max(1, n);
}

/**
 * Inverse of mzToShell — return the [low, high] m/z bounds of shell n.
 * @param {number} n
 * @returns {[number, number]}
 */
export function shellMzRange(n) {
  const low = (n - 1) * (n - 1) * M_REF;
  const high = n * n * M_REF;
  return [low, high];
}

/* -------------------------------------------------------------------- */
/* Angular complexity within a shell: l                                 */
/* -------------------------------------------------------------------- */

/**
 * Map a peak's relative position within its shell to angular complexity l.
 *
 * Within shell n, l ∈ {0, .., n−1}. We use the peak's normalised
 * position (linear interpolation of m/z within the shell's range) to
 * pick l. The lowest-mass peaks in a shell have l = 0 (spherical);
 * the highest have l = n−1 (most angularly complex).
 *
 * @param {number} mz
 * @param {number} n
 * @returns {number}  l ∈ [0, n−1]
 */
export function angularComplexity(mz, n) {
  if (n <= 1) return 0;
  const [low, high] = shellMzRange(n);
  if (high === low) return 0;
  const frac = (mz - low) / (high - low);
  const clamped = Math.max(0, Math.min(0.9999, frac));
  return Math.min(n - 1, Math.floor(clamped * n));
}

/* -------------------------------------------------------------------- */
/* Orientation within (n, l): m                                          */
/* -------------------------------------------------------------------- */

/**
 * Map a peak's intensity / isotope offset to orientation m ∈ [−l, +l].
 *
 * The simplest interpretation matches isotopologue patterns:
 *   m =  0  monoisotopic peak in its (n, l) cell
 *   m = +1  M+1 isotope
 *   m = −1  rare M−1 (e.g. neutron loss)
 *   m = +2  M+2 (e.g. ³⁷Cl, ³⁴S)
 *
 * Without isotope assignment we fall back to intensity-rank within
 * the (n, l) cell, mapped to [−l, +l].
 *
 * @param {number} intensityRank  rank within (n, l), 0 = most intense
 * @param {number} l
 * @returns {number}  m ∈ [−l, +l]
 */
export function orientation(intensityRank, l) {
  if (l === 0) return 0;
  // Map rank 0,1,2,… to ascending |m|: 0, +1, -1, +2, -2, …
  const half = Math.ceil(intensityRank / 2);
  const sign = intensityRank % 2 === 0 ? +1 : -1;
  const mag = Math.min(half, l);
  return sign * mag;
}

/* -------------------------------------------------------------------- */
/* Chirality: s                                                          */
/* -------------------------------------------------------------------- */

/**
 * Map polarity (and optional even/odd electron state) to chirality.
 *
 * Cation (positive) → +½, anion (negative) → −½.
 * For radical species a finer 4-state encoding is possible; we keep
 * the binary form to stay aligned with the {±½} algebra.
 *
 * @param {"positive"|"negative"|"unknown"} polarity
 * @param {number} [charge]
 * @returns {number}
 */
export function chirality(polarity, charge = 0) {
  if (polarity === "negative") return -0.5;
  if (polarity === "positive") return +0.5;
  // Unknown polarity: use sign of charge
  if (charge < 0) return -0.5;
  return +0.5;
}

/* -------------------------------------------------------------------- */
/* Cell indexing                                                         */
/* -------------------------------------------------------------------- */

/**
 * Capacity of shell n (number of (l, m, s) cells in shell n).
 * @param {number} n
 */
export function shellCapacity(n) {
  return 2 * n * n;
}

/**
 * Cumulative capacity through shell n_max (total cells in shells 1..n_max).
 * @param {number} nMax
 */
export function cumulativeCapacity(nMax) {
  return (nMax * (nMax + 1) * (2 * nMax + 1)) / 3;
}

/**
 * Flatten (n, l, m, s) into a single non-negative integer cell index.
 * The packing is reversible — see decodeCellIndex.
 *
 * Layout:
 *   index = cumulativeCapacity(n−1)
 *         + 2 · (l² + l + m)        ← position within shell
 *         + (s > 0 ? 1 : 0)         ← chirality bit
 *
 * The middle term l² + l + m maps (l, m) ∈ {(0,0), (1,−1), (1,0), (1,1), (2,−2), …}
 * to consecutive integers 0, 1, 2, 3, 4, … which is exactly the
 * spherical-harmonic enumeration order.
 *
 * @param {number} n
 * @param {number} l
 * @param {number} m
 * @param {number} s
 * @returns {number}  cell index ≥ 0
 */
export function packCellIndex(n, l, m, s) {
  const shellOffset = cumulativeCapacity(n - 1);
  const lmIdx = l * l + l + m;       // (l, m) → consecutive integer
  const sBit = s > 0 ? 1 : 0;
  return shellOffset + 2 * lmIdx + sBit;
}

/**
 * Inverse of packCellIndex.
 * @param {number} idx
 * @returns {{n: number, l: number, m: number, s: number}}
 */
export function unpackCellIndex(idx) {
  // Find shell — smallest n such that cumulativeCapacity(n) > idx
  let n = 1;
  while (cumulativeCapacity(n) <= idx) n++;
  const within = idx - cumulativeCapacity(n - 1);
  const sBit = within % 2;
  const lmIdx = (within - sBit) / 2;
  // l = ⌊√lmIdx⌋, m = lmIdx − l² − l
  const l = Math.floor(Math.sqrt(lmIdx));
  const m = lmIdx - l * l - l;
  const s = sBit === 1 ? +0.5 : -0.5;
  return { n, l, m, s };
}

/* -------------------------------------------------------------------- */
/* Main decomposition                                                    */
/* -------------------------------------------------------------------- */

/**
 * Decompose a spectrum into a hierarchy of oscillators.
 *
 * Each peak gets its own (n, l, m, s). The result also gives the
 * cell-occupancy distribution — how the molecule's "mass" is spread
 * across the partition hierarchy.
 *
 * @param {Float32Array|Float64Array|number[]} mz
 * @param {Float32Array|Float64Array|number[]} intensity
 * @param {Object} [opts]
 * @param {string} [opts.polarity="positive"]
 * @param {number} [opts.charge=0]
 * @param {number} [opts.topN=64]            keep this many most-intense peaks
 * @param {number} [opts.minIntensity=0]
 * @returns {HierarchicalDecomposition}
 */
export function decomposeSpectrum(mz, intensity, opts = {}) {
  const { polarity = "positive", charge = 0, topN = 64, minIntensity = 0 } = opts;

  if (!mz || !intensity || mz.length === 0) {
    return emptyDecomposition();
  }

  // 1. Pick top-N peaks by intensity (above threshold)
  const candidates = [];
  for (let i = 0; i < mz.length; i++) {
    if (intensity[i] > minIntensity) {
      candidates.push({ mz: mz[i], intensity: intensity[i] });
    }
  }
  candidates.sort((a, b) => b.intensity - a.intensity);
  const peaks = candidates.slice(0, topN);
  if (peaks.length === 0) return emptyDecomposition();

  // Total intensity for normalisation
  let totalI = 0;
  for (const p of peaks) totalI += p.intensity;
  const invTotal = totalI > 0 ? 1 / totalI : 0;

  const s = chirality(polarity, charge);

  // 2. First pass: assign (n, l) to each peak, group by (n, l)
  const cellGroups = new Map(); // "n,l" → [{peak, ...}]
  const intermediate = peaks.map((p) => {
    const n = mzToShell(p.mz);
    const l = angularComplexity(p.mz, n);
    const key = `${n},${l}`;
    if (!cellGroups.has(key)) cellGroups.set(key, []);
    cellGroups.get(key).push({ ...p, n, l });
    return { ...p, n, l };
  });

  // 3. Second pass: within each (n, l) group, assign m by intensity rank
  const oscillators = [];
  let nMax = 0;
  const occupancy = new Map();

  for (const group of cellGroups.values()) {
    group.sort((a, b) => b.intensity - a.intensity);
    for (let rank = 0; rank < group.length; rank++) {
      const p = group[rank];
      const m = orientation(rank, p.l);
      const cellIndex = packCellIndex(p.n, p.l, m, s);
      const weight = p.intensity * invTotal;

      oscillators.push({
        n: p.n,
        l: p.l,
        m,
        s,
        mz: p.mz,
        intensity: p.intensity,
        weight,
        cellIndex,
      });
      if (p.n > nMax) nMax = p.n;
      occupancy.set(cellIndex, (occupancy.get(cellIndex) || 0) + weight);
    }
  }

  // Re-sort oscillators by intensity (most intense first) for downstream consumers
  oscillators.sort((a, b) => b.intensity - a.intensity);

  return {
    oscillators,
    nMax,
    occupancy,
    totalCells: cumulativeCapacity(nMax),
    occupiedCells: occupancy.size,
  };
}

function emptyDecomposition() {
  return {
    oscillators: [],
    nMax: 0,
    occupancy: new Map(),
    totalCells: 0,
    occupiedCells: 0,
  };
}

/* -------------------------------------------------------------------- */
/* Aggregations                                                          */
/* -------------------------------------------------------------------- */

/**
 * Summarise the n-shell distribution as a probability vector p_n.
 * @param {HierarchicalDecomposition} decomp
 * @param {number} [maxN=10]
 * @returns {Float64Array}  length maxN, p_n[i] = mass fraction in shell i+1
 */
export function shellDistribution(decomp, maxN = 10) {
  const out = new Float64Array(maxN);
  for (const osc of decomp.oscillators) {
    if (osc.n - 1 < maxN) out[osc.n - 1] += osc.weight;
  }
  return out;
}

/**
 * Summarise the angular distribution at a given shell.
 * @param {HierarchicalDecomposition} decomp
 * @param {number} n
 * @returns {Float64Array}  length n, p_l[l] = fraction of mass at l within n
 */
export function angularDistribution(decomp, n) {
  const out = new Float64Array(n);
  let total = 0;
  for (const osc of decomp.oscillators) {
    if (osc.n === n) {
      out[osc.l] += osc.weight;
      total += osc.weight;
    }
  }
  if (total > 0) for (let i = 0; i < n; i++) out[i] /= total;
  return out;
}

/**
 * Compute the partition entropy of the decomposition itself.
 * S_partition = − Σ_cell p_cell log p_cell over the occupancy distribution.
 * Higher S_partition = mass spread across more cells = more complex molecule.
 *
 * @param {HierarchicalDecomposition} decomp
 * @returns {number}  in nats; divide by ln(totalCells) to normalise
 */
export function partitionEntropy(decomp) {
  let H = 0;
  for (const w of decomp.occupancy.values()) {
    if (w > 0) H -= w * Math.log(w);
  }
  return H;
}

/**
 * Cell-occupancy similarity (Jaccard-like) between two decompositions.
 * The intersection over union of occupied cells, weighted by occupancy.
 *
 * Unlike summary statistics, this captures STRUCTURAL overlap exactly:
 * two molecules sharing peaks 1, 3, and 5 of their top-10 list will
 * agree on those three cells and differ on the others.
 *
 * @param {HierarchicalDecomposition} a
 * @param {HierarchicalDecomposition} b
 * @returns {number}  in [0, 1]
 */
export function partitionSimilarity(a, b) {
  let inter = 0;
  let union = 0;
  const seen = new Set();
  for (const [idx, wa] of a.occupancy.entries()) {
    const wb = b.occupancy.get(idx) || 0;
    inter += Math.min(wa, wb);
    union += Math.max(wa, wb);
    seen.add(idx);
  }
  for (const [idx, wb] of b.occupancy.entries()) {
    if (!seen.has(idx)) union += wb;
  }
  return union > 0 ? inter / union : 0;
}

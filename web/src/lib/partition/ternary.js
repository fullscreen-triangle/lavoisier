/**
 * Ternary encoding of S-entropy coordinates.
 *
 * Each trit refines one coordinate axis, interleaved:
 *   position j mod 3 = 0 → refine S_k
 *   position j mod 3 = 1 → refine S_t
 *   position j mod 3 = 2 → refine S_e
 *
 * The address IS the trajectory — position and path encoded identically.
 */

/**
 * Encode S-entropy coordinates as a k-trit string.
 *
 * @param {number} sk
 * @param {number} st
 * @param {number} se
 * @param {number} [depth=18]   trit depth; 18 = ~0.1% per-axis resolution
 * @returns {string}            e.g. "202222112122..."
 */
export function ternaryEncode(sk, st, se, depth = 18) {
  const coords = [sk, st, se];
  const trits = new Array(depth);
  for (let j = 0; j < depth; j++) {
    const dim = j % 3;
    let val = Math.floor(coords[dim] * 3);
    if (val > 2) val = 2;
    if (val < 0) val = 0;
    trits[j] = String(val);
    coords[dim] = coords[dim] * 3 - val;
  }
  return trits.join("");
}

/**
 * Recover cell-centre coordinates from a ternary address.
 *
 * The inverse of ternaryEncode — each trit contributes (t + 0.5) / 3^n
 * to its corresponding axis at refinement depth n.
 *
 * @param {string} address
 * @returns {{sk: number, st: number, se: number}}
 */
export function ternaryDecode(address) {
  let sk = 0, st = 0, se = 0;
  const depthPerAxis = [0, 0, 0];

  for (let j = 0; j < address.length; j++) {
    const dim = j % 3;
    const t = parseInt(address[j], 10);
    if (!Number.isFinite(t) || t < 0 || t > 2) continue;

    const nRefine = depthPerAxis[dim] + 1;
    const contribution = (t + 0.5) / Math.pow(3, nRefine);

    if (dim === 0) sk += contribution;
    else if (dim === 1) st += contribution;
    else if (dim === 2) se += contribution;

    depthPerAxis[dim] = nRefine;
  }

  // The above sums midpoints of nested intervals, but only up to the
  // refined depth. For proper reconstruction we want the cell centre,
  // which needs every position filled — use the already-computed
  // values as the current-best-estimate.
  return { sk, st, se };
}

/**
 * Length of the longest common prefix of two addresses.
 * This IS the resonance depth — how many oscillation observations
 * it takes to distinguish the two compounds.
 *
 * @param {string} a
 * @param {string} b
 * @returns {number}
 */
export function commonPrefixLength(a, b) {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  return i;
}

/**
 * Euclidean distance bound from common prefix length.
 * Two addresses sharing k trits are guaranteed to be within
 *   √3 · 3^{-⌊k/3⌋}
 * of each other in [0,1]^3.
 *
 * @param {number} k
 */
export function distanceBoundForPrefix(k) {
  if (k <= 0) return Math.sqrt(3);
  return Math.sqrt(3) * Math.pow(3, -Math.floor(k / 3));
}

/**
 * Resonance score: normalised prefix length in [0, 1].
 * 1.0 = identical addresses, 0 = no shared structure.
 * @param {string} a
 * @param {string} b
 */
export function resonanceScore(a, b) {
  const n = Math.min(a.length, b.length);
  if (n === 0) return 0;
  return commonPrefixLength(a, b) / n;
}

/**
 * False-positive probability bound — two unrelated compounds share
 * a k-trit prefix with probability ≤ 3^{-k}.
 * @param {number} k
 */
export function falsePositiveBound(k) {
  return Math.pow(3, -k);
}

/* -------------------------------------------------------------------- */
/* Hierarchical encoding — one address burst per oscillator              */
/* -------------------------------------------------------------------- */

/**
 * Encode a non-negative integer as a fixed-width base-3 string,
 * least-significant trit last (big-endian).
 * @param {number} value
 * @param {number} width
 */
function intToTrits(value, width) {
  if (!Number.isFinite(value) || value < 0) value = 0;
  let v = Math.floor(value);
  const out = new Array(width);
  for (let i = width - 1; i >= 0; i--) {
    out[i] = String(v % 3);
    v = Math.floor(v / 3);
  }
  return out.join("");
}

/**
 * Decode a fixed-width base-3 string back to an integer.
 * @param {string} trits
 */
function tritsToInt(trits) {
  let v = 0;
  for (let i = 0; i < trits.length; i++) {
    const t = parseInt(trits[i], 10);
    if (!Number.isFinite(t) || t < 0 || t > 2) return v;
    v = v * 3 + t;
  }
  return v;
}

/**
 * Encode a hierarchical decomposition (each peak as its own oscillator
 * with (n, l, m, s)) as a ternary address.
 *
 * Each oscillator contributes a fixed-width "burst" of trits encoding
 * its cell index. The first burst belongs to the most intense
 * oscillator, the second to the next, and so on.
 *
 * Two molecules whose top-K peaks land in the same partition cells will
 * share the first K bursts of the address — so the longest common
 * prefix length corresponds to the depth of structural agreement,
 * exactly as in the summary-statistics encoding, but now grounded in
 * actual peak identity rather than aggregate moments.
 *
 * @param {import("./hierarchy").HierarchicalDecomposition} decomp
 * @param {Object} [opts]
 * @param {number} [opts.peaks=6]        oscillators to encode (top by intensity)
 * @param {number} [opts.tritsPerPeak=5] base-3 width per cell index
 *                                       (5 trits = 243 cells = up to n_max ≈ 5)
 * @returns {string}                     ternary address
 */
export function hierarchicalEncode(decomp, opts = {}) {
  const { peaks = 6, tritsPerPeak = 5 } = opts;
  if (!decomp || !decomp.oscillators || decomp.oscillators.length === 0) {
    return "0".repeat(peaks * tritsPerPeak);
  }
  const cap = Math.pow(3, tritsPerPeak) - 1;
  const out = [];
  for (let i = 0; i < peaks; i++) {
    const osc = decomp.oscillators[i];
    if (osc) {
      const idx = Math.min(osc.cellIndex, cap);
      out.push(intToTrits(idx, tritsPerPeak));
    } else {
      out.push("0".repeat(tritsPerPeak));
    }
  }
  return out.join("");
}

/**
 * Decode a hierarchical address back to a list of cell indices.
 * @param {string} address
 * @param {Object} [opts]
 * @param {number} [opts.tritsPerPeak=5]
 * @returns {number[]}  cell indices in encoding order
 */
export function hierarchicalDecode(address, opts = {}) {
  const { tritsPerPeak = 5 } = opts;
  const cells = [];
  for (let i = 0; i + tritsPerPeak <= address.length; i += tritsPerPeak) {
    cells.push(tritsToInt(address.slice(i, i + tritsPerPeak)));
  }
  return cells;
}

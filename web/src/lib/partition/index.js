/**
 * Partition layer — public API.
 *
 * Given a Scan (from the parser), produce a complete CategoricalState:
 *   - S-entropy coordinates (S_k, S_t, S_e)
 *   - Ternary address (18+ trits)
 *   - Analyser-specific observables
 *
 * This is the main entry point the worker and UI call.
 */

export * as sentropy from "./sentropy.js";
export * as ternary from "./ternary.js";
export * as lagrangian from "./lagrangian.js";
export * as hierarchy from "./hierarchy.js";
export { TernaryTrie } from "./trie.js";

import { computeFromSpectrum, clamp01 } from "./sentropy.js";
import { ternaryEncode, hierarchicalEncode } from "./ternary.js";
import { observe, partitionInertia } from "./lagrangian.js";
import {
  decomposeSpectrum,
  shellDistribution,
  partitionEntropy,
} from "./hierarchy.js";

/**
 * @typedef {Object} HierarchyView
 * @property {Array<{n:number,l:number,m:number,s:number,mz:number,intensity:number,weight:number,cellIndex:number}>} oscillators
 * @property {number} nMax
 * @property {number} occupiedCells
 * @property {number} totalCells
 * @property {number[]} shells          shell-mass distribution p_n (length nMax)
 * @property {number} entropyNats       partition entropy − Σ w log w
 * @property {string} address           ternary burst address (one burst per peak)
 */

/**
 * @typedef {Object} CategoricalState
 * @property {string} scanId
 * @property {number} msLevel
 * @property {number} retentionTime
 * @property {number} charge
 * @property {string} polarity
 * @property {number} basePeakMz
 * @property {number} basePeakIntensity
 * @property {number} totalIonCurrent
 * @property {{sk: number, st: number, se: number}} sentropy
 * @property {string} address              ternary address (summary)
 * @property {HierarchyView} [hierarchy]   per-peak (n,l,m,s) decomposition
 * @property {number} nPeaks
 * @property {Object} [precursor]
 * @property {Object} [observables]        analyser-specific values
 */

/**
 * Convert a parsed Scan into a complete CategoricalState.
 *
 * This is the S-entropy bridge: peaks → (S_k, S_t, S_e) → ternary address.
 *
 * @param {Scan} scan
 * @param {Object} [opts]
 * @param {number} [opts.depth=18]        ternary depth
 * @param {number} [opts.topN=32]         peaks to use
 * @param {number} [opts.minIntensity=0]
 * @param {string} [opts.analyser]        if set, attach observables
 * @param {Object} [opts.analyserCfg]
 * @returns {CategoricalState}
 */
export function encodeScan(scan, opts = {}) {
  const {
    depth = 18,
    topN = 32,
    minIntensity = 0,
    analyser = null,
    analyserCfg = {},
    hierarchical = true,
    hierPeaks = 6,
    hierTritsPerPeak = 5,
    hierTopN = 64,
  } = opts;

  const coords = clamp01(
    computeFromSpectrum(scan.mz, scan.intensity, { topN, minIntensity })
  );

  const address = ternaryEncode(coords.sk, coords.st, coords.se, depth);

  const state = {
    scanId: scan.id,
    msLevel: scan.msLevel,
    retentionTime: scan.retentionTime,
    charge: scan.charge,
    polarity: scan.polarity,
    basePeakMz: scan.basePeakMz,
    basePeakIntensity: scan.basePeakIntensity,
    totalIonCurrent: scan.totalIonCurrent,
    sentropy: coords,
    address,
    nPeaks: scan.peakCount,
    precursor: scan.precursor,
  };

  if (hierarchical && scan.mz && scan.mz.length > 0) {
    const decomp = decomposeSpectrum(scan.mz, scan.intensity, {
      polarity: scan.polarity,
      charge: scan.charge,
      topN: hierTopN,
      minIntensity,
    });
    const shells = Array.from(
      shellDistribution(decomp, Math.max(1, decomp.nMax || 1))
    );
    const hierAddress = hierarchicalEncode(decomp, {
      peaks: hierPeaks,
      tritsPerPeak: hierTritsPerPeak,
    });
    state.hierarchy = {
      oscillators: decomp.oscillators,
      nMax: decomp.nMax,
      occupiedCells: decomp.occupiedCells,
      totalCells: decomp.totalCells,
      shells,
      entropyNats: partitionEntropy(decomp),
      address: hierAddress,
    };
  }

  if (analyser && scan.basePeakMz > 0) {
    state.observables = observe(analyser, scan.basePeakMz, analyserCfg);
    state.partitionInertia = partitionInertia(scan.basePeakMz, scan.charge || 1);
  }

  return state;
}

/**
 * Batch-encode many scans (for GPU upload preparation).
 * @param {Scan[]} scans
 * @param {Object} [opts]
 */
export function encodeScans(scans, opts = {}) {
  return scans.map((s) => encodeScan(s, opts));
}

/**
 * S-entropy coordinate computation.
 *
 * Given a set of oscillation frequencies (or peak m/z values treated as
 * such), compute three algebraically independent quantities in [0,1]:
 *
 *   S_k (knowledge) — Shannon entropy of the frequency distribution
 *   S_t (temporal)  — logarithmic ratio of frequency extremes
 *   S_e (evolution) — fraction of harmonically proximate mode pairs
 *
 * These three coordinates address the molecule's position in the
 * bounded phase space S = [0,1]^3. Identical to the Python reference
 * in validate_force_free.py et al., down to the constants.
 */

export const OMEGA_REF_MAX = 4401.0;   // cm^-1 (H2)
export const OMEGA_REF_MIN = 218.0;    // cm^-1 (CCl4 lowest)
export const B_ROT_REF_MIN = 0.39;     // rotational reference (cm^-1)

export const DELTA_HARMONIC = 0.05;
export const P_MAX_HARMONIC = 8;

/**
 * Knowledge entropy.
 *
 * Polyatomic (N >= 2): normalised Shannon entropy of the frequency
 *   distribution p_i = ω_i / Σ ω_j.
 * Diatomic (N = 1): ω / ω_ref with ω_ref = 4401 cm^-1 (H2).
 *
 * @param {number[]|Float32Array|Float64Array} freqs
 * @param {boolean} isDiatomic
 * @returns {number} in [0, 1]
 */
export function computeSk(freqs, isDiatomic) {
  if (isDiatomic) {
    return freqs[0] / OMEGA_REF_MAX;
  }
  const N = freqs.length;
  if (N === 0) return 0;
  if (N === 1) return freqs[0] / OMEGA_REF_MAX;

  let total = 0;
  for (let i = 0; i < N; i++) total += freqs[i];
  if (total === 0) return 0;

  let H = 0;
  for (let i = 0; i < N; i++) {
    const p = freqs[i] / total;
    if (p > 0) H -= p * Math.log2(p);
  }
  return H / Math.log2(N);
}

/**
 * Temporal entropy.
 *
 * Polyatomic: log(ω_max / ω_min) / log(ω_ref_max / ω_ref_min).
 * Diatomic with rotational constant B_rot: log(ω / B_rot) / log(ω_ref_max / B_rot_ref_min).
 *
 * @param {number[]|Float32Array|Float64Array} freqs
 * @param {boolean} isDiatomic
 * @param {number|null} [bRot]
 * @returns {number} in [0, 1]
 */
export function computeSt(freqs, isDiatomic, bRot = null) {
  if (isDiatomic) {
    if (!bRot || bRot <= 0) return 0;
    return Math.log(freqs[0] / bRot) / Math.log(OMEGA_REF_MAX / B_ROT_REF_MIN);
  }

  let wMin = Infinity;
  let wMax = -Infinity;
  for (let i = 0; i < freqs.length; i++) {
    const w = freqs[i];
    if (w <= 0) continue;
    if (w < wMin) wMin = w;
    if (w > wMax) wMax = w;
  }
  if (!Number.isFinite(wMin) || !Number.isFinite(wMax) || wMin === wMax) return 0;
  return Math.log(wMax / wMin) / Math.log(OMEGA_REF_MAX / OMEGA_REF_MIN);
}

/**
 * Evolution entropy.
 *
 * Fraction of frequency pairs (ω_a, ω_b) where max/min is close to a rational
 * p/q with p,q <= 8 and |ratio - p/q| < δ (δ = 0.05).
 *
 * @param {number[]|Float32Array|Float64Array} freqs
 * @returns {number} in [0, 1]
 */
export function computeSe(freqs) {
  const N = freqs.length;
  if (N < 2) return 0;

  const nPairs = (N * (N - 1)) / 2;
  let nHarmonic = 0;

  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const a = Math.max(freqs[i], freqs[j]);
      const b = Math.min(freqs[i], freqs[j]);
      if (b <= 0) continue;
      const ratio = a / b;

      let matched = false;
      for (let p = 1; p <= P_MAX_HARMONIC && !matched; p++) {
        for (let q = 1; q <= p && !matched; q++) {
          if (Math.abs(ratio - p / q) < DELTA_HARMONIC) {
            nHarmonic++;
            matched = true;
          }
        }
      }
    }
  }

  return nHarmonic / Math.max(nPairs, 1);
}

/**
 * Compute all three S-entropy coordinates from a spectrum.
 *
 * For an MS spectrum, we treat the top-N most intense peaks (weighted by
 * intensity) as the effective "frequency spectrum" of the bounded
 * oscillatory system represented by the ion. This is the S-entropy
 * bridge from mzML data into partition space.
 *
 * @param {Float32Array|Float64Array|number[]} mz
 * @param {Float32Array|Float64Array|number[]} intensity
 * @param {Object} [opts]
 * @param {number} [opts.topN=32]       take this many most intense peaks
 * @param {number} [opts.minIntensity=0] filter peaks below this
 * @returns {{sk: number, st: number, se: number, nPeaks: number}}
 */
export function computeFromSpectrum(mz, intensity, opts = {}) {
  const { topN = 32, minIntensity = 0 } = opts;

  if (!mz || !intensity || mz.length === 0) {
    return { sk: 0, st: 0, se: 0, nPeaks: 0 };
  }

  // Collect indices of peaks above threshold, sorted by intensity desc
  const indices = [];
  for (let i = 0; i < mz.length; i++) {
    if (intensity[i] > minIntensity) indices.push(i);
  }
  indices.sort((a, b) => intensity[b] - intensity[a]);

  const take = Math.min(topN, indices.length);
  if (take === 0) return { sk: 0, st: 0, se: 0, nPeaks: 0 };

  // Build the effective frequency array — we use m/z values directly
  // as the oscillatory proxy (this is the S-entropy bridge)
  const freqs = new Float64Array(take);
  for (let i = 0; i < take; i++) freqs[i] = mz[indices[i]];

  const isDiatomic = take === 1;
  const sk = computeSk(freqs, isDiatomic);
  const st = computeSt(freqs, isDiatomic, null);
  const se = computeSe(freqs);

  return { sk, st, se, nPeaks: take };
}

/**
 * Euclidean distance in S-entropy space.
 * @param {{sk,st,se}} a
 * @param {{sk,st,se}} b
 */
export function distance(a, b) {
  const dk = a.sk - b.sk;
  const dt = a.st - b.st;
  const de = a.se - b.se;
  return Math.sqrt(dk * dk + dt * dt + de * de);
}

/**
 * Clamp coordinates into [0,1]^3 for downstream ternary encoding.
 */
export function clamp01(coord) {
  return {
    sk: Math.min(1, Math.max(0, coord.sk)),
    st: Math.min(1, Math.max(0, coord.st)),
    se: Math.min(1, Math.max(0, coord.se)),
  };
}

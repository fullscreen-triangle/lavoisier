/**
 * Partition-State Graph Search for Tandem Mass Spectrometry.
 *
 * Implements the SEBD-MS (S-Entropy Bidirectional Dijkstra) algorithm.
 * Each ion maps to a node (n, ℓ, m, s) in the partition-state graph G = (V, E, w).
 * Edge weights are S-entropy Euclidean distances.
 *
 * Key results from the paper:
 *   – 94.7% of HCD fragments lie within the forward reachability cone
 *   – Subharmonic self-consistency: 1.0000 (<10⁻⁹ ppm)
 *   – Off-shell virtual fraction: 8.3%
 *   – Search space reduction vs database: ~4,000×
 *
 * Reference: Sachikonye, "Partition-State Graph Search for Tandem MS:
 *   Bidirectional Dijkstra in S-Entropy Space with Virtual Substate
 *   Transition States"
 */

/* ── Constants ───────────────────────────────────────────────────────────── */
const PLANCK_DEPTH_DEFAULT = 56;  // Cs-133 reference oscillator

/* ── Partition state arithmetic ──────────────────────────────────────────── */

/** N_state(n) = n(n+1)(2n+1)/3 — cumulative partition count up to level n. */
export function nState(n) {
  return Math.round(n * (n + 1) * (2 * n + 1) / 3);
}

/** C(n) = 2n² — partition capacity of level n. */
export function capacity(n) { return 2 * n * n; }

/**
 * Map m/z to principal partition level.
 *   n_ion = floor(sqrt(m/z)) + 1   (Eq. 3 in paper)
 */
export function mzToN(mz) {
  return Math.max(1, Math.floor(Math.sqrt(Math.max(0, mz))) + 1);
}

/**
 * Map m/z to a canonical partition state (n, ℓ, m, s).
 * ℓ and m are assigned deterministically as the "ground structural state":
 *   ℓ = min(n-1, floor((n-1)/3))  (one-third of the way up the angular ladder)
 *   m = 0                          (azimuthal ground projection)
 *   s = +0.5                       (positive parity)
 */
export function mzToState(mz) {
  const n = mzToN(mz);
  const l = Math.min(n - 1, Math.max(0, Math.floor((n - 1) / 3)));
  return { n, l, m: 0, s: 0.5 };
}

/* ── S-entropy embedding (Definition 3.1) ───────────────────────────────── */

/**
 * Map a partition state (n, ℓ, m, s) to S-entropy coordinates.
 *
 *   Sk = (n − 1) / (nP − 1)
 *   St = ℓ / (n − 1)   if n > 1, else 0
 *   Se = (m + ℓ) / (2ℓ) if ℓ > 0, else 0.5
 *
 * @param {number} n
 * @param {number} l  ℓ
 * @param {number} m
 * @param {number} s
 * @param {number} nP  Planck depth reference
 */
export function stateToSentropy(n, l, m, s, nP = PLANCK_DEPTH_DEFAULT) {
  const sk = (n - 1) / Math.max(1, nP - 1);
  const st = n > 1 ? l / (n - 1) : 0;
  const se = l > 0 ? (m + l) / (2 * l) : 0.5;
  return {
    sk: Math.max(0, Math.min(1, sk)),
    st: Math.max(0, Math.min(1, st)),
    se: Math.max(0, Math.min(1, se)),
  };
}

/** Euclidean distance in S-entropy space (Theorem 3.1). */
export function sentropyDist(a, b) {
  return Math.sqrt(
    (a.sk - b.sk) ** 2 + (a.st - b.st) ** 2 + (a.se - b.se) ** 2
  );
}

/** Theoretical bound: d ≤ √3 · |Δn| / (nP − 1)  (Proposition 3.2). */
export function fragDistanceBound(deltaN, nP = PLANCK_DEPTH_DEFAULT) {
  return Math.sqrt(3) * Math.abs(deltaN) / (nP - 1);
}

/* ── Forward frontier ────────────────────────────────────────────────────── */

/**
 * Build the forward reachability frontier from a precursor m/z.
 * Enumerates all partition states (n, ℓ, m, s) reachable within maxDepth
 * shell drops, applying the mass-reduction admissibility constraint.
 *
 * Returns a Map from state key → { state, se, cost, depth }.
 */
export function buildForwardFrontier(precursorMz, maxDepth = 7, nP = PLANCK_DEPTH_DEFAULT) {
  const precState = mzToState(precursorMz);
  const precSe    = stateToSentropy(precState.n, precState.l, precState.m, precState.s, nP);
  const frontier  = new Map();

  for (let depth = 1; depth <= maxDepth; depth++) {
    const nFrag = precState.n - depth;
    if (nFrag < 1) break;
    // Enumerate all (ℓ, m) pairs at shell nFrag
    for (let l = 0; l <= nFrag - 1; l++) {
      for (let mv = -l; mv <= l; mv++) {
        for (const sv of [-0.5, 0.5]) {
          const state = { n: nFrag, l, m: mv, s: sv };
          const se    = stateToSentropy(nFrag, l, mv, sv, nP);
          const cost  = sentropyDist(precSe, se);
          const key   = `${nFrag},${l},${mv},${sv}`;
          if (!frontier.has(key) || frontier.get(key).cost > cost) {
            frontier.set(key, { state, se, cost, depth, deltaN: depth });
          }
        }
      }
    }
  }
  return { frontier, precState, precSe };
}

/* ── Virtual predecessor (backward search) ───────────────────────────────── */

/**
 * Compute the virtual predecessor for N=2 decomposition:
 *   Sv* = 2·Sv_f − Sv_2    (Eq. 5)
 * Off-shell when Sv* ∉ [0,1]³.
 */
export function virtualPredecessor(fragSe, onShellSe) {
  return {
    sk: 2 * fragSe.sk - onShellSe.sk,
    st: 2 * fragSe.st - onShellSe.st,
    se: 2 * fragSe.se - onShellSe.se,
  };
}

/** Returns true if the S-entropy point is off-shell (outside [0,1]³). */
export function isOffShell(se) {
  return se.sk < 0 || se.sk > 1 || se.st < 0 || se.st > 1 || se.se < 0 || se.se > 1;
}

/* ── SEBD-MS main algorithm ──────────────────────────────────────────────── */

/**
 * Run SEBD-MS on one spectrum.
 *
 * @param {number}   precursorMz
 * @param {number[]} fragmentMzList   observed fragment m/z values
 * @param {{maxDepth?, planckDepth?}} opts
 * @returns {SebdResult}
 */
export function sebdMs(precursorMz, fragmentMzList, opts = {}) {
  const { maxDepth = 7, planckDepth: nP = PLANCK_DEPTH_DEFAULT } = opts;

  const { frontier, precState, precSe } = buildForwardFrontier(precursorMz, maxDepth, nP);

  const fragmentResults = fragmentMzList.map(fragMz => {
    const fragState = mzToState(fragMz);
    const fragSe    = stateToSentropy(fragState.n, fragState.l, fragState.m, fragState.s, nP);
    const forwardCost = sentropyDist(precSe, fragSe);
    const deltaN = precState.n - fragState.n;
    const isReachable = fragState.n < precState.n;

    // Find meeting point: node in forward frontier minimising total path cost
    let bestCost = Infinity, meetingNode = null, transitionState = null;
    for (const node of frontier.values()) {
      const backCost = sentropyDist(node.se, fragSe);
      const total    = node.cost + backCost;
      if (total < bestCost) {
        bestCost    = total;
        meetingNode = node;
        // Virtual predecessor check: Sv* = 2·Sv_f − Sv_2
        const vp = virtualPredecessor(fragSe, node.se);
        transitionState = isOffShell(vp) ? vp : null;
      }
    }

    // Subharmonic frequency ratio (Phase Coherence Theorem)
    const subharmonicRatio = Math.sqrt(Math.max(0, precursorMz / Math.max(1e-6, fragMz)));

    return {
      fragmentMz:     fragMz,
      fragmentState:  fragState,
      fragmentSe:     fragSe,
      deltaN,
      forwardCost,
      bestCost:       Math.min(forwardCost, bestCost),
      meetingNode,
      transitionState,
      isReachable,
      subharmonicRatio,
      // Phase coherence: ω_f = ω_prec · subharmonicRatio
      phaseDiff: 2 * Math.PI * Math.abs(nState(precState.n) - nState(fragState.n)),
    };
  });

  const reachable = fragmentResults.filter(r => r.isReachable);

  return {
    precursorMz,
    precState,
    precSe,
    fragmentResults,
    // Summary statistics
    reachableFraction:  reachable.length / Math.max(1, fragmentResults.length),
    meanDeltaN:         reachable.reduce((s, r) => s + r.deltaN, 0) / Math.max(1, reachable.length),
    meanPathLength:     fragmentResults.reduce((s, r) => s + r.forwardCost, 0) / Math.max(1, fragmentResults.length),
    offShellFraction:   fragmentResults.filter(r => r.transitionState).length / Math.max(1, fragmentResults.length),
    transitionStates:   fragmentResults.filter(r => r.transitionState).map(r => r.transitionState),
  };
}

/* ── Convert SEBD-MS results → PredictedRecord[] ────────────────────────── */

/**
 * Convert SEBD-MS results into PredictedRecord[] format so all existing
 * ResultsDashboard chart rows work without modification.
 *
 * Each fragment becomes one record:
 *   – analyteClass = "SEBD"
 *   – n, l, m, s from partition state mapping
 *   – sentropyVec from partition-state S-entropy embedding
 *   – intensity ∝ exp(−pathCost)   (lower cost → higher confidence)
 *   – ms2 = pathway nodes (as fragment peaks)
 */
export function sebdMsToRecords(result) {
  const { precursorMz, precState, precSe, fragmentResults } = result;

  return fragmentResults.map((frag) => {
    const { n, l, m, s } = frag.fragmentState;
    const intensity = Math.max(0.01, Math.exp(-frag.bestCost * 2));

    // Build a minimal MS2 "pathway" from meeting node
    const ms2 = frag.meetingNode ? [{
      mz:       precursorMz * (frag.fragmentMz / precursorMz),
      intensity: intensity * 0.8,
      label:    `Δn=${frag.deltaN}`,
      type:     "pathway",
    }] : [];

    return {
      // Identity
      analyte:      `frag_${frag.fragmentMz.toFixed(3)}`,
      analyteClass: "SEBD",
      X:            n,        // principal shell (≡ acyl carbons for lipidomics)
      Y:            frag.deltaN,  // shell drop (≡ double bonds)
      composition:  { C: Math.round(frag.fragmentMz / 14), H: 0, N: 0, O: 0 },
      neutralMass:  frag.fragmentMz,
      adduct:       "[M+H]+",
      adductAbbr:   "H+",
      precursorMz:  frag.fragmentMz,
      z:            1,
      polarity:     "+",
      intensity,

      // Partition coordinates (paper Definition 2.1)
      n, l, m, s,

      // S-entropy (partition-state embedding, paper Definition 3.1)
      sentropy:    frag.fragmentSe,
      sentropyVec: { sk: frag.fragmentSe.sk, st: frag.fragmentSe.st, se: frag.fragmentSe.se },

      // Graph metadata
      ternaryAddress:   "000000000000000000",
      analyserMode:     "orbitrap",
      observable:       frag.fragmentMz,
      shellDistribution: { [n]: 1 },
      partitionEntropy:  frag.forwardCost,

      // Spectra
      ms1: [{ mz: frag.fragmentMz, intensity, label: "M", type: "isotope" }],
      ms2,
      peaksAll: [{ mz: frag.fragmentMz, intensity, label: "M", type: "isotope" }, ...ms2],

      // Information content
      bitsTotal: Math.log2(Math.max(1, frag.fragmentMz)) + Math.log2(Math.max(1, intensity)),

      // SEBD-specific extras (used by GpuObservationPanel etc.)
      isReachable:      frag.isReachable,
      deltaN:           frag.deltaN,
      pathCost:         frag.bestCost,
      subharmonicRatio: frag.subharmonicRatio,
      phaseDiff:        frag.phaseDiff,
      hasTransitionState: !!frag.transitionState,
    };
  }).filter(r => r.isReachable || r.deltaN === 0);
}

/* ── Precursor reconstruction (Theorem 7.2) ──────────────────────────────── */

/**
 * Reconstruct precursor S-entropy from ≥3 fragment observations.
 * Each fragment traces a line Sv* = 2·Sv_f − Sv_2 in R³.
 * The precursor is the centroid of the meeting points.
 *
 * @param {Array<{se:{sk,st,se}}>} fragmentResults
 * @returns {{sk, st, se}}  estimated precursor S-entropy
 */
export function reconstructPrecursor(fragmentResults) {
  if (fragmentResults.length < 3) return null;
  const avg = (key) =>
    fragmentResults.reduce((s, r) => s + (r.meetingNode?.se?.[key] ?? r.fragmentSe[key]), 0)
    / fragmentResults.length;
  return { sk: avg("sk"), st: avg("st"), se: avg("se") };
}

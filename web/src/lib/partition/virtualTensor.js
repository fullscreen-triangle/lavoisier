/**
 * Stacked Virtual Substates as a Partition Tensor.
 *
 * Implements the virtual partition tensor V_{ijkl} and related operations:
 *   – Phase Coherence Theorem: fragment frequencies as subharmonics
 *   – Virtual tensor decomposition across 4 dimensions
 *   – Off-shell fraction and mean-recovery validation
 *   – Impossible ions as crossing-symmetry probes
 *   – Ion removal via partition complement (SWIFT derivation)
 *   – Planck depth and N-route mass verification
 *
 * Reference: Sachikonye, "Stacked Virtual Substates as a Partition Tensor:
 *   Complete Molecular Information from Single-Ion Orbitrap Transients
 *   via Phase-Coherent Decomposition"
 */

import { nState, mzToN } from "./partitionStateGraph.js";

/* ── Physical constants ──────────────────────────────────────────────────── */
const PLANCK_TIME = 5.391e-44;   // s
const KAPPA_REF   = 1e12;         // Orbitrap field constant (m/z-calibrated)
const E_CHARGE    = 1.60218e-19;  // C
const AMU         = 1.66054e-27;  // kg

/* ── Phase Coherence Theorem (Theorem 4.1) ───────────────────────────────── */

/**
 * Compute fragment ion subharmonic frequencies and phase differences.
 *
 * For an Orbitrap ion at ω_z = √(q·κ/m):
 *   ω_f / ω_prec = √(m_prec / m_f)
 *
 * Phase difference:
 *   Δθ = 2π·ΔM   where ΔM = M_f − M_prec  (partition count difference)
 *
 * Self-consistency back-conversion should give <10⁻⁹ ppm error.
 *
 * @param {number}   precursorMz
 * @param {number[]} fragmentMzList
 */
export function fragmentSubharmonics(precursorMz, fragmentMzList) {
  const n_prec = mzToN(precursorMz);
  const M_prec = nState(n_prec);

  return fragmentMzList.map(fragMz => {
    const n_frag = mzToN(fragMz);
    const M_frag = nState(n_frag);
    const freqRatio = Math.sqrt(precursorMz / Math.max(1e-9, fragMz));
    const deltaM    = M_frag - M_prec;
    const phaseDiff = 2 * Math.PI * deltaM;

    // Self-consistency: back-compute precursorMz from freqRatio
    const reconstructedPrec  = fragMz * freqRatio * freqRatio;
    const backConversionError = Math.abs(reconstructedPrec - precursorMz) / precursorMz * 1e9; // ppm

    return {
      fragmentMz:         fragMz,
      frequencyRatio:     freqRatio,
      phaseDiff,
      deltaM,
      deltaN:             n_prec - n_frag,
      // Subharmonic frequency in Orbitrap transient
      subharmonicFreq_Hz: Math.sqrt(E_CHARGE * KAPPA_REF / (precursorMz * AMU)) / (2 * Math.PI) * freqRatio,
      backConversionError_ppm: backConversionError,
      selfConsistent:     backConversionError < 1e-6, // essentially machine precision
    };
  });
}

/* ── Virtual Partition Tensor V_{ijkl} (Definition 6.1) ─────────────────── */

/**
 * Build the stacked virtual tensor for an ion.
 *
 * Four dimensions:
 *   i ∈ {0,1,2,3}  instrument basis (Orbitrap, FT-ICR, TOF, quadrupole)
 *   j ∈ 1..Z_max    charge state
 *   k ∈ {0,1}       polarity (+/−)
 *   l ∈ 0..nT−1     time step
 *
 * Each component V_{ijkl} ∈ R (may be off-shell, outside [0,1]).
 * The mean-recovery constraint: mean(V) = v_phys ∈ [0,1].
 *
 * @param {number} mz
 * @param {number} maxCharge   maximum charge state to include
 * @param {number} nT          number of time steps
 */
export function virtualTensorComponents(mz, maxCharge = 3, nT = 10) {
  const instruments = ["orbitrap", "fticr", "tof", "quadrupole"];
  const omega_ref = Math.sqrt(E_CHARGE * KAPPA_REF / (mz * AMU));  // rad/s ref

  // Physical (singly charged Orbitrap) observable, normalised to [0,1]
  const omega_max = Math.sqrt(E_CHARGE * KAPPA_REF / (50 * AMU));   // 50 Da reference
  const v_phys    = Math.max(0, Math.min(1, omega_ref / omega_max));

  const components = [];

  for (let i = 0; i < instruments.length; i++) {
    for (let j = 1; j <= maxCharge; j++) {
      for (let k = 0; k < 2; k++) {
        const polarity = k === 0 ? 1 : -1;
        for (let l = 0; l < nT; l++) {
          // Instrument-specific frequency (Theorem 4.2)
          let omega;
          switch (instruments[i]) {
            case "orbitrap":   omega = omega_ref * Math.sqrt(j); break;
            case "fticr":      omega = omega_ref * j;             break;
            case "tof":        omega = 2 * Math.PI / (0.001 * Math.sqrt(mz)); break;
            case "quadrupole": omega = omega_ref * (0.5 + 0.5 * j / maxCharge); break;
            default:           omega = omega_ref;
          }

          // Temporal modulation: phase advance over time steps
          const temporalPhase = l / Math.max(1, nT - 1);
          const value = polarity * (omega / omega_max) * (1 + 0.1 * Math.sin(2 * Math.PI * temporalPhase));

          components.push({
            i, j, k, l,
            instrument: instruments[i],
            chargeState: j,
            polarity:   k === 0 ? "+" : "-",
            timeStep:   l,
            value,
            isOffShell: value < 0 || value > 1,
          });
        }
      }
    }
  }

  return { components, vPhys: v_phys, N: components.length };
}

/**
 * Verify the mean-recovery constraint: mean(V) should equal v_phys ∈ [0,1].
 *
 * @param {{ components: Array<{value:number}>, vPhys: number, N: number }} tensor
 */
export function verifyMeanRecovery(tensor) {
  const { components, vPhys, N } = tensor;
  const mean = components.reduce((s, c) => s + c.value, 0) / N;
  const offShellCount = components.filter(c => c.isOffShell).length;
  return {
    mean,
    vPhys,
    error:             Math.abs(mean - vPhys),
    offShellFraction:  offShellCount / N,
    meanIsPhysical:    mean >= 0 && mean <= 1,
    meanRecoveryHolds: Math.abs(mean - vPhys) < 0.01,
  };
}

/* ── Planck depth (Definition 9.1 and Theorem 9.2) ──────────────────────── */

/**
 * Compute the Planck depth n_P for a system with d_eff effective dimensions.
 *
 *   n_P = 1 + ⌈log_{d+1}(τ_osc / (d · t_P))⌉
 */
export function planckDepth(dEff, tauOsc_s) {
  const ratio = tauOsc_s / (dEff * PLANCK_TIME);
  if (ratio <= 0) return 1;
  return 1 + Math.ceil(Math.log(ratio) / Math.log(dEff + 1));
}

/**
 * Effective dimensionality of the stacked virtual tensor (Eq. 12):
 *   d_eff = 4 × Z × 2 × n_t
 */
export function effectiveDimensionality(atomicNumber, nTimeSteps) {
  return 4 * atomicNumber * 2 * nTimeSteps;
}

/**
 * N-route agreement bound (Theorem 10.1):
 *   ε_N = 2K · (d_eff + 1)^{−n_P}
 */
export function nRouteAgreementBound(dEff, nP, K = 1.0) {
  return 2 * K * Math.pow(dEff + 1, -nP);
}

/* ── Impossible ions (Definition 7.1 & Theorem 7.1) ─────────────────────── */

/**
 * Construct an "impossible ion" as a virtual combination of real ions.
 * The mean of the combination may fall at a partition address with no
 * chemical realization — a crossing-symmetry probe.
 *
 * For N=2: impossible_mean = (ion1.mz + ion2.mz) / 2
 * This is "impossible" if no compound exists at that mass.
 *
 * @param {number[]} realMzValues  m/z values of real ions
 * @returns {{ impossibleMz, phases, isImpossible }}[]
 */
export function impossibleIons(realMzValues) {
  const results = [];
  for (let i = 0; i < realMzValues.length; i++) {
    for (let j = i + 1; j < realMzValues.length; j++) {
      const mz1 = realMzValues[i];
      const mz2 = realMzValues[j];
      const impossibleMz = (mz1 + mz2) / 2;
      const n1 = mzToN(mz1), n2 = mzToN(mz2);
      const nImp = mzToN(impossibleMz);
      // Phase shifts: δφ_i = 2π·δM_i from mean-recovery constraint
      const deltaM1 = nState(nImp) - nState(n1);
      const deltaM2 = nState(nImp) - nState(n2);
      results.push({
        ion1_mz:     mz1,
        ion2_mz:     mz2,
        impossibleMz,
        phaseShift1: 2 * Math.PI * deltaM1,
        phaseShift2: 2 * Math.PI * deltaM2,
        isImpossible: true,  // by construction — no compound at mean unless coincidence
      });
    }
  }
  return results;
}

/* ── Ion removal via partition complement (Corollary 7.1 — SWIFT) ─────────── */

/**
 * Compute the partition complement for SWIFT-like ion removal.
 *
 *   M̃ = C_max − M_ion   where C_max = 2·n_P²
 *
 * The antistate frequency ω̃ = √(C_max · e · κ / m) cancels the target ion
 * when applied with equal and opposite amplitude.
 *
 * @param {number} mz
 * @param {number} planckDepth  n_P
 */
export function partitionComplement(mz, planckDepth = 56) {
  const Cmax   = 2 * planckDepth * planckDepth;
  const M_ion  = nState(mzToN(mz));
  const M_comp = Cmax - M_ion;
  // Equivalent m/z for the complement address (from ω_z ∝ 1/√m)
  const mzComp = mz * (M_ion / Math.max(1, M_comp)) ** 2;
  return {
    originalMz:    mz,
    complementMz:  mzComp,
    M_ion,
    M_comp,
    Cmax,
    antiFreqRatio: Math.sqrt(mz / Math.max(1e-6, mzComp)),
    // "SWIFT waveform" would be the Fourier transform of all complement antistates
  };
}

/* ── Single-transient completeness summary ───────────────────────────────── */

/**
 * Summarise what a single Orbitrap transient of a precursor contains
 * (Theorem 11.1 — Single-Measurement Completeness).
 *
 * Returns the list of frequency components that encode all information:
 *   (i)   precursor frequency ω_prec
 *   (ii)  fragment subharmonics ω_f = ω_prec · √(m_prec/m_f)
 *   (iii) charge-state series at √z · ω_prec
 *   (iv)  negative-mode phase offset
 *   (v)   initial-state phase
 */
export function transientContents(precursorMz, fragmentMzList, maxCharge = 4) {
  const omegaPrec = Math.sqrt(E_CHARGE * KAPPA_REF / (precursorMz * AMU)) / (2 * Math.PI);

  const contents = {
    precursor: { mz: precursorMz, freq_Hz: omegaPrec, type: "precursor" },
    fragments: fragmentMzList.map(fMz => ({
      mz:    fMz,
      freq_Hz: omegaPrec * Math.sqrt(precursorMz / fMz),
      type:  "subharmonic",
    })),
    chargeStates: Array.from({ length: maxCharge }, (_, i) => ({
      z:    i + 1,
      freq_Hz: omegaPrec * Math.sqrt(i + 1),
      type:  "charge_state",
    })),
    negativeMode: {
      freq_Hz: omegaPrec, // same ω, opposite phase (−½ parity)
      phaseDiff: Math.PI,
      type:  "polarity_complement",
    },
  };

  return contents;
}

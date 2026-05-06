/**
 * Virtual mass-spec instrument.
 *
 * Forward simulation pipeline:
 *
 *   analyte spec -> precursor m/z (per adduct)
 *               -> partition coordinates (n, l, m, s)
 *               -> S-entropy coordinates (S_k, S_t, S_e)
 *               -> analyser observable (T_TOF / omega_orbi / omega_c / a,q)
 *               -> MS2 fragments (per adduct, per CE)
 *               -> CategoricalState (compatible with the existing tool's viewers)
 *
 * No physical sample is required; this runs entirely on the user's specified
 * analyte set.
 */

import { LIPID_CLASSES, enumerateAnalytes } from "./lipidomics";
import {
  ADDUCTS,
  applyAdduct,
  preferredAdducts,
  adductRelativeIntensity,
} from "./adducts";
import { fragmentsFor, normaliseFragments } from "./fragmentation";
import { observe, partitionField } from "../partition/lagrangian";
import { computeFromSpectrum } from "../partition/sentropy";
import {
  decomposeSpectrum,
  shellDistribution,
  partitionEntropy,
} from "../partition/hierarchy";
import { hierarchicalEncode } from "../partition/ternary";

/**
 * Map a lipid analyte to a partition principal coordinate n.
 * Larger species occupy larger principal shells (more confinement modes).
 *
 * Heuristic: n = ceil(sqrt(M / mz_per_shell)) where mz_per_shell ~ 162
 * (approximately the hexose unit, but applied generically).
 */
function principalCoordinate(mass) {
  return Math.max(1, Math.ceil(Math.sqrt(mass / 162.0)));
}

/**
 * Map an analyte's structural complexity to angular complexity ℓ.
 * Based on chain length and degree of unsaturation.
 */
function angularCoordinate(analyte, n) {
  const cls = LIPID_CLASSES[analyte.class];
  const complexity = analyte.Y + cls.faChains - 1; // unsaturation + chain count
  return Math.min(n - 1, complexity);
}

/**
 * Magnetic coordinate m: oriented in [-l, l].
 * For deterministic mapping, use class hash + Y parity.
 */
function magneticCoordinate(analyte, l) {
  if (l === 0) return 0;
  const cls = analyte.class;
  // Stable hash on class and chain composition
  const h = (cls.charCodeAt(0) * 31 + analyte.X * 7 + analyte.Y * 13) % (2 * l + 1);
  return h - l;
}

/**
 * Spin coordinate from polarity & charge state.
 */
function spinCoordinate(adductKey) {
  return ADDUCTS[adductKey].polarity === "+" ? +0.5 : -0.5;
}

/**
 * Generate isotopologue intensities (Mp+0, +1, +2) for a composition.
 * Uses the natural-abundance approximation for C and H.
 */
function isotopePattern(composition) {
  const nC = composition.C || 0;
  const nH = composition.H || 0;
  const nN = composition.N || 0;
  const nS = composition.S || 0;
  // Approximate +1/M0 ratio = 0.0107 * nC + 0.000115 * nH + 0.00366 * nN
  const r1 = 0.0107 * nC + 0.000115 * nH + 0.00366 * nN;
  // Approximate +2/M0 ratio (mostly 13C2 + 18O + 34S)
  const r2 = (r1 * r1) / 2 + (composition.O || 0) * 0.00205 + nS * 0.0440;
  return [
    { delta_mass: 0,        relative_intensity: 1.0 },
    { delta_mass: 1.003355, relative_intensity: r1 },
    { delta_mass: 2.006710, relative_intensity: r2 },
  ];
}

/**
 * Predict relative intensity given:
 *   - class abundance (from typical biological context)
 *   - chain length (peaks around C16-18)
 *   - unsaturation (tapering toward DB > 4)
 *
 * This is purely heuristic but produces realistic-looking spectra.
 */
function predictIntensity(analyte) {
  const cls = LIPID_CLASSES[analyte.class];
  // Class baseline (rough plasma lipidomics order of magnitude)
  const baseline = {
    PC: 1.0, PE: 0.6, PS: 0.2, PI: 0.15, PG: 0.05,
    SM: 0.7, Cer: 0.3, TAG: 1.2, DAG: 0.4, LPC: 0.3,
    CE: 0.9, FA: 0.5, PA: 0.05,
  };
  const cb = baseline[analyte.class] ?? 0.3;
  const chainOptimum = 16 + (cls.faChains - 1) * 2; // optimum chain centre per chain
  const meanChain = analyte.X / cls.faChains;
  const chainFactor = Math.exp(-Math.pow((meanChain - chainOptimum) / 4, 2));
  const dbFactor = Math.exp(-analyte.Y / 4);
  return cb * chainFactor * dbFactor;
}

/**
 * Run the virtual experiment: produce per-precursor records that include
 * MS1 (precursor + isotope envelope) and MS2 (fragments).
 *
 * @param {Object} cfg
 * @param {Array<{classKey, Xmin?, Xmax?, Ymin?, Ymax?}>} cfg.classSpecs
 * @param {string[]} cfg.adductsAllowed   subset of ADDUCT keys to use
 * @param {"+"|"-"} cfg.polarity
 * @param {string} cfg.analyser     "tof"|"quadrupole"|"orbitrap"|"fticr"
 * @param {Object} cfg.analyserCfg
 * @param {number} cfg.collisionEnergy_eV
 * @param {[number, number]} cfg.mzWindow
 * @returns {Array<PredictedRecord>}
 */
export function runExperiment(cfg) {
  const {
    classSpecs,
    adductsAllowed = null,    // null = use class preferred
    polarity = "+",
    analyser = "orbitrap",
    analyserCfg = {},
    collisionEnergy_eV = 25,
    mzWindow = [100, 2000],
  } = cfg;

  const analytes = enumerateAnalytes(classSpecs);
  const records = [];

  for (const analyte of analytes) {
    const cls = LIPID_CLASSES[analyte.class];
    const adductChoices = adductsAllowed
      ? adductsAllowed.filter((a) => ADDUCTS[a].polarity === polarity)
      : preferredAdducts(analyte.class, polarity);

    const Iprime = predictIntensity(analyte);
    if (Iprime < 1e-4) continue;

    for (const adductKey of adductChoices) {
      const adduct = ADDUCTS[adductKey];
      const z = Math.abs(adduct.z);
      const mz = applyAdduct(analyte.mass, adductKey);
      if (mz < mzWindow[0] || mz > mzWindow[1]) continue;

      const I = Iprime * adductRelativeIntensity(analyte.class, adductKey);

      // Partition coordinates
      const n = principalCoordinate(analyte.mass);
      const l = angularCoordinate(analyte, n);
      const m = magneticCoordinate(analyte, l);
      const s = spinCoordinate(adductKey);

      // Analyser observable
      const observable = observe(analyser, mz, analyserCfg);

      // Isotope pattern
      const isotopes = isotopePattern(analyte.composition);
      const ms1 = isotopes.map((iso) => ({
        mz: mz + iso.delta_mass / z,
        intensity: I * iso.relative_intensity,
        label: iso.delta_mass === 0 ? `M` : `M+${iso.delta_mass.toFixed(0)}`,
        type: "isotope",
      }));

      // MS2 fragments
      const rawFrags = fragmentsFor(
        analyte,
        adductKey,
        { precursor_mz: mz, polarity },
        collisionEnergy_eV
      );
      const ms2 = normaliseFragments(rawFrags).map((f) => ({
        ...f,
        intensity: f.intensity * I,
      }));

      // S-entropy coordinates from the predicted spectrum
      const allPeaks = [...ms1, ...ms2];
      const sentropy = computeFromSpectrum(allPeaks);

      // Hierarchical decomposition (n,l,m,s) per peak
      const hier = decomposeSpectrum(allPeaks);
      const shells = shellDistribution(hier);
      const entropyNats = partitionEntropy(hier);

      // Ternary address from the observable
      const address = hierarchicalEncode(allPeaks, 18);

      // Multimodal-bit estimate
      const bitsPrecursor = Math.log2(Math.max(1, mz)) + Math.log2(Math.max(1, I));
      const bitsCoord = Math.log2(Math.max(1, 2 * n * n));
      const bitsFragments = Math.log2(Math.max(1, ms2.length));
      const bitsTotal = bitsPrecursor + bitsCoord + bitsFragments + 5;

      records.push({
        // identity
        analyte: analyte.name,
        analyteClass: analyte.class,
        X: analyte.X,
        Y: analyte.Y,
        composition: analyte.composition,
        neutralMass: analyte.mass,
        adduct: adductKey,
        adductAbbr: adduct.abbr,
        precursorMz: mz,
        z,
        polarity,
        intensity: I,

        // partition coordinates
        n, l, m, s,

        // S-entropy
        sentropy,
        ternaryAddress: address,

        // analyser
        analyserMode: analyser,
        observable,

        // hierarchy summary
        shellDistribution: shells,
        partitionEntropy: entropyNats,

        // spectra
        ms1,
        ms2,
        peaksAll: allPeaks,

        // information content
        bitsTotal,

        // CategoricalState-compatible projection
        sentropyVec: { sk: sentropy.sk, st: sentropy.st, se: sentropy.se },
      });
    }
  }

  return records;
}

/**
 * Aggregate-level statistics across the predicted record set.
 */
export function summariseRecords(records) {
  if (records.length === 0) {
    return {
      count: 0, perClass: {}, perAdduct: {},
      mzRange: [0, 0], intensityRange: [0, 0],
      shellsHistogram: [], avgEntropy: 0,
    };
  }
  const perClass = {};
  const perAdduct = {};
  let mzMin = Infinity, mzMax = -Infinity;
  let iMin = Infinity, iMax = -Infinity;
  const shellsHist = new Map();
  let sumEntropy = 0;

  for (const r of records) {
    perClass[r.analyteClass] = (perClass[r.analyteClass] || 0) + 1;
    perAdduct[r.adduct] = (perAdduct[r.adduct] || 0) + 1;
    mzMin = Math.min(mzMin, r.precursorMz);
    mzMax = Math.max(mzMax, r.precursorMz);
    iMin = Math.min(iMin, r.intensity);
    iMax = Math.max(iMax, r.intensity);
    shellsHist.set(r.n, (shellsHist.get(r.n) || 0) + 1);
    sumEntropy += r.partitionEntropy;
  }
  const sortedShells = [...shellsHist.entries()].sort((a, b) => a[0] - b[0]);

  return {
    count: records.length,
    perClass,
    perAdduct,
    mzRange: [mzMin, mzMax],
    intensityRange: [iMin, iMax],
    shellsHistogram: sortedShells.map(([n, c]) => ({ n, count: c })),
    avgEntropy: sumEntropy / records.length,
  };
}

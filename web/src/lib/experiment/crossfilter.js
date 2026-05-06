/**
 * Crossfilter setup for the predicted record set.
 *
 * Pattern lifted from the dc.js reference (sources/stocks.js): one
 * crossfilter instance, many named dimensions, groups bound to those
 * dimensions. Every chart binds to a dim+group and redraws when filters
 * change.
 */

import crossfilter from "crossfilter2";

/**
 * Build a fresh crossfilter pack from an array of predicted records.
 * Returns dims + groups keyed by chart purpose.
 */
export function buildCrossfilterPack(records) {
  const cf = crossfilter(records);
  const all = cf.groupAll();

  // ---- numeric / continuous ----
  const dimMz = cf.dimension((d) => d.precursorMz);
  const dimIntensity = cf.dimension((d) => d.intensity);
  const dimNeutralMass = cf.dimension((d) => d.neutralMass);
  const dimX = cf.dimension((d) => d.X);
  const dimY = cf.dimension((d) => d.Y);
  const dimSk = cf.dimension((d) => d.sentropy.sk);
  const dimSt = cf.dimension((d) => d.sentropy.st);
  const dimSe = cf.dimension((d) => d.sentropy.se);
  const dimEntropy = cf.dimension((d) => d.partitionEntropy);
  const dimBits = cf.dimension((d) => d.bitsTotal);
  const dimNumFragments = cf.dimension((d) => d.ms2.length);

  // ---- partition coordinates (discrete) ----
  const dimN = cf.dimension((d) => d.n);
  const dimL = cf.dimension((d) => d.l);
  const dimM = cf.dimension((d) => d.m);
  const dimS = cf.dimension((d) => d.s);

  // ---- categorical ----
  const dimClass = cf.dimension((d) => d.analyteClass);
  const dimAdduct = cf.dimension((d) => d.adduct);
  const dimPolarity = cf.dimension((d) => d.polarity);
  const dimZ = cf.dimension((d) => d.z);

  // ---- oscillatory ----
  const dimObservable = cf.dimension((d) => observableValue(d.observable));

  // ---- groups ----
  const mzBinSize = mzBinFor(records);
  const grpMzBin = dimMz.group((mz) => Math.floor(mz / mzBinSize) * mzBinSize)
    .reduceSum((d) => d.intensity);

  const grpClass = dimClass.group();
  const grpClassIntensity = dimClass.group().reduceSum((d) => d.intensity);
  const grpAdduct = dimAdduct.group();
  const grpPolarity = dimPolarity.group();
  const grpZ = dimZ.group();
  const grpN = dimN.group();
  const grpL = dimL.group();
  const grpM = dimM.group();
  const grpS = dimS.group();

  const grpSkBin = dimSk.group((v) => Math.floor(v * 20) / 20);
  const grpStBin = dimSt.group((v) => Math.floor(v * 20) / 20);
  const grpSeBin = dimSe.group((v) => Math.floor(v * 20) / 20);
  const grpEntropyBin = dimEntropy.group((v) => Math.floor(v * 10) / 10);
  const grpBitsBin = dimBits.group((v) => Math.floor(v));
  const grpFragsBin = dimNumFragments.group();

  const grpXY = cf.dimension((d) => `${d.X}|${d.Y}`).group();

  // Class bubble: aggregate (mean m/z, mean n, total intensity) per class
  const grpClassBubble = dimClass.group().reduce(
    (p, v) => {
      p.count++;
      p.sumMz += v.precursorMz;
      p.sumN  += v.n;
      p.sumI  += v.intensity;
      p.maxI = Math.max(p.maxI, v.intensity);
      return p;
    },
    (p, v) => {
      p.count--;
      p.sumMz -= v.precursorMz;
      p.sumN  -= v.n;
      p.sumI  -= v.intensity;
      return p;
    },
    () => ({ count: 0, sumMz: 0, sumN: 0, sumI: 0, maxI: 0 })
  );

  return {
    cf,
    all,

    dims: {
      mz: dimMz,
      intensity: dimIntensity,
      neutralMass: dimNeutralMass,
      X: dimX, Y: dimY,
      sk: dimSk, st: dimSt, se: dimSe,
      entropy: dimEntropy,
      bits: dimBits,
      numFragments: dimNumFragments,
      n: dimN, l: dimL, m: dimM, s: dimS,
      class: dimClass,
      adduct: dimAdduct,
      polarity: dimPolarity,
      z: dimZ,
      observable: dimObservable,
    },

    groups: {
      mzBin: grpMzBin,
      class: grpClass,
      classIntensity: grpClassIntensity,
      classBubble: grpClassBubble,
      adduct: grpAdduct,
      polarity: grpPolarity,
      z: grpZ,
      n: grpN, l: grpL, m: grpM, s: grpS,
      skBin: grpSkBin, stBin: grpStBin, seBin: grpSeBin,
      entropyBin: grpEntropyBin,
      bitsBin: grpBitsBin,
      fragsBin: grpFragsBin,
      xy: grpXY,
    },

    mzBinSize,
  };
}

function observableValue(obs) {
  if (!obs) return 0;
  if (obs.frequencyHz !== undefined) return Math.log10(Math.max(1, obs.frequencyHz));
  if (obs.T !== undefined) return Math.log10(Math.max(1e-9, obs.T));
  if (obs.q !== undefined) return obs.q;
  return 0;
}

function mzBinFor(records) {
  if (!records || records.length === 0) return 1;
  let lo = Infinity, hi = -Infinity;
  for (const r of records) {
    if (r.precursorMz < lo) lo = r.precursorMz;
    if (r.precursorMz > hi) hi = r.precursorMz;
  }
  const range = hi - lo;
  return Math.max(1, Math.round(range / 60));
}

/**
 * Drop everything attached to records. Call before discarding the pack.
 */
export function disposePack(pack) {
  if (!pack) return;
  try {
    Object.values(pack.dims || {}).forEach((d) => d.dispose && d.dispose());
  } catch (_e) {
    // ignore
  }
}

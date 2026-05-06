/**
 * GPU layer — public API.
 *
 * Provides the six-pass observation apparatus. The pipeline takes
 * CategoricalStates, renders their wave contributions with additive
 * blending (Pass 1), computes interference with generated candidates
 * (Pass 4), and extracts physical quality metrics (Pass 6).
 *
 * Passes 2, 3 and 5 are present in shaders but can be skipped when
 * not needed — the interface is pass-level granular.
 */

export { createContext, createProgram, loadShaderSource, preloadShaders } from "./context";
export { createPipeline } from "./pipeline";
export { statesToIons, coordsToIon } from "./ions";

import { createPipeline } from "./pipeline";
import { statesToIons, coordsToIon } from "./ions";
import { ternaryDecode } from "../partition/ternary";

/**
 * High-level observation session.
 *
 * Encapsulates a pipeline + the full six-pass observation workflow
 * callable from the UI and worker layers.
 *
 * @example
 *   const session = await createObservationSession(canvas, { width: 512, height: 512 });
 *   await session.observeQuery(categoricalStates);
 *   const resonance = await session.compareCandidate("202222112122");
 *   const quality = session.measureQuality();
 *   session.dispose();
 */
export async function createObservationSession(canvas, cfg) {
  const pipeline = await createPipeline(canvas, cfg);
  const { width, height } = cfg;

  let lastQueryMax = 1.0;
  let lastCandidateMax = 1.0;

  /**
   * Observe a set of CategoricalStates as the query.
   * This is Pass 1 for the query field.
   * @param {CategoricalState[]} states
   */
  function observeQuery(states) {
    const ions = statesToIons(states, { width, height });
    pipeline.renderIons(ions, "wave");

    // Cache maximum for normalisation
    const px = pipeline.readback("wave");
    lastQueryMax = maxAbs(px);
  }

  /**
   * Observe a single generated candidate address and measure interference
   * with the current query.
   * @param {string} address  ternary address
   * @returns {{resonance: number, interference: Float32Array}}
   */
  function compareCandidate(address) {
    const coords = ternaryDecode(address);
    const ion = coordsToIon(coords, { width, height });
    pipeline.renderIons([ion], "candidate");

    const px = pipeline.readback("candidate");
    lastCandidateMax = maxAbs(px);

    pipeline.renderInterference(lastQueryMax, lastCandidateMax);
    const resonance = pipeline.aggregateResonance();
    return { resonance, interference: pipeline.readback("interference") };
  }

  /**
   * Measure physical quality metrics on the current wave field.
   */
  function measureQuality() {
    pipeline.renderQuality(pipeline.textures.wave);
    return pipeline.aggregateQualityMetrics();
  }

  return {
    pipeline,
    observeQuery,
    compareCandidate,
    measureQuality,
    dispose: pipeline.dispose,
  };
}

function maxAbs(floats) {
  let m = 0;
  for (let i = 0; i < floats.length; i += 4) {
    const v = Math.abs(floats[i]);
    if (v > m) m = v;
  }
  return m > 0 ? m : 1;
}

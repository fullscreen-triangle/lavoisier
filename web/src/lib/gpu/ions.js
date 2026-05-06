/**
 * Translate CategoricalState (from the partition layer) into IonParams
 * suitable for the wave shader's uniforms.
 *
 * The S-entropy coordinates position the ion in phase space; the other
 * parameters (amplitude, wavelength, decay, angle, phase) modulate the
 * wave contribution. Exactly matches the Python ion_droplets_to_texture
 * reference implementation in gpu_wave_renderer.py.
 */

/**
 * Convert a list of CategoricalStates to IonParams for the pipeline.
 *
 * @param {CategoricalState[]} states
 * @param {Object} opts
 * @param {number} opts.width      canvas width in px
 * @param {number} opts.height     canvas height in px
 * @param {number} [opts.mzMin]    if null, computed from states
 * @param {number} [opts.mzMax]
 * @returns {IonParams[]}
 */
export function statesToIons(states, { width, height, mzMin = null, mzMax = null }) {
  if (states.length === 0) return [];

  // Derive mz range if not provided
  if (mzMin == null || mzMax == null) {
    mzMin = Infinity;
    mzMax = -Infinity;
    for (const s of states) {
      const mz = s.basePeakMz || 0;
      if (mz > 0 && mz < mzMin) mzMin = mz;
      if (mz > mzMax) mzMax = mz;
    }
    if (!Number.isFinite(mzMin) || !Number.isFinite(mzMax)) {
      mzMin = 0;
      mzMax = 1;
    }
  }

  const range = Math.max(mzMax - mzMin, 1);
  const ions = new Array(states.length);

  for (let i = 0; i < states.length; i++) {
    const s = states[i];
    const mz = s.basePeakMz || 0;

    // Position: m/z → x, retention time (or S_t) → y
    const cx = ((mz - mzMin) / range) * (width - 1);
    const cy = s.sentropy.st * (height - 1);

    // Amplitude from total ion current (log-scaled)
    const amplitude = Math.log1p(s.totalIonCurrent) / 5.0 + 0.5;

    // Wavelength from partition inertia proxy (mass scale)
    const wavelength = 2 + Math.sqrt(mz) / 10;

    // Decay rate: higher for lower-quality signals (wider scatter)
    const decayRate = 1.0;

    // Radius from peak count
    const radius = 1 + Math.log1p(s.nPeaks) / 3;

    // Directional bias from polarity + charge
    const angleRad =
      s.polarity === "negative" ? Math.PI : 0;

    // Phase from S_k — knowledge entropy drives oscillation phase
    const phaseOffset = s.sentropy.sk * Math.PI;

    ions[i] = {
      cx,
      cy,
      amplitude,
      wavelength,
      decayRate,
      radius,
      angleRad,
      phaseOffset,
    };
  }

  return ions;
}

/**
 * For a single candidate (generated from a ternary address) build a
 * "virtual scan" IonParams. Since we don't have a real spectrum for
 * generated candidates, we synthesise the canonical peak from the
 * address's coordinate centre.
 *
 * @param {{sk: number, st: number, se: number}} coords
 * @param {Object} opts
 * @returns {IonParams}
 */
export function coordsToIon(coords, { width, height }) {
  return {
    cx: coords.sk * (width - 1),
    cy: coords.st * (height - 1),
    amplitude: 1.0,
    wavelength: 3,
    decayRate: 1,
    radius: 1.5,
    angleRad: coords.se * Math.PI * 2,
    phaseOffset: coords.sk * Math.PI,
  };
}

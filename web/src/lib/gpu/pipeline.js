/**
 * Six-pass GPU observation pipeline.
 *
 * Pass 1 — Partition state observation (wave field accumulation)
 * Pass 2 — Coordinate assignment (S-entropy overlay)
 * Pass 3 — Bijective validation (visual vs numerical path)
 * Pass 4 — Resonance comparison (query vs candidate interference)
 * Pass 5 — Partition completion / detection (CPU side: threshold + readback)
 * Pass 6 — Quality metric extraction (partition sharpness, coherence, noise)
 *
 * All passes operate on the same small set of reusable textures (~25 MB).
 * The compiled probe / UI drives which passes fire for each observation.
 */

import {
  createContext,
  createProgram,
  createFullscreenQuad,
  createTexture,
  createFramebuffer,
  readPixelsFloat,
  preloadShaders,
} from "./context";

/**
 * @typedef {Object} IonParams
 * @property {number} cx          centre x in pixel coords
 * @property {number} cy          centre y in pixel coords
 * @property {number} amplitude
 * @property {number} wavelength
 * @property {number} decayRate
 * @property {number} radius
 * @property {number} angleRad    impact angle
 * @property {number} phaseOffset
 */

/**
 * @typedef {Object} PipelineConfig
 * @property {number} width
 * @property {number} height
 * @property {boolean} [preserveDrawing=true]
 */

/**
 * Build a complete six-pass pipeline on a canvas.
 * @param {HTMLCanvasElement|OffscreenCanvas} canvas
 * @param {PipelineConfig} cfg
 * @returns {Promise<Pipeline>}
 */
export async function createPipeline(canvas, cfg) {
  const gl = createContext(canvas);
  const { width, height } = cfg;
  canvas.width = width;
  canvas.height = height;
  gl.viewport(0, 0, width, height);

  // Load shader sources
  const sources = await preloadShaders({
    waveVert: "wave.vert",
    waveFrag: "wave.frag",
    physicsFrag: "physics_overlay.frag",
    bijectiveFrag: "bijective_validation.frag",
    interferenceFrag: "interference.frag",
    qualityFrag: "quality.frag",
  });

  // Build programs for each pass
  const programs = {
    wave: createProgram(gl, sources.waveVert, sources.waveFrag),
    physics: createProgram(gl, sources.waveVert, sources.physicsFrag),
    bijective: createProgram(gl, sources.waveVert, sources.bijectiveFrag),
    interference: createProgram(gl, sources.waveVert, sources.interferenceFrag),
    quality: createProgram(gl, sources.waveVert, sources.qualityFrag),
  };

  // One VAO per program (position attribute location may vary)
  const vaos = {
    wave: createFullscreenQuad(gl, programs.wave),
    physics: createFullscreenQuad(gl, programs.physics),
    bijective: createFullscreenQuad(gl, programs.bijective),
    interference: createFullscreenQuad(gl, programs.interference),
    quality: createFullscreenQuad(gl, programs.quality),
  };

  // Reusable textures + framebuffers
  const textures = {
    wave: createTexture(gl, width, height),        // Pass 1 output (query wave field)
    candidate: createTexture(gl, width, height),   // Pass 1 output (candidate)
    overlay: createTexture(gl, width, height),     // Pass 2 S-entropy colour
    bijective: createTexture(gl, width, height),   // Pass 3 validation
    interference: createTexture(gl, width, height),// Pass 4 resonance
    quality: createTexture(gl, width, height),     // Pass 6 metrics
  };

  const framebuffers = {
    wave: createFramebuffer(gl, textures.wave),
    candidate: createFramebuffer(gl, textures.candidate),
    overlay: createFramebuffer(gl, textures.overlay),
    bijective: createFramebuffer(gl, textures.bijective),
    interference: createFramebuffer(gl, textures.interference),
    quality: createFramebuffer(gl, textures.quality),
  };

  // Cache uniform locations
  const wave = programs.wave;
  const waveUniforms = {
    u_ion: gl.getUniformLocation(wave, "u_ion"),
    u_ion2: gl.getUniformLocation(wave, "u_ion2"),
    u_resolution: gl.getUniformLocation(wave, "u_resolution"),
  };

  const interference = programs.interference;
  const interferenceUniforms = {
    u_queryField: gl.getUniformLocation(interference, "u_queryField"),
    u_candidateField: gl.getUniformLocation(interference, "u_candidateField"),
    u_resolution: gl.getUniformLocation(interference, "u_resolution"),
    u_queryMax: gl.getUniformLocation(interference, "u_queryMax"),
    u_candidateMax: gl.getUniformLocation(interference, "u_candidateMax"),
    u_signalThreshold: gl.getUniformLocation(interference, "u_signalThreshold"),
  };

  const quality = programs.quality;
  const qualityUniforms = {
    u_waveField: gl.getUniformLocation(quality, "u_waveField"),
    u_resolution: gl.getUniformLocation(quality, "u_resolution"),
    u_signalThreshold: gl.getUniformLocation(quality, "u_signalThreshold"),
  };

  /**
   * Pass 1: render one ion's wave contribution to the wave framebuffer.
   * Call once per ion with additive blending.
   */
  function renderIon(ion, target = "wave") {
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers[target]);
    gl.viewport(0, 0, width, height);
    gl.useProgram(programs.wave);
    gl.bindVertexArray(vaos.wave.vao);

    gl.uniform4f(waveUniforms.u_ion, ion.cx, ion.cy, ion.amplitude, ion.wavelength);
    gl.uniform4f(
      waveUniforms.u_ion2,
      ion.decayRate,
      ion.angleRad,
      ion.phaseOffset,
      ion.radius
    );
    gl.uniform2f(waveUniforms.u_resolution, width, height);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Pass 1 batch: accumulate all ions with additive blending.
   * @param {IonParams[]} ions
   * @param {"wave"|"candidate"} target
   */
  function renderIons(ions, target = "wave") {
    // Clear target first
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers[target]);
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Additive blending accumulates ion contributions
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);

    for (const ion of ions) {
      renderIon(ion, target);
    }

    gl.disable(gl.BLEND);
  }

  /**
   * Pass 4: render interference between query and candidate fields.
   * @returns {Float32Array}  RGBA readback from interference framebuffer
   */
  function renderInterference(queryMax = 1.0, candidateMax = 1.0, signalThreshold = 0.02) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers.interference);
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(programs.interference);
    gl.bindVertexArray(vaos.interference.vao);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, textures.wave);
    gl.uniform1i(interferenceUniforms.u_queryField, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, textures.candidate);
    gl.uniform1i(interferenceUniforms.u_candidateField, 1);

    gl.uniform2f(interferenceUniforms.u_resolution, width, height);
    gl.uniform1f(interferenceUniforms.u_queryMax, queryMax);
    gl.uniform1f(interferenceUniforms.u_candidateMax, candidateMax);
    gl.uniform1f(interferenceUniforms.u_signalThreshold, signalThreshold);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Pass 6: compute quality metrics. Can be run against any wave-like field.
   * @returns {Float32Array}  RGBA = (sharpness, laplacian_noise, coherence, mask)
   */
  function renderQuality(sourceTexture = textures.wave, signalThreshold = 0.02) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers.quality);
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(programs.quality);
    gl.bindVertexArray(vaos.quality.vao);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
    gl.uniform1i(qualityUniforms.u_waveField, 0);

    gl.uniform2f(qualityUniforms.u_resolution, width, height);
    gl.uniform1f(qualityUniforms.u_signalThreshold, signalThreshold);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Read a framebuffer's pixels back to CPU.
   * @param {"wave"|"candidate"|"overlay"|"bijective"|"interference"|"quality"} target
   * @returns {Float32Array}
   */
  function readback(target) {
    return readPixelsFloat(gl, framebuffers[target], width, height);
  }

  /**
   * Aggregate quality metrics from Pass 6 output into scalar values.
   * @returns {{partitionSharpness: number, noiseLevel: number, phaseCoherence: number, signalFraction: number}}
   */
  function aggregateQualityMetrics() {
    const pixels = readback("quality");
    let sharpnessSum = 0;
    let laplacianSum = 0;
    let coherenceSum = 0;
    let maskSum = 0;
    const n = pixels.length / 4;
    for (let i = 0; i < pixels.length; i += 4) {
      const mask = pixels[i + 3];
      if (mask > 0.5) {
        sharpnessSum += pixels[i];
        laplacianSum += pixels[i + 1];
        coherenceSum += pixels[i + 2];
        maskSum++;
      }
    }
    if (maskSum === 0) {
      return { partitionSharpness: 0, noiseLevel: 0, phaseCoherence: 0, signalFraction: 0 };
    }
    return {
      partitionSharpness: sharpnessSum / maskSum,
      noiseLevel: laplacianSum / maskSum,
      phaseCoherence: coherenceSum / maskSum,
      signalFraction: maskSum / n,
    };
  }

  /**
   * Aggregate interference into scalar resonance score.
   */
  function aggregateResonance() {
    const pixels = readback("interference");
    let resSum = 0;
    let maskSum = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      if (pixels[i + 3] > 0.5) {
        resSum += pixels[i];
        maskSum++;
      }
    }
    return maskSum > 0 ? resSum / maskSum : 0;
  }

  /**
   * Dispose all GPU resources.
   */
  function dispose() {
    Object.values(programs).forEach((p) => gl.deleteProgram(p));
    Object.values(textures).forEach((t) => gl.deleteTexture(t));
    Object.values(framebuffers).forEach((fb) => gl.deleteFramebuffer(fb));
    Object.values(vaos).forEach(({ vao, buffer }) => {
      gl.deleteVertexArray(vao);
      gl.deleteBuffer(buffer);
    });
  }

  return {
    gl,
    canvas,
    width,
    height,
    textures,
    framebuffers,
    programs,
    // Pass 1
    renderIon,
    renderIons,
    // Pass 4
    renderInterference,
    aggregateResonance,
    // Pass 6
    renderQuality,
    aggregateQualityMetrics,
    // Utilities
    readback,
    dispose,
  };
}

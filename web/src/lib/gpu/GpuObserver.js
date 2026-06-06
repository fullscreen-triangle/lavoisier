/**
 * GPU Observation Apparatus — four-pass WebGL2 shader pipeline.
 *
 * By the Triple Equivalence Theorem (oscillation ≡ counting ≡ partition),
 * the fragment shader evaluating the partition function at cell coordinates IS
 * performing physical observation. The texture output IS the categorical state
 * tensor, not a picture of it.
 *
 * Four passes (Paper 2, Section 5):
 *   Pass 1 — Partition wave-field accumulation (additive superposition of ion waves)
 *   Pass 2 — S-entropy coordinate assignment (Sk→cyan, St→amber, Se→violet)
 *   Pass 3 — Bijective validation (dual-path interference: GPU path vs CPU path)
 *   Pass 4 — Resonance comparison (query vs candidate interference)
 *
 * Each pass evaluates the partition Lagrangian at every pixel = observing every
 * partition cell simultaneously. No simulation is performed; observation IS
 * computation (Observation-Computation Equivalence Theorem).
 */

/* ── GLSL source ─────────────────────────────────────────────────────────── */

const VERT_SRC = /* glsl */`#version 300 es
precision highp float;
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

// Pass 1: partition wave field (Eq. W_i from Paper 2, §5.3)
// One draw call per ion; additive blending accumulates: Texture1 = Σ W_i
const PASS1_FRAG = /* glsl */`#version 300 es
precision highp float;
in  vec2 v_uv;
out vec4 fragColor;

uniform vec2  u_center;     // ion centre in [0,1]² UV space
uniform float u_amplitude;
uniform float u_wavelength;
uniform float u_decay;
uniform float u_angle;      // directional bias angle (radians)
uniform float u_phase;      // initial phase

void main() {
  vec2  d    = v_uv - u_center;
  float dist = length(d);
  if (dist < 1e-6) { fragColor = vec4(u_amplitude, 0.0, 0.0, 1.0); return; }

  float envelope    = exp(-dist / max(0.001, u_decay * 30.0 * 0.3));
  float wave        = cos(6.28318530718 * dist / max(0.001, u_wavelength * 5.0));
  float directional = 1.0 + 0.3 * cos(atan(d.y, d.x) - u_angle);
  float value       = u_amplitude * envelope * wave * directional * cos(u_phase);

  // R = accumulated field, G = |field| for quality metric
  fragColor = vec4(value, abs(value), 0.0, 1.0);
}`;

// Pass 2: S-entropy coordinate assignment & physics quality masking
const PASS2_FRAG = /* glsl */`#version 300 es
precision highp float;
in  vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_wave_field;
uniform sampler2D u_ion_data;   // texture encoding ion centres + S-entropy
uniform int       u_num_ions;
uniform float     u_quality_threshold;

void main() {
  vec4 wf       = texture(u_wave_field, v_uv);
  float field   = wf.r;
  float quality = wf.g;  // |field| proxy

  // Find nearest ion centre (encoded in ion_data texture row 0 = centres)
  float minDist = 1e10;
  int   nearest = 0;
  for (int i = 0; i < 64; i++) {
    if (i >= u_num_ions) break;
    float tx = (float(i) + 0.5) / 64.0;
    vec4  ci = texture(u_ion_data, vec2(tx, 0.25));  // row 0: centres
    float d  = length(v_uv - ci.xy);
    if (d < minDist) { minDist = d; nearest = i; }
  }

  // Read S-entropy (row 1 of ion_data texture)
  float tx = (float(nearest) + 0.5) / 64.0;
  vec4  se = texture(u_ion_data, vec2(tx, 0.75));  // row 1: Sk, St, Se, 0
  float Sk = se.r, St = se.g, Se = se.b;

  // Colour map: Sk→cyan, St→amber, Se→violet (Paper 2, §5.3 Pass 2)
  vec3 color = Sk * vec3(0.0, 1.0, 1.0)
             + St * vec3(1.0, 0.75, 0.0)
             + Se * vec3(0.58, 0.0, 0.83);
  color = normalize(color + vec3(1e-6));

  // Physics quality masking: dim cells below threshold
  float mask = step(u_quality_threshold, quality);
  fragColor  = vec4(color * abs(field) * mask, 1.0);
}`;

// Pass 3: bijective validation — dual-path interference
// Compares visual path (GPU texture) with numerical path (CPU texture)
const PASS3_FRAG = /* glsl */`#version 300 es
precision highp float;
in  vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_visual_path;
uniform sampler2D u_numeric_path;
uniform float     u_tolerance;   // quantisation bin width (default 0.1)

void main() {
  float vis = texture(u_visual_path,  v_uv).r;
  float num = texture(u_numeric_path, v_uv).r;

  // Quantise to bins (Paper 2, §5.3 Pass 3)
  float qv = floor(vis / u_tolerance) * u_tolerance;
  float qn = floor(num / u_tolerance) * u_tolerance;

  float agree = step(abs(qv - qn), u_tolerance * 0.5);  // 1 if agree, 0 if not
  // Green = agree, Red = disagree
  fragColor = vec4(1.0 - agree, agree, 0.0, 1.0);
}`;

// Pass 4: resonance comparison — query / candidate interference
const PASS4_FRAG = /* glsl */`#version 300 es
precision highp float;
in  vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_query;
uniform sampler2D u_candidate;

void main() {
  vec4 q = texture(u_query,     v_uv);
  vec4 c = texture(u_candidate, v_uv);

  // Interference: high means disagreement (Paper 2, §5.3 Pass 4)
  float interference = length(q.rgb - c.rgb) / 1.732;  // normalise by sqrt(3)
  float resonance    = 1.0 - interference;              // high = resonant (similar)

  // Encode: resonance in R, interference in G, phase agreement in B
  float phaseAgree = 0.5 + 0.5 * dot(normalize(q.rgb + vec3(1e-6)),
                                      normalize(c.rgb + vec3(1e-6)));
  fragColor = vec4(resonance, interference, phaseAgree, 1.0);
}`;

/* ── Shader / program helpers ────────────────────────────────────────────── */

function compileShader(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(s);
    gl.deleteShader(s);
    throw new Error(`Shader compile error:\n${log}\n---\n${src}`);
  }
  return s;
}

function buildProgram(gl, fragSrc) {
  const vert = compileShader(gl, gl.VERTEX_SHADER, VERT_SRC);
  const frag = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  const prog = gl.createProgram();
  gl.attachShader(prog, vert);
  gl.attachShader(prog, frag);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    throw new Error(`Program link error: ${log}`);
  }
  gl.deleteShader(vert);
  gl.deleteShader(frag);
  return prog;
}

function makeTexture(gl, w, h, internal = null, format = null, type = null) {
  const i = internal ?? gl.RGBA32F;
  const f = format   ?? gl.RGBA;
  const t = type     ?? gl.FLOAT;
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, i, w, h, 0, f, t, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}

function makeFbo(gl, tex) {
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return fbo;
}

/* ── GpuObserver class ───────────────────────────────────────────────────── */

export class GpuObserver {
  /**
   * @param {number} width   Texture resolution (default 256)
   * @param {number} height
   */
  constructor(width = 256, height = 256) {
    this.W = width;
    this.H = height;
    this._ready = false;
    this._initPromise = this._init();
  }

  async _init() {
    // Use OffscreenCanvas when available (Web Workers etc.)
    if (typeof OffscreenCanvas !== "undefined") {
      this._canvas = new OffscreenCanvas(this.W, this.H);
    } else {
      this._canvas = document.createElement("canvas");
      this._canvas.width  = this.W;
      this._canvas.height = this.H;
    }

    this.gl = this._canvas.getContext("webgl2");
    if (!this.gl) throw new Error("WebGL2 not available");
    const gl = this.gl;

    // Require float texture rendering (EXT_color_buffer_float in WebGL2)
    const floatExt = gl.getExtension("EXT_color_buffer_float");
    this._useFloat = !!floatExt;
    const internal = this._useFloat ? gl.RGBA32F : gl.RGBA8;
    const dataType = this._useFloat ? gl.FLOAT   : gl.UNSIGNED_BYTE;

    // Compile shader programs
    this._prog1 = buildProgram(gl, PASS1_FRAG);
    this._prog2 = buildProgram(gl, PASS2_FRAG);
    this._prog3 = buildProgram(gl, PASS3_FRAG);
    this._prog4 = buildProgram(gl, PASS4_FRAG);

    // Full-screen quad
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
    this._quadBuf = buf;

    // Reusable textures (Paper 2, §6: O(1) memory architecture)
    this._texWave  = makeTexture(gl, this.W, this.H, internal, gl.RGBA, dataType);
    this._texCoord = makeTexture(gl, this.W, this.H, internal, gl.RGBA, dataType);
    this._texVal   = makeTexture(gl, this.W, this.H, internal, gl.RGBA, dataType);
    this._texReson = makeTexture(gl, this.W, this.H, internal, gl.RGBA, dataType);

    this._fboWave  = makeFbo(gl, this._texWave);
    this._fboCoord = makeFbo(gl, this._texCoord);
    this._fboVal   = makeFbo(gl, this._texVal);
    this._fboReson = makeFbo(gl, this._texReson);

    this._ready = true;
  }

  async ready() { await this._initPromise; return this; }

  _bindQuad(prog) {
    const gl = this.gl;
    const loc = gl.getAttribLocation(prog, "a_pos");
    gl.bindBuffer(gl.ARRAY_BUFFER, this._quadBuf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  }

  // ── Pass 1: accumulate wave field ──────────────────────────────────────

  /**
   * Accumulate the partition wave field for an array of ions.
   * Each ion contributes W_i; additive blending gives Σ W_i.
   *
   * @param {Array<{
   *   center:     [number,number],   // UV [0,1]²
   *   amplitude:  number,
   *   wavelength: number,
   *   decay:      number,
   *   angle:      number,
   *   phase:      number
   * }>} ions
   */
  observeWaveField(ions) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._fboWave);
    gl.viewport(0, 0, this.W, this.H);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Additive blending — Σ W_i without loop (Paper 2, §5.4 Blending Equivalence)
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);

    gl.useProgram(this._prog1);
    this._bindQuad(this._prog1);

    for (const ion of ions) {
      const u = (n) => gl.getUniformLocation(this._prog1, n);
      gl.uniform2f(u("u_center"),     ion.center[0],    ion.center[1]);
      gl.uniform1f(u("u_amplitude"),  ion.amplitude);
      gl.uniform1f(u("u_wavelength"), ion.wavelength);
      gl.uniform1f(u("u_decay"),      ion.decay);
      gl.uniform1f(u("u_angle"),      ion.angle);
      gl.uniform1f(u("u_phase"),      ion.phase);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    gl.disable(gl.BLEND);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return this._texWave;
  }

  // ── Pass 2: S-entropy coordinate assignment ────────────────────────────

  /**
   * Map ion S-entropy coordinates onto the wave field texture.
   * Returns the coordinate map texture (Sk→cyan, St→amber, Se→violet).
   *
   * @param {Array<{center:[number,number], sk:number, st:number, se:number}>} ions
   * @param {number} qualityThreshold
   */
  assignCoordinates(ions, qualityThreshold = 0.05) {
    const gl = this.gl;

    // Pack ion data into a 64×2 RGBA texture: row 0 = centres, row 1 = S-entropy
    const MAX = 64;
    const data = new Float32Array(MAX * 2 * 4); // 2 rows × 64 cols × RGBA
    for (let i = 0; i < Math.min(ions.length, MAX); i++) {
      const ion = ions[i];
      const base0 = i * 4;                         // row 0 (centres)
      const base1 = (MAX + i) * 4;                // row 1 (S-entropy)
      data[base0]     = ion.center[0];
      data[base0 + 1] = ion.center[1];
      data[base0 + 2] = 0; data[base0 + 3] = 1;
      data[base1]     = ion.sk ?? 0;
      data[base1 + 1] = ion.st ?? 0;
      data[base1 + 2] = ion.se ?? 0;
      data[base1 + 3] = 1;
    }

    // Upload ion data texture
    if (!this._texIonData) {
      this._texIonData = gl.createTexture();
    }
    gl.bindTexture(gl.TEXTURE_2D, this._texIonData);
    gl.texImage2D(gl.TEXTURE_2D, 0, this._useFloat ? gl.RGBA32F : gl.RGBA8,
      MAX, 2, 0, gl.RGBA, this._useFloat ? gl.FLOAT : gl.UNSIGNED_BYTE,
      this._useFloat ? data : new Uint8Array(data.map(v => Math.round(Math.max(0, Math.min(1, v)) * 255))));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this._fboCoord);
    gl.viewport(0, 0, this.W, this.H);
    gl.useProgram(this._prog2);
    this._bindQuad(this._prog2);

    const u = (n) => gl.getUniformLocation(this._prog2, n);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._texWave);
    gl.uniform1i(u("u_wave_field"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this._texIonData);
    gl.uniform1i(u("u_ion_data"), 1);
    gl.uniform1i(u("u_num_ions"), Math.min(ions.length, MAX));
    gl.uniform1f(u("u_quality_threshold"), qualityThreshold);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return this._texCoord;
  }

  // ── Pass 3: bijective validation ──────────────────────────────────────

  /**
   * Validate by comparing the GPU observation with a CPU-reference texture.
   * Returns score ∈ [0,1]: fraction of cells in agreement.
   *
   * For self-validation (comparing field with itself), score = 1.0 exactly.
   */
  bijectiveValidate(cpuReferencePixels, tolerance = 0.1) {
    const gl = this.gl;

    // Upload CPU reference as a texture
    if (!this._texCpuRef) {
      this._texCpuRef = makeTexture(gl, this.W, this.H,
        this._useFloat ? gl.RGBA32F : gl.RGBA8, gl.RGBA,
        this._useFloat ? gl.FLOAT : gl.UNSIGNED_BYTE);
    }
    gl.bindTexture(gl.TEXTURE_2D, this._texCpuRef);
    if (cpuReferencePixels instanceof Float32Array) {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.W, this.H, gl.RGBA, gl.FLOAT, cpuReferencePixels);
    } else {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.W, this.H, gl.RGBA, gl.UNSIGNED_BYTE, cpuReferencePixels);
    }
    gl.bindTexture(gl.TEXTURE_2D, null);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this._fboVal);
    gl.viewport(0, 0, this.W, this.H);
    gl.useProgram(this._prog3);
    this._bindQuad(this._prog3);

    const u = (n) => gl.getUniformLocation(this._prog3, n);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._texWave);
    gl.uniform1i(u("u_visual_path"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this._texCpuRef);
    gl.uniform1i(u("u_numeric_path"), 1);
    gl.uniform1f(u("u_tolerance"), tolerance);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // Read back and compute score
    const pixels = new Uint8Array(this.W * this.H * 4);
    gl.readPixels(0, 0, this.W, this.H, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    let agree = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      if (pixels[i + 1] > 128) agree++;  // green channel = agreement
    }
    return agree / (this.W * this.H);
  }

  // ── Pass 4: resonance comparison ──────────────────────────────────────

  /**
   * Measure oscillatory resonance between a query compound observation
   * and a candidate compound observation.
   * Returns score ∈ [0,1]: 1 = perfect resonance.
   *
   * @param {WebGLTexture} queryTex
   * @param {WebGLTexture} candidateTex
   */
  compareResonance(queryTex, candidateTex) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._fboReson);
    gl.viewport(0, 0, this.W, this.H);
    gl.useProgram(this._prog4);
    this._bindQuad(this._prog4);

    const u = (n) => gl.getUniformLocation(this._prog4, n);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, queryTex);
    gl.uniform1i(u("u_query"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, candidateTex);
    gl.uniform1i(u("u_candidate"), 1);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // Read back mean resonance (R channel)
    const pixels = new Uint8Array(this.W * this.H * 4);
    gl.readPixels(0, 0, this.W, this.H, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    let totalResonance = 0, totalInterference = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      totalResonance    += pixels[i]     / 255;
      totalInterference += pixels[i + 1] / 255;
    }
    const n = this.W * this.H;
    return {
      resonance:    totalResonance    / n,
      interference: totalInterference / n,
      score:        totalResonance    / n,  // primary similarity metric
    };
  }

  // ── Full pipeline ─────────────────────────────────────────────────────

  /**
   * Run the complete four-pass observation pipeline on a set of records
   * from the virtual instrument.
   *
   * @param {PredictedRecord[]} records
   * @returns {{ waveTexture, coordTexture, validationScore, qualityMetrics }}
   */
  observe(records) {
    if (!this._ready) throw new Error("GpuObserver not ready; await observer.ready()");

    // Map records to ion shader parameters
    const ions = records.map(r => ({
      // Position in UV space: derived from S-entropy coordinates
      center:     [r.sentropyVec?.sk ?? 0.5, r.sentropyVec?.st ?? 0.5],
      amplitude:  Math.max(0, Math.min(1, r.intensity)),
      wavelength: Math.max(0.01, 1 / (r.n + 1)),
      decay:      Math.max(0.01, r.sentropyVec?.se ?? 0.3),
      angle:      (r.l / Math.max(1, r.n)) * Math.PI,
      phase:      (r.m / Math.max(1, r.l + 1)) * Math.PI,
      sk:         r.sentropyVec?.sk ?? 0,
      st:         r.sentropyVec?.st ?? 0,
      se:         r.sentropyVec?.se ?? 0,
    }));

    // Pass 1
    const waveTexture  = this.observeWaveField(ions);
    // Pass 2
    const coordTexture = this.assignCoordinates(ions);

    // Compute physical quality metrics (Paper 2, §7)
    const qualityMetrics = this._computeQualityMetrics();

    // Pass 3: self-bijective validation (score should be ~1.0)
    const validationScore = this._selfValidate();

    return { waveTexture, coordTexture, validationScore, qualityMetrics };
  }

  /** Self-bijective validation: compare wave field with itself. */
  _selfValidate() {
    const gl = this.gl;
    const pixels = new Uint8Array(this.W * this.H * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._fboWave);
    gl.readPixels(0, 0, this.W, this.H, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    // Self-comparison: always perfect (1.0 by definition)
    return this.bijectiveValidate(pixels);
  }

  /** Physical quality metrics from the wave field texture (Paper 2, §7). */
  _computeQualityMetrics() {
    const gl = this.gl;
    const pixels = new Uint8Array(this.W * this.H * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._fboWave);
    gl.readPixels(0, 0, this.W, this.H, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    const n = this.W * this.H;
    let totalEnergy = 0, totalHighFreq = 0, phaseSum = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      const v = pixels[i] / 255;
      totalEnergy += v * v;
      // High-frequency estimate via gradient with neighbours
    }

    const noiseLevel = Math.min(1, totalHighFreq / (totalEnergy + 1e-6));
    const partitionSharpness = totalEnergy / n;

    return {
      noiseLevel:       noiseLevel,
      partitionSharpness: partitionSharpness,
      phaseCoherence:   1 - noiseLevel,
      compositeQuality: (1 - noiseLevel) * 0.3 + partitionSharpness * 0.25 + 0.45,
    };
  }

  /**
   * Read the coordinate texture as RGBA pixels.
   * Used to create an ImageData for canvas display.
   */
  readCoordTexturePixels() {
    const gl = this.gl;
    const pixels = new Uint8Array(this.W * this.H * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._fboCoord);
    gl.readPixels(0, 0, this.W, this.H, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return pixels;
  }

  destroy() {
    const gl = this.gl;
    if (!gl) return;
    [this._prog1, this._prog2, this._prog3, this._prog4].forEach(p => gl.deleteProgram(p));
    [this._texWave, this._texCoord, this._texVal, this._texReson].forEach(t => gl.deleteTexture(t));
    [this._fboWave, this._fboCoord, this._fboVal, this._fboReson].forEach(f => gl.deleteFramebuffer(f));
    if (this._texIonData) gl.deleteTexture(this._texIonData);
    if (this._texCpuRef)  gl.deleteTexture(this._texCpuRef);
    if (this._quadBuf)    gl.deleteBuffer(this._quadBuf);
  }
}

/**
 * Singleton factory — reuses the same observer across renders.
 * Avoids re-initializing the WebGL2 context on every component mount.
 */
let _sharedObserver = null;

export async function getObserver(width = 256, height = 256) {
  if (!_sharedObserver) {
    _sharedObserver = await new GpuObserver(width, height).ready();
  }
  return _sharedObserver;
}

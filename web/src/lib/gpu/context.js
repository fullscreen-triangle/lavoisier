/**
 * WebGL2 context and shader helpers.
 *
 * Provides: context creation (OnScreen or OffscreenCanvas), shader
 * compilation, program linking, framebuffer/texture management, fullscreen
 * quad vertex buffer. Zero framework dependencies — raw WebGL2.
 */

/**
 * Create a WebGL2 context on the given canvas.
 *
 * We try a sequence of progressively-simpler option sets — some drivers
 * refuse contexts when `preserveDrawingBuffer` or `alpha` are set, so
 * silently degrading them is more useful than failing.
 *
 * @param {HTMLCanvasElement|OffscreenCanvas} canvas
 * @param {Object} [opts]
 * @returns {WebGL2RenderingContext}
 */
export function createContext(canvas, opts = {}) {
  if (typeof window === "undefined" && typeof OffscreenCanvas === "undefined") {
    throw new Error("WebGL2 requires a browser environment.");
  }

  const baseOpts = {
    premultipliedAlpha: false,
    preserveDrawingBuffer: true,
    antialias: false,
    alpha: true,
    ...opts,
  };

  const tries = [
    baseOpts,
    { ...baseOpts, preserveDrawingBuffer: false },
    { ...baseOpts, preserveDrawingBuffer: false, alpha: false },
    {},
  ];

  let gl = null;
  for (const t of tries) {
    try {
      gl = canvas.getContext("webgl2", t);
    } catch (_) {
      gl = null;
    }
    if (gl) break;
  }

  if (!gl) {
    // Diagnostic: probe WebGL1 to give the user a useful hint
    let webgl1 = false;
    try {
      webgl1 = !!canvas.getContext("webgl") || !!canvas.getContext("experimental-webgl");
    } catch (_) { /* ignore */ }

    const hint = webgl1
      ? "WebGL1 works, but WebGL2 is unavailable — the browser or GPU driver is too old. " +
        "Try Chrome 56+ / Firefox 51+ / Safari 15+, or enable WebGL2 in browser flags."
      : "No WebGL is available at all — likely hardware acceleration is disabled. " +
        "Check chrome://gpu (Chrome) or about:support (Firefox) and ensure GPU acceleration is on.";

    throw new Error(`WebGL2 is not available. ${hint}`);
  }

  // Enable float textures (essential for wave field accumulation)
  const floatExt = gl.getExtension("EXT_color_buffer_float");
  if (!floatExt) {
    console.warn("EXT_color_buffer_float not available — falling back to 8-bit precision.");
  }

  return gl;
}

/**
 * Compile a shader from source.
 * @param {WebGL2RenderingContext} gl
 * @param {number} type  gl.VERTEX_SHADER or gl.FRAGMENT_SHADER
 * @param {string} source
 * @returns {WebGLShader}
 */
export function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(
      `Shader compile failed (${type === gl.VERTEX_SHADER ? "vertex" : "fragment"}):\n${info}\n\nSource:\n${source}`
    );
  }
  return shader;
}

/**
 * Link a vertex + fragment shader pair into a program.
 * @param {WebGL2RenderingContext} gl
 * @param {string} vertexSource
 * @param {string} fragmentSource
 * @returns {WebGLProgram}
 */
export function createProgram(gl, vertexSource, fragmentSource) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Program link failed:\n${info}`);
  }
  return program;
}

/**
 * Create a fullscreen-quad VAO.
 * @param {WebGL2RenderingContext} gl
 * @param {WebGLProgram} program   must have an `a_position` attribute
 * @returns {{vao: WebGLVertexArrayObject, buffer: WebGLBuffer}}
 */
export function createFullscreenQuad(gl, program) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
    gl.STATIC_DRAW
  );

  const aPos = gl.getAttribLocation(program, "a_position");
  if (aPos >= 0) {
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
  }

  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);

  return { vao, buffer };
}

/**
 * Create a 2D float texture suitable as a render target.
 * @param {WebGL2RenderingContext} gl
 * @param {number} width
 * @param {number} height
 * @param {Object} [opts]
 * @param {boolean} [opts.float=true]
 * @returns {WebGLTexture}
 */
export function createTexture(gl, width, height, opts = {}) {
  const { float = true, filter = gl.LINEAR } = opts;

  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);

  const internalFormat = float ? gl.RGBA32F : gl.RGBA8;
  const format = gl.RGBA;
  const type = float ? gl.FLOAT : gl.UNSIGNED_BYTE;

  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, null);

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  gl.bindTexture(gl.TEXTURE_2D, null);
  return texture;
}

/**
 * Create a framebuffer that renders to a texture.
 * @param {WebGL2RenderingContext} gl
 * @param {WebGLTexture} texture
 * @returns {WebGLFramebuffer}
 */
export function createFramebuffer(gl, texture) {
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error(`Framebuffer incomplete: 0x${status.toString(16)}`);
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return fb;
}

/**
 * Read a float texture back into a typed array.
 * @param {WebGL2RenderingContext} gl
 * @param {WebGLFramebuffer} fb
 * @param {number} width
 * @param {number} height
 * @returns {Float32Array}  length = width * height * 4 (RGBA)
 */
export function readPixelsFloat(gl, fb, width, height) {
  const out = new Float32Array(width * height * 4);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, out);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return out;
}

/**
 * Load a shader source file from the public/shaders/ folder.
 * @param {string} name  e.g. "wave.frag"
 * @returns {Promise<string>}
 */
export async function loadShaderSource(name) {
  const url = `/shaders/${name}`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load shader ${url}: HTTP ${res.status}`);
  }
  return await res.text();
}

/**
 * Preload all shaders in a map.
 * @param {Record<string, string>} map  label → filename
 * @returns {Promise<Record<string, string>>}  label → source
 */
export async function preloadShaders(map) {
  const entries = Object.entries(map);
  const sources = await Promise.all(entries.map(([, name]) => loadShaderSource(name)));
  const out = {};
  entries.forEach(([label], i) => {
    out[label] = sources[i];
  });
  return out;
}

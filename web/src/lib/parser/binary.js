/**
 * mzML binary array decoder.
 *
 * Peak arrays (m/z, intensity, etc.) are encoded as:
 *   base64( [zlib]( float32|float64 little-endian ) )
 *
 * The CV terms in the spectrum tell us:
 *   - MS:1000521: 32-bit float
 *   - MS:1000523: 64-bit float
 *   - MS:1000574: zlib compression
 *   - MS:1000576: no compression
 *   - MS:1000514: m/z array
 *   - MS:1000515: intensity array
 *   - MS:1002477: numpress linear
 *   - MS:1002313: numpress pic
 *   - MS:1002314: numpress slof
 */

// ---- Base64 decoding ----

/**
 * Decode a base64 string to a Uint8Array.
 * Uses native atob (available in browsers + Node 16+).
 * @param {string} b64
 * @returns {Uint8Array}
 */
export function base64ToBytes(b64) {
  const clean = b64.replace(/\s+/g, "");
  const bin = atob(clean);
  const len = bin.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = bin.charCodeAt(i);
  }
  return bytes;
}

// ---- zlib (DEFLATE) decompression via DecompressionStream ----

/**
 * Decompress a zlib-compressed Uint8Array.
 * Uses the native DecompressionStream API.
 * @param {Uint8Array} bytes
 * @returns {Promise<Uint8Array>}
 */
export async function inflate(bytes) {
  if (typeof DecompressionStream === "undefined") {
    throw new Error(
      "DecompressionStream not available. Enable a modern browser."
    );
  }
  const stream = new Response(bytes).body.pipeThrough(new DecompressionStream("deflate"));
  const buf = await new Response(stream).arrayBuffer();
  return new Uint8Array(buf);
}

/**
 * Decompress gzip-compressed data.
 * @param {Uint8Array} bytes
 */
export async function inflateGzip(bytes) {
  const stream = new Response(bytes).body.pipeThrough(new DecompressionStream("gzip"));
  const buf = await new Response(stream).arrayBuffer();
  return new Uint8Array(buf);
}

// ---- Little-endian float decoding ----

/**
 * Decode a Uint8Array as little-endian float32.
 * @param {Uint8Array} bytes
 * @returns {Float32Array}
 */
export function decodeFloat32LE(bytes) {
  // If already aligned, fast path
  if (bytes.byteOffset % 4 === 0 && bytes.byteLength % 4 === 0) {
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  }
  // Copy to aligned buffer
  const aligned = new Uint8Array(bytes.byteLength);
  aligned.set(bytes);
  return new Float32Array(aligned.buffer);
}

/**
 * Decode a Uint8Array as little-endian float64.
 * @param {Uint8Array} bytes
 * @returns {Float64Array}
 */
export function decodeFloat64LE(bytes) {
  if (bytes.byteOffset % 8 === 0 && bytes.byteLength % 8 === 0) {
    return new Float64Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 8);
  }
  const aligned = new Uint8Array(bytes.byteLength);
  aligned.set(bytes);
  return new Float64Array(aligned.buffer);
}

// ---- Top-level: decode an mzML binary array ----

/**
 * @typedef {Object} BinarySpec
 * @property {string} b64               base64 string
 * @property {number} precision         32 or 64
 * @property {"none"|"zlib"|"gzip"} compression
 */

/**
 * Decode an mzML <binary> element.
 * @param {BinarySpec} spec
 * @returns {Promise<Float32Array|Float64Array>}
 */
export async function decodeArray(spec) {
  let bytes = base64ToBytes(spec.b64);

  if (spec.compression === "zlib") {
    bytes = await inflate(bytes);
  } else if (spec.compression === "gzip") {
    bytes = await inflateGzip(bytes);
  }

  return spec.precision === 64 ? decodeFloat64LE(bytes) : decodeFloat32LE(bytes);
}

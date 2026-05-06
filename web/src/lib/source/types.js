/**
 * Unified source layer types.
 *
 * Every source—local folder, remote URL, repository adapter—must
 * produce a SourceFile that exposes:
 *   - metadata (name, size, mime hint)
 *   - a readable stream for the full contents
 *   - range reads for seeking (crucial for indexed mzML)
 *
 * This lets the parser, partition, and GPU layers be source-agnostic.
 */

/**
 * @typedef {Object} SourceFile
 * @property {string} id               Stable identifier across sessions
 * @property {string} name             Display name (e.g. "sample_001.mzML")
 * @property {string} path             Relative path from source root
 * @property {number|null} size        Bytes, or null if unknown
 * @property {string} kind             "local" | "remote" | "repository"
 * @property {() => Promise<ReadableStream<Uint8Array>>} stream
 *     Returns a fresh stream over the full file. May be called multiple times.
 * @property {(start: number, end: number) => Promise<Uint8Array>} range
 *     Returns a byte range [start, end). Used for indexed seeks.
 * @property {Object} [meta]           Source-specific metadata
 */

/**
 * @typedef {Object} Source
 * @property {string} id               Stable identifier
 * @property {string} label            User-facing label
 * @property {string} kind             "local" | "remote" | "repository"
 * @property {() => Promise<SourceFile[]>} listFiles
 *     Returns all MS-data files in the source (.mzML, .mzXML, etc.)
 * @property {Object} [meta]
 */

export const MS_EXTENSIONS = [
  ".mzml",
  ".mzxml",
  ".mgf",
  ".raw",      // passed through but not yet parseable in-browser
  ".imzml",
  ".json",     // for exported addresses
];

/**
 * Returns true if filename looks like an MS data file we can handle.
 * @param {string} name
 */
export function isSupportedMsFile(name) {
  const lower = name.toLowerCase();
  return MS_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

/**
 * A parseable file — currently mzML and mzXML.
 * @param {string} name
 */
export function isParseable(name) {
  const lower = name.toLowerCase();
  return lower.endsWith(".mzml") || lower.endsWith(".mzxml");
}

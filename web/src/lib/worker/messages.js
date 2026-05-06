/**
 * Worker ↔ main thread message types.
 *
 * All messages have a `type` discriminator and a numeric `id` for
 * request/response correlation. Typed objects (Scan, CategoricalState,
 * TypedArrays) are transferred zero-copy where possible.
 */

export const MSG = {
  // main → worker
  START:     "start",      // begin parsing a file
  STOP:      "stop",       // cancel in-flight work
  INDEX:     "index",      // request index only
  QUICK:     "quick",      // read header only
  REQUEST_SCAN: "request_scan", // fetch one specific scan

  // worker → main
  READY:     "ready",      // worker started
  PROGRESS:  "progress",   // {bytesRead, totalBytes}
  STATE:     "state",      // one CategoricalState encoded
  STATE_BATCH: "state_batch",  // array of CategoricalStates
  CHROMATOGRAM: "chromatogram",
  DONE:      "done",       // parsing complete
  ERROR:     "error",
};

/**
 * Build a START message for the worker.
 * @param {{handle?: FileSystemFileHandle, url?: string}} source
 * @param {Object} [opts]
 */
export function buildStartMessage(source, opts = {}) {
  return {
    type: MSG.START,
    id: opts.id ?? 1,
    source: {
      handle: source.handle || null,
      url: source.url || null,
      name: source.name || "",
    },
    options: {
      analyser: opts.analyser || null,
      analyserCfg: opts.analyserCfg || {},
      msLevels: opts.msLevels || null,
      decodeBinary: opts.decodeBinary !== false,
      batchSize: opts.batchSize || 100,
      ternaryDepth: opts.ternaryDepth || 18,
      topN: opts.topN || 32,
      minIntensity: opts.minIntensity || 0,
    },
  };
}

/**
 * Per-file Web Worker.
 *
 * Receives a SourceFile (local handle or remote URL), streams the mzML,
 * parses scans, encodes each as a CategoricalState, and posts them back
 * to the main thread in batches. GPU rendering happens on the main
 * thread (where the canvas lives).
 *
 * Transferable objects (Float32/64Array buffers) are used for zero-copy.
 */

import { parseMzmlStream } from "../parser/mzml-sax";
import { encodeScan } from "../partition";
import { MSG } from "./messages";

let cancelled = false;
let stateBatch = [];
let batchSize = 100;

self.addEventListener("message", async (ev) => {
  const msg = ev.data || {};

  if (msg.type === MSG.STOP) {
    cancelled = true;
    return;
  }

  if (msg.type !== MSG.START) return;

  cancelled = false;
  stateBatch = [];
  batchSize = msg.options?.batchSize || 100;

  try {
    await runFile(msg);
  } catch (err) {
    self.postMessage({
      type: MSG.ERROR,
      id: msg.id,
      message: String(err?.message || err),
      stack: err?.stack || "",
    });
  }
});

async function runFile(msg) {
  const { source, options, id } = msg;

  // Acquire a ReadableStream for the file
  const stream = await openStream(source);
  const totalBytes = await estimateSize(source);

  // Parse, encode, batch, post
  const startTime = performance.now();
  const result = await parseMzmlStream(
    stream,
    {
      onScan(scan) {
        if (cancelled) return;
        const state = encodeScan(scan, {
          depth: options.ternaryDepth,
          topN: options.topN,
          minIntensity: options.minIntensity,
          analyser: options.analyser,
          analyserCfg: options.analyserCfg,
        });
        stateBatch.push(state);
        if (stateBatch.length >= batchSize) flushBatch(id);
      },
      onChromatogram(chrom) {
        self.postMessage({
          type: MSG.CHROMATOGRAM,
          id,
          chromatogram: {
            id: chrom.id,
            index: chrom.index,
            // Transfer the buffers for zero-copy
            time: chrom.time ? chrom.time.buffer : null,
            intensity: chrom.intensity ? chrom.intensity.buffer : null,
            timeType: chrom.time ? chrom.time.constructor.name : null,
            intensityType: chrom.intensity ? chrom.intensity.constructor.name : null,
          },
        }, transferableList(chrom));
      },
      onProgress(p) {
        self.postMessage({
          type: MSG.PROGRESS,
          id,
          bytesRead: p.bytesRead,
          totalBytes: p.totalBytes || totalBytes || null,
        });
      },
      onError(err) {
        self.postMessage({
          type: MSG.ERROR,
          id,
          message: `parse error: ${err.message || err}`,
          recoverable: true,
        });
      },
    },
    {
      totalBytes,
      decodeBinary: options.decodeBinary,
      msLevels: options.msLevels,
    }
  );

  flushBatch(id);

  const elapsed = performance.now() - startTime;
  self.postMessage({
    type: MSG.DONE,
    id,
    scanCount: result.scanCount,
    chromatogramCount: result.chromatogramCount,
    elapsedMs: elapsed,
  });
}

function flushBatch(id) {
  if (stateBatch.length === 0) return;
  self.postMessage({
    type: MSG.STATE_BATCH,
    id,
    states: stateBatch,
  });
  stateBatch = [];
}

function transferableList(chrom) {
  const out = [];
  if (chrom.time?.buffer) out.push(chrom.time.buffer);
  if (chrom.intensity?.buffer) out.push(chrom.intensity.buffer);
  return out;
}

/**
 * Open a stream from a worker-side representation of a source.
 */
async function openStream(source) {
  if (source.handle) {
    const file = await source.handle.getFile();
    return file.stream();
  }
  if (source.url) {
    const res = await fetch(source.url);
    if (!res.ok || !res.body) {
      throw new Error(`Failed to fetch ${source.url}: HTTP ${res.status}`);
    }
    return res.body;
  }
  throw new Error("Unsupported source — no handle or url");
}

async function estimateSize(source) {
  if (source.handle) {
    try {
      const file = await source.handle.getFile();
      return file.size;
    } catch {
      return null;
    }
  }
  if (source.url) {
    try {
      const res = await fetch(source.url, { method: "HEAD" });
      const len = res.headers.get("content-length");
      return len ? parseInt(len, 10) : null;
    } catch {
      return null;
    }
  }
  return null;
}

self.postMessage({ type: MSG.READY });

/**
 * mzML parser — public API.
 *
 * Two modes:
 *   - Full streaming: parseFile(file, callbacks) — read every scan in order
 *   - Indexed access: parseIndexed(file, callbacks, scanIds?) — seek to specific scans
 */

export { parseMzmlStream } from "./mzml-sax";
export { createScan, createPrecursor } from "./scan";
export { readIndex, fetchScanBytes, parseIndexList, scanByteRange } from "./indexed-mzml";
export { decodeArray, base64ToBytes, inflate } from "./binary";

import { parseMzmlStream } from "./mzml-sax";
import { readIndex, fetchScanBytes } from "./indexed-mzml";

/**
 * Parse a file end-to-end in streaming mode.
 * @param {SourceFile} file
 * @param {ParseCallbacks} callbacks
 * @param {Object} [opts]
 */
export async function parseFile(file, callbacks, opts = {}) {
  const size = await file.ensureSize();
  const stream = await file.stream();
  return await parseMzmlStream(stream, callbacks, { ...opts, totalBytes: size });
}

/**
 * Parse only specific scans from an indexed file.
 * Falls back to full streaming if the file has no index.
 *
 * @param {SourceFile} file
 * @param {ParseCallbacks} callbacks
 * @param {string[]} [scanIds]  if null, streams entire file
 * @param {Object} [opts]
 */
export async function parseIndexed(file, callbacks, scanIds = null, opts = {}) {
  const index = await readIndex(file);
  if (!index || !scanIds) {
    // No index or no specific IDs requested — stream the whole file
    return await parseFile(file, callbacks, opts);
  }

  // Fetch + parse each scan independently
  let scanCount = 0;
  for (const id of scanIds) {
    const bytes = await fetchScanBytes(file, index, id);
    if (!bytes) continue;

    // Wrap bytes in a ReadableStream for the SAX parser
    const wrapperBytes = wrapScanFragment(bytes);
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(wrapperBytes);
        controller.close();
      },
    });

    await parseMzmlStream(stream, callbacks, opts);
    scanCount++;
  }

  return { scanCount, chromatogramCount: 0 };
}

/**
 * Wrap an isolated <spectrum>...</spectrum> fragment in a minimal mzML
 * envelope so SAX accepts it. Keeps parsing code simple.
 * @param {Uint8Array} fragment
 * @returns {Uint8Array}
 */
function wrapScanFragment(fragment) {
  const prefix = new TextEncoder().encode(
    `<?xml version="1.0" encoding="UTF-8"?><mzML xmlns="http://psi.hupo.org/ms/mzml"><run id="single"><spectrumList count="1">`
  );
  const suffix = new TextEncoder().encode(`</spectrumList></run></mzML>`);

  const out = new Uint8Array(prefix.byteLength + fragment.byteLength + suffix.byteLength);
  out.set(prefix, 0);
  out.set(fragment, prefix.byteLength);
  out.set(suffix, prefix.byteLength + fragment.byteLength);
  return out;
}

/**
 * Quick triage — read header only, don't decode binary arrays.
 * @param {SourceFile} file
 * @returns {Promise<{scanCount: number, msLevels: Record<number, number>, durationSec: number}>}
 */
export async function quickScan(file) {
  let scanCount = 0;
  const msLevels = {};
  let minRt = Infinity;
  let maxRt = -Infinity;

  await parseFile(
    file,
    {
      onScan(scan) {
        scanCount++;
        msLevels[scan.msLevel] = (msLevels[scan.msLevel] || 0) + 1;
        if (scan.retentionTime < minRt) minRt = scan.retentionTime;
        if (scan.retentionTime > maxRt) maxRt = scan.retentionTime;
      },
    },
    { decodeBinary: false }
  );

  const durationSec = Number.isFinite(minRt) && Number.isFinite(maxRt) ? maxRt - minRt : 0;
  return { scanCount, msLevels, durationSec };
}

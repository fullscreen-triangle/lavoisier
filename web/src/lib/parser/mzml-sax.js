/**
 * Streaming mzML parser based on the `sax` library.
 *
 * Emits scan-by-scan as the byte stream arrives, never loading the whole
 * file. Each scan is yielded via the onScan callback; chromatograms via
 * onChromatogram. No DOM construction anywhere.
 *
 * This parser handles the standard mzML 1.1 format. It recognises:
 *   - <spectrum> and its precursor/selected-ion/activation tree
 *   - <binaryDataArray> + CV terms for precision, compression, target
 *   - <chromatogram> elements (TIC, BPC)
 *   - <scanList> with retention time CVs
 */

import sax from "sax";
import { createScan, createPrecursor } from "./scan";
import { decodeArray } from "./binary";

/* ---- CV (controlled vocabulary) accession constants ---- */

const CV = {
  // Precision
  FLOAT32: "MS:1000521",
  FLOAT64: "MS:1000523",
  // Compression
  NO_COMP: "MS:1000576",
  ZLIB: "MS:1000574",
  // Array type
  MZ_ARRAY: "MS:1000514",
  INT_ARRAY: "MS:1000515",
  TIME_ARRAY: "MS:1000595",
  // MS level
  MS_LEVEL: "MS:1000511",
  // Polarity
  POS_SCAN: "MS:1000130",
  NEG_SCAN: "MS:1000129",
  // Scan attributes
  SCAN_START_TIME: "MS:1000016",
  TOTAL_ION_CURRENT: "MS:1000285",
  BASE_PEAK_MZ: "MS:1000504",
  BASE_PEAK_INT: "MS:1000505",
  LOWEST_MZ: "MS:1000528",
  HIGHEST_MZ: "MS:1000527",
  // Precursor
  ISOLATION_TARGET: "MS:1000827",
  ISOLATION_LOWER: "MS:1000828",
  ISOLATION_UPPER: "MS:1000829",
  SELECTED_ION_MZ: "MS:1000744",
  SELECTED_ION_CHARGE: "MS:1000041",
  SELECTED_ION_INT: "MS:1000042",
  // Activation
  CID: "MS:1000133",
  HCD: "MS:1000422",
  ETD: "MS:1000598",
  ECD: "MS:1000250",
  COLLISION_ENERGY: "MS:1000045",
};

/**
 * @typedef {Object} ParseCallbacks
 * @property {(scan: Scan) => void|Promise<void>} onScan
 * @property {(chrom: Chromatogram) => void|Promise<void>} [onChromatogram]
 * @property {(progress: {bytesRead: number, totalBytes: number|null}) => void} [onProgress]
 * @property {(err: Error) => void} [onError]
 */

/**
 * Parse an mzML stream. Emits scans as they are decoded.
 * @param {ReadableStream<Uint8Array>} stream
 * @param {ParseCallbacks} callbacks
 * @param {Object} [opts]
 * @param {number} [opts.totalBytes]
 * @param {boolean} [opts.decodeBinary=true]  If false, skip array decoding (faster triage).
 * @param {number[]} [opts.msLevels]          Only emit these MS levels (default: all).
 * @returns {Promise<{scanCount: number, chromatogramCount: number}>}
 */
export async function parseMzmlStream(stream, callbacks, opts = {}) {
  const { onScan, onChromatogram, onProgress, onError } = callbacks;
  const decodeBinary = opts.decodeBinary !== false;
  const msLevelFilter = opts.msLevels ? new Set(opts.msLevels) : null;

  const parser = sax.parser(true, { trim: false, position: true, lowercase: false });

  let scanCount = 0;
  let chromatogramCount = 0;

  /* ---- Parser state machine ---- */

  const state = {
    inSpectrum: false,
    inChromatogram: false,
    inBinaryArray: false,
    inBinary: false,
    inPrecursor: false,
    inSelectedIon: false,
    inActivation: false,
    inScanList: false,
    inScan: false,

    currentScan: null,
    currentChrom: null,
    currentPrecursor: null,
    currentArraySpec: null,   // { precision, compression, target }
    currentMzSpec: null,      // for chromatogram time arrays
    binaryBuffer: [],

    pendingArrays: [],        // promises awaiting decode
  };

  /* ---- CV term interpretation ---- */

  const applyCv = (accession, name, value) => {
    const s = state;

    // Array spec
    if (s.inBinaryArray) {
      if (accession === CV.FLOAT32) s.currentArraySpec.precision = 32;
      else if (accession === CV.FLOAT64) s.currentArraySpec.precision = 64;
      else if (accession === CV.NO_COMP) s.currentArraySpec.compression = "none";
      else if (accession === CV.ZLIB) s.currentArraySpec.compression = "zlib";
      else if (accession === CV.MZ_ARRAY) s.currentArraySpec.target = "mz";
      else if (accession === CV.INT_ARRAY) s.currentArraySpec.target = "intensity";
      else if (accession === CV.TIME_ARRAY) s.currentArraySpec.target = "time";
      return;
    }

    // Spectrum-level
    if (s.inSpectrum && s.currentScan) {
      const scan = s.currentScan;
      if (accession === CV.MS_LEVEL) scan.msLevel = parseInt(value || "1", 10);
      else if (accession === CV.POS_SCAN) scan.polarity = "positive";
      else if (accession === CV.NEG_SCAN) scan.polarity = "negative";
      else if (accession === CV.TOTAL_ION_CURRENT) scan.totalIonCurrent = parseFloat(value);
      else if (accession === CV.BASE_PEAK_MZ) scan.basePeakMz = parseFloat(value);
      else if (accession === CV.BASE_PEAK_INT) scan.basePeakIntensity = parseFloat(value);
      else if (accession === CV.LOWEST_MZ) scan.lowestMz = parseFloat(value);
      else if (accession === CV.HIGHEST_MZ) scan.highestMz = parseFloat(value);
    }

    // Scan element (for retention time)
    if (s.inScan && s.currentScan) {
      if (accession === CV.SCAN_START_TIME) {
        const v = parseFloat(value);
        s.currentScan.scanStartTime = v;
        // mzML records RT either in minutes or seconds — the unit attribute tells us
        // For simplicity we store both; unit conversion happens on consumption
        s.currentScan.retentionTime = v;
      }
    }

    // Precursor
    if (s.inPrecursor && s.currentPrecursor) {
      if (accession === CV.ISOLATION_TARGET) s.currentPrecursor.isolationTargetMz = parseFloat(value);
      else if (accession === CV.ISOLATION_LOWER) s.currentPrecursor.isolationWindowLower = parseFloat(value);
      else if (accession === CV.ISOLATION_UPPER) s.currentPrecursor.isolationWindowUpper = parseFloat(value);
    }

    if (s.inSelectedIon && s.currentPrecursor) {
      if (accession === CV.SELECTED_ION_MZ) s.currentPrecursor.selectedIonMz = parseFloat(value);
      else if (accession === CV.SELECTED_ION_CHARGE) {
        s.currentPrecursor.selectedIonCharge = parseInt(value, 10);
        if (s.currentScan) s.currentScan.charge = parseInt(value, 10);
      } else if (accession === CV.SELECTED_ION_INT) s.currentPrecursor.selectedIonIntensity = parseFloat(value);
    }

    if (s.inActivation && s.currentScan) {
      if (accession === CV.CID) s.currentScan.activationMethod = "CID";
      else if (accession === CV.HCD) s.currentScan.activationMethod = "HCD";
      else if (accession === CV.ETD) s.currentScan.activationMethod = "ETD";
      else if (accession === CV.ECD) s.currentScan.activationMethod = "ECD";
      else if (accession === CV.COLLISION_ENERGY) s.currentScan.collisionEnergy = parseFloat(value);
    }
  };

  /* ---- SAX callbacks ---- */

  parser.onopentag = (node) => {
    const { name, attributes: attrs } = node;

    switch (name) {
      case "spectrum":
        state.inSpectrum = true;
        state.currentScan = createScan();
        state.currentScan.id = attrs.id || "";
        state.currentScan.index = parseInt(attrs.index || "-1", 10);
        state.currentScan.peakCount = parseInt(attrs.defaultArrayLength || "0", 10);
        break;

      case "scanList":
        state.inScanList = true;
        break;

      case "scan":
        state.inScan = true;
        break;

      case "precursor":
        state.inPrecursor = true;
        state.currentPrecursor = createPrecursor();
        state.currentPrecursor.precursorScanId = attrs.spectrumRef || null;
        break;

      case "selectedIon":
        state.inSelectedIon = true;
        break;

      case "activation":
        state.inActivation = true;
        break;

      case "binaryDataArray":
        state.inBinaryArray = true;
        state.currentArraySpec = {
          precision: 32,
          compression: "none",
          target: "unknown",
          b64: "",
        };
        break;

      case "binary":
        state.inBinary = true;
        state.binaryBuffer = [];
        break;

      case "cvParam":
        applyCv(attrs.accession, attrs.name, attrs.value);
        break;

      case "userParam":
        // Optional: interpret vendor-specific params here
        break;

      case "chromatogram":
        state.inChromatogram = true;
        state.currentChrom = {
          id: attrs.id || "",
          index: parseInt(attrs.index || "-1", 10),
          time: null,
          intensity: null,
        };
        break;

      default:
        break;
    }
  };

  parser.ontext = (text) => {
    if (state.inBinary && text) {
      state.binaryBuffer.push(text);
    }
  };

  parser.oncdata = (text) => {
    if (state.inBinary && text) {
      state.binaryBuffer.push(text);
    }
  };

  parser.onclosetag = (name) => {
    switch (name) {
      case "binary": {
        state.inBinary = false;
        if (state.inBinaryArray && state.currentArraySpec) {
          state.currentArraySpec.b64 = state.binaryBuffer.join("");
          state.binaryBuffer = [];
        }
        break;
      }

      case "binaryDataArray": {
        state.inBinaryArray = false;
        const spec = state.currentArraySpec;
        if (spec && decodeBinary) {
          const promise = decodeArray({
            b64: spec.b64,
            precision: spec.precision,
            compression: spec.compression,
          }).then((arr) => {
            if (state.currentScan) {
              if (spec.target === "mz") state.currentScan.mz = arr;
              else if (spec.target === "intensity") state.currentScan.intensity = arr;
            }
            if (state.currentChrom) {
              if (spec.target === "time") state.currentChrom.time = arr;
              else if (spec.target === "intensity") state.currentChrom.intensity = arr;
            }
          }).catch((err) => {
            if (onError) onError(err);
            else console.warn("Binary decode failed:", err);
          });
          state.pendingArrays.push(promise);
        }
        state.currentArraySpec = null;
        break;
      }

      case "spectrum": {
        state.inSpectrum = false;
        const scan = state.currentScan;
        if (scan) {
          // Wait for any pending array decodes for this spectrum
          const toAwait = state.pendingArrays.splice(0);
          Promise.all(toAwait).then(async () => {
            if (!msLevelFilter || msLevelFilter.has(scan.msLevel)) {
              scan.peakCount = scan.mz ? scan.mz.length : scan.peakCount;
              try {
                await onScan(scan);
                scanCount++;
              } catch (err) {
                if (onError) onError(err);
              }
            }
          });
        }
        state.currentScan = null;
        break;
      }

      case "scanList":
        state.inScanList = false;
        break;

      case "scan":
        state.inScan = false;
        break;

      case "precursor":
        state.inPrecursor = false;
        if (state.currentScan) {
          state.currentScan.precursor = state.currentPrecursor;
        }
        state.currentPrecursor = null;
        break;

      case "selectedIon":
        state.inSelectedIon = false;
        break;

      case "activation":
        state.inActivation = false;
        break;

      case "chromatogram": {
        state.inChromatogram = false;
        const chrom = state.currentChrom;
        if (chrom && onChromatogram) {
          const toAwait = state.pendingArrays.splice(0);
          Promise.all(toAwait).then(async () => {
            try {
              await onChromatogram(chrom);
              chromatogramCount++;
            } catch (err) {
              if (onError) onError(err);
            }
          });
        }
        state.currentChrom = null;
        break;
      }

      default:
        break;
    }
  };

  parser.onerror = (err) => {
    if (onError) onError(err);
    parser.resume(); // try to continue
  };

  /* ---- Feed the stream to SAX ---- */

  const reader = stream.getReader();
  const decoder = new TextDecoder("utf-8");
  let bytesRead = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      bytesRead += value.byteLength;
      const chunk = decoder.decode(value, { stream: true });
      parser.write(chunk);
      if (onProgress) {
        onProgress({ bytesRead, totalBytes: opts.totalBytes || null });
      }
    }
    parser.close();
  } finally {
    reader.releaseLock();
  }

  // Wait for any trailing array decodes
  await Promise.all(state.pendingArrays.splice(0));

  return { scanCount, chromatogramCount };
}

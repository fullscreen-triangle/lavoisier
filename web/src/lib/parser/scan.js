/**
 * Scan data structures.
 *
 * A Scan is the atomic unit produced by the parser — one spectrum.
 * The parser emits Scan objects which the partition layer consumes.
 */

/**
 * @typedef {Object} Scan
 * @property {string} id             mzML scan id (native id or "scan=N")
 * @property {number} index          0-based index within the file
 * @property {number} msLevel        1 for MS1, 2 for MS2, etc.
 * @property {number} retentionTime  seconds (converted from minutes if needed)
 * @property {number} scanStartTime  raw value from mzML
 * @property {number} totalIonCurrent
 * @property {number} basePeakMz
 * @property {number} basePeakIntensity
 * @property {number} lowestMz
 * @property {number} highestMz
 * @property {Float64Array|Float32Array} mz        m/z array
 * @property {Float64Array|Float32Array} intensity intensity array
 * @property {number} peakCount                    length of arrays
 * @property {number} charge                       0 if not specified
 * @property {"positive"|"negative"|"unknown"} polarity
 * @property {PrecursorInfo|null} precursor        for MS2+ scans
 * @property {string} [activationMethod]           CID, HCD, ETD for MS2
 * @property {number} [collisionEnergy]
 */

/**
 * @typedef {Object} PrecursorInfo
 * @property {number} isolationTargetMz
 * @property {number} isolationWindowLower
 * @property {number} isolationWindowUpper
 * @property {number} selectedIonMz
 * @property {number} selectedIonCharge
 * @property {number} selectedIonIntensity
 * @property {string|null} precursorScanId
 */

/**
 * @typedef {Object} Chromatogram
 * @property {string} id
 * @property {number} index
 * @property {Float64Array|Float32Array} time
 * @property {Float64Array|Float32Array} intensity
 */

/**
 * Build an empty Scan.
 * @returns {Scan}
 */
export function createScan() {
  return {
    id: "",
    index: -1,
    msLevel: 1,
    retentionTime: 0,
    scanStartTime: 0,
    totalIonCurrent: 0,
    basePeakMz: 0,
    basePeakIntensity: 0,
    lowestMz: 0,
    highestMz: 0,
    mz: null,
    intensity: null,
    peakCount: 0,
    charge: 0,
    polarity: "unknown",
    precursor: null,
    activationMethod: undefined,
    collisionEnergy: undefined,
  };
}

/**
 * Build an empty PrecursorInfo.
 * @returns {PrecursorInfo}
 */
export function createPrecursor() {
  return {
    isolationTargetMz: 0,
    isolationWindowLower: 0,
    isolationWindowUpper: 0,
    selectedIonMz: 0,
    selectedIonCharge: 0,
    selectedIonIntensity: 0,
    precursorScanId: null,
  };
}

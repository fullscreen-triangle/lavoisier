/**
 * Indexed mzML helpers.
 *
 * Many mzML files include a <indexList> section near the end that records
 * the byte offset of every <spectrum> and <chromatogram>. This lets us
 * fetch individual scans with HTTP Range requests or file.slice without
 * reading the preceding GB of data.
 *
 * Format (simplified):
 *   ...
 *   <indexList count="N">
 *     <index name="spectrum">
 *       <offset idRef="scan=1">12345</offset>
 *       <offset idRef="scan=2">67890</offset>
 *       ...
 *     </index>
 *     <index name="chromatogram">...</index>
 *   </indexList>
 *   <indexListOffset>12345678</indexListOffset>
 *   <fileChecksum>...</fileChecksum>
 *   </indexedmzML>
 */

/**
 * @typedef {Object} ScanIndex
 * @property {Map<string, number>} spectrumOffsets   scan id → byte offset
 * @property {Map<string, number>} chromatogramOffsets
 * @property {number} indexListOffset
 * @property {number} totalBytes
 */

const INDEX_LIST_LOCATOR = /<indexListOffset>\s*(\d+)\s*<\/indexListOffset>/;
const INDEX_BLOCK_START = /<indexList[^>]*>/;
const INDEX_BLOCK_END = /<\/indexList>/;

/**
 * Read the index from an indexed mzML file via its SourceFile.
 * Returns null if the file is not indexed.
 *
 * Strategy:
 *   1. Read the last 64 KB to find <indexListOffset>
 *   2. Range-read from that offset to EOF to get the full indexList
 *   3. Parse with a simple regex (the indexList is pure structure, no CDATA)
 *
 * @param {SourceFile} file
 * @returns {Promise<ScanIndex|null>}
 */
export async function readIndex(file) {
  const size = await file.ensureSize();
  if (!size) return null;

  const TAIL = Math.min(64 * 1024, size);
  const tailBytes = await file.range(size - TAIL, size);
  const tail = new TextDecoder("utf-8").decode(tailBytes);

  const locatorMatch = tail.match(INDEX_LIST_LOCATOR);
  if (!locatorMatch) return null;
  const indexListOffset = parseInt(locatorMatch[1], 10);
  if (!Number.isFinite(indexListOffset) || indexListOffset >= size) {
    return null;
  }

  // Fetch the indexList block
  const blockBytes = await file.range(indexListOffset, size);
  const block = new TextDecoder("utf-8").decode(blockBytes);
  return parseIndexList(block, size, indexListOffset);
}

/**
 * Parse an <indexList> XML fragment.
 * @param {string} xml
 * @param {number} totalBytes
 * @param {number} indexListOffset
 * @returns {ScanIndex}
 */
export function parseIndexList(xml, totalBytes, indexListOffset) {
  const spectrumOffsets = new Map();
  const chromatogramOffsets = new Map();

  const startIdx = xml.search(INDEX_BLOCK_START);
  const endIdx = xml.search(INDEX_BLOCK_END);
  const fragment =
    startIdx >= 0 && endIdx >= 0 ? xml.substring(startIdx, endIdx) : xml;

  // Split into named <index> blocks
  const indexBlocks = fragment.split(/<index\s+name=/i);
  for (const block of indexBlocks) {
    if (!block.trim()) continue;

    const nameMatch = block.match(/^"([^"]+)"/);
    const name = nameMatch ? nameMatch[1] : "unknown";
    const target = name === "spectrum" ? spectrumOffsets : name === "chromatogram" ? chromatogramOffsets : null;
    if (!target) continue;

    // Parse <offset idRef="...">NNN</offset> entries
    const offsetRegex = /<offset\s+idRef="([^"]+)"[^>]*>\s*(\d+)\s*<\/offset>/g;
    let m;
    while ((m = offsetRegex.exec(block)) !== null) {
      target.set(m[1], parseInt(m[2], 10));
    }
  }

  return {
    spectrumOffsets,
    chromatogramOffsets,
    indexListOffset,
    totalBytes,
  };
}

/**
 * Compute the byte-range for a scan given its offset and the next offset.
 * @param {ScanIndex} index
 * @param {string} scanId
 * @returns {{start: number, end: number}|null}
 */
export function scanByteRange(index, scanId) {
  const start = index.spectrumOffsets.get(scanId);
  if (start == null) return null;

  // Find the next higher offset (from any index) to bound the read
  let next = index.indexListOffset;
  for (const off of index.spectrumOffsets.values()) {
    if (off > start && off < next) next = off;
  }
  for (const off of index.chromatogramOffsets.values()) {
    if (off > start && off < next) next = off;
  }

  return { start, end: next };
}

/**
 * Fetch a single scan's XML bytes using the index.
 * Returns the raw bytes; pair with a mini SAX parser to extract one spectrum.
 * @param {SourceFile} file
 * @param {ScanIndex} index
 * @param {string} scanId
 * @returns {Promise<Uint8Array|null>}
 */
export async function fetchScanBytes(file, index, scanId) {
  const range = scanByteRange(index, scanId);
  if (!range) return null;
  return await file.range(range.start, range.end);
}

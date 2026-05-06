/**
 * Unified source dispatcher.
 *
 * Given user input (URL, accession, or nothing), figure out which adapter
 * to use and return a Source. Keeps the UI layer simple.
 */

export {
  isLocalFolderSupported,
  isFilePickerSupported,
  pickLocalFolder,
  pickLocalFiles,
  createLocalSource,
  createSourceFromFiles,
  extractFilesFromDataTransfer,
} from "./local";
export { createRemoteSource, probeRemote, isReachable } from "./remote";
export * as metabolights from "./adapters/metabolights";
export * as massive from "./adapters/massive";
export * as zenodo from "./adapters/zenodo";
export { isSupportedMsFile, isParseable, MS_EXTENSIONS } from "./types";

import { parseAccession as parseMtbls, createMetaboLightsSource } from "./adapters/metabolights";
import { parseAccession as parseMsv, createMassiveSource } from "./adapters/massive";
import { parseRecordId as parseZenodo, createZenodoSource } from "./adapters/zenodo";
import { createRemoteSource } from "./remote";

/**
 * Inspect user input and route to the right adapter.
 *
 * Order of checks:
 *   1. MetaboLights accession (MTBLSxxxx)
 *   2. MassIVE accession (MSVxxxxxx)
 *   3. Zenodo record (number, DOI, or zenodo.org URL)
 *   4. Direct URL(s), comma- or newline-separated
 *
 * @param {string} input
 * @returns {Promise<Source>}
 */
export async function createSourceFromInput(input) {
  const trimmed = (input || "").trim();
  if (!trimmed) throw new Error("Empty input");

  // MetaboLights
  if (parseMtbls(trimmed)) {
    return await createMetaboLightsSource(trimmed);
  }

  // MassIVE — accession alone is insufficient, but we still route here
  // so the UI can prompt for explicit file URLs.
  if (parseMsv(trimmed)) {
    throw new Error(
      `MassIVE accessions require explicit file URLs. Paste the direct download URLs.`
    );
  }

  // Zenodo
  if (parseZenodo(trimmed)) {
    return await createZenodoSource(trimmed);
  }

  // Direct URLs
  const urls = trimmed
    .split(/[\n,]/)
    .map((s) => s.trim())
    .filter((s) => /^https?:\/\//.test(s));

  if (urls.length > 0) {
    return await createRemoteSource(urls, urls.length === 1 ? urls[0] : `${urls.length} URLs`);
  }

  throw new Error(
    `Unrecognised input. Provide a MetaboLights accession (MTBLSxxxx), a Zenodo DOI/ID, or direct HTTPS URLs.`
  );
}

/**
 * Detect what kind of input the user has pasted (for UI hints).
 * @param {string} input
 * @returns {"metabolights"|"massive"|"zenodo"|"url"|"unknown"}
 */
export function detectInputKind(input) {
  const trimmed = (input || "").trim();
  if (!trimmed) return "unknown";
  if (parseMtbls(trimmed)) return "metabolights";
  if (parseMsv(trimmed)) return "massive";
  if (parseZenodo(trimmed)) return "zenodo";
  if (/^https?:\/\//.test(trimmed)) return "url";
  return "unknown";
}

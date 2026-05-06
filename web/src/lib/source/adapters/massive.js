/**
 * MassIVE (UCSD) repository adapter.
 *
 * MassIVE hosts public proteomics/metabolomics datasets. Dataset IDs are
 * MSV000XXXXXX. File listing and downloads are served via:
 *   https://massive.ucsd.edu/ProteoSAFe/datasets_files.jsp?task={TASK_ID}
 *   ftp://massive.ucsd.edu/{MSV_ID}/...
 *
 * MassIVE's JSP-based API is less modern than MetaboLights but we can work
 * with it. For CORS-restricted content, a small proxy (Cloudflare Worker)
 * may be required.
 */

import { createRemoteSource } from "../remote";
import { isSupportedMsFile } from "../types";

/**
 * Parse a MassIVE accession.
 * Accepts: "MSV000087855", full URL
 */
export function parseAccession(input) {
  const trimmed = (input || "").trim();
  const match = trimmed.match(/MSV\d+/i);
  return match ? match[0].toUpperCase() : null;
}

/**
 * For now, MassIVE support accepts a user-provided file list URL
 * (since their public API is heterogeneous). In a full implementation,
 * this would call their FTP listing endpoint.
 *
 * @param {string} accession
 * @param {string[]} [fileUrls]  optional explicit URLs
 * @returns {Promise<Source>}
 */
export async function createMassiveSource(accession, fileUrls = []) {
  const normalised = parseAccession(accession);
  if (!normalised) {
    throw new Error(`Invalid MassIVE accession: ${accession}`);
  }

  // Filter to MS files
  const msUrls = fileUrls.filter((url) => {
    try {
      const name = new URL(url).pathname.split("/").pop();
      return isSupportedMsFile(name);
    } catch {
      return false;
    }
  });

  if (msUrls.length === 0) {
    throw new Error(
      `No MS files provided for ${normalised}. MassIVE's anonymous FTP may block browser access; paste explicit file URLs.`
    );
  }

  const src = await createRemoteSource(msUrls, `MassIVE · ${normalised}`);
  src.kind = "repository";
  src.meta = {
    ...src.meta,
    repository: "massive",
    accession: normalised,
  };
  return src;
}

export function datasetUrl(accession) {
  const a = parseAccession(accession);
  if (!a) return null;
  return `https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=${a}`;
}

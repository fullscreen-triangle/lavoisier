/**
 * MetaboLights (EBI) repository adapter.
 *
 * MetaboLights hosts public metabolomics studies. Every study has a stable
 * ID like MTBLS1707 and files accessible at:
 *   https://www.ebi.ac.uk/metabolights/ws/studies/{ID}/files
 *   ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/{ID}/
 *
 * The REST API returns JSON listing all files with sizes and types.
 * Files themselves are served over HTTPS with CORS headers.
 */

import { createRemoteSource } from "../remote";
import { isSupportedMsFile } from "../types";

const API_BASE = "https://www.ebi.ac.uk/metabolights/ws/studies";
const FILE_BASE = "https://www.ebi.ac.uk/metabolights/ws/studies";

/**
 * Parse a MetaboLights accession from user input.
 * Accepts: "MTBLS1707", "mtbls1707", full URL
 * @param {string} input
 * @returns {string|null} normalised accession or null
 */
export function parseAccession(input) {
  const trimmed = (input || "").trim();
  const match = trimmed.match(/MTBLS\d+/i);
  return match ? match[0].toUpperCase() : null;
}

/**
 * Fetch study metadata + file list for a MetaboLights accession.
 * @param {string} accession  e.g. "MTBLS1707"
 * @returns {Promise<{accession: string, title: string, files: Array}>}
 */
export async function fetchStudy(accession) {
  const normalised = parseAccession(accession);
  if (!normalised) {
    throw new Error(`Invalid MetaboLights accession: ${accession}`);
  }

  // Files endpoint returns a JSON array of {file, size, type, ...}
  const filesUrl = `${API_BASE}/${normalised}/files`;
  const filesRes = await fetch(filesUrl);
  if (!filesRes.ok) {
    throw new Error(
      `MetaboLights: failed to fetch ${normalised} file list (HTTP ${filesRes.status})`
    );
  }
  const filesData = await filesRes.json();

  // Try to fetch study metadata for the title (optional)
  let title = normalised;
  try {
    const metaRes = await fetch(`${API_BASE}/${normalised}`);
    if (metaRes.ok) {
      const meta = await metaRes.json();
      title = meta?.content?.title || meta?.title || normalised;
    }
  } catch {
    /* title is optional */
  }

  // Filter to MS data files
  const all = Array.isArray(filesData) ? filesData : filesData?.study?.files || [];
  const files = all
    .map((f) => ({
      name: f.file || f.name || "",
      size: f.size || null,
      type: f.type || "",
    }))
    .filter((f) => isSupportedMsFile(f.name));

  return { accession: normalised, title, files };
}

/**
 * Build a Source for a MetaboLights study.
 * @param {string} accession
 * @returns {Promise<Source>}
 */
export async function createMetaboLightsSource(accession) {
  const study = await fetchStudy(accession);

  const urls = study.files.map(
    (f) => `${FILE_BASE}/${study.accession}/download/${encodeURIComponent(f.name)}`
  );

  const src = await createRemoteSource(urls, `MetaboLights · ${study.accession}`);
  src.kind = "repository";
  src.meta = {
    ...src.meta,
    repository: "metabolights",
    accession: study.accession,
    title: study.title,
    fileCount: study.files.length,
  };

  return src;
}

/**
 * Build a direct study URL for display.
 * @param {string} accession
 */
export function studyUrl(accession) {
  const a = parseAccession(accession);
  if (!a) return null;
  return `https://www.ebi.ac.uk/metabolights/${a}`;
}

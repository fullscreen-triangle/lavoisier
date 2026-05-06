/**
 * Zenodo repository adapter.
 *
 * Zenodo is a generic scientific data repository at CERN. Records have DOIs
 * and public REST API:
 *   https://zenodo.org/api/records/{RECORD_ID}
 *
 * Files are served with CORS + Range support. Clean, modern API.
 */

import { createRemoteSource } from "../remote";
import { isSupportedMsFile } from "../types";

const API_BASE = "https://zenodo.org/api/records";

/**
 * Parse a Zenodo record ID from user input.
 * Accepts: "7654321", DOI "10.5281/zenodo.7654321", full URL
 */
export function parseRecordId(input) {
  const trimmed = (input || "").trim();
  // Full URL form
  const urlMatch = trimmed.match(/zenodo\.org\/records?\/(\d+)/i);
  if (urlMatch) return urlMatch[1];
  // DOI form
  const doiMatch = trimmed.match(/10\.5281\/zenodo\.(\d+)/i);
  if (doiMatch) return doiMatch[1];
  // Bare number
  if (/^\d+$/.test(trimmed)) return trimmed;
  return null;
}

/**
 * Fetch record metadata and file listing.
 * @param {string} id
 */
export async function fetchRecord(id) {
  const recordId = parseRecordId(id);
  if (!recordId) throw new Error(`Invalid Zenodo record: ${id}`);

  const res = await fetch(`${API_BASE}/${recordId}`);
  if (!res.ok) {
    throw new Error(`Zenodo: fetch record ${recordId} → HTTP ${res.status}`);
  }
  const data = await res.json();

  const files = (data.files || [])
    .map((f) => ({
      name: f.key || f.filename || "",
      size: f.size || null,
      url: f.links?.self || f.links?.download || null,
    }))
    .filter((f) => f.url && isSupportedMsFile(f.name));

  return {
    id: recordId,
    title: data.metadata?.title || `Zenodo ${recordId}`,
    doi: data.doi || `10.5281/zenodo.${recordId}`,
    files,
  };
}

/**
 * Build a Source for a Zenodo record.
 * @param {string} id
 * @returns {Promise<Source>}
 */
export async function createZenodoSource(id) {
  const record = await fetchRecord(id);
  if (record.files.length === 0) {
    throw new Error(`Zenodo record ${record.id} contains no supported MS files`);
  }

  const urls = record.files.map((f) => f.url);
  const src = await createRemoteSource(urls, `Zenodo · ${record.id}`);
  src.kind = "repository";
  src.meta = {
    ...src.meta,
    repository: "zenodo",
    recordId: record.id,
    title: record.title,
    doi: record.doi,
  };
  return src;
}

export function recordUrl(id) {
  const r = parseRecordId(id);
  return r ? `https://zenodo.org/records/${r}` : null;
}

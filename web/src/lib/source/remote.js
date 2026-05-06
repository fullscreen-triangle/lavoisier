/**
 * Remote URL source — direct HTTP(S) with Range requests.
 *
 * Works with any CORS-enabled URL serving mzML. The HEAD request gets the
 * size; Range requests let us read individual scans from indexed files
 * without downloading the whole file.
 *
 * For repository-specific integration see adapters/ (MetaboLights, MassIVE, etc.)
 */

import { isSupportedMsFile, isParseable } from "./types";

/**
 * Create a Source from one or more remote URLs.
 * @param {string[]} urls
 * @param {string} [label]
 * @returns {Promise<Source>}
 */
export async function createRemoteSource(urls, label = "Remote URLs") {
  const files = await Promise.all(urls.map(makeRemoteFile));
  const filtered = files.filter(Boolean);

  return {
    id: `remote:${Date.now()}`,
    label,
    kind: "remote",
    meta: { urls },

    async listFiles() {
      return filtered;
    },
  };
}

/**
 * HEAD request to get size + validate range support.
 * @param {string} url
 * @returns {Promise<{size: number|null, acceptRanges: boolean}>}
 */
export async function probeRemote(url) {
  const res = await fetch(url, { method: "HEAD" });
  if (!res.ok) {
    throw new Error(`Remote ${url}: HTTP ${res.status}`);
  }
  const size = res.headers.get("content-length");
  const ranges = res.headers.get("accept-ranges");
  return {
    size: size ? parseInt(size, 10) : null,
    acceptRanges: ranges === "bytes",
    contentType: res.headers.get("content-type") || "",
  };
}

/**
 * Build a SourceFile for a remote URL.
 * @param {string} url
 * @returns {Promise<SourceFile|null>}
 */
async function makeRemoteFile(url) {
  const name = filenameFromUrl(url);
  if (!isSupportedMsFile(name)) {
    console.warn(`Skipping unsupported remote file: ${name}`);
    return null;
  }

  let probe;
  try {
    probe = await probeRemote(url);
  } catch (err) {
    console.warn(`HEAD failed for ${url}:`, err);
    probe = { size: null, acceptRanges: false };
  }

  return {
    id: `remote:${url}`,
    name,
    path: url,
    size: probe.size,
    kind: "remote",
    meta: { url, acceptRanges: probe.acceptRanges },

    async stream() {
      const res = await fetch(url);
      if (!res.ok || !res.body) {
        throw new Error(`Failed to stream ${url}: HTTP ${res.status}`);
      }
      return res.body;
    },

    async range(start, end) {
      if (!probe.acceptRanges) {
        // Fallback: fetch whole file, slice client-side.
        // Used only if server doesn't support Range.
        const res = await fetch(url);
        const buf = new Uint8Array(await res.arrayBuffer());
        return buf.slice(start, end);
      }
      const headers = { Range: `bytes=${start}-${end - 1}` };
      const res = await fetch(url, { headers });
      if (!res.ok && res.status !== 206) {
        throw new Error(`Range ${start}-${end} failed for ${url}: HTTP ${res.status}`);
      }
      const buf = await res.arrayBuffer();
      return new Uint8Array(buf);
    },

    async ensureSize() {
      return probe.size;
    },
  };
}

/**
 * Extract a filename from a URL.
 * @param {string} url
 */
function filenameFromUrl(url) {
  try {
    const u = new URL(url);
    const parts = u.pathname.split("/").filter(Boolean);
    return parts[parts.length - 1] || u.hostname;
  } catch {
    return url.substring(url.lastIndexOf("/") + 1) || url;
  }
}

/**
 * Check whether a URL is reachable with CORS.
 * @param {string} url
 * @returns {Promise<boolean>}
 */
export async function isReachable(url) {
  try {
    const res = await fetch(url, { method: "HEAD", mode: "cors" });
    return res.ok;
  } catch {
    return false;
  }
}

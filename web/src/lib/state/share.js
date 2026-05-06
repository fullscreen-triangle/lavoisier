/**
 * URL sharing helpers.
 *
 * The ternary address IS the molecular identity, so a shareable link
 * just needs the address. Optionally include the analyser and
 * source reference so the recipient lands in the same context.
 *
 * URL schema: /tool?addr=202222112122&analyser=orbitrap&src=mtbls:1707
 */

/**
 * Build a query string for the current observation.
 *
 * @param {Object} args
 * @param {string|null} [args.address]
 * @param {string|null} [args.analyser]
 * @param {Object|null} [args.source]   {kind, accession, recordId, urls}
 * @returns {string}                    starts with "?" or empty
 */
export function buildShareQuery({ address, analyser, source }) {
  const params = new URLSearchParams();
  if (address) params.set("addr", address);
  if (analyser) params.set("analyser", analyser);

  if (source) {
    if (source.kind === "repository" && source.meta?.repository === "metabolights") {
      params.set("src", `mtbls:${source.meta.accession}`);
    } else if (source.kind === "repository" && source.meta?.repository === "zenodo") {
      params.set("src", `zenodo:${source.meta.recordId}`);
    } else if (source.kind === "remote" && source.meta?.urls?.length) {
      params.set("src", `url:${source.meta.urls[0]}`);
    } else if (source.kind === "local") {
      params.set("src", `local:${source.label || ""}`);
    }
  }
  const qs = params.toString();
  return qs ? `?${qs}` : "";
}

/**
 * Parse a query string into a workspace state seed.
 *
 * @param {URLSearchParams|string} input
 * @returns {{address: string|null, analyser: string|null, source: Object|null}}
 */
export function parseShareQuery(input) {
  const params =
    input instanceof URLSearchParams
      ? input
      : new URLSearchParams(input || "");

  const address = params.get("addr");
  const analyser = params.get("analyser");
  const src = params.get("src");

  let source = null;
  if (src) {
    if (src.startsWith("mtbls:")) {
      source = { kind: "metabolights", accession: src.substring(6) };
    } else if (src.startsWith("zenodo:")) {
      source = { kind: "zenodo", recordId: src.substring(7) };
    } else if (src.startsWith("url:")) {
      source = { kind: "url", url: src.substring(4) };
    } else if (src.startsWith("local:")) {
      source = { kind: "local", hint: src.substring(6) };
    }
  }

  return { address, analyser, source };
}

/**
 * Build a full sharable URL for the current workspace state.
 *
 * @param {string} origin   e.g. "https://lavoisier.bitspark.com"
 * @param {Object} args     same as buildShareQuery
 */
export function buildShareUrl(origin, args) {
  return `${origin}/tool${buildShareQuery(args)}`;
}

/**
 * Copy a string to clipboard with graceful fallback.
 * @param {string} text
 * @returns {Promise<boolean>}
 */
export async function copyToClipboard(text) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    document.body.removeChild(textarea);
    return true;
  } catch {
    return false;
  }
}

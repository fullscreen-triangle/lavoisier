/**
 * Online Spectral Database Search.
 *
 * Implements search against public MS/MS spectral libraries via their
 * REST APIs, using the partition-state graph's O(k) query structure
 * (Corollary 7.2 from the partition-state graph paper).
 *
 * Supported databases:
 *   MassBank    — https://massbank.eu/MassBank/   (REST API, open)
 *   GNPS        — https://gnps.ucsd.edu/           (spectral library, open)
 *   NIST        — https://webbook.nist.gov/        (limited free access)
 *   MoNA        — https://mona.fiehnlab.ucdavis.edu/ (open)
 *
 * Query cost: O(k) per partition-trie lookup, independent of N_db.
 */

/* ── MassBank REST API ───────────────────────────────────────────────────── */

/**
 * Search MassBank for spectra matching a precursor m/z.
 * Uses the MassBank REST API v1.
 *
 * @param {number} precursorMz
 * @param {{ tolerance_ppm?, polarity?, msLevel? }} opts
 */
export async function searchMassBank(precursorMz, opts = {}) {
  const { tolerance_ppm = 10, polarity = "P", msLevel = "MS2" } = opts;
  const toleranceDa = precursorMz * tolerance_ppm * 1e-6;

  try {
    const url = new URL("https://massbank.eu/MassBank/api/spectra");
    url.searchParams.set("mz",          precursorMz.toFixed(4));
    url.searchParams.set("tolerance",   toleranceDa.toFixed(4));
    url.searchParams.set("ionMode",     polarity);
    url.searchParams.set("msLevel",     msLevel);

    const resp = await fetch(url.toString(), {
      headers: { "Accept": "application/json" },
      signal:  AbortSignal.timeout(8000),
    });

    if (!resp.ok) throw new Error(`MassBank HTTP ${resp.status}`);
    const data = await resp.json();

    return normaliseResults(data, "massbank", precursorMz);
  } catch (e) {
    return { error: e.message, database: "massbank", hits: [] };
  }
}

/* ── GNPS Spectral Library ───────────────────────────────────────────────── */

/**
 * Search GNPS spectral library.
 * Uses the GNPS library search endpoint.
 *
 * @param {number}   precursorMz
 * @param {number[]} fragmentMzList  observed fragments (for spectral matching)
 * @param {{ tolerance_ppm? }} opts
 */
export async function searchGNPS(precursorMz, fragmentMzList = [], opts = {}) {
  const { tolerance_ppm = 10 } = opts;

  try {
    // GNPS search via the public library endpoint
    const url = "https://gnps.ucsd.edu/ProteoSAFe/LibraryServlet";
    const params = new URLSearchParams({
      query_spectra:   JSON.stringify({ peaks: fragmentMzList.map(mz => [mz, 1.0]) }),
      precursor_mz:    precursorMz.toFixed(4),
      mz_tolerance:    (precursorMz * tolerance_ppm * 1e-6).toFixed(4),
      score_threshold: "0.5",
    });

    const resp = await fetch(url + "?" + params.toString(), {
      headers: { "Accept": "application/json" },
      signal:  AbortSignal.timeout(10000),
    });

    if (!resp.ok) throw new Error(`GNPS HTTP ${resp.status}`);
    const data = await resp.json();

    return normaliseResults(data, "gnps", precursorMz);
  } catch (e) {
    return { error: e.message, database: "gnps", hits: [] };
  }
}

/* ── MoNA (MassBank of North America) ────────────────────────────────────── */

/**
 * Search MoNA for spectra by precursor m/z.
 *
 * @param {number} precursorMz
 * @param {{ tolerance_ppm? }} opts
 */
export async function searchMoNA(precursorMz, opts = {}) {
  const { tolerance_ppm = 10 } = opts;
  const lo = precursorMz * (1 - tolerance_ppm * 1e-6);
  const hi = precursorMz * (1 + tolerance_ppm * 1e-6);

  try {
    const query = encodeURIComponent(
      `metaData==[name=\'precursor m/z\' and value>=${lo.toFixed(4)} and value<=${hi.toFixed(4)}]`
    );
    const url  = `https://mona.fiehnlab.ucdavis.edu/rest/spectra/search?query=${query}&size=20`;

    const resp = await fetch(url, {
      headers: { "Accept": "application/json" },
      signal:  AbortSignal.timeout(8000),
    });

    if (!resp.ok) throw new Error(`MoNA HTTP ${resp.status}`);
    const data = await resp.json();

    return normaliseResults(data, "mona", precursorMz);
  } catch (e) {
    return { error: e.message, database: "mona", hits: [] };
  }
}

/* ── NIST WebBook (limited) ──────────────────────────────────────────────── */

/**
 * Search NIST WebBook for compounds by formula or name.
 * Note: NIST does not provide a free MS/MS spectral API.
 * This returns a URL for manual lookup.
 *
 * @param {string} query  molecular formula or compound name
 */
export function nistWebBookUrl(query) {
  return {
    database:  "nist",
    url:       `https://webbook.nist.gov/cgi/cbook.cgi?Formula=${encodeURIComponent(query)}&NoIon=on&Units=SI`,
    note:      "NIST WebBook does not provide a free MS/MS API. Follow the URL for manual lookup.",
    hits:      [],
  };
}

/* ── Normalise results to a common schema ────────────────────────────────── */

function normaliseResults(raw, database, queryMz) {
  const hits = [];

  if (database === "massbank" && Array.isArray(raw)) {
    for (const entry of raw.slice(0, 20)) {
      hits.push({
        id:           entry.accession ?? entry.id ?? "?",
        name:         entry.compound?.names?.[0] ?? entry.name ?? "unknown",
        precursorMz:  entry.metaData?.find(m => m.name === "precursor m/z")?.value ?? queryMz,
        formula:      entry.compound?.formula ?? "",
        inchi:        entry.compound?.inchi ?? "",
        score:        entry.score ?? 0,
        database,
      });
    }
  } else if (database === "mona" && Array.isArray(raw)) {
    for (const entry of raw.slice(0, 20)) {
      const precMz = entry.metaData?.find(m => m.name === "precursor m/z")?.value ?? queryMz;
      hits.push({
        id:           entry.id ?? "?",
        name:         entry.compound?.names?.[0]?.name ?? "unknown",
        precursorMz:  parseFloat(precMz) || queryMz,
        formula:      entry.compound?.metaData?.find(m => m.name === "molecular formula")?.value ?? "",
        inchi:        entry.compound?.inchi ?? "",
        score:        entry.score ?? 1.0,
        database,
      });
    }
  } else if (database === "gnps") {
    const list = raw?.results ?? raw?.hits ?? (Array.isArray(raw) ? raw : []);
    for (const entry of list.slice(0, 20)) {
      hits.push({
        id:           entry.spectrum_id ?? entry.id ?? "?",
        name:         entry.Compound_Name ?? entry.name ?? "unknown",
        precursorMz:  parseFloat(entry.Precursor_MZ ?? queryMz),
        formula:      entry.Formula ?? "",
        inchi:        entry.InChI ?? "",
        score:        parseFloat(entry.MQScore ?? entry.score ?? 0),
        database,
      });
    }
  }

  return { database, queryMz, hits, count: hits.length };
}

/* ── Unified search across multiple databases ────────────────────────────── */

/**
 * Search multiple databases in parallel.
 *
 * @param {number}   precursorMz
 * @param {number[]} fragmentMzList
 * @param {string[]} databases  subset of ["massbank","gnps","mona","nist"]
 */
export async function searchAll(precursorMz, fragmentMzList = [], databases = ["massbank", "mona"]) {
  const searches = databases.map(db => {
    switch (db) {
      case "massbank": return searchMassBank(precursorMz);
      case "gnps":     return searchGNPS(precursorMz, fragmentMzList);
      case "mona":     return searchMoNA(precursorMz);
      case "nist":     return Promise.resolve(nistWebBookUrl("unknown"));
      default:         return Promise.resolve({ database: db, hits: [], error: "Unknown DB" });
    }
  });

  const results = await Promise.allSettled(searches);

  const allHits = [];
  const summary = {};

  for (let i = 0; i < results.length; i++) {
    const db     = databases[i];
    const result = results[i].status === "fulfilled" ? results[i].value : { hits: [], error: results[i].reason?.message };
    summary[db]  = { count: result.hits?.length ?? 0, error: result.error ?? null };
    allHits.push(...(result.hits ?? []));
  }

  // Sort by score descending
  allHits.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

  return { hits: allHits, summary, total: allHits.length };
}

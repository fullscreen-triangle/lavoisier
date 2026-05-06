/**
 * Export utilities — CSV / JSON / address-list dumps.
 *
 * Output is generated entirely client-side (no server round-trip)
 * and triggered as a Blob download.
 */

/**
 * Trigger a browser download for arbitrary text content.
 *
 * @param {string} content
 * @param {string} filename
 * @param {string} [mime]
 */
export function downloadText(content, filename, mime = "text/plain;charset=utf-8") {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 100);
}

/**
 * Convert CategoricalStates to CSV.
 *
 * Columns: scanId, msLevel, retentionTime, charge, polarity,
 *          basePeakMz, basePeakIntensity, totalIonCurrent,
 *          Sk, St, Se, address, nPeaks, observable
 *
 * @param {CategoricalState[]} states
 * @returns {string}
 */
export function statesToCsv(states) {
  if (states.length === 0) return "";

  const cols = [
    "scanId",
    "msLevel",
    "retentionTime",
    "charge",
    "polarity",
    "basePeakMz",
    "basePeakIntensity",
    "totalIonCurrent",
    "nPeaks",
    "Sk",
    "St",
    "Se",
    "address",
    "hierAddress",
    "nMax",
    "occupiedCells",
    "partitionEntropy",
    "analyser",
    "observableValue",
  ];

  const lines = [cols.join(",")];
  for (const s of states) {
    const obs = s.observables || {};
    const obsKeys = ["T", "frequencyHz", "omega", "omegaC", "q"];
    const observableValue =
      obsKeys.map((k) => obs[k]).find((v) => v != null) ?? "";
    const obsName = obs.observable || "";
    const h = s.hierarchy;
    lines.push([
      escapeCsv(s.scanId),
      s.msLevel,
      s.retentionTime,
      s.charge,
      escapeCsv(s.polarity),
      s.basePeakMz,
      s.basePeakIntensity,
      s.totalIonCurrent,
      s.nPeaks,
      s.sentropy.sk.toFixed(6),
      s.sentropy.st.toFixed(6),
      s.sentropy.se.toFixed(6),
      escapeCsv(s.address),
      escapeCsv(h?.address || ""),
      h?.nMax ?? "",
      h?.occupiedCells ?? "",
      h?.entropyNats != null ? h.entropyNats.toFixed(6) : "",
      escapeCsv(obsName),
      observableValue,
    ].join(","));
  }
  return lines.join("\n");
}

function escapeCsv(value) {
  if (value == null) return "";
  const s = String(value);
  if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

/**
 * Convert CategoricalStates to JSON-compatible objects.
 * Strips internal references (Float arrays etc.) so it's safe to JSON.stringify.
 *
 * @param {CategoricalState[]} states
 * @returns {Object[]}
 */
export function statesToJson(states) {
  return states.map((s) => ({
    scanId: s.scanId,
    msLevel: s.msLevel,
    retentionTime: s.retentionTime,
    charge: s.charge,
    polarity: s.polarity,
    basePeakMz: s.basePeakMz,
    basePeakIntensity: s.basePeakIntensity,
    totalIonCurrent: s.totalIonCurrent,
    nPeaks: s.nPeaks,
    sentropy: { ...s.sentropy },
    address: s.address,
    hierarchy: s.hierarchy
      ? {
          nMax: s.hierarchy.nMax,
          occupiedCells: s.hierarchy.occupiedCells,
          totalCells: s.hierarchy.totalCells,
          shells: s.hierarchy.shells,
          entropyNats: s.hierarchy.entropyNats,
          address: s.hierarchy.address,
          oscillators: s.hierarchy.oscillators,
        }
      : null,
    precursor: s.precursor || null,
    observables: s.observables || null,
    partitionInertia: s.partitionInertia,
  }));
}

/**
 * Export just the unique ternary addresses (for sharing / re-querying).
 * @param {CategoricalState[]} states
 * @returns {string} newline-separated addresses
 */
export function statesToAddressList(states) {
  const seen = new Set();
  const out = [];
  for (const s of states) {
    if (!seen.has(s.address)) {
      seen.add(s.address);
      out.push(s.address);
    }
  }
  return out.join("\n");
}

/**
 * Trigger a CSV download.
 * @param {CategoricalState[]} states
 * @param {string} [filename]
 */
export function downloadCsv(states, filename = "lavoisier-results.csv") {
  downloadText(statesToCsv(states), filename, "text/csv;charset=utf-8");
}

/**
 * Trigger a JSON download.
 * @param {CategoricalState[]} states
 * @param {Object} [meta]                     optional metadata header
 * @param {string} [filename]
 */
export function downloadJson(states, meta = {}, filename = "lavoisier-results.json") {
  const payload = {
    framework: "Lavoisier",
    version: "0.1.0",
    generated: new Date().toISOString(),
    ...meta,
    stateCount: states.length,
    states: statesToJson(states),
  };
  downloadText(JSON.stringify(payload, null, 2), filename, "application/json");
}

/**
 * Trigger an address-list download.
 * @param {CategoricalState[]} states
 * @param {string} [filename]
 */
export function downloadAddressList(states, filename = "lavoisier-addresses.txt") {
  downloadText(statesToAddressList(states), filename);
}

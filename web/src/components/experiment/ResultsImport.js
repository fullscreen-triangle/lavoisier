import React, { useRef, useState } from "react";
import { useStore } from "@/lib/state/store";
import { summariseRecords } from "@/lib/experiment/virtualinstrument";
import { PALETTE } from "./cf/chartUtils";

/**
 * Normalise a record that came from the CLI (or any external source) so every
 * field the crossfilter expects is present and has the right shape.
 */
function normalise(r) {
  const analyteClass = r.analyteClass ?? r.class ?? "Unknown";
  const X = r.X ?? 0;
  const Y = r.Y ?? 0;
  const n = r.n ?? Math.max(1, Math.ceil(Math.sqrt((r.neutralMass ?? r.precursorMz ?? 0) / 162)));
  const sentropy = r.sentropy ?? { sk: 0, st: 0, se: 0 };
  return {
    analyte:       r.analyte       ?? `${analyteClass}(${X}:${Y})`,
    analyteClass,
    X,
    Y,
    composition:   r.composition   ?? {},
    neutralMass:   r.neutralMass   ?? r.precursorMz ?? 0,
    adduct:        r.adduct        ?? "[M+H]+",
    adductAbbr:    r.adductAbbr    ?? (r.adduct ?? "+H"),
    precursorMz:   r.precursorMz   ?? 0,
    z:             r.z             ?? 1,
    polarity:      r.polarity      ?? "+",
    intensity:     r.intensity     ?? 0,
    n,
    l:             r.l             ?? 0,
    m:             r.m             ?? 0,
    s:             r.s             ?? 0.5,
    sentropy,
    ternaryAddress:    r.ternaryAddress    ?? "",
    analyserMode:      r.analyserMode     ?? r.analyser ?? "orbitrap",
    observable:        r.observable       ?? null,
    shellDistribution: r.shellDistribution ?? {},
    partitionEntropy:  r.partitionEntropy  ?? 0,
    ms1:      r.ms1      ?? [],
    ms2:      r.ms2      ?? [],
    peaksAll: r.peaksAll ?? [],
    bitsTotal: r.bitsTotal ?? 0,
    sentropyVec: r.sentropyVec ?? sentropy,
  };
}

export default function ResultsImport() {
  const setRecords = useStore((s) => s.setExperimentRecords);
  const fileRef   = useRef(null);
  const [error,   setError]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [loaded,  setLoaded]  = useState(null);

  const onFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    setLoaded(null);
    setLoading(true);
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      const raw  = Array.isArray(data) ? data : (data.records ?? []);
      if (!Array.isArray(raw) || raw.length === 0) {
        throw new Error("no records found — expected { records: [...] } or a bare array");
      }
      const records = raw.map(normalise);
      const t0      = performance.now();
      const summary = summariseRecords(records);
      const dt      = performance.now() - t0;
      setRecords(records, summary, dt);
      setLoaded({ count: records.length, name: file.name });
    } catch (err) {
      setError(String(err?.message ?? err));
    } finally {
      setLoading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  return (
    <div className="space-y-1.5">
      <div className="text-[9px] uppercase tracking-wider"
        style={{ color: PALETTE.muted }}>
        Or load from CLI
      </div>
      <input
        ref={fileRef}
        type="file"
        accept=".json,.lavoisier.json"
        className="hidden"
        onChange={onFile}
      />
      <button
        onClick={() => fileRef.current?.click()}
        disabled={loading}
        className="w-full py-2 rounded text-[11px] tracking-wide border text-left px-3
          transition-opacity hover:opacity-80 disabled:opacity-40"
        style={{
          borderColor: PALETTE.grid,
          color:       PALETTE.text,
          background:  "transparent",
        }}
      >
        {loading ? "parsing…" : "↑  Load .lavoisier.json"}
      </button>

      {loaded && (
        <div className="text-[10px]" style={{ color: PALETTE.muted }}>
          {loaded.count.toLocaleString()} records ← {loaded.name}
        </div>
      )}
      {error && (
        <div className="text-[10px]" style={{ color: "#e66" }}>{error}</div>
      )}
    </div>
  );
}

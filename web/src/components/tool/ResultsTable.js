import React, { useMemo, useState } from "react";
import { useStore } from "@/lib/state/store";
import { downloadCsv, downloadJson, downloadAddressList } from "@/lib/state/export";

/**
 * ResultsTable — live stream of CategoricalStates as workers parse files.
 *
 * Selecting a row drives selectedAddress in the store, which the 3D
 * viewer and shader canvas both observe.
 */
export default function ResultsTable() {
  const states = useStore((s) => s.states);
  const totalScanCount = useStore((s) => s.totalScanCount);
  const selectedAddress = useStore((s) => s.selectedAddress);
  const selectAddress = useStore((s) => s.selectAddress);
  const analyser = useStore((s) => s.analyser);
  const source = useStore((s) => s.source);

  const [sortKey, setSortKey] = useState("retentionTime");
  const [sortDir, setSortDir] = useState("asc");
  const [search, setSearch] = useState("");

  const sorted = useMemo(() => {
    const filtered = search
      ? states.filter((s) =>
          (s.scanId || "").toLowerCase().includes(search.toLowerCase()) ||
          (s.address || "").startsWith(search)
        )
      : states;

    const key = sortKey;
    const dir = sortDir === "asc" ? 1 : -1;
    return [...filtered].sort((a, b) => {
      const av = pickKey(a, key);
      const bv = pickKey(b, key);
      if (av < bv) return -1 * dir;
      if (av > bv) return 1 * dir;
      return 0;
    }).slice(0, 500); // cap rendering for performance
  }, [states, search, sortKey, sortDir]);

  const toggleSort = (key) => {
    if (sortKey === key) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  };

  const handleExportCsv = () => downloadCsv(states, suggestedFilename("csv"));
  const handleExportJson = () =>
    downloadJson(
      states,
      {
        analyser,
        source: source ? { kind: source.kind, label: source.label } : null,
      },
      suggestedFilename("json")
    );
  const handleExportAddresses = () =>
    downloadAddressList(states, suggestedFilename("addresses.txt"));

  if (states.length === 0) {
    return (
      <div className="text-xs text-dark/40 dark:text-light/40 italic py-4">
        No observations yet. Process a file to begin.
      </div>
    );
  }

  return (
    <div className="space-y-2 flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs text-dark/70 dark:text-light/70">
          <strong>{states.length}</strong>
          {sorted.length < states.length && (
            <span className="text-dark/40 dark:text-light/40"> · {sorted.length} shown</span>
          )}
          {totalScanCount > states.length && (
            <span className="text-dark/40 dark:text-light/40"> / {totalScanCount}</span>
          )}
        </span>
        <div className="flex gap-1 text-[10px]">
          <button
            onClick={handleExportCsv}
            className="px-2 py-0.5 rounded border border-dark/10 dark:border-light/10
              hover:border-primary dark:hover:border-primaryDark"
            title="Export as CSV"
          >
            CSV
          </button>
          <button
            onClick={handleExportJson}
            className="px-2 py-0.5 rounded border border-dark/10 dark:border-light/10
              hover:border-primary dark:hover:border-primaryDark"
            title="Export as JSON"
          >
            JSON
          </button>
          <button
            onClick={handleExportAddresses}
            className="px-2 py-0.5 rounded border border-dark/10 dark:border-light/10
              hover:border-primary dark:hover:border-primaryDark"
            title="Export ternary addresses"
          >
            Addr
          </button>
        </div>
      </div>

      {/* Search */}
      <input
        type="text"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="Filter by scan ID or address prefix"
        className="w-full px-2 py-1 text-xs rounded border border-dark/10 dark:border-light/10
          bg-light dark:bg-dark focus:outline-none focus:border-primary dark:focus:border-primaryDark"
      />

      {/* Sort headers */}
      <div className="grid grid-cols-[80px_50px_60px_1fr_60px] gap-1 text-[10px] uppercase tracking-wider
        font-bold text-dark/60 dark:text-light/60 px-1">
        <SortHeader k="scanId" cur={sortKey} dir={sortDir} onClick={toggleSort}>Scan</SortHeader>
        <SortHeader k="msLevel" cur={sortKey} dir={sortDir} onClick={toggleSort}>MS</SortHeader>
        <SortHeader k="retentionTime" cur={sortKey} dir={sortDir} onClick={toggleSort}>RT</SortHeader>
        <SortHeader k="address" cur={sortKey} dir={sortDir} onClick={toggleSort}>Address</SortHeader>
        <SortHeader k="basePeakMz" cur={sortKey} dir={sortDir} onClick={toggleSort}>m/z</SortHeader>
      </div>

      {/* Rows */}
      <div className="flex-1 overflow-y-auto space-y-0.5 pr-1">
        {sorted.map((s) => (
          <ResultRow
            key={`${s.scanId}-${s.address}`}
            s={s}
            selected={s.address === selectedAddress}
            onClick={() =>
              selectAddress(s.address === selectedAddress ? null : s.address)
            }
          />
        ))}
      </div>
    </div>
  );
}

function SortHeader({ k, cur, dir, onClick, children }) {
  const active = cur === k;
  return (
    <button
      onClick={() => onClick(k)}
      className={`text-left ${active ? "text-primary dark:text-primaryDark" : ""}`}
    >
      {children}
      {active && (dir === "asc" ? " ↑" : " ↓")}
    </button>
  );
}

function ResultRow({ s, selected, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`w-full grid grid-cols-[80px_50px_60px_1fr_60px] gap-1 text-[10px] px-1 py-1 rounded
        text-left transition-colors
        ${
          selected
            ? "bg-primary/15 dark:bg-primaryDark/15 ring-1 ring-primary dark:ring-primaryDark"
            : "hover:bg-dark/5 dark:hover:bg-light/5"
        }`}
    >
      <span className="truncate text-dark/80 dark:text-light/80" title={s.scanId}>
        {shortScanId(s.scanId)}
      </span>
      <span className="font-mono">
        MS{s.msLevel}
      </span>
      <span className="font-mono text-dark/70 dark:text-light/70">
        {s.retentionTime.toFixed(2)}
      </span>
      <span className="font-mono text-primary dark:text-primaryDark truncate" title={s.address}>
        {s.address.substring(0, 12)}
      </span>
      <span className="font-mono text-right text-dark/70 dark:text-light/70">
        {s.basePeakMz.toFixed(2)}
      </span>
    </button>
  );
}

function shortScanId(id) {
  if (!id) return "";
  if (id.length <= 12) return id;
  // Many mzML scan ids are like "controllerType=0 controllerNumber=1 scan=42"
  // or "scanId=spectrum=42" — keep just the trailing number-like portion
  const m = id.match(/scan=(\d+)/) || id.match(/(\d+)\s*$/);
  return m ? `s${m[1]}` : id.substring(0, 12);
}

function pickKey(s, key) {
  switch (key) {
    case "scanId":        return s.scanId || "";
    case "msLevel":       return s.msLevel || 0;
    case "retentionTime": return s.retentionTime || 0;
    case "address":       return s.address || "";
    case "basePeakMz":    return s.basePeakMz || 0;
    default:              return 0;
  }
}

function suggestedFilename(ext) {
  const ts = new Date().toISOString().replace(/[:.]/g, "-").substring(0, 19);
  return `lavoisier-${ts}.${ext}`;
}

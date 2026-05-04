import React, { useState, useMemo } from "react";
import { useStore } from "@/lib/state/store";
import { classColor } from "./charts/palette";

/**
 * Searchable, sortable, filterable list of predicted records.
 * Compact rows; row click selects the record for the detail panel.
 */
export default function RecordsTable() {
  const records = useStore((s) => s.experimentRecords);
  const selectedId = useStore((s) => s.selectedRecordId);
  const select = useStore((s) => s.selectRecord);

  const [query, setQuery] = useState("");
  const [classFilter, setClassFilter] = useState("all");
  const [sortBy, setSortBy] = useState("intensity");
  const [sortDir, setSortDir] = useState(-1);

  const classes = useMemo(
    () => Array.from(new Set(records.map((r) => r.analyteClass))),
    [records]
  );

  const filtered = useMemo(() => {
    let r = records;
    if (classFilter !== "all") r = r.filter((x) => x.analyteClass === classFilter);
    if (query) {
      const q = query.toLowerCase();
      r = r.filter((x) => x.analyte.toLowerCase().includes(q) || x.adduct.includes(q));
    }
    r = [...r].sort((a, b) => sortDir * ((a[sortBy] ?? 0) - (b[sortBy] ?? 0)));
    return r.slice(0, 600);
  }, [records, query, classFilter, sortBy, sortDir]);

  if (records.length === 0) return null;

  const setSort = (col) => {
    if (sortBy === col) setSortDir(-sortDir);
    else { setSortBy(col); setSortDir(-1); }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="filter (e.g. PC(34:1) or [M+H]+)"
          className="flex-1 px-2 py-1 text-[11px] rounded border border-dark/15 dark:border-light/15 bg-light dark:bg-dark"
        />
        <select
          value={classFilter}
          onChange={(e) => setClassFilter(e.target.value)}
          className="text-[11px] px-2 py-1 rounded border border-dark/15 dark:border-light/15 bg-light dark:bg-dark"
        >
          <option value="all">all classes</option>
          {classes.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <span className="text-[10px] text-dark/50 dark:text-light/50">
          {filtered.length} / {records.length}
        </span>
      </div>
      <div className="border border-dark/10 dark:border-light/10 rounded overflow-hidden">
        <table className="w-full text-[11px]">
          <thead className="bg-dark/5 dark:bg-light/5 text-dark/70 dark:text-light/70">
            <tr>
              <Th onClick={() => setSort("analyteClass")}>cls</Th>
              <Th onClick={() => setSort("analyte")}>name</Th>
              <Th onClick={() => setSort("precursorMz")}>m/z</Th>
              <Th onClick={() => setSort("intensity")}>I</Th>
              <Th onClick={() => setSort("n")}>n</Th>
              <Th>(ℓ,m,s)</Th>
            </tr>
          </thead>
          <tbody className="max-h-72 overflow-auto">
            {filtered.map((r, i) => {
              const id = `${r.analyte}${r.adduct}`;
              const sel = id === selectedId;
              return (
                <tr key={id + i}
                  onClick={() => select(id)}
                  className={`cursor-pointer transition ${
                    sel
                      ? "bg-primary/10 dark:bg-primaryDark/10"
                      : i % 2 === 0
                      ? "bg-light dark:bg-dark"
                      : "bg-dark/[0.025] dark:bg-light/[0.025]"
                  }`}
                >
                  <td className="px-2 py-1">
                    <span style={{
                      display: "inline-block", width: 8, height: 8, borderRadius: 8,
                      background: classColor(r.analyteClass), marginRight: 4,
                    }} />
                    {r.analyteClass}
                  </td>
                  <td className="px-2 py-1 font-mono">{r.analyte}{r.adductAbbr ? ` ${r.adductAbbr}` : ""}</td>
                  <td className="px-2 py-1 font-mono">{r.precursorMz.toFixed(4)}</td>
                  <td className="px-2 py-1 font-mono">{r.intensity.toExponential(1)}</td>
                  <td className="px-2 py-1 font-mono">{r.n}</td>
                  <td className="px-2 py-1 font-mono">{r.l},{r.m},{r.s.toFixed(1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Th({ children, onClick }) {
  return (
    <th onClick={onClick}
      className="px-2 py-1 text-left font-bold uppercase tracking-wider cursor-pointer
        select-none hover:text-primary dark:hover:text-primaryDark"
    >
      {children}
    </th>
  );
}

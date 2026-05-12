import React from "react";
import { useStore } from "@/lib/state/store";
import { LIPID_CLASSES, LIPID_CLASS_KEYS } from "@/lib/experiment/lipidomics";
import {
  PROTEIN_CLASSES,
  PROTEIN_CLASS_KEYS,
  countMissedCleavages,
} from "@/lib/experiment/proteomics";
import { classColor } from "./charts/palette";

export default function AnalyteBuilder() {
  const design           = useStore((s) => s.experimentDesign);
  const setExperimentType = useStore((s) => s.setExperimentType);

  // lipidomics actions
  const setClassSpec    = useStore((s) => s.setClassSpec);
  const toggleClass     = useStore((s) => s.toggleClass);
  const addClassSpec    = useStore((s) => s.addClassSpec);
  const removeClassSpec = useStore((s) => s.removeClassSpec);

  // proteomics actions
  const setProteinSpec    = useStore((s) => s.setProteinSpec);
  const toggleProtein     = useStore((s) => s.toggleProtein);
  const addProteinSpec    = useStore((s) => s.addProteinSpec);
  const removeProteinSpec = useStore((s) => s.removeProteinSpec);

  const experimentType = design.experimentType || "lipidomics";
  const proteinSpecs   = design.proteinSpecs   || [];

  return (
    <div className="space-y-3">
      {/* header + experiment type toggle */}
      <div className="flex items-center gap-2">
        <div className="text-xs uppercase tracking-wider font-bold text-dark/60 dark:text-light/60">
          Analyte design space
        </div>
        <div className="flex gap-0.5 ml-auto rounded overflow-hidden border border-dark/15 dark:border-light/15">
          {["lipidomics", "proteomics"].map((t) => (
            <button
              key={t}
              onClick={() => setExperimentType(t)}
              className={`text-[9px] uppercase tracking-wider px-2 py-0.5 transition ${
                experimentType === t
                  ? "bg-dark text-light dark:bg-light dark:text-dark"
                  : "text-dark/40 dark:text-light/40 hover:bg-dark/5 dark:hover:bg-light/5"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* ── Lipidomics mode ─────────────────────────────────────── */}
      {experimentType === "lipidomics" && (
        <LipidomicsPanel
          design={design}
          setClassSpec={setClassSpec}
          toggleClass={toggleClass}
          addClassSpec={addClassSpec}
          removeClassSpec={removeClassSpec}
        />
      )}

      {/* ── Proteomics mode ─────────────────────────────────────── */}
      {experimentType === "proteomics" && (
        <ProteomicsPanel
          proteinSpecs={proteinSpecs}
          setProteinSpec={setProteinSpec}
          toggleProtein={toggleProtein}
          addProteinSpec={addProteinSpec}
          removeProteinSpec={removeProteinSpec}
        />
      )}
    </div>
  );
}

// ─── Lipidomics panel (unchanged logic, extracted to component) ──────────────

function LipidomicsPanel({ design, setClassSpec, toggleClass, addClassSpec, removeClassSpec }) {
  const inUse         = new Set(design.classSpecs.map((cs) => cs.classKey));
  const additionalKeys = LIPID_CLASS_KEYS.filter((k) => !inUse.has(k));

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 gap-3">
        {design.classSpecs.map((cs) => {
          const cls = LIPID_CLASSES[cs.classKey];
          if (!cls) return null;
          return (
            <div
              key={cs.classKey}
              className={`rounded-md border p-3 ${
                cs.enabled
                  ? "border-dark/15 dark:border-light/15"
                  : "border-dark/5 dark:border-light/5 opacity-60"
              }`}
              style={{ borderLeft: `4px solid ${classColor(cs.classKey)}` }}
            >
              <div className="flex items-center justify-between mb-2">
                <div>
                  <span className="font-bold text-sm">{cls.abbr}</span>
                  <span className="text-xs text-dark/50 dark:text-light/50 ml-2">{cls.name}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-dark/60 dark:text-light/60">
                    ~{estimateLipidSpecies(cs)} species
                  </span>
                  <button
                    onClick={() => toggleClass(cs.classKey)}
                    className={`text-[10px] px-2 py-0.5 rounded ${
                      cs.enabled
                        ? "bg-dark text-light dark:bg-light dark:text-dark"
                        : "bg-dark/10 dark:bg-light/10"
                    }`}
                  >
                    {cs.enabled ? "on" : "off"}
                  </button>
                  <button
                    onClick={() => removeClassSpec(cs.classKey)}
                    className="text-[10px] text-dark/40 hover:text-red-500"
                  >
                    ✕
                  </button>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3 text-[11px]">
                <RangeRow
                  label="acyl C"
                  min={Math.max(2 * cls.faChains, 8)}
                  max={70}
                  lo={cs.Xmin}
                  hi={cs.Xmax}
                  onChange={(lo, hi) => setClassSpec(cs.classKey, { Xmin: lo, Xmax: hi })}
                />
                <RangeRow
                  label="DB"
                  min={0}
                  max={9}
                  lo={cs.Ymin}
                  hi={cs.Ymax}
                  onChange={(lo, hi) => setClassSpec(cs.classKey, { Ymin: lo, Ymax: hi })}
                />
              </div>
            </div>
          );
        })}
      </div>

      {additionalKeys.length > 0 && (
        <div className="border-t border-dark/10 dark:border-light/10 pt-3">
          <div className="text-[10px] text-dark/50 dark:text-light/50 mb-1.5">Add class</div>
          <div className="flex flex-wrap gap-1">
            {additionalKeys.map((k) => (
              <button
                key={k}
                onClick={() => addClassSpec(k)}
                className="text-[10px] px-2 py-1 rounded border border-dark/15 dark:border-light/15
                  hover:bg-dark/5 dark:hover:bg-light/5"
                style={{ borderColor: classColor(k) }}
                title={LIPID_CLASSES[k]?.name}
              >
                + {k}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Proteomics panel ────────────────────────────────────────────────────────

function ProteomicsPanel({
  proteinSpecs,
  setProteinSpec,
  toggleProtein,
  addProteinSpec,
  removeProteinSpec,
}) {
  const inUse          = new Set(proteinSpecs.map((ps) => ps.classKey));
  const additionalKeys = PROTEIN_CLASS_KEYS.filter((k) => !inUse.has(k));

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 gap-3">
        {proteinSpecs.map((ps) => {
          const cls = PROTEIN_CLASSES[ps.classKey];
          if (!cls) return null;
          const active = estimateActivePeptides(ps, cls);
          return (
            <div
              key={ps.classKey}
              className={`rounded-md border p-3 ${
                ps.enabled
                  ? "border-dark/15 dark:border-light/15"
                  : "border-dark/5 dark:border-light/5 opacity-60"
              }`}
              style={{ borderLeft: `4px solid ${classColor(ps.classKey)}` }}
            >
              <div className="flex items-center justify-between mb-2">
                <div>
                  <span className="font-bold text-sm">{cls.abbr}</span>
                  <span className="text-xs text-dark/50 dark:text-light/50 ml-2">{cls.name}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-dark/60 dark:text-light/60">
                    {active}/{cls.peptides.length} peptides
                  </span>
                  <button
                    onClick={() => toggleProtein(ps.classKey)}
                    className={`text-[10px] px-2 py-0.5 rounded ${
                      ps.enabled
                        ? "bg-dark text-light dark:bg-light dark:text-dark"
                        : "bg-dark/10 dark:bg-light/10"
                    }`}
                  >
                    {ps.enabled ? "on" : "off"}
                  </button>
                  <button
                    onClick={() => removeProteinSpec(ps.classKey)}
                    className="text-[10px] text-dark/40 hover:text-red-500"
                  >
                    ✕
                  </button>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3 text-[11px]">
                <RangeRow
                  label="length (aa)"
                  min={4}
                  max={40}
                  lo={ps.lengthMin}
                  hi={ps.lengthMax}
                  onChange={(lo, hi) =>
                    setProteinSpec(ps.classKey, { lengthMin: lo, lengthMax: hi })
                  }
                />
                <RangeRow
                  label="missed cleavages"
                  min={0}
                  max={3}
                  lo={ps.mcMin}
                  hi={ps.mcMax}
                  onChange={(lo, hi) =>
                    setProteinSpec(ps.classKey, { mcMin: lo, mcMax: hi })
                  }
                />
              </div>
            </div>
          );
        })}
      </div>

      {additionalKeys.length > 0 && (
        <div className="border-t border-dark/10 dark:border-light/10 pt-3">
          <div className="text-[10px] text-dark/50 dark:text-light/50 mb-1.5">Add protein standard</div>
          <div className="flex flex-wrap gap-1">
            {additionalKeys.map((k) => (
              <button
                key={k}
                onClick={() => addProteinSpec(k)}
                className="text-[10px] px-2 py-1 rounded border border-dark/15 dark:border-light/15
                  hover:bg-dark/5 dark:hover:bg-light/5"
                style={{ borderColor: classColor(k) }}
                title={PROTEIN_CLASSES[k]?.name}
              >
                + {k}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Shared range-row input ───────────────────────────────────────────────────

function RangeRow({ label, min, max, lo, hi, onChange }) {
  return (
    <div>
      <div className="flex justify-between mb-1">
        <span className="text-dark/60 dark:text-light/60">{label}</span>
        <span className="font-mono">
          {lo} – {hi}
        </span>
      </div>
      <div className="flex gap-2 items-center">
        <input
          type="number" value={lo} min={min} max={hi}
          onChange={(e) => onChange(Math.max(min, +e.target.value), hi)}
          className="w-12 px-1 py-0.5 rounded border border-dark/15 dark:border-light/15
            bg-light dark:bg-dark text-[11px]"
        />
        <input
          type="range" min={min} max={max} value={lo}
          onChange={(e) => onChange(Math.max(min, Math.min(+e.target.value, hi)), hi)}
          className="flex-1 accent-current"
        />
        <input
          type="range" min={min} max={max} value={hi}
          onChange={(e) => onChange(lo, Math.max(lo, Math.min(+e.target.value, max)))}
          className="flex-1 accent-current"
        />
        <input
          type="number" value={hi} min={lo} max={max}
          onChange={(e) => onChange(lo, Math.min(max, +e.target.value))}
          className="w-12 px-1 py-0.5 rounded border border-dark/15 dark:border-light/15
            bg-light dark:bg-dark text-[11px]"
        />
      </div>
    </div>
  );
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function estimateLipidSpecies(cs) {
  return Math.max(0, cs.Xmax - cs.Xmin + 1) * Math.max(0, cs.Ymax - cs.Ymin + 1);
}

function estimateActivePeptides(ps, cls) {
  return cls.peptides.filter((seq) => {
    const mc = countMissedCleavages(seq);
    return (
      seq.length >= ps.lengthMin &&
      seq.length <= ps.lengthMax &&
      mc >= ps.mcMin &&
      mc <= ps.mcMax
    );
  }).length;
}

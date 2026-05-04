import React from "react";
import { useStore } from "@/lib/state/store";

const ANALYSERS = [
  { key: "tof",        label: "TOF" },
  { key: "quadrupole", label: "Quad" },
  { key: "orbitrap",   label: "Orbitrap" },
  { key: "fticr",      label: "FT-ICR" },
];

const SAMPLE_TYPES = ["plasma", "serum", "tissue", "cell lysate", "yeast", "bacteria", "urine"];
const EXTRACTIONS = ["MTBE", "Bligh-Dyer", "Folch", "IPA", "BUME", "SPE"];
const COLUMNS = [
  "RPLC C18 30 min gradient",
  "RPLC C30 60 min gradient",
  "HILIC 25 min gradient",
  "Normal phase silica",
];

export default function AcquisitionConfig() {
  const design = useStore((s) => s.experimentDesign);
  const setDesign = useStore((s) => s.setExperimentDesign);

  return (
    <div className="space-y-3">
      <div className="text-xs uppercase tracking-wider font-bold text-dark/60 dark:text-light/60">
        Acquisition
      </div>

      <div className="grid grid-cols-2 gap-2 text-[11px]">
        <Select label="Analyser"
          value={design.analyser}
          options={ANALYSERS.map((a) => ({ value: a.key, label: a.label }))}
          onChange={(v) => setDesign({ analyser: v })}
        />
        <Select label="Sample"
          value={design.sampleType}
          options={SAMPLE_TYPES.map((s) => ({ value: s, label: s }))}
          onChange={(v) => setDesign({ sampleType: v })}
        />
        <Select label="Extraction"
          value={design.extraction}
          options={EXTRACTIONS.map((s) => ({ value: s, label: s }))}
          onChange={(v) => setDesign({ extraction: v })}
        />
        <Select label="Chromatography"
          value={design.chromatography}
          options={COLUMNS.map((s) => ({ value: s, label: s }))}
          onChange={(v) => setDesign({ chromatography: v })}
        />
      </div>

      <div className="grid grid-cols-2 gap-3 text-[11px]">
        <Numeric
          label={`Collision energy (eV)`}
          value={design.collisionEnergy_eV}
          min={5} max={80} step={1}
          onChange={(v) => setDesign({ collisionEnergy_eV: v })}
        />
        <div>
          <div className="text-dark/60 dark:text-light/60 mb-1">m/z window</div>
          <div className="flex gap-1 items-center">
            <input
              type="number" value={design.mzWindow[0]} min={50} max={5000}
              onChange={(e) => setDesign({
                mzWindow: [+e.target.value, design.mzWindow[1]],
              })}
              className="flex-1 px-2 py-1 rounded border border-dark/15 dark:border-light/15
                bg-light dark:bg-dark"
            />
            <span>–</span>
            <input
              type="number" value={design.mzWindow[1]} min={100} max={5000}
              onChange={(e) => setDesign({
                mzWindow: [design.mzWindow[0], +e.target.value],
              })}
              className="flex-1 px-2 py-1 rounded border border-dark/15 dark:border-light/15
                bg-light dark:bg-dark"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function Select({ label, value, options, onChange }) {
  return (
    <label className="block">
      <div className="text-dark/60 dark:text-light/60 mb-1">{label}</div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-2 py-1 rounded border border-dark/15 dark:border-light/15 bg-light dark:bg-dark"
      >
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </label>
  );
}

function Numeric({ label, value, min, max, step, onChange }) {
  return (
    <label className="block">
      <div className="text-dark/60 dark:text-light/60 mb-1">{label}</div>
      <input
        type="number" value={value} min={min} max={max} step={step}
        onChange={(e) => onChange(+e.target.value)}
        className="w-full px-2 py-1 rounded border border-dark/15 dark:border-light/15 bg-light dark:bg-dark"
      />
    </label>
  );
}

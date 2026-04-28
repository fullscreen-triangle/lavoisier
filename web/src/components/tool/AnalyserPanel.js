import React from "react";
import { motion } from "framer-motion";
import { useStore } from "@/lib/state/store";

const ANALYSERS = [
  { id: "tof",        name: "TOF",        equation: "T ∝ √(m/z)",      tag: "linear gradient" },
  { id: "quadrupole", name: "Quadrupole", equation: "a, q ∝ 1/(m/z)", tag: "saddle RF" },
  { id: "orbitrap",   name: "Orbitrap",   equation: "ω ∝ √(z/m)",      tag: "quadro-log" },
  { id: "fticr",      name: "FT-ICR",     equation: "ωc ∝ z/m",          tag: "vector A_M" },
];

const CONFIG_FIELDS = {
  tof: [
    { key: "accelV",       label: "Accelerating V",  unit: "V",  step: 100 },
    { key: "flightLength", label: "Flight length",   unit: "m",  step: 0.1 },
  ],
  quadrupole: [
    { key: "dcVoltage",   label: "DC voltage",  unit: "V",   step: 10 },
    { key: "rfVoltage",   label: "RF voltage",  unit: "V",   step: 50 },
    { key: "rfFrequency", label: "RF freq",      unit: "Hz", step: 1e5 },
  ],
  orbitrap: [
    { key: "kField",      label: "Field curvature", unit: "N/m·C", step: 1e11 },
  ],
  fticr: [
    { key: "B", label: "Magnetic field", unit: "T", step: 0.5 },
  ],
};

/**
 * AnalyserPanel — selects which partition depth field topology
 * the GPU pipeline uses. Same Lagrangian, different M(x,t).
 */
export default function AnalyserPanel({ compact = false }) {
  const analyser = useStore((s) => s.analyser);
  const setAnalyser = useStore((s) => s.setAnalyser);
  const cfg = useStore((s) => s.analyserCfg[s.analyser]);
  const setAnalyserCfg = useStore((s) => s.setAnalyserCfg);

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        <span className="text-xs uppercase tracking-wider font-bold text-dark/60 dark:text-light/60">
          Analyser:
        </span>
        {ANALYSERS.map((a) => {
          const active = a.id === analyser;
          return (
            <motion.button
              key={a.id}
              onClick={() => setAnalyser(a.id)}
              whileTap={{ scale: 0.95 }}
              className={`px-3 py-1 text-xs rounded-md border-2 font-medium transition-colors
                ${
                  active
                    ? "border-primary bg-primary/10 text-primary dark:border-primaryDark dark:bg-primaryDark/10 dark:text-primaryDark"
                    : "border-dark/10 dark:border-light/10 hover:border-dark/30 dark:hover:border-light/30"
                }`}
            >
              {a.name}
            </motion.button>
          );
        })}
        <span className="ml-auto text-xs font-mono text-dark/40 dark:text-light/40">
          L_M = ½μ|ẋ|² + μẋ·A_M − M(x,t)
        </span>
      </div>
    );
  }

  // Expanded view (could be shown in a settings panel)
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-2">
        {ANALYSERS.map((a) => {
          const active = a.id === analyser;
          return (
            <button
              key={a.id}
              onClick={() => setAnalyser(a.id)}
              className={`p-3 rounded-lg border-2 text-left transition-colors
                ${
                  active
                    ? "border-primary dark:border-primaryDark bg-primary/10 dark:bg-primaryDark/10"
                    : "border-dark/10 dark:border-light/10 hover:border-dark/30 dark:hover:border-light/30"
                }`}
            >
              <div className="text-sm font-bold">{a.name}</div>
              <div className="text-xs text-dark/60 dark:text-light/60 mt-0.5 font-mono">
                {a.equation}
              </div>
              <div className="text-[10px] uppercase tracking-wider text-dark/40 dark:text-light/40 mt-1">
                {a.tag}
              </div>
            </button>
          );
        })}
      </div>

      {/* Per-analyser config */}
      <div className="border-t border-dark/10 dark:border-light/10 pt-4">
        <div className="text-xs uppercase tracking-wider text-dark/60 dark:text-light/60 mb-2">
          {ANALYSERS.find((a) => a.id === analyser).name} configuration
        </div>
        <div className="space-y-2">
          {(CONFIG_FIELDS[analyser] || []).map((field) => (
            <div key={field.key} className="flex items-center gap-2">
              <label className="text-xs text-dark/70 dark:text-light/70 w-32 flex-shrink-0">
                {field.label}
              </label>
              <input
                type="number"
                step={field.step}
                value={cfg[field.key] ?? 0}
                onChange={(e) =>
                  setAnalyserCfg(analyser, { [field.key]: parseFloat(e.target.value) || 0 })
                }
                className="flex-1 min-w-0 px-2 py-1 text-xs rounded border border-dark/20 dark:border-light/20
                  bg-light dark:bg-dark"
              />
              <span className="text-[10px] text-dark/40 dark:text-light/40 w-10 flex-shrink-0">
                {field.unit}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/**
 * GPU Observation Panel — renders the four-pass observation apparatus output.
 *
 * By the Triple Equivalence Theorem, the GPU wave-field texture IS the physical
 * observation of partition states. This panel shows:
 *   - The categorical state tensor (wave-field texture as canvas)
 *   - S-entropy coordinate map (Pass 2 output)
 *   - Bijective validation score (Pass 3)
 *   - Physical quality metrics (Paper 2, §7)
 *   - Ion-droplet convergence (dual-path validation)
 */

import React, { useEffect, useRef, useState, useCallback } from "react";
import { useStore } from "@/lib/state/store";
import { PALETTE } from "./cf/chartUtils";
import { dualPathValidate } from "@/lib/partition/ionDroplet";

/* ── Lazy-load GpuObserver (SSR-safe) ───────────────────────────────────── */
let _observerPromise = null;
function getObserverLazy(w, h) {
  if (!_observerPromise) {
    _observerPromise = import("@/lib/gpu/GpuObserver")
      .then(m => m.getObserver(w, h))
      .catch(() => null);
  }
  return _observerPromise;
}

/* ── Pass 1+2 canvas display ─────────────────────────────────────────────── */
function ObservationCanvas({ records, width = 256, height = 256 }) {
  const canvasRef = useRef(null);
  const [status, setStatus] = useState("idle");

  useEffect(() => {
    if (!records || records.length === 0) return;
    let cancelled = false;
    setStatus("observing");

    (async () => {
      try {
        const observer = await getObserverLazy(width, height);
        if (!observer || cancelled) return;

        const { coordTexture } = observer.observe(records);
        const pixels = observer.readCoordTexturePixels();

        if (cancelled) return;
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const imgData = ctx.createImageData(width, height);
        imgData.data.set(pixels);
        ctx.putImageData(imgData, 0, 0);
        setStatus("ready");
      } catch (e) {
        if (!cancelled) setStatus("error: " + e.message);
      }
    })();

    return () => { cancelled = true; };
  }, [records, width, height]);

  return (
    <div className="relative rounded overflow-hidden"
      style={{ border: `1px solid ${PALETTE.grid}` }}>
      <canvas ref={canvasRef} width={width} height={height}
        className="block" style={{ imageRendering: "pixelated", width: "100%", height: "auto" }} />
      {status !== "ready" && (
        <div className="absolute inset-0 flex items-center justify-center text-[10px]"
          style={{ background: "rgba(0,0,0,0.6)", color: PALETTE.muted }}>
          {status === "observing" ? "observing partition states…" : status}
        </div>
      )}
    </div>
  );
}

/* ── Dual-path validation (ion-droplet bijection) ────────────────────────── */
function DualPathValidator({ records }) {
  const [results, setResults] = useState([]);

  useEffect(() => {
    if (!records || records.length === 0) { setResults([]); return; }
    const sample = records.slice(0, 8);
    const validated = sample.map(r => {
      try {
        const ionSE = r.sentropyVec ?? { sk: 0, st: 0, se: 0 };
        const ionParams = {
          mass: r.neutralMass ?? 500,
          kineticEnergy: 1.0,
          composition: r.composition ?? {},
        };
        return { name: r.analyte, ...dualPathValidate(ionSE, ionParams, 10) };
      } catch {
        return { name: r.analyte, convergenceScore: 0, commonPrefixLen: 0, falsePosProb: 1 };
      }
    });
    setResults(validated);
  }, [records]);

  if (!results.length) return null;

  return (
    <div className="space-y-1">
      <div className="text-[9px] uppercase tracking-wider" style={{ color: PALETTE.muted }}>
        Dual-path interference (ion ↔ droplet bijection)
      </div>
      <div className="space-y-0.5">
        {results.map((r, i) => (
          <div key={i} className="flex items-center gap-2 text-[10px]">
            <span className="truncate font-mono" style={{ color: PALETTE.text, maxWidth: 130 }}>
              {r.name}
            </span>
            <div className="flex-1 h-1.5 rounded-full overflow-hidden"
              style={{ background: PALETTE.grid }}>
              <div className="h-full rounded-full"
                style={{
                  width: `${r.convergenceScore * 100}%`,
                  background: r.convergenceScore > 0.7 ? "#34d399"
                    : r.convergenceScore > 0.4 ? "#dcdcaa" : "#f48771",
                }} />
            </div>
            <span className="font-mono w-8 text-right" style={{ color: PALETTE.muted }}>
              {r.commonPrefixLen}/10
            </span>
          </div>
        ))}
      </div>
      <p className="text-[9px]" style={{ color: PALETTE.muted }}>
        Common prefix length k → false-positive probability ≤ 3⁻ᵏ
      </p>
    </div>
  );
}

/* ── Quality metrics display ─────────────────────────────────────────────── */
function QualityMetrics({ metrics }) {
  if (!metrics) return null;
  const items = [
    { label: "Partition sharpness", value: metrics.partitionSharpness ?? 0, color: "#60a5fa" },
    { label: "Phase coherence",     value: metrics.phaseCoherence ?? 0,     color: "#34d399" },
    { label: "Noise level",         value: 1 - (metrics.noiseLevel ?? 0),   color: "#f87171" },
    { label: "Composite Q",         value: metrics.compositeQuality ?? 0,   color: "#a78bfa" },
  ];
  return (
    <div className="grid grid-cols-2 gap-1.5">
      {items.map(({ label, value, color }) => (
        <div key={label} className="rounded p-2 text-center"
          style={{ background: "rgba(255,255,255,0.03)", border: `1px solid ${PALETTE.grid}` }}>
          <div className="text-[8px] uppercase tracking-wider mb-1" style={{ color: PALETTE.muted }}>{label}</div>
          <div className="font-mono text-[13px]" style={{ color }}>{value.toFixed(3)}</div>
        </div>
      ))}
    </div>
  );
}

/* ── S-entropy legend ────────────────────────────────────────────────────── */
function SEntropyLegend() {
  return (
    <div className="flex gap-3 text-[9px]">
      {[
        { label: "Sₖ knowledge", color: "rgba(0,255,255,0.7)" },
        { label: "Sₜ temporal",  color: "rgba(255,191,0,0.7)" },
        { label: "Sₑ evolution", color: "rgba(148,0,211,0.7)" },
      ].map(({ label, color }) => (
        <span key={label} className="flex items-center gap-1">
          <span className="w-3 h-1.5 rounded-full" style={{ background: color }} />
          <span style={{ color: PALETTE.muted }}>{label}</span>
        </span>
      ))}
    </div>
  );
}

/* ── Main panel ─────────────────────────────────────────────────────────── */
export default function GpuObservationPanel() {
  const records   = useStore(s => s.experimentRecords);
  const [gpuMetrics, setGpuMetrics] = useState(null);
  const [valScore,   setValScore]   = useState(null);
  const [gpuAvailable, setGpuAvailable] = useState(true);

  useEffect(() => {
    if (!records.length) return;
    let cancelled = false;
    (async () => {
      try {
        const obs = await getObserverLazy(256, 256);
        if (!obs || cancelled) return;
        const { validationScore, qualityMetrics } = obs.observe(records);
        if (!cancelled) {
          setGpuMetrics(qualityMetrics);
          setValScore(validationScore);
        }
      } catch (e) {
        if (!cancelled) setGpuAvailable(false);
      }
    })();
    return () => { cancelled = true; };
  }, [records]);

  if (!gpuAvailable) {
    return (
      <div className="rounded p-4 text-[11px]"
        style={{ background: PALETTE.bg, border: `1px solid ${PALETTE.grid}`, color: PALETTE.muted }}>
        WebGL2 not available in this environment. GPU observation requires a browser with WebGL2 support.
      </div>
    );
  }

  if (!records.length) {
    return (
      <div className="flex items-center justify-center py-8 text-[11px]"
        style={{ color: PALETTE.muted }}>
        run a virtual experiment to observe the partition state tensor
      </div>
    );
  }

  return (
    <div className="space-y-4 text-[11px]" style={{ color: PALETTE.text }}>

      {/* Explanation */}
      <p className="text-[10px] leading-relaxed" style={{ color: PALETTE.muted }}>
        By the Triple Equivalence Theorem, the GPU fragment shader evaluating the
        partition function at cell coordinates IS performing physical observation.
        The texture below is the categorical state tensor — not a picture of the data,
        but the data itself, in the natural language of bounded phase space.
      </p>

      {/* Four-pass pipeline indicator */}
      <div className="flex gap-2">
        {["Pass 1 Wave field", "Pass 2 S-entropy", "Pass 3 Bijective", "Pass 4 Resonance"].map((p, i) => (
          <div key={i} className="flex-1 rounded px-1.5 py-1 text-center text-[9px]"
            style={{
              background: records.length > 0 ? "rgba(14,99,156,0.3)" : "rgba(255,255,255,0.03)",
              border: `1px solid ${records.length > 0 ? "#0e639c" : PALETTE.grid}`,
              color: records.length > 0 ? "#9cdcfe" : PALETTE.muted,
            }}>
            {p}
          </div>
        ))}
      </div>

      {/* Categorical state tensor (Pass 2 output) */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <div className="text-[9px] uppercase tracking-wider" style={{ color: PALETTE.muted }}>
            Categorical state tensor (S-entropy map)
          </div>
          {valScore !== null && (
            <span className="text-[9px] font-mono"
              style={{ color: valScore > 0.9 ? "#34d399" : "#dcdcaa" }}>
              bijective score: {valScore.toFixed(4)}
            </span>
          )}
        </div>
        <ObservationCanvas records={records} width={256} height={256} />
        <SEntropyLegend />
      </div>

      {/* Physical quality metrics (Paper 2, §7) */}
      <div className="space-y-1.5">
        <div className="text-[9px] uppercase tracking-wider" style={{ color: PALETTE.muted }}>
          Physical quality metrics (GPU observables)
        </div>
        <QualityMetrics metrics={gpuMetrics} />
        <p className="text-[9px]" style={{ color: PALETTE.muted }}>
          Deterministic functions of the GPU output — not heuristics.
          No human labels required. Objective training signal.
        </p>
      </div>

      {/* Dual-path interference validation */}
      <DualPathValidator records={records} />

      {/* Partition depth statistics */}
      <PartitionDepthStats records={records} />
    </div>
  );
}

/* ── Partition depth distribution (from the six-pass ion journey) ─────── */
function PartitionDepthStats({ records }) {
  if (!records.length) return null;

  const byN = {};
  for (const r of records) {
    const n = r.n ?? 0;
    byN[n] = (byN[n] || 0) + 1;
  }
  const entries = Object.entries(byN).sort((a, b) => +a[0] - +b[0]);
  const maxCount = Math.max(...entries.map(([, c]) => c));

  return (
    <div className="space-y-1.5">
      <div className="text-[9px] uppercase tracking-wider" style={{ color: PALETTE.muted }}>
        Partition depth distribution — n (principal quantum number)
      </div>
      <div className="flex items-end gap-1" style={{ height: 40 }}>
        {entries.map(([n, count]) => (
          <div key={n} className="flex flex-col items-center gap-0.5 flex-1">
            <div className="w-full rounded-sm"
              style={{
                height: Math.max(2, (count / maxCount) * 36),
                background: "#0e639c",
                opacity: 0.7 + 0.3 * (count / maxCount),
              }} />
            <span className="text-[8px] font-mono" style={{ color: PALETTE.muted }}>{n}</span>
          </div>
        ))}
      </div>
      <p className="text-[9px]" style={{ color: PALETTE.muted }}>
        Partition inertia μ = α(m/z). Ions follow −∇M through the partition landscape.
        Higher n = more confinement modes.
      </p>
    </div>
  );
}

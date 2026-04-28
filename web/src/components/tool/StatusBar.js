import React from "react";
import { useStore } from "@/lib/state/store";
import { copyToClipboard, buildShareUrl } from "@/lib/state/share";

/**
 * Status bar at the top of the workspace.
 *
 * Surfaces: GPU readiness, source kind, scan throughput,
 *           live quality metrics, and a share button.
 */
export default function StatusBar() {
  const gpuReady = useStore((s) => s.gpuReady);
  const source = useStore((s) => s.source);
  const totalScanCount = useStore((s) => s.totalScanCount);
  const quality = useStore((s) => s.quality);
  const selectedAddress = useStore((s) => s.selectedAddress);
  const analyser = useStore((s) => s.analyser);

  const handleShare = async () => {
    if (typeof window === "undefined") return;
    const url = buildShareUrl(window.location.origin, {
      address: selectedAddress,
      analyser,
      source,
    });
    const ok = await copyToClipboard(url);
    if (ok) {
      // Could show a toast here; for now just announce via title
      const el = document.activeElement;
      if (el && el.blur) el.blur();
    }
  };

  return (
    <div className="flex items-center justify-between px-6 py-2 border-b-2 border-dark/10 dark:border-light/10
      text-xs gap-3">
      <div className="flex items-center gap-3">
        <Dot ok={gpuReady} label={gpuReady ? "GPU ready" : "GPU init"} />
        <Dot ok={!!source} label={sourceLabel(source)} />
        <Sep />
        <span className="font-mono">
          <strong className="text-primary dark:text-primaryDark">{totalScanCount}</strong>
          <span className="text-dark/50 dark:text-light/50 ml-1">scans observed</span>
        </span>
      </div>

      {quality && (
        <div className="flex items-center gap-3 text-[10px] font-mono">
          <Metric label="sharpness"  value={quality.partitionSharpness} fmt={(v) => v.toFixed(2)} />
          <Metric label="noise"      value={quality.noiseLevel}        fmt={(v) => v.toFixed(2)} />
          <Metric label="coherence"  value={quality.phaseCoherence}    fmt={(v) => v.toFixed(2)} />
          <Metric label="signal frac" value={quality.signalFraction}    fmt={(v) => `${(v*100).toFixed(0)}%`} />
        </div>
      )}

      <div className="flex items-center gap-2">
        {selectedAddress && (
          <span className="font-mono text-primary dark:text-primaryDark text-[11px]">
            {selectedAddress.substring(0, 12)}…
          </span>
        )}
        <button
          onClick={handleShare}
          disabled={!source}
          className={`px-2 py-1 rounded text-[10px] uppercase tracking-wider font-bold transition-colors
            ${
              source
                ? "border border-dark/20 dark:border-light/20 hover:border-primary dark:hover:border-primaryDark"
                : "border border-dark/5 dark:border-light/5 text-dark/30 dark:text-light/30 cursor-not-allowed"
            }`}
        >
          Share
        </button>
      </div>
    </div>
  );
}

function Dot({ ok, label }) {
  return (
    <div className="flex items-center gap-1.5">
      <div
        className={`w-2 h-2 rounded-full ${
          ok ? "bg-green-500" : "bg-dark/30 dark:bg-light/30"
        }`}
      />
      <span className={ok ? "" : "text-dark/50 dark:text-light/50"}>{label}</span>
    </div>
  );
}

function Sep() {
  return <span className="text-dark/20 dark:text-light/20">·</span>;
}

function Metric({ label, value, fmt }) {
  return (
    <span title={label}>
      <span className="text-dark/50 dark:text-light/50">{label}:</span>
      <span className="ml-1 text-dark dark:text-light">{fmt(value)}</span>
    </span>
  );
}

function sourceLabel(source) {
  if (!source) return "No source";
  if (source.kind === "local") return `Local: ${source.label}`;
  if (source.kind === "repository") {
    const r = source.meta?.repository || "repo";
    const a = source.meta?.accession || source.meta?.recordId || "";
    return `${r}: ${a}`;
  }
  if (source.kind === "remote") return `URL · ${source.label}`;
  return source.label || "Source";
}

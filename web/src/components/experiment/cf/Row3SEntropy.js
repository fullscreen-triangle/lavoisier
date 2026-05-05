import React from "react";
import BrushableHistogram from "./BrushableHistogram";
import { PALETTE, CLASS_PALETTE } from "./chartUtils";

/**
 * Row 3: four S-entropy charts. The fourth is partition entropy (nats), which
 * accompanies the S-entropy triple as the framework's information measure.
 */
export default function Row3SEntropy() {
  return (
    <div className="grid grid-cols-4 gap-3 lg:grid-cols-2">
      <Tile label="Sₖ (knowledge)">
        <BrushableHistogram
          dimKey="sk" groupKey="skBin"
          xLabel="Sₖ" color="#1f77b4"
          tickFmt={(v) => v.toFixed(2)}
        />
      </Tile>
      <Tile label="Sₜ (time / coverage)">
        <BrushableHistogram
          dimKey="st" groupKey="stBin"
          xLabel="Sₜ" color="#9467bd"
          tickFmt={(v) => v.toFixed(2)}
        />
      </Tile>
      <Tile label="Sₑ (entropy / completion)">
        <BrushableHistogram
          dimKey="se" groupKey="seBin"
          xLabel="Sₑ" color="#2ca02c"
          tickFmt={(v) => v.toFixed(2)}
        />
      </Tile>
      <Tile label="partition entropy (nats)">
        <BrushableHistogram
          dimKey="entropy" groupKey="entropyBin"
          xLabel="−Σ w log w" color="#ff7f0e"
        />
      </Tile>
    </div>
  );
}

function Tile({ label, children }) {
  return (
    <div className="rounded border p-2"
      style={{ background: PALETTE.bg, borderColor: PALETTE.grid }}>
      <div className="text-[9px] uppercase tracking-wider mb-1 font-normal"
        style={{ color: PALETTE.muted }}>
        {label}
      </div>
      {children}
    </div>
  );
}

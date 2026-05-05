import React from "react";
import BrushableHistogram from "./BrushableHistogram";

/**
 * Row 6: oscillatory & information coordinates.
 *   - log10 analyser observable (frequency or time)
 *   - bits/record total
 *   - fragment count per record
 *   - intensity (predicted I)
 */
export default function Row6Oscillatory() {
  return (
    <div className="grid grid-cols-4 gap-3 lg:grid-cols-2">
      <Tile label="log₁₀ analyser observable">
        <BrushableHistogram
          dimKey="observable" groupKey="entropyBin"
          xLabel="log₁₀ ω" color="#1f77b4"
        />
      </Tile>
      <Tile label="bits / record">
        <BrushableHistogram
          dimKey="bits" groupKey="bitsBin"
          xLabel="bits" color="#9467bd"
        />
      </Tile>
      <Tile label="fragments / record">
        <BrushableHistogram
          dimKey="numFragments" groupKey="fragsBin"
          xLabel="# MS² peaks" color="#2ca02c"
        />
      </Tile>
      <Tile label="predicted I">
        <BrushableHistogram
          dimKey="intensity" groupKey="bitsBin"
          xLabel="I" color="#ff7f0e"
        />
      </Tile>
    </div>
  );
}

function Tile({ label, children }) {
  return (
    <div className="rounded-md border border-dark/10 dark:border-light/10 p-2 bg-light dark:bg-dark">
      <div className="text-[10px] uppercase tracking-wider font-bold text-dark/60 dark:text-light/60 mb-1">
        {label}
      </div>
      {children}
    </div>
  );
}

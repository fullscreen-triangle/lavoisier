import React from "react";
import RowChart from "./RowChart";
import { PALETTE } from "./chartUtils";

/**
 * Row 4: partition coordinates n, ℓ, m, s.
 *  - n, ℓ, |m| as row charts (discrete; click-to-filter).
 *  - s as row chart with two values.
 */
export default function Row4Partition() {
  return (
    <div className="grid grid-cols-4 gap-3 lg:grid-cols-2">
      <Tile label="n (principal)">
        <RowChart
          dimKey="n" groupKey="n"
          colorFn={() => "#1f77b4"}
          labelFn={(k) => `n=${k}`}
          height={210}
        />
      </Tile>
      <Tile label="ℓ (angular)">
        <RowChart
          dimKey="l" groupKey="l"
          colorFn={() => "#d62728"}
          labelFn={(k) => `ℓ=${k}`}
          height={210}
        />
      </Tile>
      <Tile label="m (orientation)">
        <RowChart
          dimKey="m" groupKey="m"
          colorFn={() => "#9467bd"}
          labelFn={(k) => `m=${k}`}
          height={210}
        />
      </Tile>
      <Tile label="s (chirality)">
        <RowChart
          dimKey="s" groupKey="s"
          colorFn={() => "#2ca02c"}
          labelFn={(k) => `s=${(+k).toFixed(1)}`}
          height={210}
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

import React from "react";
import PieChart from "./PieChart";
import RowChart from "./RowChart";
import { classColor } from "./chartUtils";

const POLARITY_COLORS = { "+": "#1f77b4", "-": "#d62728" };
const Z_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"];

/**
 * Row 5: categorical coordinates — class, adduct, polarity, charge state.
 *   - class: pie (class-coloured)
 *   - adduct: row chart
 *   - polarity: pie
 *   - z: pie
 */
export default function Row5Categorical() {
  return (
    <div className="grid grid-cols-4 gap-3 lg:grid-cols-2">
      <Tile label="lipid class">
        <PieChart
          dimKey="class" groupKey="class"
          colorFn={classColor}
          height={200} innerRadius={32}
        />
      </Tile>
      <Tile label="adduct">
        <RowChart
          dimKey="adduct" groupKey="adduct"
          colorFn={() => "#17becf"}
          labelFn={(k) => k}
          height={210}
        />
      </Tile>
      <Tile label="polarity">
        <PieChart
          dimKey="polarity" groupKey="polarity"
          colorFn={(k) => POLARITY_COLORS[k] || "#888"}
          height={200} innerRadius={32}
        />
      </Tile>
      <Tile label="charge state |z|">
        <PieChart
          dimKey="z" groupKey="z"
          colorFn={(k) => Z_COLORS[(k - 1) % Z_COLORS.length]}
          height={200} innerRadius={32}
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

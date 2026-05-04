import React from "react";
import * as d3 from "d3";
import { useD3 } from "./useD3";
import { PALETTE, FRAGMENT_TYPE_COLOR } from "./palette";

/**
 * Fragment tree visualisation: precursor at top, fragments connected by
 * neutral-loss edges. Edges labelled with the loss mass.
 */
export default function D3FragmentationTree({ record, width = 460, height = 320 }) {
  const ref = useD3((el) => {
    const svg = d3.select(el);
    svg.selectAll("*").remove();
    if (!record || !record.ms2 || record.ms2.length === 0) return;

    const allPeaks = [
      { ...record.ms2.find((p) => p.type === "precursor") || record.ms1[0], _index: -1 },
      ...record.ms2.filter((p) => p.type !== "precursor").map((p, i) => ({ ...p, _index: i })),
    ];
    // Dedupe by m/z
    const seen = new Set();
    const peaks = allPeaks.filter((p) => {
      const k = p.mz.toFixed(4);
      if (seen.has(k)) return false;
      seen.add(k); return true;
    });

    // Build a tree: parent = first peak with mz > current and small loss difference
    const parentLink = peaks.map(() => -1);
    for (let i = 1; i < peaks.length; i++) {
      let best = -1, bestDiff = Infinity;
      for (let j = 0; j < i; j++) {
        const dm = peaks[j].mz - peaks[i].mz;
        if (dm > 5 && dm < bestDiff) {
          bestDiff = dm; best = j;
        }
      }
      parentLink[i] = best === -1 ? 0 : best;
    }

    // Layout: depth via mz descending (precursor topmost). x by index.
    const xs = d3.scaleBand().domain(peaks.map((_, i) => i)).range([40, width - 20]).padding(0.2);
    const ys = d3.scaleLinear()
      .domain(d3.extent(peaks, (d) => d.mz)).range([height - 30, 30]);

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .append("g");

    // Edges
    peaks.forEach((p, i) => {
      const parent = parentLink[i];
      if (parent < 0 || parent === i) return;
      const x1 = xs(parent) + xs.bandwidth() / 2;
      const y1 = ys(peaks[parent].mz);
      const x2 = xs(i) + xs.bandwidth() / 2;
      const y2 = ys(p.mz);
      g.append("path")
        .attr("d", `M${x1},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2},${y2}`)
        .attr("fill", "none").attr("stroke", PALETTE.muted)
        .attr("stroke-width", 0.6);
      const dm = peaks[parent].mz - p.mz;
      if (dm > 5 && dm < 200) {
        g.append("text")
          .attr("x", (x1 + x2) / 2).attr("y", (y1 + y2) / 2)
          .attr("text-anchor", "middle")
          .style("font-size", "8px").style("fill", PALETTE.muted)
          .text(`-${dm.toFixed(1)}`);
      }
    });

    // Nodes
    g.selectAll("circle.node").data(peaks).enter().append("circle")
      .attr("class", "node")
      .attr("cx", (_, i) => xs(i) + xs.bandwidth() / 2)
      .attr("cy", (d) => ys(d.mz))
      .attr("r", (d) => 3 + 6 * d.intensity)
      .attr("fill", (d) => FRAGMENT_TYPE_COLOR[d.type] || "#444")
      .attr("stroke", "white").attr("stroke-width", 0.6)
      .append("title").text((d) => `${d.label || ""}\nm/z ${d.mz.toFixed(4)}`);

    g.selectAll("text.lab").data(peaks).enter().append("text")
      .attr("class", "lab")
      .attr("x", (_, i) => xs(i) + xs.bandwidth() / 2)
      .attr("y", (d) => ys(d.mz) + 14)
      .attr("text-anchor", "middle")
      .style("font-size", "8px").style("fill", PALETTE.text)
      .text((d) => d.mz.toFixed(1));
  }, [record, width, height]);

  return <svg ref={ref} width={width} height={height} style={{ background: PALETTE.bg }} />;
}

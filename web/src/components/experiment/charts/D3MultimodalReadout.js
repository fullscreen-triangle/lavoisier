import React from "react";
import * as d3 from "d3";
import { useD3 } from "./useD3";
import { PALETTE, classColor } from "./palette";

/**
 * Radar/spider chart of multimodal coordinates per record.
 * Axes: m/z, intensity, n, l, |m|, fragments, bits.
 */
export default function D3MultimodalReadout({ records, width = 360, height = 320 }) {
  const ref = useD3((el) => {
    const svg = d3.select(el);
    svg.selectAll("*").remove();
    if (!records || records.length === 0) return;

    const cx = width / 2;
    const cy = height / 2 + 8;
    const radius = Math.min(width, height) * 0.36;

    // Aggregate per class for the radar
    const classes = Array.from(new Set(records.map((d) => d.analyteClass)));

    const dimensions = [
      { key: "precursorMz",  label: "m/z",       max: 2000 },
      { key: "intensity",    label: "I",         max: 1.5 },
      { key: "n",            label: "n",         max: 8 },
      { key: "l",            label: "ℓ",         max: 5 },
      { key: "m_abs",        label: "|m|",       max: 4 },
      { key: "n_frags",      label: "frags",     max: 30 },
      { key: "bitsTotal",    label: "bits",      max: 30 },
    ];
    const N = dimensions.length;
    const angle = (i) => (i / N) * 2 * Math.PI - Math.PI / 2;

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .append("g")
      .attr("transform", `translate(${cx},${cy})`);

    // Concentric grid
    [0.25, 0.5, 0.75, 1.0].forEach((r) => {
      g.append("circle")
        .attr("cx", 0).attr("cy", 0).attr("r", radius * r)
        .attr("fill", "none").attr("stroke", PALETTE.grid).attr("stroke-width", 0.5);
    });
    // Axes
    dimensions.forEach((d, i) => {
      const a = angle(i);
      g.append("line")
        .attr("x1", 0).attr("y1", 0)
        .attr("x2", radius * Math.cos(a)).attr("y2", radius * Math.sin(a))
        .attr("stroke", PALETTE.grid).attr("stroke-width", 0.5);
      g.append("text")
        .attr("x", (radius + 12) * Math.cos(a))
        .attr("y", (radius + 12) * Math.sin(a))
        .attr("text-anchor", "middle").attr("dy", "0.35em")
        .style("font-size", "10px").style("fill", PALETTE.text)
        .text(d.label);
    });

    // Per-class polygon
    for (const cls of classes) {
      const subset = records.filter((r) => r.analyteClass === cls);
      const stat = dimensions.map((d) => {
        let avg;
        if (d.key === "m_abs")    avg = d3.mean(subset, (r) => Math.abs(r.m));
        else if (d.key === "n_frags") avg = d3.mean(subset, (r) => r.ms2.length);
        else avg = d3.mean(subset, (r) => r[d.key]);
        return Math.min(1, (avg || 0) / d.max);
      });
      const points = stat.map((v, i) => [
        radius * v * Math.cos(angle(i)),
        radius * v * Math.sin(angle(i)),
      ]);
      const path = d3.line().curve(d3.curveLinearClosed)(points);
      g.append("path")
        .attr("d", path)
        .attr("fill", classColor(cls))
        .attr("fill-opacity", 0.18)
        .attr("stroke", classColor(cls))
        .attr("stroke-width", 1.4)
        .append("title").text(`${cls} averages`);
    }
  }, [records, width, height]);

  return <svg ref={ref} width={width} height={height} style={{ background: PALETTE.bg }} />;
}

import React from "react";
import * as d3 from "d3";
import { useD3 } from "./useD3";
import { PALETTE, classColor, MARGIN } from "./palette";

/**
 * Pseudo-3D scatter of partition coordinates (n, l, m), colour by class.
 * Uses an axonometric projection (no Three.js needed; matches the panel
 * aesthetic).
 */
export default function D3PartitionScatter3D({
  records, width = 460, height = 360,
}) {
  const ref = useD3((el) => {
    const svg = d3.select(el);
    svg.selectAll("*").remove();
    if (!records || records.length === 0) return;

    const margin = { top: 28, right: 18, bottom: 26, left: 18 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const nMax = d3.max(records, (d) => d.n) || 8;
    const lMax = d3.max(records, (d) => d.l) || 8;
    const mMax = Math.max(d3.max(records, (d) => Math.abs(d.m)) || 4, 4);

    // Axonometric projection: x' = a*n - b*l, y' = c*m + d*l + e*n
    const a = w / (nMax * 1.7);
    const b = a * 0.5;
    const c = h / (3 * mMax + 2);
    const d_l = a * 0.35;
    const e_n = -a * 0.15;

    const project = (n, l, m) => ({
      x: margin.left + a * n - b * l + 50,
      y: margin.top + h - (c * m + d_l * l + e_n * n),
    });

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .append("g");

    // Axes
    const drawAxis = (from, to, label) => {
      g.append("line")
        .attr("x1", from.x).attr("y1", from.y)
        .attr("x2", to.x).attr("y2", to.y)
        .attr("stroke", PALETTE.axis).attr("stroke-width", 0.8);
      g.append("text")
        .attr("x", to.x).attr("y", to.y)
        .attr("text-anchor", "start")
        .style("font-size", "10px").style("fill", PALETTE.text)
        .text(label);
    };
    const o = project(0, 0, 0);
    drawAxis(o, project(nMax, 0, 0), "n");
    drawAxis(o, project(0, lMax, 0), "ℓ");
    drawAxis(o, project(0, 0, mMax), "+m");
    drawAxis(o, project(0, 0, -mMax), "−m");

    // Scatter
    g.selectAll("circle.pt")
      .data(records).enter().append("circle")
      .attr("class", "pt")
      .attr("cx", (d) => project(d.n, d.l, d.m).x)
      .attr("cy", (d) => project(d.n, d.l, d.m).y)
      .attr("r", (d) => 2.5 + 1.2 * Math.log10(1 + d.intensity * 1000))
      .attr("fill", (d) => classColor(d.analyteClass))
      .attr("opacity", 0.78)
      .attr("stroke", "white")
      .attr("stroke-width", 0.5)
      .append("title")
      .text((d) =>
        `${d.analyte}${d.adduct}\n(n,ℓ,m,s)=(${d.n},${d.l},${d.m},${d.s.toFixed(1)})`
      );
  }, [records, width, height]);

  return <svg ref={ref} width={width} height={height} style={{ background: PALETTE.bg }} />;
}

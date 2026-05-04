import React from "react";
import * as d3 from "d3";
import { useD3 } from "./useD3";
import { PALETTE, classColor } from "./palette";

/**
 * S-entropy unit cube [0,1]^3 with axonometric projection, points coloured
 * by lipid class. Mirrors the panel_E12 / spike-protein S-entropy plot.
 */
export default function D3SEntropyCube({ records, width = 460, height = 360 }) {
  const ref = useD3((el) => {
    const svg = d3.select(el);
    svg.selectAll("*").remove();
    if (!records || records.length === 0) return;

    const cx = width / 2 - 20;
    const cy = height / 2 + 30;
    const sz = Math.min(width, height) * 0.36;

    // Axonometric projection
    const project = (s_k, s_t, s_e) => ({
      x: cx + sz * (s_k - 0.5) - sz * 0.5 * (s_t - 0.5),
      y: cy - sz * (s_e - 0.5) - sz * 0.3 * (s_t - 0.5),
    });

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .append("g");

    // Cube edges
    const corners = [
      [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
      [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ];
    const edges = [
      [0, 1], [1, 2], [2, 3], [3, 0],
      [4, 5], [5, 6], [6, 7], [7, 4],
      [0, 4], [1, 5], [2, 6], [3, 7],
    ];
    edges.forEach(([a, b]) => {
      const p1 = project(...corners[a]);
      const p2 = project(...corners[b]);
      g.append("line")
        .attr("x1", p1.x).attr("y1", p1.y)
        .attr("x2", p2.x).attr("y2", p2.y)
        .attr("stroke", PALETTE.muted).attr("stroke-width", 0.6)
        .attr("stroke-dasharray", "2 2");
    });

    // Axis labels
    const lo = project(0, 0, 0);
    g.append("text").attr("x", project(1.05, 0, 0).x).attr("y", project(1.05, 0, 0).y)
      .text("Sₖ").style("font-size", "11px").style("fill", PALETTE.text);
    g.append("text").attr("x", project(0, 1.05, 0).x).attr("y", project(0, 1.05, 0).y)
      .text("Sₜ").style("font-size", "11px").style("fill", PALETTE.text);
    g.append("text").attr("x", project(0, 0, 1.05).x).attr("y", project(0, 0, 1.05).y)
      .text("Sₑ").style("font-size", "11px").style("fill", PALETTE.text);

    // Points
    g.selectAll("circle.s")
      .data(records).enter().append("circle")
      .attr("class", "s")
      .attr("cx", (d) => project(d.sentropy.sk, d.sentropy.st, d.sentropy.se).x)
      .attr("cy", (d) => project(d.sentropy.sk, d.sentropy.st, d.sentropy.se).y)
      .attr("r", 3)
      .attr("fill", (d) => classColor(d.analyteClass))
      .attr("opacity", 0.7)
      .attr("stroke", "white").attr("stroke-width", 0.4)
      .append("title").text((d) =>
        `${d.analyte}${d.adduct}\nSₖ ${d.sentropy.sk.toFixed(2)} Sₜ ${d.sentropy.st.toFixed(2)} Sₑ ${d.sentropy.se.toFixed(2)}`
      );
  }, [records, width, height]);

  return <svg ref={ref} width={width} height={height} style={{ background: PALETTE.bg }} />;
}

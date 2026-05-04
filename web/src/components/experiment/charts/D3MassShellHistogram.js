import React from "react";
import * as d3 from "d3";
import { useD3 } from "./useD3";
import { PALETTE, classColor, MARGIN } from "./palette";

/**
 * 2D heatmap of (precursor m/z, principal coord n), coloured by class density.
 * Demonstrates the mass-shell correspondence.
 */
export default function D3MassShellHistogram({ records, width = 460, height = 260 }) {
  const ref = useD3((el) => {
    const svg = d3.select(el);
    svg.selectAll("*").remove();
    if (!records || records.length === 0) return;

    const w = width - MARGIN.left - MARGIN.right;
    const h = height - MARGIN.top - MARGIN.bottom;

    const x = d3.scaleLinear()
      .domain(d3.extent(records, (d) => d.precursorMz)).nice()
      .range([0, w]);
    const ys = Array.from(new Set(records.map((d) => d.n))).sort((a, b) => a - b);
    const y = d3.scaleBand().domain(ys).range([h, 0]).padding(0.1);

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    g.append("g")
      .attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(6).tickSizeOuter(0))
      .selectAll("text").style("font-size", "10px");
    g.append("g")
      .call(d3.axisLeft(y).tickSizeOuter(0))
      .selectAll("text").style("font-size", "10px");

    g.append("text").attr("x", w / 2).attr("y", h + 30)
      .attr("text-anchor", "middle").style("font-size", "11px")
      .style("fill", PALETTE.text).text("precursor m/z");
    g.append("text").attr("transform", "rotate(-90)")
      .attr("x", -h / 2).attr("y", -36)
      .attr("text-anchor", "middle").style("font-size", "11px")
      .style("fill", PALETTE.text).text("n");

    // Capacity boundary lines (m/z scaling estimate)
    g.selectAll("line.cap").data(ys).enter().append("line")
      .attr("class", "cap")
      .attr("x1", 0).attr("x2", w)
      .attr("y1", (n) => y(n) + y.bandwidth() / 2)
      .attr("y2", (n) => y(n) + y.bandwidth() / 2)
      .attr("stroke", PALETTE.muted).attr("stroke-width", 0.4)
      .attr("stroke-dasharray", "2 2");

    g.selectAll("circle.r")
      .data(records).enter().append("circle")
      .attr("class", "r")
      .attr("cx", (d) => x(d.precursorMz))
      .attr("cy", (d) => y(d.n) + y.bandwidth() / 2)
      .attr("r", (d) => 2 + 1.5 * Math.log10(1 + d.intensity * 1000))
      .attr("fill", (d) => classColor(d.analyteClass))
      .attr("opacity", 0.7)
      .append("title").text((d) => `${d.analyte}${d.adduct}\nm/z ${d.precursorMz.toFixed(4)}, n=${d.n}`);
  }, [records, width, height]);

  return <svg ref={ref} width={width} height={height} style={{ background: PALETTE.bg }} />;
}

import React from "react";
import * as d3 from "d3";
import { useD3 } from "./useD3";
import { PALETTE } from "./palette";

/**
 * (l, m) occupancy heatmap at a chosen principal coordinate n.
 * Mirrors panel_E34 column D.
 */
export default function D3PartitionCellGrid({ records, n = null, width = 320, height = 280 }) {
  const ref = useD3((el) => {
    const svg = d3.select(el);
    svg.selectAll("*").remove();
    if (!records || records.length === 0) return;

    const nFix = n ?? d3.mode(records.map((d) => d.n));
    if (!nFix) return;
    const subset = records.filter((r) => r.n === nFix);

    const margin = { top: 16, right: 16, bottom: 30, left: 30 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const lDomain = d3.range(0, nFix);
    const mDomain = d3.range(-(nFix - 1), nFix);

    const xs = d3.scaleBand().domain(mDomain).range([0, w]).padding(0.04);
    const ys = d3.scaleBand().domain(lDomain).range([h, 0]).padding(0.04);

    const counts = d3.rollup(subset, (v) => v.length, (d) => d.l, (d) => d.m);
    const cells = [];
    for (const l of lDomain) {
      for (const m of mDomain) {
        const c = counts.get(l)?.get(m);
        const valid = Math.abs(m) <= l;
        cells.push({ l, m, count: c || 0, valid });
      }
    }
    const maxC = d3.max(cells, (d) => d.count) || 1;
    const color = d3.scaleSequential(d3.interpolateBlues).domain([0, maxC]);

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    g.append("g").attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(xs).tickValues(mDomain.filter((_, i) => i % 2 === 0)).tickSizeOuter(0))
      .selectAll("text").style("font-size", "9px");
    g.append("g").call(d3.axisLeft(ys).tickSizeOuter(0))
      .selectAll("text").style("font-size", "9px");
    g.append("text").attr("x", w / 2).attr("y", h + 24).attr("text-anchor", "middle")
      .style("font-size", "10px").style("fill", PALETTE.text).text("m");
    g.append("text").attr("transform", "rotate(-90)").attr("x", -h / 2).attr("y", -22)
      .attr("text-anchor", "middle").style("font-size", "10px").style("fill", PALETTE.text)
      .text("ℓ");

    g.append("text").attr("x", w / 2).attr("y", -4).attr("text-anchor", "middle")
      .style("font-size", "10px").style("fill", PALETTE.muted)
      .text(`n = ${nFix}, capacity = ${2 * nFix * nFix}`);

    g.selectAll("rect.cell").data(cells).enter().append("rect")
      .attr("class", "cell")
      .attr("x", (d) => xs(d.m))
      .attr("y", (d) => ys(d.l))
      .attr("width", xs.bandwidth())
      .attr("height", ys.bandwidth())
      .attr("fill", (d) => (!d.valid ? "#fbeded" : d.count === 0 ? "#f5f5f5" : color(d.count)))
      .attr("stroke", "white").attr("stroke-width", 0.4)
      .append("title").text((d) => `(ℓ=${d.l}, m=${d.m})${d.valid ? "" : " forbidden"}: ${d.count}`);
  }, [records, n, width, height]);

  return <svg ref={ref} width={width} height={height} style={{ background: PALETTE.bg }} />;
}

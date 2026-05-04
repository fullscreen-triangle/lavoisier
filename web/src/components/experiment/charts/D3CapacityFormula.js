import React from "react";
import * as d3 from "d3";
import { useD3 } from "./useD3";
import { PALETTE, MARGIN } from "./palette";

/**
 * Capacity formula plot: theoretical C(n)=2n^2 vs observed shell occupancy.
 * Matches panel_E34 column B.
 */
export default function D3CapacityFormula({ records, width = 360, height = 220 }) {
  const ref = useD3((el) => {
    const svg = d3.select(el);
    svg.selectAll("*").remove();
    if (!records || records.length === 0) return;

    const w = width - MARGIN.left - MARGIN.right;
    const h = height - MARGIN.top - MARGIN.bottom;

    const shellCount = d3.rollup(records, (v) => v.length, (d) => d.n);
    const ns = Array.from(shellCount.keys()).sort((a, b) => a - b);
    const nMax = Math.max(...ns, 8);
    const observed = ns.map((n) => ({ n, count: shellCount.get(n) }));

    const x = d3.scaleLinear().domain([1, nMax + 1]).range([0, w]);
    const yMax = Math.max(d3.max(observed, (d) => d.count) || 0, 2 * Math.pow(nMax, 2));
    const y = d3.scaleLinear().domain([0, yMax * 1.1]).range([h, 0]);

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    g.append("g")
      .attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(8).tickSizeOuter(0))
      .selectAll("text").style("font-size", "10px");
    g.append("g")
      .call(d3.axisLeft(y).ticks(5).tickSizeOuter(0))
      .selectAll("text").style("font-size", "10px");
    g.append("text").attr("x", w / 2).attr("y", h + 30)
      .attr("text-anchor", "middle").style("font-size", "11px").style("fill", PALETTE.text)
      .text("n");
    g.append("text").attr("transform", "rotate(-90)")
      .attr("x", -h / 2).attr("y", -36)
      .attr("text-anchor", "middle").style("font-size", "11px").style("fill", PALETTE.text)
      .text("count");

    // Theoretical line C(n) = 2n^2
    const theory = d3.range(1, nMax + 1, 0.1).map((n) => ({ n, c: 2 * n * n }));
    const line = d3.line().x((d) => x(d.n)).y((d) => y(d.c)).curve(d3.curveMonotoneX);
    g.append("path").datum(theory)
      .attr("fill", "none").attr("stroke", PALETTE.PC).attr("stroke-width", 1.4)
      .attr("d", line);

    // Observed points
    g.selectAll("circle.obs")
      .data(observed).enter().append("circle")
      .attr("class", "obs")
      .attr("cx", (d) => x(d.n))
      .attr("cy", (d) => y(d.count))
      .attr("r", 4).attr("fill", "#d62728").attr("stroke", "white").attr("stroke-width", 0.6);
  }, [records, width, height]);

  return <svg ref={ref} width={width} height={height} style={{ background: PALETTE.bg }} />;
}

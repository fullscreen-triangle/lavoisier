import React from "react";
import * as d3 from "d3";
import { useD3 } from "./useD3";
import { PALETTE, classColor, MARGIN } from "./palette";

/**
 * Per-class chain coverage matrix: cell colour = predicted intensity at
 * (X, Y); empty cells = not in design space.
 */
export default function D3CoverageMatrix({ records, width = 460, height = 320 }) {
  const ref = useD3((el) => {
    const svg = d3.select(el);
    svg.selectAll("*").remove();
    if (!records || records.length === 0) return;

    const w = width - MARGIN.left - MARGIN.right;
    const h = height - MARGIN.top - MARGIN.bottom;

    // Aggregate intensity per (class, X, Y), summing across adducts
    const grouped = d3.rollups(
      records,
      (v) => d3.sum(v, (d) => d.intensity),
      (d) => d.analyteClass, (d) => d.X, (d) => d.Y
    );

    const cells = [];
    grouped.forEach(([cls, byX]) => {
      byX.forEach(([X, byY]) => {
        byY.forEach(([Y, val]) => {
          cells.push({ cls, X, Y, intensity: val });
        });
      });
    });

    const xExtent = d3.extent(cells, (d) => d.X);
    const yExtent = [0, d3.max(cells, (d) => d.Y)];

    const xs = d3.scaleLinear().domain([xExtent[0] - 1, xExtent[1] + 1]).range([0, w]);
    const ys = d3.scaleLinear().domain([yExtent[0] - 0.5, yExtent[1] + 0.5]).range([h, 0]);

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    g.append("g").attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(xs).ticks(8).tickSizeOuter(0))
      .selectAll("text").style("font-size", "10px");
    g.append("g").call(d3.axisLeft(ys).ticks(yExtent[1] + 1).tickSizeOuter(0))
      .selectAll("text").style("font-size", "10px");
    g.append("text").attr("x", w / 2).attr("y", h + 30).attr("text-anchor", "middle")
      .style("font-size", "11px").style("fill", PALETTE.text).text("X (acyl carbons)");
    g.append("text").attr("transform", "rotate(-90)").attr("x", -h / 2).attr("y", -36)
      .attr("text-anchor", "middle").style("font-size", "11px").style("fill", PALETTE.text)
      .text("Y (double bonds)");

    g.selectAll("rect.cv").data(cells).enter().append("rect")
      .attr("class", "cv")
      .attr("x", (d) => xs(d.X) - 5)
      .attr("y", (d) => ys(d.Y) - 5)
      .attr("width", 10).attr("height", 10)
      .attr("fill", (d) => classColor(d.cls))
      .attr("opacity", (d) => Math.min(1, 0.2 + d.intensity * 1.5))
      .append("title").text((d) => `${d.cls}(${d.X}:${d.Y}): I=${d.intensity.toFixed(3)}`);
  }, [records, width, height]);

  return <svg ref={ref} width={width} height={height} style={{ background: PALETTE.bg }} />;
}

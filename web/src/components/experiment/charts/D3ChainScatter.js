import React from "react";
import * as d3 from "d3";
import { useD3 } from "./useD3";
import { PALETTE, classColor, MARGIN } from "./palette";

/**
 * Chain composition scatter: total acyl carbons (X) vs total double bonds (Y),
 * coloured by class, sized by intensity. Mirrors the typical "lipid map"
 * presentation in lipidomics papers.
 */
export default function D3ChainScatter({ records, width = 460, height = 280 }) {
  const ref = useD3((el) => {
    const svg = d3.select(el);
    svg.selectAll("*").remove();
    if (!records || records.length === 0) return;

    const w = width - MARGIN.left - MARGIN.right;
    const h = height - MARGIN.top - MARGIN.bottom;

    const xExtent = d3.extent(records, (d) => d.X);
    const yExtent = [0, d3.max(records, (d) => d.Y) + 1];
    const x = d3.scaleLinear().domain([xExtent[0] - 1, xExtent[1] + 1]).range([0, w]);
    const y = d3.scaleLinear().domain(yExtent).range([h, 0]);

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    g.append("g").attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(8).tickSizeOuter(0))
      .selectAll("text").style("font-size", "10px");
    g.append("g")
      .call(d3.axisLeft(y).ticks(6).tickSizeOuter(0))
      .selectAll("text").style("font-size", "10px");

    g.append("text").attr("x", w / 2).attr("y", h + 30)
      .attr("text-anchor", "middle").style("font-size", "11px")
      .style("fill", PALETTE.text).text("acyl carbons (X)");
    g.append("text").attr("transform", "rotate(-90)")
      .attr("x", -h / 2).attr("y", -30)
      .attr("text-anchor", "middle").style("font-size", "11px")
      .style("fill", PALETTE.text).text("double bonds (Y)");

    g.selectAll("circle.r").data(records).enter().append("circle")
      .attr("class", "r")
      .attr("cx", (d) => x(d.X))
      .attr("cy", (d) => y(d.Y))
      .attr("r", (d) => 3 + 8 * Math.sqrt(d.intensity))
      .attr("fill", (d) => classColor(d.analyteClass))
      .attr("opacity", 0.65)
      .attr("stroke", "white").attr("stroke-width", 0.6)
      .append("title").text((d) => `${d.analyte}${d.adduct}\nm/z ${d.precursorMz.toFixed(3)}`);
  }, [records, width, height]);

  return <svg ref={ref} width={width} height={height} style={{ background: PALETTE.bg }} />;
}

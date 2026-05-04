import React from "react";
import * as d3 from "d3";
import { useD3 } from "./useD3";
import { PALETTE, MARGIN } from "./palette";

/**
 * Predicted isotope envelope for the currently-selected record.
 * Shows three peaks (M, M+1, M+2) at relative natural abundance.
 */
export default function D3IsotopeEnvelope({ record, width = 320, height = 200 }) {
  const ref = useD3((el) => {
    const svg = d3.select(el);
    svg.selectAll("*").remove();
    if (!record) return;
    const peaks = record.ms1;
    if (!peaks || peaks.length === 0) return;

    const w = width - MARGIN.left - MARGIN.right;
    const h = height - MARGIN.top - MARGIN.bottom;
    const xExtent = [
      peaks[0].mz - 0.5,
      peaks[peaks.length - 1].mz + 0.5,
    ];
    const x = d3.scaleLinear().domain(xExtent).range([0, w]);
    const yMax = d3.max(peaks, (d) => d.intensity);
    const y = d3.scaleLinear().domain([0, yMax * 1.1]).range([h, 0]);

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    g.append("g").attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(5, ".4f").tickSizeOuter(0))
      .selectAll("text").style("font-size", "9px");
    g.append("g")
      .call(d3.axisLeft(y).ticks(4).tickFormat(d3.format(".1e")).tickSizeOuter(0))
      .selectAll("text").style("font-size", "9px");

    g.append("text").attr("x", w / 2).attr("y", h + 28)
      .attr("text-anchor", "middle").style("font-size", "10px")
      .style("fill", PALETTE.text).text("m/z");

    g.selectAll("line.iso").data(peaks).enter().append("line")
      .attr("class", "iso")
      .attr("x1", (d) => x(d.mz)).attr("x2", (d) => x(d.mz))
      .attr("y1", y(0)).attr("y2", (d) => y(d.intensity))
      .attr("stroke", PALETTE.PC).attr("stroke-width", 2.5);

    g.selectAll("text.lab").data(peaks).enter().append("text")
      .attr("class", "lab")
      .attr("x", (d) => x(d.mz))
      .attr("y", (d) => y(d.intensity) - 4)
      .attr("text-anchor", "middle")
      .style("font-size", "9px").style("fill", PALETTE.text)
      .text((d) => d.label);
  }, [record, width, height]);

  return <svg ref={ref} width={width} height={height} style={{ background: PALETTE.bg }} />;
}

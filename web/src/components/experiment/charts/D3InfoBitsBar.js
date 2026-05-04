import React from "react";
import * as d3 from "d3";
import { useD3 } from "./useD3";
import { PALETTE, classColor, MARGIN } from "./palette";

/**
 * Per-record total information content (bits), grouped by class.
 * Mirrors the multimodal-bits panel from the publications.
 */
export default function D3InfoBitsBar({ records, width = 460, height = 220 }) {
  const ref = useD3((el) => {
    const svg = d3.select(el);
    svg.selectAll("*").remove();
    if (!records || records.length === 0) return;

    const w = width - MARGIN.left - MARGIN.right;
    const h = height - MARGIN.top - MARGIN.bottom;

    const data = d3.rollups(
      records,
      (v) => ({
        count: v.length,
        mean: d3.mean(v, (d) => d.bitsTotal),
        std: d3.deviation(v, (d) => d.bitsTotal) || 0,
      }),
      (d) => d.analyteClass
    ).map(([cls, s]) => ({ cls, ...s }))
      .sort((a, b) => b.mean - a.mean);

    const x = d3.scaleBand().domain(data.map((d) => d.cls)).range([0, w]).padding(0.2);
    const y = d3.scaleLinear().domain([0, d3.max(data, (d) => d.mean + d.std) * 1.05]).range([h, 0]);

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    g.append("g").attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).tickSizeOuter(0)).selectAll("text").style("font-size", "10px");
    g.append("g").call(d3.axisLeft(y).ticks(5).tickSizeOuter(0))
      .selectAll("text").style("font-size", "10px");
    g.append("text").attr("x", w / 2).attr("y", h + 30).attr("text-anchor", "middle")
      .style("font-size", "11px").style("fill", PALETTE.text).text("class");
    g.append("text").attr("transform", "rotate(-90)").attr("x", -h / 2).attr("y", -36)
      .attr("text-anchor", "middle").style("font-size", "11px").style("fill", PALETTE.text)
      .text("bits / record");

    g.selectAll("rect.bar").data(data).enter().append("rect")
      .attr("class", "bar")
      .attr("x", (d) => x(d.cls))
      .attr("y", (d) => y(d.mean))
      .attr("width", x.bandwidth())
      .attr("height", (d) => h - y(d.mean))
      .attr("fill", (d) => classColor(d.cls))
      .attr("opacity", 0.85);

    g.selectAll("line.err").data(data).enter().append("line")
      .attr("class", "err")
      .attr("x1", (d) => x(d.cls) + x.bandwidth() / 2)
      .attr("x2", (d) => x(d.cls) + x.bandwidth() / 2)
      .attr("y1", (d) => y(d.mean - d.std))
      .attr("y2", (d) => y(d.mean + d.std))
      .attr("stroke", PALETTE.text).attr("stroke-width", 1);

    // Reference line at conventional MS bit rate (~20 bits)
    g.append("line").attr("x1", 0).attr("x2", w)
      .attr("y1", y(20)).attr("y2", y(20))
      .attr("stroke", PALETTE.muted).attr("stroke-dasharray", "3 3").attr("stroke-width", 0.8);
    g.append("text").attr("x", 4).attr("y", y(20) - 4)
      .style("font-size", "9px").style("fill", PALETTE.muted)
      .text("conventional ~20 bits");
  }, [records, width, height]);

  return <svg ref={ref} width={width} height={height} style={{ background: PALETTE.bg }} />;
}

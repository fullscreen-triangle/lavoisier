import React from "react";
import * as d3 from "d3";
import { useD3 } from "./useD3";
import { PALETTE, classColor, MARGIN } from "./palette";

/**
 * Per-class precursor m/z distribution as overlapping violins.
 */
export default function D3MassRangeViolin({ records, width = 460, height = 280 }) {
  const ref = useD3((el) => {
    const svg = d3.select(el);
    svg.selectAll("*").remove();
    if (!records || records.length === 0) return;

    const w = width - MARGIN.left - MARGIN.right;
    const h = height - MARGIN.top - MARGIN.bottom;

    const grouped = d3.rollups(
      records, (v) => v.map((d) => d.precursorMz), (d) => d.analyteClass
    );
    const classes = grouped.map(([k]) => k);

    const xs = d3.scaleBand().domain(classes).range([0, w]).padding(0.2);
    const yExtent = d3.extent(records, (d) => d.precursorMz);
    const y = d3.scaleLinear().domain(yExtent).nice().range([h, 0]);

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    g.append("g").attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(xs).tickSizeOuter(0)).selectAll("text").style("font-size", "10px");
    g.append("g").call(d3.axisLeft(y).ticks(6).tickSizeOuter(0))
      .selectAll("text").style("font-size", "10px");
    g.append("text").attr("x", w / 2).attr("y", h + 30).attr("text-anchor", "middle")
      .style("font-size", "11px").style("fill", PALETTE.text).text("class");
    g.append("text").attr("transform", "rotate(-90)").attr("x", -h / 2).attr("y", -36)
      .attr("text-anchor", "middle").style("font-size", "11px").style("fill", PALETTE.text)
      .text("precursor m/z");

    const histGen = d3.histogram().domain(y.domain()).thresholds(20);

    grouped.forEach(([cls, values]) => {
      const bins = histGen(values);
      const xMid = xs(cls) + xs.bandwidth() / 2;
      const maxBin = d3.max(bins, (b) => b.length) || 1;
      const widthScale = d3.scaleLinear()
        .domain([0, maxBin])
        .range([0, xs.bandwidth() / 2]);
      const area = d3.area()
        .x0((b) => xMid - widthScale(b.length))
        .x1((b) => xMid + widthScale(b.length))
        .y((b) => y((b.x0 + b.x1) / 2))
        .curve(d3.curveBasis);
      g.append("path").datum(bins)
        .attr("d", area)
        .attr("fill", classColor(cls))
        .attr("opacity", 0.6)
        .attr("stroke", classColor(cls))
        .attr("stroke-width", 0.8);
    });
  }, [records, width, height]);

  return <svg ref={ref} width={width} height={height} style={{ background: PALETTE.bg }} />;
}

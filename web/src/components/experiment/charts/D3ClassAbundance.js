import React from "react";
import * as d3 from "d3";
import { useD3 } from "./useD3";
import { PALETTE, classColor, MARGIN } from "./palette";

/**
 * Stacked bar of per-class counts and summed predicted intensity, plus a
 * small donut showing fractional intensity by class.
 */
export default function D3ClassAbundance({ records, width = 460, height = 280 }) {
  const ref = useD3((el) => {
    const svg = d3.select(el);
    svg.selectAll("*").remove();
    if (!records || records.length === 0) return;

    const summary = d3.rollups(
      records,
      (v) => ({
        count: v.length,
        intensity: d3.sum(v, (d) => d.intensity),
      }),
      (d) => d.analyteClass
    ).map(([cls, s]) => ({ cls, ...s }))
      .sort((a, b) => b.intensity - a.intensity);

    const w = width - MARGIN.left - MARGIN.right;
    const h = height - MARGIN.top - MARGIN.bottom - 20;

    const x = d3.scaleBand().domain(summary.map((d) => d.cls)).range([0, w * 0.55]).padding(0.15);
    const y = d3.scaleLinear().domain([0, d3.max(summary, (d) => d.intensity) * 1.05]).range([h, 0]);

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    g.append("g")
      .attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).tickSizeOuter(0))
      .selectAll("text").style("font-size", "10px");
    g.append("g")
      .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".1f")).tickSizeOuter(0))
      .selectAll("text").style("font-size", "10px");
    g.append("text").attr("x", (w * 0.55) / 2).attr("y", h + 30)
      .attr("text-anchor", "middle").style("font-size", "11px")
      .style("fill", PALETTE.text).text("class");
    g.append("text").attr("transform", "rotate(-90)")
      .attr("x", -h / 2).attr("y", -36)
      .attr("text-anchor", "middle").style("font-size", "11px")
      .style("fill", PALETTE.text).text("∑ predicted I");

    g.selectAll("rect.bar")
      .data(summary).enter().append("rect")
      .attr("class", "bar")
      .attr("x", (d) => x(d.cls))
      .attr("y", (d) => y(d.intensity))
      .attr("width", x.bandwidth())
      .attr("height", (d) => h - y(d.intensity))
      .attr("fill", (d) => classColor(d.cls))
      .append("title").text((d) => `${d.cls}: ${d.count} species, I=${d.intensity.toFixed(3)}`);

    // Donut on right side (count distribution)
    const cx = w * 0.78, cy = h / 2;
    const radius = Math.min(w * 0.18, h * 0.5);
    const pie = d3.pie().value((d) => d.count).sort(null)(summary);
    const arc = d3.arc().innerRadius(radius * 0.55).outerRadius(radius);
    const dg = svg.select("g").append("g").attr("transform", `translate(${cx},${cy})`);
    dg.selectAll("path.slice").data(pie).enter().append("path")
      .attr("d", arc)
      .attr("fill", (d) => classColor(d.data.cls))
      .attr("stroke", "white").attr("stroke-width", 1)
      .append("title").text((d) => `${d.data.cls}: ${d.data.count}`);
    dg.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "0.3em")
      .style("font-size", "12px").style("fill", PALETTE.text)
      .text(`${records.length}`);
    dg.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "1.6em")
      .style("font-size", "9px").style("fill", PALETTE.muted)
      .text("species");
  }, [records, width, height]);

  return <svg ref={ref} width={width} height={height} style={{ background: PALETTE.bg }} />;
}

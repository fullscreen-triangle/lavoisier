import React from "react";
import * as d3 from "d3";
import { useD3 } from "./useD3";
import { PALETTE, FRAGMENT_TYPE_COLOR, MARGIN } from "./palette";

/**
 * Predicted MS spectrum (m/z vs intensity). Stems coloured by fragment type.
 *
 * @param {Object} props
 * @param {Array} props.peaks  [{mz, intensity, type, label}]
 * @param {[number,number]} [props.mzRange]
 * @param {number} props.width
 * @param {number} props.height
 */
export default function D3SpectrumChart({ peaks, mzRange, width = 800, height = 280 }) {
  const ref = useD3((el) => {
    const svg = d3.select(el);
    svg.selectAll("*").remove();
    if (!peaks || peaks.length === 0) return;

    const w = width - MARGIN.left - MARGIN.right;
    const h = height - MARGIN.top - MARGIN.bottom;

    const xExtent = mzRange ?? d3.extent(peaks, (d) => d.mz);
    const yMax = d3.max(peaks, (d) => d.intensity) ?? 1;

    const x = d3.scaleLinear().domain(xExtent).nice().range([0, w]);
    const y = d3.scaleLinear().domain([0, yMax * 1.05]).range([h, 0]);

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    // axes
    g.append("g")
      .attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(8).tickSizeOuter(0))
      .selectAll("text").style("font-size", "10px");
    g.append("g")
      .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".1e")).tickSizeOuter(0))
      .selectAll("text").style("font-size", "10px");

    g.append("text")
      .attr("x", w / 2).attr("y", h + 30).attr("text-anchor", "middle")
      .style("font-size", "11px").style("fill", PALETTE.text)
      .text("m/z");
    g.append("text")
      .attr("transform", "rotate(-90)").attr("x", -h / 2).attr("y", -36)
      .attr("text-anchor", "middle").style("font-size", "11px")
      .style("fill", PALETTE.text).text("intensity");

    // baseline
    g.append("line")
      .attr("x1", 0).attr("x2", w)
      .attr("y1", y(0)).attr("y2", y(0))
      .attr("stroke", PALETTE.muted).attr("stroke-width", 0.5);

    // stems
    const stems = g.selectAll("line.stem")
      .data(peaks).enter().append("line")
      .attr("class", "stem")
      .attr("x1", (d) => x(d.mz)).attr("x2", (d) => x(d.mz))
      .attr("y1", y(0)).attr("y2", (d) => y(d.intensity))
      .attr("stroke", (d) => FRAGMENT_TYPE_COLOR[d.type] || "#444")
      .attr("stroke-width", (d) => (d.type === "precursor" ? 1.5 : 0.9));

    // tooltips
    const tip = d3.select(el.parentNode).select("div.tooltip-box");
    stems
      .on("mouseenter", function (e, d) {
        d3.select(this).attr("stroke-width", 2.4);
        tip
          .style("opacity", 1)
          .html(
            `<b>${d.label || ""}</b><br/>m/z ${d.mz.toFixed(4)}<br/>I ${d.intensity.toExponential(2)}`
          )
          .style("left", e.offsetX + 12 + "px")
          .style("top", e.offsetY - 6 + "px");
      })
      .on("mouseleave", function (_, d) {
        d3.select(this).attr("stroke-width", d.type === "precursor" ? 1.5 : 0.9);
        tip.style("opacity", 0);
      });
  }, [peaks, mzRange, width, height]);

  return (
    <div className="relative w-full">
      <svg ref={ref} width={width} height={height} style={{ background: PALETTE.bg }} />
      <div
        className="tooltip-box absolute pointer-events-none px-2 py-1 rounded
        bg-dark/95 dark:bg-light/95 text-light dark:text-dark text-[10px]
        leading-tight shadow-lg"
        style={{ opacity: 0, transition: "opacity 80ms" }}
      />
    </div>
  );
}

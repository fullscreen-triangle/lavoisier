import React from "react";
import * as d3 from "d3";
import { useD3 } from "./useD3";
import { PALETTE, MARGIN } from "./palette";

/**
 * R(omega, T) ridge plot: resolving power as a function of frequency and
 * residence time. Renders multiple traces (one per residence time) on a
 * log-log axis. Mirrors panel_E5E10 column A flattened to 2D.
 */
export default function D3ResolutionSurface({ width = 460, height = 280 }) {
  const ref = useD3((el) => {
    const svg = d3.select(el);
    svg.selectAll("*").remove();

    const w = width - MARGIN.left - MARGIN.right;
    const h = height - MARGIN.top - MARGIN.bottom;

    const Ts = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6];
    const omegas = d3.range(3, 8.5, 0.05).map((logw) => Math.pow(10, logw));

    const x = d3.scaleLog().domain([1e3, 1e8]).range([0, w]);
    const y = d3.scaleLog().domain([1, 1e15]).range([h, 0]);

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    g.append("g")
      .attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(5, ".1e").tickSizeOuter(0))
      .selectAll("text").style("font-size", "10px");
    g.append("g")
      .call(d3.axisLeft(y).ticks(8, ".0e").tickSizeOuter(0))
      .selectAll("text").style("font-size", "10px");

    g.append("text").attr("x", w / 2).attr("y", h + 30)
      .attr("text-anchor", "middle").style("font-size", "11px")
      .style("fill", PALETTE.text).text("ω (Hz)");
    g.append("text").attr("transform", "rotate(-90)")
      .attr("x", -h / 2).attr("y", -38)
      .attr("text-anchor", "middle").style("font-size", "11px")
      .style("fill", PALETTE.text).text("R");

    const cmap = d3.scaleSequential(d3.interpolateViridis).domain([Math.log10(Ts[Ts.length - 1]), Math.log10(Ts[0])]);
    const line = d3.line()
      .x((d) => x(d.w))
      .y((d) => y(d.R));

    Ts.forEach((T, i) => {
      const data = omegas.map((w_) => ({ w: w_, R: w_ * T / (2 * Math.PI) }));
      g.append("path").datum(data)
        .attr("fill", "none").attr("stroke", cmap(Math.log10(T)))
        .attr("stroke-width", 1.2).attr("d", line);
      g.append("text")
        .attr("x", x(omegas[omegas.length - 1])).attr("y", y(omegas[omegas.length - 1] * T / (2 * Math.PI)))
        .attr("dx", 4)
        .style("font-size", "8px").style("fill", cmap(Math.log10(T)))
        .text(`T=${T < 1 ? T.toFixed(3) : T}s`);
    });

    // Best-Orbitrap reference line
    g.append("line").attr("x1", 0).attr("x2", w)
      .attr("y1", y(1e6)).attr("y2", y(1e6))
      .attr("stroke", PALETTE.muted).attr("stroke-width", 0.6).attr("stroke-dasharray", "4 3");
    g.append("text").attr("x", 4).attr("y", y(1e6) - 4)
      .style("font-size", "9px").style("fill", PALETTE.muted)
      .text("best Orbitrap R ~ 10⁶");
  }, [width, height]);

  return <svg ref={ref} width={width} height={height} style={{ background: PALETTE.bg }} />;
}

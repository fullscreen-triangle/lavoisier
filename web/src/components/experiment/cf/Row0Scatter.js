import React, { useCallback, useRef } from "react";
import * as d3 from "d3";
import { useCrossfilter, useChartRedraw } from "./CrossfilterContext";
import {
  PALETTE, TYPO, classColor, setupSvg, drawAxis, axisLabel,
  attachXYBrush, applyTextStyle,
} from "./chartUtils";

/**
 * Row 0: full-width 2D scatter of every record.
 *   x = precursor m/z  (linear)
 *   y = predicted intensity (log)
 *   color = lipid class
 *   size = sqrt(intensity)
 * 2D brush sets both dim.mz and dim.intensity filters simultaneously.
 */
export default function Row0Scatter({ height = 220 }) {
  const { pack, redrawAll } = useCrossfilter();
  const ref = useRef(null);

  const render = useCallback(() => {
    const node = ref.current;
    if (!node) return;
    const width = node.clientWidth || 920;
    const all = pack.dims.mz.bottom(Infinity);
    const margin = { top: 10, right: 14, bottom: 26, left: 44 };
    const { svg, g, w, h } = setupSvg(node, width, height, margin);
    if (all.length === 0) return;

    const x = d3.scaleLinear()
      .domain(d3.extent(all, (d) => d.precursorMz)).nice()
      .range([0, w]);

    const iExtent = d3.extent(all, (d) => d.intensity);
    const iLo = Math.max(iExtent[0], 1e-6);
    const y = d3.scaleLog().domain([iLo, iExtent[1] * 1.05]).range([h, 0]);

    // grid (very faint)
    g.append("g").attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(8).tickSize(-h).tickFormat(""))
      .call((s) => s.selectAll("line").attr("stroke", PALETTE.grid).attr("stroke-width", 0.4))
      .call((s) => s.select("path").remove());
    g.append("g")
      .call(d3.axisLeft(y).ticks(4).tickSize(-w).tickFormat(""))
      .call((s) => s.selectAll("line").attr("stroke", PALETTE.grid).attr("stroke-width", 0.4))
      .call((s) => s.select("path").remove());

    drawAxis(g, x, "bottom", h, 8);
    drawAxis(g, y, "left", h, 4);
    axisLabel(g, w, h, "precursor m/z", "predicted I (log)");

    // points — small, semi-transparent, no stroke
    g.selectAll("circle.pt").data(all).enter().append("circle")
      .attr("class", "pt")
      .attr("cx", (d) => x(d.precursorMz))
      .attr("cy", (d) => y(Math.max(iLo, d.intensity)))
      .attr("r", (d) => 1.4 + 1.6 * Math.sqrt(d.intensity))
      .attr("fill", (d) => classColor(d.analyteClass))
      .attr("fill-opacity", 0.55)
      .append("title").text((d) =>
        `${d.analyte}${d.adduct}\nm/z ${d.precursorMz.toFixed(3)}\nI ${d.intensity.toExponential(2)}`
      );

    attachXYBrush(g, w, h,
      () => {},
      (extent) => {
        if (!extent) {
          pack.dims.mz.filterAll();
          pack.dims.intensity.filterAll();
        } else {
          const [[x0, y0], [x1, y1]] = extent;
          pack.dims.mz.filterRange([x.invert(x0), x.invert(x1)]);
          // y is log; remember y0 is at top (smaller Y px == larger I)
          pack.dims.intensity.filterRange([y.invert(y1), y.invert(y0)]);
        }
        redrawAll();
      });
  }, [pack, height, redrawAll]);

  useChartRedraw(render);

  return <svg ref={ref} className="w-full" style={{ height }} />;
}

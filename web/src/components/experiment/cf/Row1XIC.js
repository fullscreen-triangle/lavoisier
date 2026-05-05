import React, { useCallback, useRef } from "react";
import * as d3 from "d3";
import { useCrossfilter, useChartRedraw } from "./CrossfilterContext";
import { PALETTE, setupSvg, drawAxis, axisLabel, attachXBrush, snapshot } from "./chartUtils";

/**
 * Row 1: Full width XIC line + brushable m/z bar histogram beneath.
 * The XIC here is the predicted m/z spectrum (m/z vs total intensity within bin).
 * Brushing the bottom bars filters to that m/z range; all other charts redraw.
 */
export default function Row1XIC({ height = 240 }) {
  const { pack, redrawAll } = useCrossfilter();
  const lineRef = useRef(null);
  const barRef = useRef(null);
  const lastBrushRef = useRef(null);

  const lineHeight = Math.round(height * 0.62);
  const barHeight = height - lineHeight - 4;

  const renderLine = useCallback(() => {
    const node = lineRef.current;
    if (!node) return;
    const width = node.clientWidth || 920;

    const data = snapshot(pack.groups.mzBin)
      .filter((d) => d.value > 0)
      .sort((a, b) => a.key - b.key);

    const { g, w, h } = setupSvg(node, width, lineHeight);
    if (data.length === 0) return;
    const x = d3.scaleLinear()
      .domain(d3.extent(data, (d) => d.key)).nice()
      .range([0, w]);
    const y = d3.scaleLinear()
      .domain([0, d3.max(data, (d) => d.value) * 1.05]).nice()
      .range([h, 0]);

    drawAxis(g, x, "bottom", h, 8);
    drawAxis(g, y, "left", h, 4);
    axisLabel(g, w, h, "m/z", "Σ I (XIC)");

    const line = d3.line()
      .x((d) => x(d.key + pack.mzBinSize / 2))
      .y((d) => y(d.value))
      .curve(d3.curveMonotoneX);
    g.append("path").datum(data)
      .attr("fill", "none").attr("stroke", PALETTE.fill).attr("stroke-width", 1.4)
      .attr("d", line);
    g.append("path").datum(data)
      .attr("fill", PALETTE.fill).attr("opacity", 0.18)
      .attr("d", d3.area()
        .x((d) => x(d.key + pack.mzBinSize / 2))
        .y0(h).y1((d) => y(d.value))
        .curve(d3.curveMonotoneX));
  }, [pack, lineHeight]);

  const renderBars = useCallback(() => {
    const node = barRef.current;
    if (!node) return;
    const width = node.clientWidth || 920;

    const data = snapshot(pack.groups.mzBin)
      .filter((d) => d.value > 0)
      .sort((a, b) => a.key - b.key);

    const { svg, g, w, h } = setupSvg(node, width, barHeight,
      { top: 4, right: 14, bottom: 24, left: 38 });
    if (data.length === 0) return;
    const x = d3.scaleLinear()
      .domain(d3.extent(data, (d) => d.key)).nice()
      .range([0, w]);
    const y = d3.scaleLinear()
      .domain([0, d3.max(data, (d) => d.value) * 1.05])
      .range([h, 0]);

    drawAxis(g, x, "bottom", h, 8);

    const barW = Math.max(1, w / Math.max(1, data.length) - 1);
    g.selectAll("rect.b").data(data).enter().append("rect")
      .attr("class", "b")
      .attr("x", (d) => x(d.key))
      .attr("y", (d) => y(d.value))
      .attr("width", barW)
      .attr("height", (d) => h - y(d.value))
      .attr("fill", PALETTE.fill).attr("opacity", 0.65);

    attachXBrush(g, w, h,
      () => {},
      (extent) => {
        if (!extent) {
          pack.dims.mz.filterAll();
        } else {
          const [a, b] = extent;
          pack.dims.mz.filterRange([x.invert(a), x.invert(b)]);
        }
        lastBrushRef.current = extent;
        redrawAll();
      });
  }, [pack, barHeight, redrawAll]);

  const redraw = useCallback(() => { renderLine(); renderBars(); }, [renderLine, renderBars]);
  useChartRedraw(redraw);

  return (
    <div className="space-y-1">
      <svg ref={lineRef} className="w-full" style={{ height: lineHeight }} />
      <svg ref={barRef} className="w-full" style={{ height: barHeight }} />
    </div>
  );
}

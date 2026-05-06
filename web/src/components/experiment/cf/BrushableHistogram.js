import React, { useCallback, useEffect, useRef } from "react";
import * as d3 from "d3";
import { useCrossfilter, useChartRedraw } from "./CrossfilterContext";
import {
  PALETTE, TYPO, applyTextStyle,
  setupSvg, drawAxis, axisLabel, attachXBrush, snapshot,
} from "./chartUtils";

/**
 * Generic brushable bar histogram bound to a crossfilter dim+group.
 * The brush selection persists across redraws so it can be dragged around.
 */
export default function BrushableHistogram({
  dimKey, groupKey,
  xLabel, yLabel = "n",
  height = 170,
  tickFmt = null,
  color = PALETTE.fill,
}) {
  const { pack, redrawAll } = useCrossfilter();
  const ref = useRef(null);
  // Persistent brush state in data coordinates [a, b]; null = no brush.
  const brushDataRef = useRef(null);

  // Reset brush state if the underlying records change.
  useEffect(() => { brushDataRef.current = null; }, [pack]);

  const render = useCallback(() => {
    const node = ref.current;
    if (!node) return;
    const width = node.clientWidth || 220;
    const data = snapshot(pack.groups[groupKey])
      .filter((d) => d.value > 0)
      .sort((a, b) => a.key - b.key);

    const { g, w, h } = setupSvg(node, width, height,
      { top: 8, right: 10, bottom: 22, left: 32 });
    if (data.length === 0) return;

    const x = d3.scaleLinear()
      .domain(d3.extent(data, (d) => d.key)).nice()
      .range([0, w]);
    const y = d3.scaleLinear()
      .domain([0, d3.max(data, (d) => d.value) * 1.1])
      .range([h, 0]);

    const xAxis = d3.axisBottom(x).ticks(4).tickSizeOuter(0).tickSize(3);
    if (tickFmt) xAxis.tickFormat(tickFmt);
    const xg = g.append("g").attr("transform", `translate(0,${h})`).call(xAxis);
    applyTextStyle(xg.selectAll("text"), TYPO.axis);
    xg.selectAll("path,line").attr("stroke", PALETTE.axis).attr("stroke-width", 0.5);

    drawAxis(g, y, "left", h, 3);
    axisLabel(g, w, h, xLabel, yLabel);

    const binW = data.length > 1
      ? Math.max(1, x(data[1].key) - x(data[0].key) - 2.5)
      : Math.max(2, w / 8 - 2);

    g.selectAll("rect.b").data(data).enter().append("rect")
      .attr("class", "b")
      .attr("x", (d) => x(d.key))
      .attr("y", (d) => y(d.value))
      .attr("width", binW)
      .attr("height", (d) => Math.max(0.5, h - y(d.value)))
      .attr("fill", color).attr("fill-opacity", 0.65);

    // Reapply persisted brush selection
    const initialPx = brushDataRef.current
      ? [x(brushDataRef.current[0]), x(brushDataRef.current[1])]
      : null;

    attachXBrush(g, w, h,
      (extent) => {
        if (!extent) {
          pack.dims[dimKey].filterAll();
          brushDataRef.current = null;
        } else {
          const [a, b] = extent;
          const range = [x.invert(a), x.invert(b)];
          pack.dims[dimKey].filterRange(range);
          brushDataRef.current = range;
        }
        redrawAll();
      },
      initialPx);
  }, [pack, dimKey, groupKey, xLabel, yLabel, height, tickFmt, color, redrawAll]);

  useChartRedraw(render);

  return <svg ref={ref} className="w-full" style={{ height }} />;
}

import React, { useCallback, useRef } from "react";
import * as d3 from "d3";
import { useCrossfilter, useChartRedraw } from "./CrossfilterContext";
import { PALETTE, setupSvg, drawAxis, axisLabel, attachXBrush, snapshot } from "./chartUtils";

/**
 * Generic brushable bar histogram bound to a crossfilter dim+group.
 * Brushing applies dim.filterRange; clicking outside clears.
 */
export default function BrushableHistogram({
  dimKey, groupKey,
  xLabel, yLabel = "n",
  height = 180,
  tickFmt = null,
  color = PALETTE.fill,
}) {
  const { pack, redrawAll } = useCrossfilter();
  const ref = useRef(null);

  const render = useCallback(() => {
    const node = ref.current;
    if (!node) return;
    const width = node.clientWidth || 220;
    const data = snapshot(pack.groups[groupKey])
      .filter((d) => d.value > 0)
      .sort((a, b) => a.key - b.key);

    const { g, w, h } = setupSvg(node, width, height);
    if (data.length === 0) return;

    const x = d3.scaleLinear()
      .domain(d3.extent(data, (d) => d.key)).nice()
      .range([0, w]);
    const y = d3.scaleLinear()
      .domain([0, d3.max(data, (d) => d.value) * 1.1])
      .range([h, 0]);

    const xAxis = d3.axisBottom(x).ticks(4).tickSizeOuter(0);
    if (tickFmt) xAxis.tickFormat(tickFmt);
    g.append("g").attr("transform", `translate(0,${h})`)
      .call(xAxis)
      .call((s) => s.selectAll("text").style("font-size", "9px"))
      .call((s) => s.selectAll("path,line").attr("stroke", PALETTE.axis));
    drawAxis(g, y, "left", h, 3);
    axisLabel(g, w, h, xLabel, yLabel);

    const binW = data.length > 1
      ? Math.max(1, x(data[1].key) - x(data[0].key) - 1)
      : Math.max(1, w / 8 - 1);

    g.selectAll("rect.b").data(data).enter().append("rect")
      .attr("class", "b")
      .attr("x", (d) => x(d.key))
      .attr("y", (d) => y(d.value))
      .attr("width", binW)
      .attr("height", (d) => h - y(d.value))
      .attr("fill", color).attr("opacity", 0.78);

    attachXBrush(g, w, h,
      () => {},
      (extent) => {
        if (!extent) {
          pack.dims[dimKey].filterAll();
        } else {
          const [a, b] = extent;
          pack.dims[dimKey].filterRange([x.invert(a), x.invert(b)]);
        }
        redrawAll();
      });
  }, [pack, dimKey, groupKey, xLabel, yLabel, height, tickFmt, color, redrawAll]);

  useChartRedraw(render);

  return <svg ref={ref} className="w-full" style={{ height }} />;
}

import React, { useCallback, useRef, useState } from "react";
import * as d3 from "d3";
import { useCrossfilter, useChartRedraw } from "./CrossfilterContext";
import { PALETTE, classColor } from "./chartUtils";

/**
 * dc.js-style PieChart bound to a categorical crossfilter dim/group.
 * Click a slice to filter; click again to clear.
 */
export default function PieChart({
  dimKey, groupKey,
  colorFn,
  height = 180,
  innerRadius = 0,
}) {
  const { pack, redrawAll } = useCrossfilter();
  const ref = useRef(null);
  const filterRef = useRef(null);

  const palette = colorFn || ((k) => classColor(k));

  const render = useCallback(() => {
    const node = ref.current;
    if (!node) return;
    const width = node.clientWidth || 200;
    const data = pack.groups[groupKey].all().filter((d) => d.value > 0);

    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) * 0.42;

    const svg = d3.select(node);
    svg.selectAll("*").remove();
    svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet");
    const g = svg.append("g").attr("transform", `translate(${cx},${cy})`);

    if (data.length === 0) return;

    const pie = d3.pie().value((d) => d.value).sort(null)(data);
    const arc = d3.arc().innerRadius(innerRadius).outerRadius(radius);

    const slice = g.selectAll("path.s").data(pie);
    const enter = slice.enter().append("path").attr("class", "s")
      .style("cursor", "pointer")
      .on("click", (e, d) => {
        const k = d.data.key;
        if (filterRef.current === k) {
          pack.dims[dimKey].filterAll();
          filterRef.current = null;
        } else {
          pack.dims[dimKey].filter(k);
          filterRef.current = k;
        }
        redrawAll();
      });

    enter
      .attr("d", arc)
      .attr("fill", (d) => palette(d.data.key))
      .attr("opacity", (d) =>
        filterRef.current === null || filterRef.current === d.data.key ? 0.92 : 0.3
      )
      .attr("stroke", "white").attr("stroke-width", 1.5);

    g.selectAll("text.lbl").data(pie).enter().append("text")
      .attr("class", "lbl")
      .attr("transform", (d) => `translate(${arc.centroid(d)})`)
      .attr("dy", "0.35em").attr("text-anchor", "middle")
      .style("font-size", "10px")
      .style("fill", PALETTE.text)
      .text((d) => (d.endAngle - d.startAngle > 0.18 ? d.data.key : ""));

    enter.append("title").text((d) => `${d.data.key}: ${d.data.value}`);
  }, [pack, dimKey, groupKey, palette, height, innerRadius, redrawAll]);

  useChartRedraw(render);

  return <svg ref={ref} className="w-full" style={{ height }} />;
}

import React, { useCallback, useRef } from "react";
import * as d3 from "d3";
import { useCrossfilter, useChartRedraw } from "./CrossfilterContext";
import { PALETTE, classColor } from "./chartUtils";

/**
 * dc.js-style RowChart for categorical / ordinal dims (e.g. n, ℓ).
 * Bars are horizontal, sorted descending by default.
 */
export default function RowChart({
  dimKey, groupKey,
  height = 200,
  colorFn,
  labelFn = (k) => String(k),
  sortDesc = true,
}) {
  const { pack, redrawAll } = useCrossfilter();
  const ref = useRef(null);
  const filterRef = useRef(null);

  const render = useCallback(() => {
    const node = ref.current;
    if (!node) return;
    const width = node.clientWidth || 220;
    let data = pack.groups[groupKey].all().filter((d) => d.value > 0);
    if (sortDesc) data = [...data].sort((a, b) => b.value - a.value);

    const margin = { top: 8, right: 14, bottom: 22, left: 50 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const svg = d3.select(node);
    svg.selectAll("*").remove();
    svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet");
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    if (data.length === 0) return;

    const y = d3.scaleBand()
      .domain(data.map((d) => d.key)).range([0, h]).padding(0.18);
    const x = d3.scaleLinear()
      .domain([0, d3.max(data, (d) => d.value)]).nice().range([0, w]);

    g.append("g").attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(4).tickSizeOuter(0))
      .call((s) => s.selectAll("text").style("font-size", "9px"))
      .call((s) => s.selectAll("path,line").attr("stroke", PALETTE.axis));
    g.append("g")
      .call(d3.axisLeft(y).tickSizeOuter(0).tickFormat((d) => labelFn(d)))
      .call((s) => s.selectAll("text").style("font-size", "10px"))
      .call((s) => s.selectAll("path,line").attr("stroke", PALETTE.axis));

    g.selectAll("rect.r").data(data).enter().append("rect")
      .attr("class", "r")
      .attr("x", 0)
      .attr("y", (d) => y(d.key))
      .attr("width", (d) => x(d.value))
      .attr("height", y.bandwidth())
      .attr("fill", (d) => (colorFn ? colorFn(d.key) : classColor(d.key)))
      .attr("opacity", (d) =>
        filterRef.current === null || filterRef.current === d.key ? 0.85 : 0.28
      )
      .style("cursor", "pointer")
      .on("click", (e, d) => {
        const k = d.key;
        if (filterRef.current === k) {
          pack.dims[dimKey].filterAll();
          filterRef.current = null;
        } else {
          pack.dims[dimKey].filter(k);
          filterRef.current = k;
        }
        redrawAll();
      });

    g.selectAll("text.lab").data(data).enter().append("text")
      .attr("class", "lab")
      .attr("x", (d) => x(d.value) + 4)
      .attr("y", (d) => y(d.key) + y.bandwidth() / 2 + 3)
      .style("font-size", "9px").style("fill", PALETTE.text)
      .text((d) => d.value);
  }, [pack, dimKey, groupKey, height, colorFn, labelFn, sortDesc, redrawAll]);

  useChartRedraw(render);

  return <svg ref={ref} className="w-full" style={{ height }} />;
}

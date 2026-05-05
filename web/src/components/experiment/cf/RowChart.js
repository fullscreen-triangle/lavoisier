import React, { useCallback, useRef } from "react";
import * as d3 from "d3";
import { useCrossfilter, useChartRedraw } from "./CrossfilterContext";
import { PALETTE, TYPO, classColor, applyTextStyle } from "./chartUtils";

export default function RowChart({
  dimKey, groupKey,
  height = 190,
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

    const margin = { top: 6, right: 12, bottom: 20, left: 50 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const svg = d3.select(node);
    svg.selectAll("*").remove();
    svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .style("background", PALETTE.bg);
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    if (data.length === 0) return;

    const y = d3.scaleBand()
      .domain(data.map((d) => d.key)).range([0, h]).padding(0.32);
    const x = d3.scaleLinear()
      .domain([0, d3.max(data, (d) => d.value)]).nice().range([0, w]);

    const xg = g.append("g").attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(4).tickSizeOuter(0).tickSize(3));
    applyTextStyle(xg.selectAll("text"), TYPO.axis);
    xg.selectAll("path,line").attr("stroke", PALETTE.axis).attr("stroke-width", 0.5);

    const yg = g.append("g")
      .call(d3.axisLeft(y).tickSizeOuter(0).tickSize(0).tickFormat((d) => labelFn(d)));
    applyTextStyle(yg.selectAll("text"), TYPO.axis);
    yg.selectAll("path").attr("stroke", PALETTE.axis).attr("stroke-width", 0.5);

    g.selectAll("rect.r").data(data).enter().append("rect")
      .attr("class", "r")
      .attr("x", 0)
      .attr("y", (d) => y(d.key))
      .attr("width", (d) => x(d.value))
      .attr("height", Math.min(y.bandwidth(), 10))
      .attr("fill", (d) => (colorFn ? colorFn(d.key) : classColor(d.key)))
      .attr("fill-opacity", (d) =>
        filterRef.current === null || filterRef.current === d.key ? 0.7 : 0.22
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

    const labels = g.selectAll("text.lab").data(data).enter().append("text")
      .attr("class", "lab")
      .attr("x", (d) => x(d.value) + 4)
      .attr("y", (d) => y(d.key) + Math.min(y.bandwidth(), 10) / 2 + 3)
      .text((d) => d.value);
    applyTextStyle(labels, TYPO.inline);
  }, [pack, dimKey, groupKey, height, colorFn, labelFn, sortDesc, redrawAll]);

  useChartRedraw(render);

  return <svg ref={ref} className="w-full" style={{ height }} />;
}

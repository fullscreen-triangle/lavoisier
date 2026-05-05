import React, { useCallback, useRef } from "react";
import * as d3 from "d3";
import { useCrossfilter, useChartRedraw } from "./CrossfilterContext";
import {
  PALETTE, TYPO, classColor,
  setupSvg, drawAxis, axisLabel, applyTextStyle,
} from "./chartUtils";

export default function Row2ClassBubble({ height = 280 }) {
  const { pack, redrawAll } = useCrossfilter();
  const ref = useRef(null);
  const filteredClassRef = useRef(null);

  const render = useCallback(() => {
    const node = ref.current;
    if (!node) return;
    const width = node.clientWidth || 920;

    const groupsAll = pack.groups.classBubble.all().filter((d) => d.value.count > 0);
    const data = groupsAll.map((g) => ({
      cls: g.key,
      meanMz: g.value.sumMz / g.value.count,
      meanN: g.value.sumN / g.value.count,
      sumI: g.value.sumI,
      count: g.value.count,
    }));

    const { svg, g, w, h } = setupSvg(node, width, height,
      { top: 10, right: 14, bottom: 24, left: 40 });
    if (data.length === 0) return;

    const xExtent = d3.extent(data, (d) => d.meanMz);
    const yExtent = d3.extent(data, (d) => d.meanN);
    const x = d3.scaleLinear()
      .domain([xExtent[0] * 0.92, xExtent[1] * 1.08]).range([0, w]);
    const y = d3.scaleLinear()
      .domain([Math.max(0, yExtent[0] - 1), yExtent[1] + 1]).range([h, 0]);
    const rScale = d3.scaleSqrt()
      .domain([0, d3.max(data, (d) => d.sumI) || 1]).range([5, 56]);

    g.append("g")
      .attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).tickSize(-h).tickFormat(""))
      .call((s) => s.selectAll("line").attr("stroke", PALETTE.grid).attr("stroke-width", 0.4))
      .call((s) => s.select("path").remove());
    g.append("g")
      .call(d3.axisLeft(y).tickSize(-w).tickFormat(""))
      .call((s) => s.selectAll("line").attr("stroke", PALETTE.grid).attr("stroke-width", 0.4))
      .call((s) => s.select("path").remove());

    drawAxis(g, x, "bottom", h, 6);
    drawAxis(g, y, "left", h, 5);
    axisLabel(g, w, h, "mean precursor m/z", "mean partition n");

    g.append("rect")
      .attr("x", 0).attr("y", 0).attr("width", w).attr("height", h)
      .attr("fill", "transparent")
      .on("click", () => {
        pack.dims.class.filterAll();
        filteredClassRef.current = null;
        redrawAll();
      });

    const bubbles = g.selectAll("g.bub").data(data, (d) => d.cls).enter().append("g")
      .attr("class", "bub")
      .attr("transform", (d) => `translate(${x(d.meanMz)},${y(d.meanN)})`)
      .style("cursor", "pointer")
      .on("click", (e, d) => {
        e.stopPropagation();
        if (filteredClassRef.current === d.cls) {
          pack.dims.class.filterAll();
          filteredClassRef.current = null;
        } else {
          pack.dims.class.filter(d.cls);
          filteredClassRef.current = d.cls;
        }
        redrawAll();
      });

    bubbles.append("circle")
      .attr("r", (d) => rScale(d.sumI))
      .attr("fill", (d) => classColor(d.cls))
      .attr("fill-opacity", 0.34)
      .attr("stroke", (d) => classColor(d.cls))
      .attr("stroke-width", (d) => filteredClassRef.current === d.cls ? 1.6 : 0.8);

    const labels = bubbles.append("text")
      .attr("text-anchor", "middle").attr("dy", 3).text((d) => d.cls);
    applyTextStyle(labels, { ...TYPO.title, size: "10px" });

    bubbles.append("title").text((d) =>
      `${d.cls}\n${d.count} species\nmean m/z ${d.meanMz.toFixed(1)}\nmean n ${d.meanN.toFixed(2)}\n∑I ${d.sumI.toFixed(3)}`
    );
  }, [pack, height, redrawAll]);

  useChartRedraw(render);
  return <svg ref={ref} className="w-full" style={{ height }} />;
}

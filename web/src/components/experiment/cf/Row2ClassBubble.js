import React, { useCallback, useRef } from "react";
import * as d3 from "d3";
import { useCrossfilter, useChartRedraw } from "./CrossfilterContext";
import { PALETTE, classColor, setupSvg, drawAxis, axisLabel } from "./chartUtils";

/**
 * Row 2: Bubble chart of lipid classes.
 *   x = mean precursor m/z within filter
 *   y = mean partition principal coordinate n
 *   r = sqrt(total intensity)
 * Click a bubble to filter to that class. Click empty space to clear.
 */
export default function Row2ClassBubble({ height = 320 }) {
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
      { top: 14, right: 18, bottom: 30, left: 44 });

    if (data.length === 0) return;

    const xExtent = d3.extent(data, (d) => d.meanMz);
    const yExtent = d3.extent(data, (d) => d.meanN);
    const x = d3.scaleLinear()
      .domain([xExtent[0] * 0.92, xExtent[1] * 1.08])
      .range([0, w]);
    const y = d3.scaleLinear()
      .domain([Math.max(0, yExtent[0] - 1), yExtent[1] + 1])
      .range([h, 0]);
    const rScale = d3.scaleSqrt()
      .domain([0, d3.max(data, (d) => d.sumI) || 1])
      .range([6, 64]);

    // grid
    g.append("g").attr("class", "grid")
      .attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).tickSize(-h).tickFormat(""))
      .call((s) => s.selectAll("line").attr("stroke", PALETTE.grid))
      .call((s) => s.select("path").remove());
    g.append("g").attr("class", "grid")
      .call(d3.axisLeft(y).tickSize(-w).tickFormat(""))
      .call((s) => s.selectAll("line").attr("stroke", PALETTE.grid))
      .call((s) => s.select("path").remove());

    drawAxis(g, x, "bottom", h, 6);
    drawAxis(g, y, "left", h, 5);
    axisLabel(g, w, h, "mean precursor m/z", "mean partition n");

    // bubble background to capture deselect clicks
    g.append("rect")
      .attr("x", 0).attr("y", 0).attr("width", w).attr("height", h)
      .attr("fill", "transparent")
      .on("click", () => {
        pack.dims.class.filterAll();
        filteredClassRef.current = null;
        redrawAll();
      });

    const bubbles = g.selectAll("g.bub").data(data, (d) => d.cls);
    const bg = bubbles.enter().append("g").attr("class", "bub")
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

    bg.append("circle")
      .attr("r", (d) => rScale(d.sumI))
      .attr("fill", (d) => classColor(d.cls))
      .attr("fill-opacity", 0.55)
      .attr("stroke", (d) => classColor(d.cls))
      .attr("stroke-width", (d) =>
        filteredClassRef.current === d.cls ? 3 : 1.3
      );

    bg.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", 4)
      .style("font-size", "11px")
      .style("font-weight", 700)
      .style("fill", PALETTE.text)
      .text((d) => d.cls);

    bg.append("title").text((d) =>
      `${d.cls}\n${d.count} species\nmean m/z ${d.meanMz.toFixed(1)}\nmean n ${d.meanN.toFixed(2)}\n∑I ${d.sumI.toFixed(3)}`
    );
  }, [pack, height, redrawAll]);

  useChartRedraw(render);

  return <svg ref={ref} className="w-full" style={{ height }} />;
}

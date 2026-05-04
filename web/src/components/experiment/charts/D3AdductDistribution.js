import React from "react";
import * as d3 from "d3";
import { useD3 } from "./useD3";
import { PALETTE, classColor, MARGIN } from "./palette";

/**
 * Heatmap of adduct distribution across classes. Cells = count.
 */
export default function D3AdductDistribution({ records, width = 460, height = 260 }) {
  const ref = useD3((el) => {
    const svg = d3.select(el);
    svg.selectAll("*").remove();
    if (!records || records.length === 0) return;

    const w = width - MARGIN.left - MARGIN.right;
    const h = height - MARGIN.top - MARGIN.bottom;

    const classes = Array.from(new Set(records.map((d) => d.analyteClass)));
    const adducts = Array.from(new Set(records.map((d) => d.adduct)));

    const xs = d3.scaleBand().domain(classes).range([0, w]).padding(0.04);
    const ys = d3.scaleBand().domain(adducts).range([0, h]).padding(0.04);

    const counts = d3.rollup(
      records, (v) => ({ count: v.length, intensity: d3.mean(v, (d) => d.intensity) }),
      (d) => d.analyteClass, (d) => d.adduct
    );

    const cells = [];
    for (const cls of classes) {
      for (const ad of adducts) {
        const cell = counts.get(cls)?.get(ad);
        cells.push({ cls, ad, count: cell?.count || 0, intensity: cell?.intensity || 0 });
      }
    }

    const maxCount = d3.max(cells, (d) => d.count) || 1;
    const color = d3.scaleSequential(d3.interpolateBlues).domain([0, maxCount]);

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    g.append("g")
      .attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(xs).tickSizeOuter(0))
      .selectAll("text").style("font-size", "10px");
    g.append("g")
      .call(d3.axisLeft(ys).tickSizeOuter(0))
      .selectAll("text").style("font-size", "9px");

    g.selectAll("rect.cell").data(cells).enter().append("rect")
      .attr("class", "cell")
      .attr("x", (d) => xs(d.cls))
      .attr("y", (d) => ys(d.ad))
      .attr("width", xs.bandwidth())
      .attr("height", ys.bandwidth())
      .attr("fill", (d) => (d.count === 0 ? "#f5f5f5" : color(d.count)))
      .attr("stroke", "white").attr("stroke-width", 0.5)
      .append("title").text((d) => `${d.cls} × ${d.ad}: ${d.count}`);

    // Cell text
    g.selectAll("text.lab").data(cells.filter((c) => c.count > 0)).enter().append("text")
      .attr("class", "lab")
      .attr("x", (d) => xs(d.cls) + xs.bandwidth() / 2)
      .attr("y", (d) => ys(d.ad) + ys.bandwidth() / 2 + 3)
      .attr("text-anchor", "middle")
      .style("font-size", "9px")
      .style("fill", (d) => (d.count > maxCount * 0.6 ? "white" : PALETTE.text))
      .text((d) => d.count);
  }, [records, width, height]);

  return <svg ref={ref} width={width} height={height} style={{ background: PALETTE.bg }} />;
}

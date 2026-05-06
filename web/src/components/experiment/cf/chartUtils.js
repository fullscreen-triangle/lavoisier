/**
 * Shared d3 helpers for the crossfilter charts.
 * Dark background, thin lines, academic-feeling typography.
 */
import * as d3 from "d3";

export const PALETTE = {
  bg: "#0d0f12",
  axis: "#9aa3ad",
  grid: "#222831",
  text: "#cdd5df",
  muted: "#6b7280",
  fill: "#5fa8d3",
  selFill: "#e07a7a",
  brushFill: "rgba(95,168,211,0.16)",
  brushStroke: "#5fa8d3",
};

export const CLASS_PALETTE = {
  PC:  "#5fa8d3", PE: "#e07a7a", PS: "#b388eb",
  PG:  "#e493b3", PI: "#5dc0d8", SM: "#7cc77c",
  Cer: "#e6a456", TAG: "#cdc15c", DAG: "#a07a5e",
  LPC: "#a8b2bd", CE:  "#9cc4d8", FA:  "#e8c598",
};

export function classColor(c) {
  return CLASS_PALETTE[c] || "#7c8794";
}

const SERIF_STACK = '"Computer Modern", "CMU Serif", "Latin Modern", Georgia, "Times New Roman", serif';
const SANS_STACK  = '"Inter", "IBM Plex Sans", -apple-system, sans-serif';

export const TYPO = {
  axis:   { size: "8.5px", weight: 400, family: SANS_STACK,  fill: PALETTE.axis },
  label:  { size: "9px",   weight: 400, family: SANS_STACK,  fill: PALETTE.text },
  inline: { size: "9px",   weight: 400, family: SANS_STACK,  fill: PALETTE.text },
  title:  { size: "10px",  weight: 500, family: SERIF_STACK, fill: PALETTE.text },
};

export function applyTextStyle(sel, t) {
  return sel
    .style("font-size", t.size)
    .style("font-weight", t.weight)
    .style("font-family", t.family)
    .style("fill", t.fill);
}

export function setupSvg(node, width, height, margin = { top: 12, right: 12, bottom: 24, left: 36 }) {
  const svg = d3.select(node);
  svg.selectAll("*").remove();
  svg
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("preserveAspectRatio", "xMidYMid meet")
    .style("background", PALETTE.bg);
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  return { svg, g, w, h, margin };
}

export function drawAxis(g, scale, orient, h, ticks = 5) {
  if (orient === "bottom") {
    const ax = g.append("g")
      .attr("class", "x axis")
      .attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(scale).ticks(ticks).tickSizeOuter(0).tickSize(3));
    applyTextStyle(ax.selectAll("text"), TYPO.axis);
    ax.selectAll("path,line")
      .attr("stroke", PALETTE.axis).attr("stroke-width", 0.6);
  } else {
    const ax = g.append("g")
      .attr("class", "y axis")
      .call(d3.axisLeft(scale).ticks(ticks).tickSizeOuter(0).tickSize(3));
    applyTextStyle(ax.selectAll("text"), TYPO.axis);
    ax.selectAll("path,line")
      .attr("stroke", PALETTE.axis).attr("stroke-width", 0.6);
  }
}

export function axisLabel(g, w, h, xLabel, yLabel) {
  if (xLabel) {
    applyTextStyle(
      g.append("text")
        .attr("x", w / 2).attr("y", h + 18)
        .attr("text-anchor", "middle").text(xLabel),
      TYPO.label
    );
  }
  if (yLabel) {
    applyTextStyle(
      g.append("text").attr("transform", "rotate(-90)")
        .attr("x", -h / 2).attr("y", -24)
        .attr("text-anchor", "middle").text(yLabel),
      TYPO.label
    );
  }
}

/**
 * Attach a horizontal brush to the inner group g.
 *
 * @param {[number,number]|null} initialPx  pixel range to restore (so the
 *   brush rectangle persists across redraws and remains draggable).
 */
export function attachXBrush(g, w, h, onBrushEnd, initialPx = null) {
  const brush = d3.brushX()
    .extent([[0, 0], [w, h]])
    .on("end", (e) => {
      // Ignore programmatic brush.move calls (those have null sourceEvent).
      if (!e.sourceEvent) return;
      onBrushEnd(e.selection);
    });
  const grp = g.append("g").attr("class", "brush").call(brush);
  grp.selectAll(".selection")
    .attr("fill", PALETTE.brushFill)
    .attr("stroke", PALETTE.brushStroke)
    .attr("stroke-width", 0.7);
  // Hide the leading "overlay" rect handles so the visible brush is just
  // the selection (matches the dc.js feel — a draggable rectangle).
  if (initialPx) {
    const a = Math.max(0, Math.min(w, initialPx[0]));
    const b = Math.max(0, Math.min(w, initialPx[1]));
    if (b > a) brush.move(grp, [a, b]);
  }
  return { brush, grp };
}

/**
 * Attach a 2D brush.
 *
 * @param {[[number,number],[number,number]]|null} initialPx  rectangle in
 *   pixel coordinates to restore.
 */
export function attachXYBrush(g, w, h, onBrushEnd, initialPx = null) {
  const brush = d3.brush()
    .extent([[0, 0], [w, h]])
    .on("end", (e) => {
      if (!e.sourceEvent) return;
      onBrushEnd(e.selection);
    });
  const grp = g.append("g").attr("class", "brush").call(brush);
  grp.selectAll(".selection")
    .attr("fill", PALETTE.brushFill)
    .attr("stroke", PALETTE.brushStroke)
    .attr("stroke-width", 0.7);
  if (initialPx) {
    const [[x0, y0], [x1, y1]] = initialPx;
    if (x1 > x0 && y1 > y0) brush.move(grp, [[x0, y0], [x1, y1]]);
  }
  return { brush, grp };
}

export function snapshot(group) {
  return group.all().map((d) => ({ key: d.key, value: d.value }));
}

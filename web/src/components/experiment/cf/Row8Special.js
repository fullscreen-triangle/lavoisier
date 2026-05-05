import React, { useCallback, useRef, useEffect } from "react";
import * as d3 from "d3";
import * as THREE from "three";
import { useCrossfilter, useChartRedraw } from "./CrossfilterContext";
import { PALETTE, classColor, setupSvg, drawAxis, axisLabel } from "./chartUtils";

/**
 * Row 8 (final row): three special views, all reflecting the current
 * crossfilter selection.
 *   A — Droplet image spectra: bijective Ion-to-Drip rendering per filtered
 *       record (Weber-style concentric structure).
 *   B — Heatmap: m/z bin × class, intensity-coloured.
 *   C — 3D peak plot (Three.js): surface of (m/z, n, intensity).
 */
export default function Row8Special({ height = 320 }) {
  return (
    <div className="grid grid-cols-3 gap-3 lg:grid-cols-1">
      <Tile label="Droplet bijection (top filtered)">
        <DropletSpectra height={height} />
      </Tile>
      <Tile label="m/z × class intensity heatmap">
        <IntensityHeatmap height={height} />
      </Tile>
      <Tile label="3D peak surface">
        <Peak3D height={height} />
      </Tile>
    </div>
  );
}

function Tile({ label, children }) {
  return (
    <div className="rounded-md border border-dark/10 dark:border-light/10 p-2 bg-light dark:bg-dark">
      <div className="text-[10px] uppercase tracking-wider font-bold text-dark/60 dark:text-light/60 mb-1">
        {label}
      </div>
      {children}
    </div>
  );
}

/* -------------------------------------------------------------- */
/* Droplet bijection: each filtered record → a concentric "drip"  */
/* with shells = partition n, hue = class, radius = sqrt(I).      */
/* Mirrors the proteomics paper's bijective Weber/Reynolds view.  */
/* -------------------------------------------------------------- */
function DropletSpectra({ height }) {
  const { pack } = useCrossfilter();
  const ref = useRef(null);

  const render = useCallback(() => {
    const node = ref.current;
    if (!node) return;
    const width = node.clientWidth || 360;
    const top = pack.dims.intensity.top(36);

    const svg = d3.select(node);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`)
       .attr("preserveAspectRatio", "xMidYMid meet")
       .style("background", PALETTE.bg);

    if (top.length === 0) return;

    const cols = 6;
    const rows = Math.ceil(top.length / cols);
    const cellW = width / cols;
    const cellH = height / rows;

    top.forEach((r, i) => {
      const cx = (i % cols) * cellW + cellW / 2;
      const cy = Math.floor(i / cols) * cellH + cellH / 2;
      const radius = Math.min(cellW, cellH) * 0.36 * (0.4 + 0.6 * Math.sqrt(r.intensity));
      const color = classColor(r.analyteClass);
      const g = svg.append("g").attr("transform", `translate(${cx},${cy})`);

      // shells: one ring per partition n
      for (let k = 1; k <= r.n; k++) {
        g.append("circle")
          .attr("r", radius * (k / r.n))
          .attr("fill", "none").attr("stroke", color)
          .attr("stroke-width", 0.7)
          .attr("opacity", 0.65);
      }
      // body
      g.append("circle")
        .attr("r", radius)
        .attr("fill", color).attr("opacity", 0.18);
      g.append("circle")
        .attr("r", radius * 0.05 + 1.5)
        .attr("fill", color);

      // chirality slash (s)
      g.append("line")
        .attr("x1", -radius * 0.7).attr("x2", radius * 0.7)
        .attr("y1", r.s > 0 ? -radius * 0.3 : radius * 0.3)
        .attr("y2", r.s > 0 ? radius * 0.3 : -radius * 0.3)
        .attr("stroke", color).attr("stroke-width", 0.6).attr("opacity", 0.5);

      // label
      g.append("text")
        .attr("y", radius + 9).attr("text-anchor", "middle")
        .style("font-size", "8px").style("fill", PALETTE.text)
        .text(`${r.analyteClass}(${r.X}:${r.Y})`);

      g.append("title").text(
        `${r.analyte}${r.adduct}\nm/z ${r.precursorMz.toFixed(3)}\nI ${r.intensity.toExponential(2)}\n(n,ℓ,m,s)=(${r.n},${r.l},${r.m},${r.s.toFixed(1)})`
      );
    });
  }, [pack, height]);

  useChartRedraw(render);
  return <svg ref={ref} className="w-full" style={{ height }} />;
}

/* -------------------------------------------------------------- */
/* m/z × class intensity heatmap                                  */
/* -------------------------------------------------------------- */
function IntensityHeatmap({ height }) {
  const { pack } = useCrossfilter();
  const ref = useRef(null);

  const render = useCallback(() => {
    const node = ref.current;
    if (!node) return;
    const width = node.clientWidth || 360;

    const records = pack.dims.mz.bottom(Infinity);
    const classes = Array.from(new Set(records.map((d) => d.analyteClass)));
    if (records.length === 0 || classes.length === 0) {
      d3.select(node).selectAll("*").remove();
      return;
    }
    const mzExtent = d3.extent(records, (d) => d.precursorMz);
    const nBins = 24;
    const binW = (mzExtent[1] - mzExtent[0]) / nBins || 1;
    const cell = new Map();
    for (const r of records) {
      const bi = Math.min(nBins - 1, Math.floor((r.precursorMz - mzExtent[0]) / binW));
      const k = `${bi}|${r.analyteClass}`;
      cell.set(k, (cell.get(k) || 0) + r.intensity);
    }
    const cells = [];
    for (const [k, v] of cell.entries()) {
      const [bi, cls] = k.split("|");
      cells.push({ bi: +bi, cls, v });
    }

    const margin = { top: 6, right: 14, bottom: 26, left: 50 };
    const { svg, g, w, h } = setupSvg(node, width, height, margin);

    const x = d3.scaleLinear()
      .domain([0, nBins]).range([0, w]);
    const y = d3.scaleBand().domain(classes).range([0, h]).padding(0.04);
    const colour = d3.scaleSequentialLog(d3.interpolateMagma)
      .domain([Math.max(1e-6, d3.min(cells, (c) => c.v)), d3.max(cells, (c) => c.v) || 1]);

    g.append("g").attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(5).tickFormat((v) => (mzExtent[0] + v * binW).toFixed(0)).tickSizeOuter(0))
      .call((s) => s.selectAll("text").style("font-size", "9px"))
      .call((s) => s.selectAll("path,line").attr("stroke", PALETTE.axis));
    g.append("g")
      .call(d3.axisLeft(y).tickSizeOuter(0))
      .call((s) => s.selectAll("text").style("font-size", "10px"))
      .call((s) => s.selectAll("path,line").attr("stroke", PALETTE.axis));
    axisLabel(g, w, h, "m/z", "class");

    g.selectAll("rect.cell").data(cells).enter().append("rect")
      .attr("class", "cell")
      .attr("x", (d) => x(d.bi))
      .attr("y", (d) => y(d.cls))
      .attr("width", x(1) - x(0) - 0.5)
      .attr("height", y.bandwidth())
      .attr("fill", (d) => colour(d.v))
      .append("title").text((d) => `${d.cls} m/z ~${(mzExtent[0] + d.bi * binW).toFixed(0)}: ${d.v.toExponential(2)}`);
  }, [pack, height]);

  useChartRedraw(render);
  return <svg ref={ref} className="w-full" style={{ height }} />;
}

/* -------------------------------------------------------------- */
/* 3D peak surface in Three.js: x = m/z bin, y = class index,     */
/* z = log(intensity).                                            */
/* -------------------------------------------------------------- */
function Peak3D({ height }) {
  const ref = useRef(null);
  const { pack, tick } = useCrossfilter();
  const stateRef = useRef(null);

  // Initialise Three.js once
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const width = node.clientWidth || 360;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(width, height);
    renderer.setClearColor(0xffffff, 1);
    node.innerHTML = "";
    node.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
    camera.position.set(2.6, 2.2, 2.6);
    camera.lookAt(0, 0, 0);

    const ambient = new THREE.AmbientLight(0xffffff, 0.6); scene.add(ambient);
    const dir = new THREE.DirectionalLight(0xffffff, 0.7);
    dir.position.set(2, 4, 3); scene.add(dir);

    const surfaceGroup = new THREE.Group(); scene.add(surfaceGroup);

    // Axes helper
    const axes = new THREE.AxesHelper(1.4);
    axes.material.opacity = 0.6;
    axes.material.transparent = true;
    scene.add(axes);

    let raf;
    let theta = 0;
    const animate = () => {
      theta += 0.0028;
      camera.position.x = 2.6 * Math.cos(theta);
      camera.position.z = 2.6 * Math.sin(theta);
      camera.lookAt(0, 0, 0);
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    };
    animate();

    stateRef.current = { renderer, scene, camera, surfaceGroup };
    return () => {
      cancelAnimationFrame(raf);
      renderer.dispose();
      try { node.removeChild(renderer.domElement); } catch {}
    };
  }, [height]);

  // Rebuild the surface whenever crossfilter ticks
  useEffect(() => {
    const st = stateRef.current;
    if (!st) return;
    const { surfaceGroup } = st;
    while (surfaceGroup.children.length) {
      const c = surfaceGroup.children.pop();
      c.geometry?.dispose();
      c.material?.dispose();
    }
    const records = pack.dims.mz.bottom(Infinity);
    if (records.length === 0) return;

    const classes = Array.from(new Set(records.map((d) => d.analyteClass)));
    const classIdx = new Map(classes.map((c, i) => [c, i]));
    const mzExtent = d3.extent(records, (d) => d.precursorMz);
    const nBins = 24;
    const binW = (mzExtent[1] - mzExtent[0]) / nBins || 1;
    const grid = new Map();
    for (const r of records) {
      const bi = Math.min(nBins - 1, Math.floor((r.precursorMz - mzExtent[0]) / binW));
      const k = `${bi}|${r.analyteClass}`;
      grid.set(k, (grid.get(k) || 0) + r.intensity);
    }

    const xSpan = 2.0;
    const zSpan = 1.6;
    const ySpan = 1.4;
    const maxI = d3.max(Array.from(grid.values())) || 1;

    for (const [key, v] of grid.entries()) {
      const [biStr, cls] = key.split("|");
      const bi = +biStr;
      const ci = classIdx.get(cls);
      const x = (bi / nBins - 0.5) * xSpan;
      const z = (ci / Math.max(1, classes.length - 1) - 0.5) * zSpan;
      const yh = (Math.log10(1 + v / maxI * 999) / 3) * ySpan;

      const geom = new THREE.BoxGeometry(xSpan / nBins * 0.9, yh, zSpan / Math.max(1, classes.length) * 0.9);
      const colorHex = classColor(cls);
      const material = new THREE.MeshLambertMaterial({ color: colorHex });
      const mesh = new THREE.Mesh(geom, material);
      mesh.position.set(x, yh / 2, z);
      surfaceGroup.add(mesh);
    }
    // ground plane
    const planeGeom = new THREE.PlaneGeometry(xSpan, zSpan);
    const planeMat = new THREE.MeshBasicMaterial({ color: 0xeeeeee, side: THREE.DoubleSide });
    const plane = new THREE.Mesh(planeGeom, planeMat);
    plane.rotation.x = -Math.PI / 2;
    plane.position.y = -0.001;
    surfaceGroup.add(plane);
  }, [pack, tick]);

  return <div ref={ref} className="w-full" style={{ height }} />;
}

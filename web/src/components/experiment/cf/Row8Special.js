import React, { useCallback, useRef, useEffect } from "react";
import * as d3 from "d3";
import * as THREE from "three";
import { useCrossfilter, useChartRedraw } from "./CrossfilterContext";
import {
  PALETTE, TYPO, classColor, applyTextStyle,
  setupSvg, drawAxis, axisLabel,
} from "./chartUtils";

export default function Row8Special({ height = 300 }) {
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
    <div className="rounded border p-2"
      style={{ background: PALETTE.bg, borderColor: PALETTE.grid }}>
      <div className="text-[9px] uppercase tracking-wider mb-1 font-normal"
        style={{ color: PALETTE.muted }}>
        {label}
      </div>
      {children}
    </div>
  );
}

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
      const radius = Math.min(cellW, cellH) * 0.34
        * (0.4 + 0.6 * Math.sqrt(r.intensity));
      const color = classColor(r.analyteClass);
      const g = svg.append("g").attr("transform", `translate(${cx},${cy})`);

      for (let k = 1; k <= r.n; k++) {
        g.append("circle")
          .attr("r", radius * (k / r.n))
          .attr("fill", "none").attr("stroke", color)
          .attr("stroke-width", 0.4).attr("opacity", 0.6);
      }
      g.append("circle")
        .attr("r", radius)
        .attr("fill", color).attr("opacity", 0.12);
      g.append("circle")
        .attr("r", radius * 0.05 + 1.2)
        .attr("fill", color);
      g.append("line")
        .attr("x1", -radius * 0.7).attr("x2", radius * 0.7)
        .attr("y1", r.s > 0 ? -radius * 0.3 : radius * 0.3)
        .attr("y2", r.s > 0 ? radius * 0.3 : -radius * 0.3)
        .attr("stroke", color).attr("stroke-width", 0.4).attr("opacity", 0.45);

      const txt = g.append("text").attr("y", radius + 8).attr("text-anchor", "middle")
        .text(`${r.analyteClass}(${r.X}:${r.Y})`);
      applyTextStyle(txt, { ...TYPO.inline, size: "8px" });

      g.append("title").text(
        `${r.analyte}${r.adduct}\nm/z ${r.precursorMz.toFixed(3)}\nI ${r.intensity.toExponential(2)}\n(n,ℓ,m,s)=(${r.n},${r.l},${r.m},${r.s.toFixed(1)})`
      );
    });
  }, [pack, height]);

  useChartRedraw(render);
  return <svg ref={ref} className="w-full" style={{ height }} />;
}

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

    const margin = { top: 6, right: 12, bottom: 22, left: 50 };
    const { g, w, h } = setupSvg(node, width, height, margin);

    const x = d3.scaleLinear().domain([0, nBins]).range([0, w]);
    const y = d3.scaleBand().domain(classes).range([0, h]).padding(0.06);
    const colour = d3.scaleSequentialLog(d3.interpolateInferno)
      .domain([Math.max(1e-6, d3.min(cells, (c) => c.v)),
               d3.max(cells, (c) => c.v) || 1]);

    const xg = g.append("g").attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(5)
        .tickFormat((v) => (mzExtent[0] + v * binW).toFixed(0))
        .tickSizeOuter(0).tickSize(3));
    applyTextStyle(xg.selectAll("text"), TYPO.axis);
    xg.selectAll("path,line").attr("stroke", PALETTE.axis).attr("stroke-width", 0.5);

    const yg = g.append("g").call(d3.axisLeft(y).tickSizeOuter(0).tickSize(0));
    applyTextStyle(yg.selectAll("text"), TYPO.axis);
    yg.selectAll("path").attr("stroke", PALETTE.axis).attr("stroke-width", 0.5);

    axisLabel(g, w, h, "m/z", "class");

    g.selectAll("rect.cell").data(cells).enter().append("rect")
      .attr("class", "cell")
      .attr("x", (d) => x(d.bi))
      .attr("y", (d) => y(d.cls))
      .attr("width", x(1) - x(0) - 0.5)
      .attr("height", y.bandwidth())
      .attr("fill", (d) => colour(d.v))
      .append("title").text((d) =>
        `${d.cls} m/z ~${(mzExtent[0] + d.bi * binW).toFixed(0)}: ${d.v.toExponential(2)}`
      );
  }, [pack, height]);

  useChartRedraw(render);
  return <svg ref={ref} className="w-full" style={{ height }} />;
}

function Peak3D({ height }) {
  const ref = useRef(null);
  const { pack, tick } = useCrossfilter();
  const stateRef = useRef(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const width = node.clientWidth || 360;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(width, height);
    renderer.setClearColor(new THREE.Color(PALETTE.bg), 1);
    node.innerHTML = "";
    node.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(PALETTE.bg);

    const camera = new THREE.PerspectiveCamera(36, width / height, 0.1, 100);
    camera.position.set(2.6, 2.2, 2.6);
    camera.lookAt(0, 0, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const dir = new THREE.DirectionalLight(0xffffff, 0.7);
    dir.position.set(2, 4, 3); scene.add(dir);

    const surfaceGroup = new THREE.Group(); scene.add(surfaceGroup);

    const axes = new THREE.AxesHelper(1.4);
    axes.material.opacity = 0.35;
    axes.material.transparent = true;
    scene.add(axes);

    let raf, theta = 0;
    const animate = () => {
      theta += 0.0024;
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

    const xSpan = 2.0, zSpan = 1.6, ySpan = 1.4;
    const maxI = d3.max(Array.from(grid.values())) || 1;

    for (const [key, v] of grid.entries()) {
      const [biStr, cls] = key.split("|");
      const bi = +biStr;
      const ci = classIdx.get(cls);
      const x = (bi / nBins - 0.5) * xSpan;
      const z = (ci / Math.max(1, classes.length - 1) - 0.5) * zSpan;
      const yh = (Math.log10(1 + v / maxI * 999) / 3) * ySpan;

      const geom = new THREE.BoxGeometry(
        xSpan / nBins * 0.78, yh,
        zSpan / Math.max(1, classes.length) * 0.78
      );
      const material = new THREE.MeshLambertMaterial({
        color: classColor(cls), transparent: true, opacity: 0.85,
      });
      const mesh = new THREE.Mesh(geom, material);
      mesh.position.set(x, yh / 2, z);
      surfaceGroup.add(mesh);
    }
    const planeGeom = new THREE.PlaneGeometry(xSpan, zSpan);
    const planeMat = new THREE.MeshBasicMaterial({
      color: 0x14171c, side: THREE.DoubleSide,
    });
    const plane = new THREE.Mesh(planeGeom, planeMat);
    plane.rotation.x = -Math.PI / 2;
    plane.position.y = -0.001;
    surfaceGroup.add(plane);
  }, [pack, tick]);

  return <div ref={ref} className="w-full" style={{ height }} />;
}

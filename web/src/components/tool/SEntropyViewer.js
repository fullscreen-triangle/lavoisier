import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { useStore } from "@/lib/state/store";

/**
 * 3D viewer of S-entropy space [0,1]³.
 *
 * Each observed CategoricalState is plotted at (S_k, S_t, S_e). The
 * cube is the bounded phase space; points are partition-cell positions.
 * Click selects (sets selectedAddress in the store).
 *
 * Uses Three.js with a manual orbital control loop — no extras.
 */
export default function SEntropyViewer() {
  const mountRef = useRef(null);
  const stateRef = useRef(null); // holds renderer/scene/camera
  const states = useStore((s) => s.states);
  const selectedAddress = useStore((s) => s.selectedAddress);
  const selectAddress = useStore((s) => s.selectAddress);

  // Init
  useEffect(() => {
    if (!mountRef.current) return;
    const mount = mountRef.current;

    const width = mount.clientWidth || 320;
    const height = mount.clientHeight || 320;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000); // transparent via setClearAlpha
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 100);
    camera.position.set(2.4, 2.0, 2.4);
    camera.lookAt(0.5, 0.5, 0.5);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(width, height);
    renderer.setClearColor(0x000000, 0);
    mount.appendChild(renderer.domElement);

    // Lighting (soft, since points are unlit but the cube needs depth)
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const light = new THREE.DirectionalLight(0xffffff, 0.4);
    light.position.set(2, 3, 2);
    scene.add(light);

    // Unit cube wireframe — the bounded phase space
    const cube = new THREE.Group();
    const wire = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
      new THREE.LineBasicMaterial({ color: 0x888888, opacity: 0.5, transparent: true })
    );
    wire.position.set(0.5, 0.5, 0.5);
    cube.add(wire);
    scene.add(cube);

    // Axis labels (using a sprite per axis end)
    const axisGeom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0), new THREE.Vector3(1.1, 0, 0),
      new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1.1, 0),
      new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 1.1),
    ]);
    const axisMat = new THREE.LineBasicMaterial({ color: 0xb63e96 });
    scene.add(new THREE.LineSegments(axisGeom, axisMat));

    // Points — populated later
    const pointsGeom = new THREE.BufferGeometry();
    pointsGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
    pointsGeom.setAttribute("color", new THREE.BufferAttribute(new Float32Array(0), 3));
    const pointsMat = new THREE.PointsMaterial({
      size: 0.025,
      vertexColors: true,
      sizeAttenuation: true,
    });
    const points = new THREE.Points(pointsGeom, pointsMat);
    scene.add(points);

    // Selection marker
    const markerGeom = new THREE.SphereGeometry(0.022, 16, 12);
    const markerMat = new THREE.MeshBasicMaterial({
      color: 0x58e6d9,
      transparent: true,
      opacity: 0.85,
    });
    const marker = new THREE.Mesh(markerGeom, markerMat);
    marker.visible = false;
    scene.add(marker);

    // Manual orbit controls (drag to rotate, wheel to zoom)
    let theta = Math.PI / 4;
    let phi = Math.PI / 4;
    let radius = 2.8;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    const updateCamera = () => {
      const x = 0.5 + radius * Math.sin(phi) * Math.cos(theta);
      const y = 0.5 + radius * Math.cos(phi);
      const z = 0.5 + radius * Math.sin(phi) * Math.sin(theta);
      camera.position.set(x, y, z);
      camera.lookAt(0.5, 0.5, 0.5);
    };
    updateCamera();

    const handleDown = (e) => {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
    };
    const handleUp = () => { dragging = false; };
    const handleMove = (e) => {
      if (!dragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      theta -= dx * 0.01;
      phi = Math.max(0.05, Math.min(Math.PI - 0.05, phi - dy * 0.01));
      updateCamera();
    };
    const handleWheel = (e) => {
      e.preventDefault();
      radius = Math.max(1.2, Math.min(8, radius + e.deltaY * 0.002));
      updateCamera();
    };

    // Click → ray-pick the nearest point and select its address
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 0.03 };
    const mouseVec = new THREE.Vector2();
    const handleClick = (e) => {
      if (dragging) return;
      const rect = renderer.domElement.getBoundingClientRect();
      mouseVec.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouseVec.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouseVec, camera);
      const hits = raycaster.intersectObject(points);
      if (hits.length > 0) {
        const idx = hits[0].index;
        const allStates = useStore.getState().states;
        if (allStates[idx]) {
          selectAddress(allStates[idx].address);
        }
      }
    };

    renderer.domElement.addEventListener("pointerdown", handleDown);
    window.addEventListener("pointerup", handleUp);
    window.addEventListener("pointermove", handleMove);
    renderer.domElement.addEventListener("wheel", handleWheel, { passive: false });
    renderer.domElement.addEventListener("click", handleClick);

    // Animate
    let raf = 0;
    const animate = () => {
      raf = requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    // Resize observer
    const resize = () => {
      if (!mount) return;
      const w = mount.clientWidth || 320;
      const h = mount.clientHeight || 320;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    const ro = new ResizeObserver(resize);
    ro.observe(mount);

    stateRef.current = {
      scene, camera, renderer, points, pointsGeom, pointsMat, marker,
    };

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.domElement.removeEventListener("pointerdown", handleDown);
      window.removeEventListener("pointerup", handleUp);
      window.removeEventListener("pointermove", handleMove);
      renderer.domElement.removeEventListener("wheel", handleWheel);
      renderer.domElement.removeEventListener("click", handleClick);
      pointsGeom.dispose();
      pointsMat.dispose();
      markerGeom.dispose();
      markerMat.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
      stateRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-build point cloud when states change
  useEffect(() => {
    const ref = stateRef.current;
    if (!ref) return;

    const n = states.length;
    const positions = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);

    for (let i = 0; i < n; i++) {
      const s = states[i];
      positions[i * 3]     = s.sentropy.sk;
      positions[i * 3 + 1] = s.sentropy.st;
      positions[i * 3 + 2] = s.sentropy.se;

      const c = colorForState(s);
      colors[i * 3]     = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }

    ref.pointsGeom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    ref.pointsGeom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    ref.pointsGeom.attributes.position.needsUpdate = true;
    ref.pointsGeom.attributes.color.needsUpdate = true;
    ref.pointsGeom.computeBoundingSphere();
  }, [states.length]);

  // Update selection marker
  useEffect(() => {
    const ref = stateRef.current;
    if (!ref) return;
    if (!selectedAddress) {
      ref.marker.visible = false;
      return;
    }
    const target = states.find((s) => s.address === selectedAddress);
    if (!target) {
      ref.marker.visible = false;
      return;
    }
    ref.marker.position.set(target.sentropy.sk, target.sentropy.st, target.sentropy.se);
    ref.marker.visible = true;
  }, [selectedAddress, states]);

  return (
    <div className="w-full h-full min-h-[200px]" ref={mountRef} />
  );
}

/**
 * Map a CategoricalState to an RGB colour.
 * Hue from S_k, lightness from charge, saturation high.
 */
function colorForState(s) {
  const c = new THREE.Color();
  c.setHSL(0.55 + s.sentropy.se * 0.4, 0.7, 0.4 + s.sentropy.sk * 0.3);
  return c;
}

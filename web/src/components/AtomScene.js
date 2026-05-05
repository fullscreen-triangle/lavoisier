import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";

/**
 * Animated GLB atom on transparent background. Plays the embedded
 * animation clips; gently rotates if none are present.
 */
export default function AtomScene({
  src = "/model/particle_atom_loop_animaton.glb",
  className = "",
}) {
  const ref = useRef(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const w = node.clientWidth || 720;
    const h = node.clientHeight || 720;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(w, h);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    node.innerHTML = "";
    node.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(36, w / h, 0.05, 100);
    camera.position.set(0, 0, 4.4);

    // Lighting — neutral so the atom reads on either bg
    scene.add(new THREE.AmbientLight(0xffffff, 0.45));
    const key = new THREE.DirectionalLight(0xffffff, 1.0);
    key.position.set(3, 3, 5);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x88aaff, 0.6);
    rim.position.set(-3, -1, -2);
    scene.add(rim);

    const root = new THREE.Group();
    scene.add(root);

    let mixer = null;
    const clock = new THREE.Clock();
    let raf = 0;

    const loader = new GLTFLoader();
    loader.load(src,
      (gltf) => {
        const model = gltf.scene;
        // Centre + scale to fit
        const box = new THREE.Box3().setFromObject(model);
        const size = new THREE.Vector3();
        const center = new THREE.Vector3();
        box.getSize(size);
        box.getCenter(center);
        model.position.sub(center);
        const maxDim = Math.max(size.x, size.y, size.z) || 1;
        const scale = 2.4 / maxDim;
        model.scale.setScalar(scale);

        root.add(model);

        if (gltf.animations && gltf.animations.length) {
          mixer = new THREE.AnimationMixer(model);
          gltf.animations.forEach((clip) => mixer.clipAction(clip).play());
        }

        // Tracker for resize
        const onResize = () => {
          const cw = node.clientWidth || 720;
          const ch = node.clientHeight || 720;
          renderer.setSize(cw, ch);
          camera.aspect = cw / ch;
          camera.updateProjectionMatrix();
        };
        window.addEventListener("resize", onResize);
        node.__resize = onResize;
      },
      undefined,
      (err) => {
        // Gracefully degrade: small placeholder if the GLB can't load
        console.warn("AtomScene GLB load failed:", err);
        const geom = new THREE.IcosahedronGeometry(1.0, 1);
        const mat = new THREE.MeshStandardMaterial({
          color: 0x4499ff, metalness: 0.4, roughness: 0.4, wireframe: true,
        });
        root.add(new THREE.Mesh(geom, mat));
      }
    );

    const animate = () => {
      const dt = clock.getDelta();
      if (mixer) mixer.update(dt);
      // gentle continual rotation (atoms don't sit still)
      root.rotation.y += 0.003;
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      cancelAnimationFrame(raf);
      if (node.__resize) window.removeEventListener("resize", node.__resize);
      renderer.dispose();
      scene.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
          else obj.material.dispose();
        }
      });
      try { node.removeChild(renderer.domElement); } catch {}
    };
  }, [src]);

  return <div ref={ref} className={className} style={{ width: "100%", height: "100%" }} />;
}

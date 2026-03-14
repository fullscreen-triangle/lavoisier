/* eslint-disable react/no-unknown-property */
'use client'
import React, { useRef, useEffect, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, useGLTF, Environment, PerspectiveCamera, useAnimations } from "@react-three/drei";
import * as THREE from "three";

function Model({ url, autoRotate }) {
  const group = useRef();
  const { scene, animations } = useGLTF(url);
  const { actions, names } = useAnimations(animations, group);

  // Play animations if they exist
  useEffect(() => {
    if (names.length > 0) {
      console.log("Available animations:", names);
      // Play all animations
      names.forEach((name) => {
        actions[name]?.play();
      });
    }
  }, [actions, names]);

  // Auto-rotate if enabled and no animations
  useFrame((state, delta) => {
    if (autoRotate && group.current) {
      group.current.rotation.y += delta * 0.5;
    }
  });

  // Center and scale the model
  useEffect(() => {
    if (scene) {
      const box = new THREE.Box3().setFromObject(scene);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());

      // Center the model
      scene.position.sub(center);

      // Scale to fit
      const maxDim = Math.max(size.x, size.y, size.z);
      const scale = 3 / maxDim;
      scene.scale.setScalar(scale);

      console.log("Model loaded successfully");
    }
  }, [scene]);

  return <primitive ref={group} object={scene} />;
}

export default function GLBViewer({ 
  modelPath = "/particle_atom_loop_animaton.glb", 
  autoRotate = false,
  showControls = true,
  backgroundColor = "#1a1a1a"
}) {
  const [error, setError] = useState(null);

  return (
    <div style={{ width: "100%", height: "100vh", background: backgroundColor }}>
      {error && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          color: '#ff4444',
          fontSize: '1.2rem',
          zIndex: 100,
          textAlign: 'center',
          padding: '20px'
        }}>
          Error loading model: {error}
        </div>
      )}

      <Canvas
        shadows
        camera={{        position: [0, 0, 10],  // Move camera further back (was 5)
                    fov: 45,               // Narrower FOV (was 50) - less distortion
                    near: 0.1,             // Near clipping plane
                    far: 1000 }}
        gl={{ 
          antialias: true, 
          alpha: true,
          powerPreference: "high-performance"
        }}
        onCreated={({ gl }) => {
          gl.setClearColor(backgroundColor);
        }}
      >
        {/* Lighting Setup */}
        <ambientLight intensity={0.6} />
        <directionalLight
          position={[10, 10, 5]}
          intensity={1}
          castShadow
          shadow-mapSize-width={1024}
          shadow-mapSize-height={1024}
        />
        <pointLight position={[-10, -10, -5]} intensity={0.3} />
        <spotLight
          position={[0, 10, 0]}
          angle={0.3}
          penumbra={1}
          intensity={0.5}
          castShadow
        />

        {/* Environment for realistic reflections */}
        <Environment preset="city" />

        {/* The 3D Model */}
        <Model url={modelPath} autoRotate={autoRotate} scale={0.5}/>

        {/* Camera Controls */}
        {showControls && (
          <OrbitControls
            enableZoom={true}
            enablePan={true}
            enableRotate={true}
            minDistance={2}
            maxDistance={20}
            autoRotate={false}
            autoRotateSpeed={2}
          />
        )}
      </Canvas>

    </div>
  );
}

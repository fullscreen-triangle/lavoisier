// GLBViewer.jsx - Updated for background
'use client'
import { Environment, OrbitControls, useAnimations, useGLTF } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

function Model({ url, autoRotate }) {
    const group = useRef();
    const { scene, animations } = useGLTF(url);
    const { actions, names } = useAnimations(animations, group);

    useEffect(() => {
        if (names.length > 0) {
            console.log("Available animations:", names);
            names.forEach((name) => {
                actions[name]?.play();
            });
        }
    }, [actions, names]);

    useFrame((state, delta) => {
        if (autoRotate && group.current) {
            group.current.rotation.y += delta * 0.3; // Slower rotation
        }
    });

    useEffect(() => {
        if (scene) {
            const box = new THREE.Box3().setFromObject(scene);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());

            scene.position.sub(center);

            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 2 / maxDim;
            scene.scale.setScalar(scale);

            // Ensure materials are visible
            scene.traverse((child) => {
                if (child.isMesh && child.material) {
                    child.material.transparent = false;
                    child.material.opacity = 1;
                }
            });
        }
    }, [scene]);

    return <primitive ref={group} object={scene} />;
}

export default function GLBViewer({
    modelPath = "/particle_atom_loop_animaton.glb",
    autoRotate = false,
    showControls = false,
    backgroundColor = "transparent"
}) {
    const [error, setError] = useState(null);

    return (
        <div style={{ 
            width: "100%", 
            height: "100%", 
            background: backgroundColor,
            position: 'absolute',
            top: 0,
            left: 0
        }}>
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
                    Error: {error}
                </div>
            )}

            <Canvas
                shadows
                camera={{ 
                    position: [0, 0, 10],
                    fov: 45,
                    near: 0.1,
                    far: 1000
                }}
                gl={{
                    antialias: true,
                    alpha: true,
                    powerPreference: "high-performance"
                }}
                style={{ 
                    width: '100%', 
                    height: '100%' 
                }}
            >
                <ambientLight intensity={0.7} />
                <directionalLight
                    position={[10, 10, 5]}
                    intensity={1.2}
                    castShadow
                />
                <pointLight position={[-10, -10, -5]} intensity={0.4} />
                
                <Environment preset="city" />
                
                <Model url={modelPath} autoRotate={autoRotate} />
            </Canvas>
        </div>
    );
}

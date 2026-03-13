/* eslint-disable react/no-unknown-property */
// components/AnimatedModel.jsx
'use client';

import { useRef, useEffect } from 'react';
import { useGLTF, useAnimations } from '@react-three/drei';

export function AnimatedModel({ modelPath, scale = 1, position = [0, 0, 0] }) {
  const group = useRef();
  const { scene, animations } = useGLTF(modelPath);
  const { actions, names } = useAnimations(animations, group);

  // Spiele alle Animationen ab
  useEffect(() => {
    if (names.length > 0) {
      // Spiele die erste Animation ab (oder wähle eine bestimmte)
      actions[names[0]]?.play();
      
      // Oder spiele alle Animationen ab:
      // names.forEach((name) => actions[name]?.play());
    }
  }, [actions, names]);

  return (
    <primitive 
      ref={group} 
      object={scene} 
      scale={scale} 
      position={position} 
    />
  );
}

// Preload das Model für bessere Performance
useGLTF.preload('/path/to/your/model.glb');

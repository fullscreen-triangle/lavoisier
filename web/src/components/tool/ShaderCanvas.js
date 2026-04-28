import React, { useEffect, useRef, useState } from "react";
import { useStore } from "@/lib/state/store";
import { createObservationSession } from "@/lib/gpu";

/**
 * ShaderCanvas — mounts the WebGL2 observation session.
 *
 * Lifecycle:
 *   1. On mount, create a 512×512 canvas + observation session
 *   2. Whenever the states array grows, replay Pass 1 (wave field)
 *   3. After each Pass 1, run Pass 6 (quality metrics) and update store
 *   4. On unmount, dispose all GPU resources
 *
 * The canvas is the visual readout of the partition depth landscape —
 * not a "rendered image" but the categorical state tensor displayed.
 */
export default function ShaderCanvas({ width = 512, height = 512, onUnavailable }) {
  const canvasRef = useRef(null);
  const sessionRef = useRef(null);
  const setGpuReady = useStore((s) => s.setGpuReady);
  const setQuality = useStore((s) => s.setQuality);
  const states = useStore((s) => s.states);

  const [error, setError] = useState(null);
  const [initialised, setInitialised] = useState(false);

  // Mount: create the observation session
  useEffect(() => {
    let cancelled = false;
    let session = null;

    (async () => {
      try {
        if (!canvasRef.current) return;
        session = await createObservationSession(canvasRef.current, { width, height });
        if (cancelled) {
          session.dispose();
          return;
        }
        sessionRef.current = session;
        setInitialised(true);
        setGpuReady(true);
      } catch (err) {
        console.error("GPU init failed:", err);
        setError(String(err?.message || err));
        if (typeof onUnavailable === "function") {
          try { onUnavailable(err); } catch (_) { /* ignore */ }
        }
      }
    })();

    return () => {
      cancelled = true;
      if (session) {
        try {
          session.dispose();
        } catch (e) {
          /* ignore */
        }
      }
      sessionRef.current = null;
      setGpuReady(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-render whenever the state count changes (debounced)
  useEffect(() => {
    if (!sessionRef.current || states.length === 0) return;
    const session = sessionRef.current;

    // Throttle: don't re-render on every single state — coalesce to ~30 fps
    const handle = requestAnimationFrame(() => {
      try {
        // Pass 1: accumulate query wave field
        session.observeQuery(states);
        // Pass 6: physical quality metrics
        const q = session.measureQuality();
        setQuality(q);
      } catch (err) {
        console.error("GPU render failed:", err);
      }
    });

    return () => cancelAnimationFrame(handle);
  }, [states.length, setQuality]);

  if (error) {
    return (
      <div className="flex-1 rounded-lg border-2 border-amber-500/30 bg-amber-500/5
        flex items-center justify-center min-h-[400px] p-6">
        <div className="text-center max-w-md">
          <div className="text-amber-700 dark:text-amber-300 font-bold mb-2">
            Shader pipeline unavailable
          </div>
          <div className="text-sm text-dark/70 dark:text-light/70">{error}</div>
          <div className="text-xs text-dark/50 dark:text-light/50 mt-3">
            The observation apparatus needs WebGL2 + EXT_color_buffer_float for
            the wave-field passes. The rest of Lavoisier still works — parsing,
            partition decomposition, and the 3D S-Entropy viewer all run without it.
          </div>
          {typeof onUnavailable === "function" ? (
            <div className="mt-4 text-xs text-dark/60 dark:text-light/60">
              The S-Entropy 3D view above has been activated automatically.
            </div>
          ) : null}
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col">
      <div
        className="flex-1 relative rounded-lg overflow-hidden border-2 border-dark/10 dark:border-light/10
          bg-gradient-to-br from-dark/5 to-primary/5 dark:from-light/5 dark:to-primaryDark/5
          flex items-center justify-center min-h-[400px]"
      >
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className="max-w-full max-h-full object-contain"
          style={{ imageRendering: "pixelated" }}
        />
        {!initialised && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="text-sm text-dark/60 dark:text-light/60 animate-pulse">
              Initialising observation apparatus…
            </div>
          </div>
        )}
        {initialised && states.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="text-center text-xs text-dark/40 dark:text-light/40">
              <div className="font-bold mb-1">Pass 1 — Partition State Observation</div>
              <div>Awaiting categorical states…</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

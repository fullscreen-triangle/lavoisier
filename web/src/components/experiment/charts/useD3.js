import { useEffect, useRef } from "react";

/**
 * Tiny hook that sets up a ref and (re)renders a D3 effect whenever
 * dependencies change. Avoids react-vs-d3 reconciliation conflicts.
 */
export function useD3(renderFn, deps) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    renderFn(ref.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return ref;
}

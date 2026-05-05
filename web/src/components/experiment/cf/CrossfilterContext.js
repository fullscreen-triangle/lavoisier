import React, {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useCallback,
  useState,
} from "react";
import { buildCrossfilterPack, disposePack } from "@/lib/experiment/crossfilter";

const CrossfilterCtx = createContext(null);

/**
 * Provider wraps the entire dashboard. Owns the crossfilter pack and a
 * subscription bus for redrawing charts on filter changes.
 */
export function CrossfilterProvider({ records, children }) {
  const pack = useMemo(() => buildCrossfilterPack(records || []), [records]);

  // Disposable cleanup if records change
  const prevPackRef = useRef(null);
  useEffect(() => {
    return () => {
      if (prevPackRef.current && prevPackRef.current !== pack) {
        disposePack(prevPackRef.current);
      }
      prevPackRef.current = pack;
    };
  }, [pack]);

  // Redraw bus
  const subscribers = useRef(new Set());
  const [tick, setTick] = useState(0);

  const subscribe = useCallback((fn) => {
    subscribers.current.add(fn);
    return () => subscribers.current.delete(fn);
  }, []);

  const redrawAll = useCallback(() => {
    subscribers.current.forEach((fn) => {
      try { fn(); } catch (e) { /* noop */ }
    });
    setTick((t) => t + 1);
  }, []);

  // First render: nothing yet to redraw, only after charts mount
  const value = useMemo(
    () => ({ pack, subscribe, redrawAll, tick }),
    [pack, subscribe, redrawAll, tick]
  );

  return (
    <CrossfilterCtx.Provider value={value}>
      {children}
    </CrossfilterCtx.Provider>
  );
}

export function useCrossfilter() {
  const v = useContext(CrossfilterCtx);
  if (!v) throw new Error("useCrossfilter must be inside <CrossfilterProvider>");
  return v;
}

/**
 * A chart hook: subscribes to the redraw bus and exposes redrawAll for
 * the chart's own filter actions.
 */
export function useChartRedraw(redrawFn) {
  const { subscribe } = useCrossfilter();
  useEffect(() => {
    if (!redrawFn) return;
    const unsub = subscribe(redrawFn);
    redrawFn(); // initial paint after subscribe
    return unsub;
  }, [subscribe, redrawFn]);
}

import React, { useEffect, useRef, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useStore } from "@/lib/state/store";
import { createWorkerManager } from "@/lib/worker/manager";
import {
  createSourceFromFiles,
  extractFilesFromDataTransfer,
} from "@/lib/source";

import SourcePicker from "./SourcePicker";
import FileTree from "./FileTree";
import AnalyserPanel from "./AnalyserPanel";
import ShaderCanvas from "./ShaderCanvas";
import SEntropyViewer from "./SEntropyViewer";
import ResultsTable from "./ResultsTable";
import StatusBar from "./StatusBar";
import DetailPanel from "./DetailPanel";
import DeepLinkLoader from "./DeepLinkLoader";

/**
 * The Lavoisier workspace — the complete force-free mass spectrometer.
 *
 * Layout:
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │ status bar (GPU, source, scan count, quality metrics)           │
 *   ├──────────────┬───────────────────────────┬──────────────────────┤
 *   │  Source      │  AnalyserPanel            │  ResultsTable        │
 *   │  FileTree    │  ShaderCanvas (Pass 1)    │                      │
 *   │              │  SEntropyViewer (Three.js)│  DetailPanel         │
 *   └──────────────┴───────────────────────────┴──────────────────────┘
 *
 * Drop a file or folder anywhere on the workspace and it will load.
 */
export default function Workspace() {
  const files = useStore((s) => s.files);
  const selectedFiles = useStore((s) => s.selectedFiles);
  const analyser = useStore((s) => s.analyser);
  const analyserCfg = useStore((s) => s.analyserCfg);
  const appendStates = useStore((s) => s.appendStates);
  const setTaskState = useStore((s) => s.setTaskState);
  const resetResults = useStore((s) => s.resetResults);
  const setSource = useStore((s) => s.setSource);
  const setFiles = useStore((s) => s.setFiles);

  const managerRef = useRef(null);
  const [view, setView] = useState("shader"); // "shader" | "viewer"
  const [processing, setProcessing] = useState(false);
  const [globalDrag, setGlobalDrag] = useState(false);
  const [dropError, setDropError] = useState(null);
  const [shaderUnavailable, setShaderUnavailable] = useState(false);

  const handleShaderUnavailable = useCallback(() => {
    setShaderUnavailable(true);
    // Auto-flip to the S-Entropy viewer so the workspace stays useful.
    setView("viewer");
  }, []);

  // Lazy-create the worker manager on first need
  useEffect(() => {
    if (typeof window === "undefined") return;
    return () => {
      if (managerRef.current) {
        managerRef.current.cancelAll();
      }
    };
  }, []);

  /* --------------------------------------------------------------- */
  /* Workspace-wide drag-and-drop                                    */
  /* --------------------------------------------------------------- */

  // Track drag state via document-level listeners. dragenter/leave fire
  // for every child element so we counter-balance with a counter.
  const dragCounter = useRef(0);

  useEffect(() => {
    const onEnter = (e) => {
      // Only react if the drag actually carries files
      if (!e.dataTransfer?.types?.includes("Files")) return;
      e.preventDefault();
      dragCounter.current += 1;
      if (dragCounter.current === 1) setGlobalDrag(true);
    };
    const onLeave = (e) => {
      e.preventDefault();
      dragCounter.current -= 1;
      if (dragCounter.current <= 0) {
        dragCounter.current = 0;
        setGlobalDrag(false);
      }
    };
    const onOver = (e) => {
      if (e.dataTransfer?.types?.includes("Files")) e.preventDefault();
    };
    const onDrop = async (e) => {
      e.preventDefault();
      dragCounter.current = 0;
      setGlobalDrag(false);
      if (!e.dataTransfer?.types?.includes("Files")) return;
      try {
        const dropped = await extractFilesFromDataTransfer(
          e.dataTransfer.items?.length ? e.dataTransfer.items : e.dataTransfer.files
        );
        if (dropped.length === 0) {
          setDropError("No files were dropped.");
          return;
        }
        const src = createSourceFromFiles(dropped);
        setSource(src);
        const list = await src.listFiles();
        setFiles(list);
        if (list.length === 0) {
          setDropError(
            "No supported MS files in drop. Lavoisier reads .mzML, .mzXML, .imzML, .mgf and .json."
          );
        } else {
          setDropError(null);
        }
      } catch (err) {
        setDropError(String(err?.message || err));
      }
    };

    window.addEventListener("dragenter", onEnter);
    window.addEventListener("dragleave", onLeave);
    window.addEventListener("dragover", onOver);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragenter", onEnter);
      window.removeEventListener("dragleave", onLeave);
      window.removeEventListener("dragover", onOver);
      window.removeEventListener("drop", onDrop);
    };
  }, [setSource, setFiles]);

  // Auto-clear drop errors after a few seconds
  useEffect(() => {
    if (!dropError) return;
    const t = setTimeout(() => setDropError(null), 6000);
    return () => clearTimeout(t);
  }, [dropError]);

  /* --------------------------------------------------------------- */
  /* Process selected files                                           */
  /* --------------------------------------------------------------- */

  const handleProcess = useCallback(async () => {
    if (typeof window === "undefined") return;
    if (!managerRef.current) {
      managerRef.current = createWorkerManager({ concurrency: 4 });
    }
    const mgr = managerRef.current;

    resetResults();
    setProcessing(true);

    const filesToProcess = files.filter((f) => selectedFiles.has(f.id));

    const taskPromises = filesToProcess.map((file) => {
      setTaskState(file.id, { status: "running", scanCount: 0, pct: 0 });

      const task = mgr.processFile(
        file,
        {
          analyser,
          analyserCfg: analyserCfg[analyser],
          ternaryDepth: 18,
          topN: 32,
          batchSize: 100,
          decodeBinary: true,
        },
        {
          onStateBatch(states) {
            appendStates(states);
            // Note: setTaskState patches with an object, not a function;
            // worker scanCount is the source of truth via onDone.
          },
          onProgress(p) {
            setTaskState(file.id, { pct: p.pct ?? 0 });
          },
          onDone(summary) {
            setTaskState(file.id, {
              status: "done",
              scanCount: summary.scanCount,
              pct: 1,
              elapsedMs: summary.elapsedMs,
            });
          },
          onError(err) {
            console.warn(`[${file.name}]`, err);
          },
        }
      );

      return task.done.catch((err) => {
        if (err.message !== "cancelled") {
          setTaskState(file.id, { status: "error", error: err.message });
        }
      });
    });

    await Promise.allSettled(taskPromises);
    setProcessing(false);
  }, [files, selectedFiles, analyser, analyserCfg, appendStates, setTaskState, resetResults]);

  return (
    <div className="flex flex-col w-full min-h-[calc(100vh-160px)] bg-light dark:bg-dark text-dark dark:text-light relative">
      <DeepLinkLoader />
      <StatusBar />

      {/* Global drop overlay */}
      <AnimatePresence>
        {globalDrag && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 pointer-events-none flex items-center justify-center"
          >
            <div className="absolute inset-0 bg-primary/15 dark:bg-primaryDark/15 backdrop-blur-sm" />
            <div className="relative rounded-2xl border-4 border-dashed border-primary dark:border-primaryDark
              bg-light dark:bg-dark px-12 py-10 shadow-2xl">
              <div className="text-center">
                <div className="text-5xl mb-3">⬇</div>
                <div className="text-2xl font-bold text-primary dark:text-primaryDark">
                  Drop to load
                </div>
                <div className="text-sm text-dark/60 dark:text-light/60 mt-1">
                  .mzML, .mzXML, .imzML, .mgf, .json
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Drop error toast */}
      <AnimatePresence>
        {dropError && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="fixed top-20 left-1/2 -translate-x-1/2 z-50
              rounded-lg bg-red-500/10 border border-red-500/40
              px-4 py-2 text-xs text-red-700 dark:text-red-300 max-w-md shadow-lg"
          >
            <div className="font-bold">Drop failed</div>
            <div className="mt-0.5">{dropError}</div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="grid grid-cols-[280px_1fr_360px] gap-0 flex-1
        lg:grid-cols-[260px_1fr] xl:grid-cols-1 min-h-0">
        {/* Left: source + files */}
        <aside className="border-r-2 border-dark/10 dark:border-light/10 p-4 overflow-y-auto">
          <h3 className="text-xs uppercase tracking-wider font-bold text-dark/60 dark:text-light/60 mb-3">
            Source
          </h3>
          <SourcePicker />

          <h3 className="text-xs uppercase tracking-wider font-bold text-dark/60 dark:text-light/60 mt-6 mb-3">
            Files
          </h3>
          <FileTree onProcess={handleProcess} />
        </aside>

        {/* Centre: analyser + canvas/viewer */}
        <main className="flex flex-col p-4 gap-3 min-w-0 min-h-0">
          <div className="flex items-center justify-between gap-3 pb-3 border-b-2 border-dark/10 dark:border-light/10">
            <AnalyserPanel compact />
            <div className="flex items-center gap-1 text-xs">
              <ViewToggle current={view} onChange={setView} shaderDisabled={shaderUnavailable} />
            </div>
          </div>

          <div className="flex-1 min-h-0 flex flex-col">
            {view === "shader" ? (
              <ShaderCanvas onUnavailable={handleShaderUnavailable} />
            ) : (
              <SEntropyViewer />
            )}
          </div>
        </main>

        {/* Right: results + detail */}
        <aside className="border-l-2 border-dark/10 dark:border-light/10 p-4 flex flex-col gap-3 lg:hidden xl:hidden">
          <div className="flex items-center justify-between">
            <h3 className="text-xs uppercase tracking-wider font-bold text-dark/60 dark:text-light/60">
              Live Results
            </h3>
            {processing && (
              <span className="text-[10px] text-primary dark:text-primaryDark animate-pulse">
                streaming…
              </span>
            )}
          </div>
          <div className="flex-1 min-h-0">
            <ResultsTable />
          </div>
          <DetailPanel />
        </aside>
      </div>
    </div>
  );
}

function ViewToggle({ current, onChange, shaderDisabled = false }) {
  return (
    <div className="flex rounded-md border border-dark/10 dark:border-light/10 overflow-hidden text-xs">
      <button
        onClick={() => !shaderDisabled && onChange("shader")}
        disabled={shaderDisabled}
        title={shaderDisabled ? "WebGL2 unavailable in this browser" : "Wave-field shader pipeline"}
        className={`px-3 py-1 ${
          shaderDisabled
            ? "opacity-40 cursor-not-allowed line-through"
            : current === "shader"
            ? "bg-dark text-light dark:bg-light dark:text-dark"
            : "hover:bg-dark/5 dark:hover:bg-light/5"
        }`}
      >
        Shader
      </button>
      <button
        onClick={() => onChange("viewer")}
        className={`px-3 py-1 ${
          current === "viewer"
            ? "bg-dark text-light dark:bg-light dark:text-dark"
            : "hover:bg-dark/5 dark:hover:bg-light/5"
        }`}
      >
        S-Entropy
      </button>
    </div>
  );
}

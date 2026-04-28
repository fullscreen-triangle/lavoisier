import React, { useState, useCallback, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useStore } from "@/lib/state/store";
import {
  pickLocalFolder,
  pickLocalFiles,
  createSourceFromFiles,
  extractFilesFromDataTransfer,
  isLocalFolderSupported,
  isFilePickerSupported,
  createSourceFromInput,
  detectInputKind,
} from "@/lib/source";

/**
 * SourcePicker — connect a local folder, individual files (drag-drop or
 * picker), or a remote source.
 *
 * Three independent local paths so users on any browser can get going.
 */
export default function SourcePicker() {
  const source = useStore((s) => s.source);
  const setSource = useStore((s) => s.setSource);
  const setFiles = useStore((s) => s.setFiles);
  const clearSource = useStore((s) => s.clearSource);

  const [mode, setMode] = useState("idle"); // idle | loading | error
  const [input, setInput] = useState("");
  const [error, setError] = useState(null);
  const [folderSupported, setFolderSupported] = useState(false);
  const [filePickerSupported, setFilePickerSupported] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const inputRef = useRef(null);

  useEffect(() => {
    setFolderSupported(isLocalFolderSupported());
    setFilePickerSupported(isFilePickerSupported());
  }, []);

  const inputKind = detectInputKind(input);

  /* --------------------------------------------------------------- */
  /* Handlers                                                        */
  /* --------------------------------------------------------------- */

  const applySource = useCallback(
    async (src) => {
      setSource(src);
      const files = await src.listFiles();
      setFiles(files);
      if (files.length === 0) {
        setError(
          "No supported MS files found. Lavoisier reads .mzML, .mzXML, .imzML, .mgf and .json."
        );
        setMode("error");
        return;
      }
      setMode("idle");
      setError(null);
    },
    [setSource, setFiles]
  );

  const handleLocalFolder = useCallback(async () => {
    setMode("loading");
    setError(null);
    try {
      const src = await pickLocalFolder();
      await applySource(src);
    } catch (err) {
      if (err?.name === "AbortError") {
        setMode("idle");
        return;
      }
      setError(String(err?.message || err));
      setMode("error");
    }
  }, [applySource]);

  const handleLocalFiles = useCallback(async () => {
    setMode("loading");
    setError(null);
    try {
      const src = await pickLocalFiles();
      await applySource(src);
    } catch (err) {
      if (err?.name === "AbortError") {
        setMode("idle");
        return;
      }
      setError(String(err?.message || err));
      setMode("error");
    }
  }, [applySource]);

  const handleFileInput = useCallback(
    async (e) => {
      const fileList = e.target.files;
      if (!fileList || fileList.length === 0) return;
      setMode("loading");
      setError(null);
      try {
        const src = createSourceFromFiles(Array.from(fileList));
        await applySource(src);
        // Clear the input so re-picking the same file fires again
        if (inputRef.current) inputRef.current.value = "";
      } catch (err) {
        setError(String(err?.message || err));
        setMode("error");
      }
    },
    [applySource]
  );

  const handleRemote = useCallback(async () => {
    if (!input.trim()) return;
    setMode("loading");
    setError(null);
    try {
      const src = await createSourceFromInput(input);
      await applySource(src);
      setInput("");
    } catch (err) {
      setError(String(err?.message || err));
      setMode("error");
    }
  }, [input, applySource]);

  const handleDisconnect = useCallback(() => {
    clearSource();
    setMode("idle");
    setError(null);
  }, [clearSource]);

  /* --------------------------------------------------------------- */
  /* Drag-and-drop                                                   */
  /* --------------------------------------------------------------- */

  const onDragEnter = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.currentTarget === e.target) setIsDragging(false);
  }, []);

  const onDrop = useCallback(
    async (e) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      setMode("loading");
      setError(null);
      try {
        const files = await extractFilesFromDataTransfer(
          e.dataTransfer.items?.length
            ? e.dataTransfer.items
            : e.dataTransfer.files
        );
        if (files.length === 0) {
          setError("No files were dropped.");
          setMode("error");
          return;
        }
        const src = createSourceFromFiles(files);
        await applySource(src);
      } catch (err) {
        setError(String(err?.message || err));
        setMode("error");
      }
    },
    [applySource]
  );

  /* --------------------------------------------------------------- */
  /* Render                                                          */
  /* --------------------------------------------------------------- */

  // Connected state
  if (source) {
    return (
      <div className="space-y-2">
        <div className="rounded-lg border-2 border-primary/30 dark:border-primaryDark/30 bg-primary/5 dark:bg-primaryDark/5 p-3">
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0 flex-1">
              <div className="text-xs uppercase tracking-wider text-primary dark:text-primaryDark font-bold">
                {sourceTypeLabel(source.kind, source.meta)}
              </div>
              <div className="text-sm font-semibold mt-0.5 truncate">{source.label}</div>
              {source.meta?.title && (
                <div className="text-xs text-dark/60 dark:text-light/60 mt-1 line-clamp-2">
                  {source.meta.title}
                </div>
              )}
            </div>
            <button
              onClick={handleDisconnect}
              className="text-xs px-2 py-1 rounded hover:bg-dark/10 dark:hover:bg-light/10 text-dark/60 dark:text-light/60"
              title="Disconnect source"
            >
              ✕
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* DROP ZONE — spans the entire local section */}
      <div
        onDragEnter={onDragEnter}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className={`relative rounded-lg border-2 border-dashed p-3 transition-colors
          ${
            isDragging
              ? "border-primary dark:border-primaryDark bg-primary/10 dark:bg-primaryDark/10"
              : "border-dark/15 dark:border-light/15"
          }`}
      >
        <div className="text-center text-xs text-dark/60 dark:text-light/60 mb-3 font-medium">
          {isDragging ? (
            <span className="text-primary dark:text-primaryDark font-bold">
              Drop to load
            </span>
          ) : (
            <>Drag .mzML files here, or use the buttons below</>
          )}
        </div>

        <div className="space-y-2">
          {/* Folder picker — best UX, Chromium only */}
          {folderSupported && (
            <button
              onClick={handleLocalFolder}
              disabled={mode === "loading"}
              className="w-full px-3 py-2 text-sm rounded-md border-2 border-dark/15 dark:border-light/15
                hover:border-primary dark:hover:border-primaryDark hover:bg-primary/5 dark:hover:bg-primaryDark/5
                transition-colors text-left flex items-center gap-2"
            >
              <span>📁</span>
              <span className="font-medium flex-1">Open Local Folder</span>
              <span className="text-[10px] text-dark/40 dark:text-light/40 uppercase tracking-wider">
                recursive
              </span>
            </button>
          )}

          {/* File picker — Chromium */}
          {filePickerSupported && (
            <button
              onClick={handleLocalFiles}
              disabled={mode === "loading"}
              className="w-full px-3 py-2 text-sm rounded-md border-2 border-dark/15 dark:border-light/15
                hover:border-primary dark:hover:border-primaryDark hover:bg-primary/5 dark:hover:bg-primaryDark/5
                transition-colors text-left flex items-center gap-2"
            >
              <span>📄</span>
              <span className="font-medium flex-1">Pick Files…</span>
              <span className="text-[10px] text-dark/40 dark:text-light/40 uppercase tracking-wider">
                multi-select
              </span>
            </button>
          )}

          {/* Universal fallback — works in every browser */}
          <label
            className="block w-full cursor-pointer"
            title="Browse for one or more .mzML files"
          >
            <input
              ref={inputRef}
              type="file"
              multiple
              accept=".mzML,.mzXML,.imzML,.mgf,.json"
              onChange={handleFileInput}
              className="sr-only"
            />
            <div
              className="w-full px-3 py-2 text-sm rounded-md border-2 border-dark/15 dark:border-light/15
                hover:border-primary dark:hover:border-primaryDark hover:bg-primary/5 dark:hover:bg-primaryDark/5
                transition-colors flex items-center gap-2"
            >
              <span>⬆</span>
              <span className="font-medium flex-1">Browse Files</span>
              <span className="text-[10px] text-dark/40 dark:text-light/40 uppercase tracking-wider">
                any browser
              </span>
            </div>
          </label>
        </div>

        {!folderSupported && !filePickerSupported && (
          <div className="mt-2 text-[10px] text-dark/50 dark:text-light/50">
            Folder picker unavailable in this browser. Use the file
            browser or drag-and-drop.
          </div>
        )}
      </div>

      <div className="text-center text-xs text-dark/40 dark:text-light/40">— or —</div>

      {/* Remote / repository */}
      <div className="space-y-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleRemote();
          }}
          placeholder="MTBLS1707, Zenodo DOI, or HTTPS URL"
          className="w-full px-3 py-2 text-sm rounded-lg border-2 border-dark/20 dark:border-light/20
            bg-light dark:bg-dark text-dark dark:text-light
            focus:outline-none focus:border-primary dark:focus:border-primaryDark"
        />
        <AnimatePresence>
          {input.trim() && (
            <motion.div
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="text-xs text-dark/60 dark:text-light/60 px-1"
            >
              Detected: <strong>{inputKindLabel(inputKind)}</strong>
            </motion.div>
          )}
        </AnimatePresence>
        <button
          onClick={handleRemote}
          disabled={!input.trim() || mode === "loading"}
          className={`w-full px-3 py-2 text-sm rounded-lg font-medium transition-colors
            ${
              !input.trim() || mode === "loading"
                ? "bg-dark/10 dark:bg-light/10 text-dark/40 dark:text-light/40 cursor-not-allowed"
                : "bg-dark text-light dark:bg-light dark:text-dark hover:bg-primary dark:hover:bg-primaryDark"
            }`}
        >
          {mode === "loading" ? "Connecting…" : "Connect"}
        </button>
      </div>

      {/* Error display */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="rounded-lg bg-red-500/10 border border-red-500/30 p-3 text-xs text-red-700 dark:text-red-300"
          >
            <div className="font-bold mb-1">Connection failed</div>
            <div className="break-words">{error}</div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Quick links */}
      <div className="pt-3 border-t border-dark/10 dark:border-light/10">
        <div className="text-xs text-dark/40 dark:text-light/40 mb-2">Try a public study:</div>
        <div className="flex flex-wrap gap-1">
          {[
            { label: "MTBLS1707", input: "MTBLS1707" },
            { label: "MTBLS90", input: "MTBLS90" },
            { label: "Zenodo example", input: "10.5281/zenodo.7654321" },
          ].map((sample) => (
            <button
              key={sample.label}
              onClick={() => setInput(sample.input)}
              className="text-xs px-2 py-1 rounded border border-dark/10 dark:border-light/10
                hover:border-primary dark:hover:border-primaryDark text-dark/70 dark:text-light/70"
            >
              {sample.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function sourceTypeLabel(kind, meta) {
  if (kind === "local") {
    if (meta?.mode === "directory") return "Local folder";
    if (meta?.mode === "flat") {
      const n = meta.count || 0;
      return n === 1 ? "Local file" : `${n} local files`;
    }
    return "Local";
  }
  if (kind === "repository") {
    if (meta?.repository === "metabolights") return "MetaboLights";
    if (meta?.repository === "zenodo") return "Zenodo";
    if (meta?.repository === "massive") return "MassIVE";
    return "Repository";
  }
  if (kind === "remote") return "Remote URL";
  return "Source";
}

function inputKindLabel(kind) {
  switch (kind) {
    case "metabolights":
      return "MetaboLights study";
    case "massive":
      return "MassIVE dataset";
    case "zenodo":
      return "Zenodo record";
    case "url":
      return "Direct URL";
    default:
      return "Unknown — try MTBLSxxxx, a Zenodo DOI, or HTTPS URLs";
  }
}

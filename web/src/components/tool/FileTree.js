import React, { useMemo } from "react";
import { useStore } from "@/lib/state/store";
import { isParseable } from "@/lib/source";

/**
 * FileTree — list discovered files with selection, status, progress.
 *
 * Status comes from store.tasks (per-file). Selection is a Set in
 * store.selectedFiles. Files are grouped by directory path.
 */
export default function FileTree({ onProcess }) {
  const files = useStore((s) => s.files);
  const selectedFiles = useStore((s) => s.selectedFiles);
  const tasks = useStore((s) => s.tasks);
  const toggleFile = useStore((s) => s.toggleFile);
  const selectAllFiles = useStore((s) => s.selectAllFiles);
  const clearSelection = useStore((s) => s.clearSelection);

  const grouped = useMemo(() => groupByPath(files), [files]);

  if (files.length === 0) {
    return (
      <div className="text-xs text-dark/40 dark:text-light/40 italic py-4">
        No files yet — connect a source above.
      </div>
    );
  }

  const allSelected = selectedFiles.size === files.length;
  const noneSelected = selectedFiles.size === 0;
  const parseableCount = files.filter((f) => isParseable(f.name)).length;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs">
        <span className="text-dark/60 dark:text-light/60">
          <strong>{files.length}</strong> files
          {parseableCount < files.length && (
            <span className="text-dark/40 dark:text-light/40 ml-1">
              ({parseableCount} parseable)
            </span>
          )}
        </span>
        <div className="flex gap-1">
          <button
            onClick={selectAllFiles}
            disabled={allSelected}
            className={`px-2 py-0.5 rounded ${
              allSelected
                ? "text-dark/30 dark:text-light/30"
                : "hover:bg-dark/10 dark:hover:bg-light/10 text-dark/70 dark:text-light/70"
            }`}
          >
            All
          </button>
          <button
            onClick={clearSelection}
            disabled={noneSelected}
            className={`px-2 py-0.5 rounded ${
              noneSelected
                ? "text-dark/30 dark:text-light/30"
                : "hover:bg-dark/10 dark:hover:bg-light/10 text-dark/70 dark:text-light/70"
            }`}
          >
            None
          </button>
        </div>
      </div>

      <div className="max-h-[40vh] overflow-y-auto pr-1 space-y-3">
        {Object.entries(grouped).map(([dir, dirFiles]) => (
          <div key={dir}>
            {dir && (
              <div className="text-xs font-mono text-dark/40 dark:text-light/40 mb-1 truncate">
                {dir}/
              </div>
            )}
            <div className="space-y-1">
              {dirFiles.map((file) => (
                <FileRow
                  key={file.id}
                  file={file}
                  selected={selectedFiles.has(file.id)}
                  onToggle={() => toggleFile(file.id)}
                  task={tasks.get(file.id)}
                />
              ))}
            </div>
          </div>
        ))}
      </div>

      {selectedFiles.size > 0 && onProcess && (
        <button
          onClick={onProcess}
          className="w-full mt-2 px-3 py-2 text-sm font-medium rounded-lg
            bg-dark text-light dark:bg-light dark:text-dark
            hover:bg-primary dark:hover:bg-primaryDark transition-colors"
        >
          Process {selectedFiles.size} {selectedFiles.size === 1 ? "file" : "files"} →
        </button>
      )}
    </div>
  );
}

function FileRow({ file, selected, onToggle, task }) {
  const parseable = isParseable(file.name);
  const status = task?.status || "idle"; // idle | running | done | error

  return (
    <button
      onClick={parseable ? onToggle : undefined}
      disabled={!parseable}
      className={`w-full text-left p-2 rounded-md border transition-colors
        ${
          !parseable
            ? "border-dark/5 dark:border-light/5 opacity-40 cursor-not-allowed"
            : selected
            ? "border-primary dark:border-primaryDark bg-primary/10 dark:bg-primaryDark/10"
            : "border-dark/10 dark:border-light/10 hover:border-dark/30 dark:hover:border-light/30"
        }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="text-xs font-medium truncate flex items-center gap-1.5">
            {selected && <span className="text-primary dark:text-primaryDark">✓</span>}
            <span>{file.name}</span>
          </div>
          <div className="text-[10px] text-dark/50 dark:text-light/50 mt-0.5 flex items-center gap-2">
            {file.size != null && <span>{formatSize(file.size)}</span>}
            <span className="font-mono">{fileExt(file.name).toUpperCase()}</span>
            {!parseable && <span className="italic">read-only</span>}
          </div>
        </div>
        {task && <TaskBadge task={task} />}
      </div>

      {task?.status === "running" && (
        <div className="mt-1.5 h-0.5 rounded-full bg-dark/10 dark:bg-light/10 overflow-hidden">
          <div
            className="h-full bg-primary dark:bg-primaryDark transition-all"
            style={{ width: `${(task.pct || 0) * 100}%` }}
          />
        </div>
      )}
    </button>
  );
}

function TaskBadge({ task }) {
  const { status, scanCount, pct } = task;
  if (status === "done") {
    return (
      <span className="text-[10px] text-green-600 dark:text-green-400 font-mono">
        {scanCount} scans ✓
      </span>
    );
  }
  if (status === "running") {
    return (
      <span className="text-[10px] text-primary dark:text-primaryDark font-mono">
        {pct != null ? `${(pct * 100).toFixed(0)}%` : "…"}
      </span>
    );
  }
  if (status === "error") {
    return <span className="text-[10px] text-red-500">error</span>;
  }
  return null;
}

function groupByPath(files) {
  const groups = {};
  for (const f of files) {
    const dir = f.path.includes("/") ? f.path.substring(0, f.path.lastIndexOf("/")) : "";
    if (!groups[dir]) groups[dir] = [];
    groups[dir].push(f);
  }
  return groups;
}

function fileExt(name) {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.substring(i + 1) : "";
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

/**
 * Worker pool manager.
 *
 * Spawns one Web Worker per file (up to N concurrent). Handles
 * lifecycle, message routing, and aggregating results back into
 * the shared state store.
 *
 * Usage (simplified):
 *   const mgr = createWorkerManager({ concurrency: 4 });
 *   const task = mgr.processFile(sourceFile, options, {
 *     onStateBatch(states) { ... },
 *     onDone(summary)      { ... },
 *     onProgress(pct)      { ... },
 *   });
 *   task.cancel();
 */

import { MSG, buildStartMessage } from "./messages";

/**
 * @typedef {Object} FileTask
 * @property {string} id
 * @property {SourceFile} file
 * @property {Worker} worker
 * @property {() => void} cancel
 * @property {Promise<TaskSummary>} done
 */

/**
 * @typedef {Object} TaskSummary
 * @property {string} fileId
 * @property {number} scanCount
 * @property {number} chromatogramCount
 * @property {number} elapsedMs
 */

let nextTaskId = 1;

/**
 * @param {Object} [opts]
 * @param {number} [opts.concurrency=navigator.hardwareConcurrency || 4]
 */
export function createWorkerManager(opts = {}) {
  const concurrency = Math.max(1, opts.concurrency ?? navigator?.hardwareConcurrency ?? 4);
  const activeTasks = new Map();
  const queue = [];

  function spawnWorker() {
    // Next.js-compatible worker instantiation
    return new Worker(
      new URL("./file.worker.js", import.meta.url),
      { type: "module" }
    );
  }

  function canStart() {
    return activeTasks.size < concurrency;
  }

  function processFile(file, options, callbacks = {}) {
    const id = String(nextTaskId++);

    const start = () =>
      new Promise((resolve, reject) => {
        const worker = spawnWorker();
        const task = { id, file, worker };
        activeTasks.set(id, task);

        const summary = {
          fileId: id,
          scanCount: 0,
          chromatogramCount: 0,
          elapsedMs: 0,
          states: [],
        };

        let cancelled = false;

        task.cancel = () => {
          if (cancelled) return;
          cancelled = true;
          try {
            worker.postMessage({ type: MSG.STOP });
          } catch (_) { /* ignore */ }
          setTimeout(() => {
            try { worker.terminate(); } catch (_) { /* ignore */ }
            activeTasks.delete(id);
            drain();
            reject(new Error("cancelled"));
          }, 50);
        };

        worker.addEventListener("message", (ev) => {
          const msg = ev.data || {};
          switch (msg.type) {
            case MSG.READY: {
              // Worker is up — send the START message
              const startMsg = buildStartMessage(
                { handle: file.meta?.handle, url: file.meta?.url, name: file.name },
                { ...options, id }
              );
              worker.postMessage(startMsg);
              break;
            }
            case MSG.PROGRESS:
              if (callbacks.onProgress) {
                const pct = msg.totalBytes ? msg.bytesRead / msg.totalBytes : null;
                callbacks.onProgress({
                  bytesRead: msg.bytesRead,
                  totalBytes: msg.totalBytes,
                  pct,
                });
              }
              break;
            case MSG.STATE_BATCH:
              summary.states.push(...msg.states);
              if (callbacks.onStateBatch) callbacks.onStateBatch(msg.states);
              break;
            case MSG.STATE:
              summary.states.push(msg.state);
              if (callbacks.onStateBatch) callbacks.onStateBatch([msg.state]);
              break;
            case MSG.CHROMATOGRAM:
              summary.chromatogramCount++;
              if (callbacks.onChromatogram) callbacks.onChromatogram(msg.chromatogram);
              break;
            case MSG.DONE:
              summary.scanCount = msg.scanCount;
              summary.chromatogramCount = msg.chromatogramCount;
              summary.elapsedMs = msg.elapsedMs;
              worker.terminate();
              activeTasks.delete(id);
              drain();
              if (callbacks.onDone) callbacks.onDone(summary);
              resolve(summary);
              break;
            case MSG.ERROR:
              if (msg.recoverable) {
                if (callbacks.onError) callbacks.onError(new Error(msg.message));
                return;
              }
              worker.terminate();
              activeTasks.delete(id);
              drain();
              reject(new Error(msg.message || "worker error"));
              break;
            default:
              break;
          }
        });

        worker.addEventListener("error", (ev) => {
          activeTasks.delete(id);
          drain();
          reject(new Error(ev.message || "worker crashed"));
        });

        return task;
      });

    const runOrQueue = () =>
      new Promise((resolve, reject) => {
        const go = () => {
          const p = start();
          p.then(resolve, reject);
        };
        if (canStart()) go();
        else queue.push(go);
      });

    const done = runOrQueue();
    const task = { id, cancel: () => {
      // Before start: remove from queue
      const idx = queue.findIndex((fn) => fn === task._queuedFn);
      if (idx >= 0) queue.splice(idx, 1);
      // After start: actual worker cancel handled by the start() promise
      const active = activeTasks.get(id);
      if (active && active.cancel) active.cancel();
    }, done };
    return task;
  }

  function drain() {
    while (canStart() && queue.length > 0) {
      const next = queue.shift();
      next();
    }
  }

  function cancelAll() {
    for (const task of activeTasks.values()) {
      if (task.cancel) task.cancel();
    }
    queue.length = 0;
  }

  function getStatus() {
    return {
      active: activeTasks.size,
      queued: queue.length,
      concurrency,
    };
  }

  return {
    processFile,
    cancelAll,
    getStatus,
  };
}

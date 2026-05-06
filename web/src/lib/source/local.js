/**
 * Local source — three independent paths:
 *
 *   1. pickLocalFolder()        FileSystemDirectoryHandle (best, lets us re-read)
 *   2. pickLocalFiles()         FileSystemFileHandle[] (Chromium file picker)
 *   3. createSourceFromFiles()  raw File[] (works in ALL browsers — drag-drop, <input>)
 *
 * All three return the same Source shape so downstream code is unaware
 * of which path was taken.
 */

import { isSupportedMsFile } from "./types";

/* -------------------------------------------------------------------- */
/* Capability checks                                                    */
/* -------------------------------------------------------------------- */

export function isLocalFolderSupported() {
  if (typeof window === "undefined") return false;
  return typeof window.showDirectoryPicker === "function";
}

export function isFilePickerSupported() {
  if (typeof window === "undefined") return false;
  return typeof window.showOpenFilePicker === "function";
}

/* -------------------------------------------------------------------- */
/* Path 1: Directory picker (Chromium)                                  */
/* -------------------------------------------------------------------- */

export async function pickLocalFolder() {
  if (!isLocalFolderSupported()) {
    throw new Error(
      "Folder picker unavailable in this browser. Use the file picker or drag-and-drop instead."
    );
  }

  const handle = await window.showDirectoryPicker({
    mode: "read",
    id: "lavoisier-workspace",
  });

  return createLocalSource(handle);
}

export function createLocalSource(dirHandle) {
  return {
    id: `local:dir:${dirHandle.name}:${Date.now()}`,
    label: dirHandle.name,
    kind: "local",
    meta: { handle: dirHandle, mode: "directory" },

    async listFiles() {
      const files = [];
      await walkDirectory(dirHandle, "", files);
      return files;
    },
  };
}

async function walkDirectory(dir, prefix, out) {
  for await (const [name, entry] of dir.entries()) {
    const path = prefix ? `${prefix}/${name}` : name;
    if (entry.kind === "directory") {
      await walkDirectory(entry, path, out);
    } else if (entry.kind === "file" && isSupportedMsFile(name)) {
      out.push(makeFromHandle(entry, name, path));
    }
  }
}

/* -------------------------------------------------------------------- */
/* Path 2: File picker (Chromium)                                       */
/* -------------------------------------------------------------------- */

/**
 * Open the OS file picker for one or more individual MS files.
 * @returns {Promise<Source>}
 */
export async function pickLocalFiles() {
  if (!isFilePickerSupported()) {
    throw new Error(
      "File picker unavailable in this browser. Try drag-and-drop or use Chromium-based browser."
    );
  }

  const handles = await window.showOpenFilePicker({
    multiple: true,
    types: [
      {
        description: "Mass spectrometry data",
        accept: {
          "application/xml": [".mzML", ".mzXML", ".imzML"],
          "text/plain": [".mgf"],
          "application/json": [".json"],
        },
      },
    ],
    excludeAcceptAllOption: false,
  });

  const files = [];
  for (const h of handles) {
    if (isSupportedMsFile(h.name)) {
      files.push(makeFromHandle(h, h.name, h.name));
    }
  }

  return makeFlatSource(files, files.length === 1 ? files[0].name : `${files.length} files`);
}

/* -------------------------------------------------------------------- */
/* Path 3: Raw File[] (drag-drop, <input type="file">)                  */
/* -------------------------------------------------------------------- */

/**
 * Build a Source from a list of regular File objects.
 * Works in every browser. The trade-off vs FileSystemFileHandle is
 * that we can't re-read after the page reloads — but for a single
 * session this is irrelevant.
 *
 * @param {File[]} fileList
 * @param {string} [label]
 * @returns {Source}
 */
export function createSourceFromFiles(fileList, label = null) {
  const files = [];
  for (const f of fileList) {
    if (!isSupportedMsFile(f.name)) continue;
    // Use webkitRelativePath if it exists (folder drop), else just name
    const path = f.webkitRelativePath || f.name;
    files.push(makeFromFile(f, f.name, path));
  }

  const labelToUse =
    label ||
    (files.length === 1
      ? files[0].name
      : files.length > 1
      ? `${files.length} files`
      : "No files");

  return makeFlatSource(files, labelToUse);
}

/* -------------------------------------------------------------------- */
/* Internal helpers                                                     */
/* -------------------------------------------------------------------- */

function makeFlatSource(files, label) {
  return {
    id: `local:flat:${Date.now()}`,
    label,
    kind: "local",
    meta: { mode: "flat", count: files.length },
    async listFiles() {
      return files;
    },
  };
}

/** SourceFile from a FileSystemFileHandle (re-readable). */
function makeFromHandle(handle, name, path) {
  let cachedFile = null;
  let cachedSize = null;

  const grabFile = async () => {
    if (cachedFile) return cachedFile;
    cachedFile = await handle.getFile();
    cachedSize = cachedFile.size;
    return cachedFile;
  };

  return {
    id: `local:${path}`,
    name,
    path,
    size: null,
    kind: "local",
    meta: { handle, mode: "handle" },

    async stream() {
      const file = await grabFile();
      return file.stream();
    },
    async range(start, end) {
      const file = await grabFile();
      const slice = file.slice(start, end);
      return new Uint8Array(await slice.arrayBuffer());
    },
    async ensureSize() {
      if (cachedSize != null) return cachedSize;
      const file = await grabFile();
      return file.size;
    },
  };
}

/** SourceFile from a raw File (drag-drop / file input). */
function makeFromFile(file, name, path) {
  return {
    id: `local:file:${path}:${file.size}:${file.lastModified}`,
    name,
    path,
    size: file.size,
    kind: "local",
    meta: { mode: "file" },

    async stream() {
      return file.stream();
    },
    async range(start, end) {
      const slice = file.slice(start, end);
      return new Uint8Array(await slice.arrayBuffer());
    },
    async ensureSize() {
      return file.size;
    },
  };
}

/* -------------------------------------------------------------------- */
/* Utility: filter a DataTransferItemList into Files (handles folders)  */
/* -------------------------------------------------------------------- */

/**
 * Extract all File objects from a drop event, recursing into directories
 * via the legacy webkitGetAsEntry API. Used by the drag-drop UI.
 *
 * @param {DataTransferItemList|FileList} dataTransfer
 * @returns {Promise<File[]>}
 */
export async function extractFilesFromDataTransfer(dt) {
  // Plain FileList (e.g. <input type="file">)
  if (dt instanceof FileList) {
    return Array.from(dt);
  }

  // DataTransferItemList from a drop event
  const items = Array.from(dt);
  const out = [];

  // Use webkitGetAsEntry for directory recursion when available
  for (const item of items) {
    if (item.kind !== "file") continue;
    const entry = item.webkitGetAsEntry?.();
    if (entry) {
      await walkEntry(entry, "", out);
    } else {
      const f = item.getAsFile();
      if (f) out.push(f);
    }
  }
  return out;
}

async function walkEntry(entry, prefix, out) {
  if (entry.isFile) {
    const f = await new Promise((resolve, reject) =>
      entry.file(resolve, reject)
    );
    // Attach the relative path so createSourceFromFiles preserves structure
    if (prefix) {
      try {
        Object.defineProperty(f, "webkitRelativePath", {
          value: prefix + f.name,
          configurable: true,
        });
      } catch (_) { /* read-only in some engines */ }
    }
    out.push(f);
  } else if (entry.isDirectory) {
    const reader = entry.createReader();
    const entries = await new Promise((resolve, reject) =>
      reader.readEntries(resolve, reject)
    );
    for (const e of entries) {
      await walkEntry(e, prefix + entry.name + "/", out);
    }
  }
}

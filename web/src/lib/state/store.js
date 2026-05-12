/**
 * Zustand store — the workspace's single source of truth.
 *
 * State surface:
 *   source       — connected Source (local folder | repository | URLs)
 *   files        — SourceFile[] discovered from the source
 *   selectedFiles — Set of file IDs the user has chosen to process
 *   tasks        — Map<fileId, TaskState> tracking each worker's progress
 *   states       — flat array of CategoricalStates streamed from workers
 *   analyser     — "tof" | "quadrupole" | "orbitrap" | "fticr"
 *   analyserCfg  — per-analyser uniform values
 *   selectedAddress — currently focused ternary address (for detail/3D view)
 *   trie         — TernaryTrie holding all observations (rebuilt on reset)
 *   gpuReady     — whether the WebGL2 pipeline has initialised
 *   quality      — last GPU quality metrics
 */

import { create } from "zustand";
import { TernaryTrie } from "../partition/trie";

const initialAnalyserCfg = {
  tof:        { accelV: 5000, flightLength: 1.0 },
  quadrupole: { dcVoltage: 100, rfVoltage: 500, rfFrequency: 1e6, r0: 5e-3 },
  orbitrap:   { kField: 1e12, Rm: 1e-2 },
  fticr:      { B: 7.0 },
};

export const useStore = create((set, get) => ({
  // ---- source ----
  source: null,
  files: [],
  selectedFiles: new Set(),

  setSource: (source) => set({ source, files: [], selectedFiles: new Set() }),
  clearSource: () => set({
    source: null,
    files: [],
    selectedFiles: new Set(),
    tasks: new Map(),
  }),
  setFiles: (files) => set({ files }),
  toggleFile: (fileId) => set((s) => {
    const next = new Set(s.selectedFiles);
    if (next.has(fileId)) next.delete(fileId);
    else next.add(fileId);
    return { selectedFiles: next };
  }),
  selectAllFiles: () => set((s) => ({
    selectedFiles: new Set(s.files.map((f) => f.id)),
  })),
  clearSelection: () => set({ selectedFiles: new Set() }),

  // ---- tasks ----
  tasks: new Map(),
  setTaskState: (fileId, patch) => set((s) => {
    const next = new Map(s.tasks);
    const cur = next.get(fileId) || {};
    next.set(fileId, { ...cur, ...patch });
    return { tasks: next };
  }),
  clearTasks: () => set({ tasks: new Map() }),

  // ---- streaming results ----
  states: [],
  trie: new TernaryTrie(),
  totalScanCount: 0,

  appendStates: (newStates) => set((s) => {
    const trie = s.trie; // mutate in place — Zustand re-renders via state reassign below
    for (const st of newStates) {
      trie.insert(st.address, st);
    }
    return {
      states: s.states.concat(newStates),
      trie,
      totalScanCount: s.totalScanCount + newStates.length,
    };
  }),

  resetResults: () => set({
    states: [],
    trie: new TernaryTrie(),
    totalScanCount: 0,
    selectedAddress: null,
  }),

  // ---- analyser ----
  analyser: "orbitrap",
  analyserCfg: initialAnalyserCfg,

  setAnalyser: (analyser) => set({ analyser }),
  setAnalyserCfg: (analyser, patch) => set((s) => ({
    analyserCfg: {
      ...s.analyserCfg,
      [analyser]: { ...s.analyserCfg[analyser], ...patch },
    },
  })),

  // ---- selection / focus ----
  selectedAddress: null,
  selectAddress: (address) => set({ selectedAddress: address }),

  // ---- GPU state ----
  gpuReady: false,
  quality: null,
  setGpuReady: (ready) => set({ gpuReady: ready }),
  setQuality: (q) => set({ quality: q }),

  // ---- UI / theme ----
  showHelp: false,
  toggleHelp: () => set((s) => ({ showHelp: !s.showHelp })),

  // ===========================================================
  // Experiment / virtual-instrument slice
  // ===========================================================
  experimentDesign: {
    experimentType: "lipidomics",
    classSpecs: [
      { classKey: "PC",  Xmin: 30, Xmax: 40, Ymin: 0, Ymax: 4, enabled: true  },
      { classKey: "PE",  Xmin: 30, Xmax: 40, Ymin: 0, Ymax: 4, enabled: true  },
      { classKey: "SM",  Xmin: 16, Xmax: 24, Ymin: 0, Ymax: 2, enabled: true  },
      { classKey: "Cer", Xmin: 16, Xmax: 24, Ymin: 0, Ymax: 2, enabled: true  },
      { classKey: "TAG", Xmin: 46, Xmax: 56, Ymin: 0, Ymax: 6, enabled: false },
    ],
    proteinSpecs: [
      { classKey: "HSA",  lengthMin: 7, lengthMax: 20, mcMin: 0, mcMax: 1, enabled: true  },
      { classKey: "ENO1", lengthMin: 7, lengthMax: 20, mcMin: 0, mcMax: 0, enabled: true  },
      { classKey: "CYCS", lengthMin: 6, lengthMax: 20, mcMin: 0, mcMax: 1, enabled: false },
    ],
    polarity: "+",
    adductsAllowed: ["[M+H]+", "[M+Na]+", "[M+NH4]+"],
    analyser: "orbitrap",
    analyserCfg: { kField: 1e12, Rm: 1e-2 },
    collisionEnergy_eV: 25,
    mzWindow: [200, 1500],
    sampleType: "plasma",
    extraction: "MTBE",
    chromatography: "RPLC C18 30 min gradient",
  },
  experimentRecords: [],         // PredictedRecord[]
  experimentSummary: null,       // summariseRecords output
  experimentRunning: false,
  experimentLastRunMs: 0,

  setExperimentDesign: (patch) => set((s) => ({
    experimentDesign: { ...s.experimentDesign, ...patch },
  })),
  setClassSpec: (classKey, patch) => set((s) => ({
    experimentDesign: {
      ...s.experimentDesign,
      classSpecs: s.experimentDesign.classSpecs.map((cs) =>
        cs.classKey === classKey ? { ...cs, ...patch } : cs
      ),
    },
  })),
  toggleClass: (classKey) => set((s) => ({
    experimentDesign: {
      ...s.experimentDesign,
      classSpecs: s.experimentDesign.classSpecs.map((cs) =>
        cs.classKey === classKey ? { ...cs, enabled: !cs.enabled } : cs
      ),
    },
  })),
  addClassSpec: (classKey) => set((s) => {
    if (s.experimentDesign.classSpecs.find((cs) => cs.classKey === classKey)) return {};
    return {
      experimentDesign: {
        ...s.experimentDesign,
        classSpecs: [
          ...s.experimentDesign.classSpecs,
          { classKey, Xmin: 30, Xmax: 40, Ymin: 0, Ymax: 4, enabled: true },
        ],
      },
    };
  }),
  removeClassSpec: (classKey) => set((s) => ({
    experimentDesign: {
      ...s.experimentDesign,
      classSpecs: s.experimentDesign.classSpecs.filter((cs) => cs.classKey !== classKey),
    },
  })),
  setExperimentRecords: (records, summary, elapsedMs) => set({
    experimentRecords: records,
    experimentSummary: summary,
    experimentLastRunMs: elapsedMs,
  }),
  setExperimentRunning: (b) => set({ experimentRunning: b }),
  resetExperiment: () => set({
    experimentRecords: [],
    experimentSummary: null,
    experimentLastRunMs: 0,
  }),

  // Switch between lipidomics and proteomics — resets adduct defaults
  setExperimentType: (type) => set((s) => ({
    experimentDesign: {
      ...s.experimentDesign,
      experimentType: type,
      polarity: "+",
      adductsAllowed: type === "proteomics"
        ? ["[M+2H]2+", "[M+3H]3+", "[M+H]+"]
        : ["[M+H]+", "[M+Na]+", "[M+NH4]+"],
    },
  })),

  // Proteomics protein-spec actions (mirror the lipidomics classSpec actions)
  setProteinSpec: (classKey, patch) => set((s) => ({
    experimentDesign: {
      ...s.experimentDesign,
      proteinSpecs: (s.experimentDesign.proteinSpecs || []).map((ps) =>
        ps.classKey === classKey ? { ...ps, ...patch } : ps
      ),
    },
  })),
  toggleProtein: (classKey) => set((s) => ({
    experimentDesign: {
      ...s.experimentDesign,
      proteinSpecs: (s.experimentDesign.proteinSpecs || []).map((ps) =>
        ps.classKey === classKey ? { ...ps, enabled: !ps.enabled } : ps
      ),
    },
  })),
  addProteinSpec: (classKey) => set((s) => {
    const specs = s.experimentDesign.proteinSpecs || [];
    if (specs.find((ps) => ps.classKey === classKey)) return {};
    return {
      experimentDesign: {
        ...s.experimentDesign,
        proteinSpecs: [
          ...specs,
          { classKey, lengthMin: 7, lengthMax: 20, mcMin: 0, mcMax: 1, enabled: true },
        ],
      },
    };
  }),
  removeProteinSpec: (classKey) => set((s) => ({
    experimentDesign: {
      ...s.experimentDesign,
      proteinSpecs: (s.experimentDesign.proteinSpecs || []).filter(
        (ps) => ps.classKey !== classKey
      ),
    },
  })),

  selectedRecordId: null,
  selectRecord: (id) => set({ selectedRecordId: id }),
}));

/**
 * Selectors — derive useful slices without re-rendering on unrelated state.
 */

export const useSource = () => useStore((s) => s.source);
export const useFiles = () => useStore((s) => s.files);
export const useSelectedFiles = () => useStore((s) => s.selectedFiles);
export const useStates = () => useStore((s) => s.states);
export const useAnalyser = () => useStore((s) => s.analyser);
export const useAnalyserCfg = () => useStore((s) => s.analyserCfg[s.analyser]);
export const useSelectedAddress = () => useStore((s) => s.selectedAddress);
export const useGpuReady = () => useStore((s) => s.gpuReady);
export const useQuality = () => useStore((s) => s.quality);

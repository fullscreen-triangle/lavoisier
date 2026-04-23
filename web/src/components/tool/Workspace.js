import React, { useState } from "react";

/**
 * The Lavoisier workspace — the complete force-free mass spectrometer in the browser.
 *
 * Layout:
 *   [  SourcePicker  ]  [  AnalyserPanel  ]  [  StatusBar  ]
 *   [  FileTree    ]  [   ShaderCanvas (S-entropy viewer)   ]
 *                     [   ResultsTable (live stream)        ]
 *
 * This is a skeleton. Individual panels will be wired in as layers come online.
 */
export default function Workspace() {
  const [gpuReady, setGpuReady] = useState(false);
  const [source, setSource] = useState(null);

  return (
    <div className="flex flex-col w-full min-h-[calc(100vh-180px)] bg-light dark:bg-dark text-dark dark:text-light">
      {/* Header bar */}
      <div className="flex items-center justify-between px-8 py-3 border-b-2 border-dark/10 dark:border-light/10">
        <div>
          <div className="text-sm uppercase tracking-wider text-dark/60 dark:text-light/60">
            Lavoisier Workspace
          </div>
          <div className="text-xs text-dark/40 dark:text-light/40 mt-0.5">
            Force-free mass spectrometer · GPU observation apparatus
          </div>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <StatusDot ok={gpuReady} label={gpuReady ? "GPU ready" : "GPU waiting"} />
          <StatusDot ok={!!source} label={source ? "Source connected" : "No source"} />
        </div>
      </div>

      {/* Main 3-column layout */}
      <div className="grid grid-cols-[260px_1fr_360px] gap-0 flex-1 lg:grid-cols-1">
        {/* Left: Source + File tree */}
        <aside className="border-r-2 border-dark/10 dark:border-light/10 p-4 overflow-y-auto">
          <h3 className="text-xs uppercase tracking-wider font-bold text-dark/60 dark:text-light/60 mb-3">
            Source
          </h3>
          <SourcePlaceholder onSource={setSource} />

          <h3 className="text-xs uppercase tracking-wider font-bold text-dark/60 dark:text-light/60 mt-6 mb-3">
            Files
          </h3>
          <FileTreePlaceholder />
        </aside>

        {/* Centre: Shader canvas + analyser panel */}
        <main className="flex flex-col p-4 overflow-hidden">
          <AnalyserPanelPlaceholder />
          <ShaderCanvasPlaceholder onReady={() => setGpuReady(true)} />
        </main>

        {/* Right: Results table */}
        <aside className="border-l-2 border-dark/10 dark:border-light/10 p-4 overflow-y-auto">
          <h3 className="text-xs uppercase tracking-wider font-bold text-dark/60 dark:text-light/60 mb-3">
            Live Results
          </h3>
          <ResultsTablePlaceholder />
        </aside>
      </div>
    </div>
  );
}

function StatusDot({ ok, label }) {
  return (
    <div className="flex items-center gap-1.5">
      <div
        className={`w-2 h-2 rounded-full ${
          ok ? "bg-green-500" : "bg-dark/30 dark:bg-light/30"
        }`}
      />
      <span className={ok ? "text-dark dark:text-light" : "text-dark/50 dark:text-light/50"}>
        {label}
      </span>
    </div>
  );
}

function SourcePlaceholder({ onSource }) {
  return (
    <div className="space-y-2">
      <button
        className="w-full px-3 py-2 text-sm rounded border-2 border-dashed border-dark/20 dark:border-light/20
          hover:border-primary dark:hover:border-primaryDark hover:bg-primary/5 dark:hover:bg-primaryDark/5 transition-colors"
        onClick={() => {
          alert("Local folder picker — coming in source layer");
        }}
      >
        📁 Open Local Folder
      </button>
      <div className="text-center text-xs text-dark/40 dark:text-light/40 py-1">or</div>
      <button
        className="w-full px-3 py-2 text-sm rounded border-2 border-dashed border-dark/20 dark:border-light/20
          hover:border-primary dark:hover:border-primaryDark hover:bg-primary/5 dark:hover:bg-primaryDark/5 transition-colors"
        onClick={() => {
          alert("Repository linker — coming in source layer");
        }}
      >
        🔗 Link Repository
      </button>
    </div>
  );
}

function FileTreePlaceholder() {
  return (
    <div className="text-xs text-dark/40 dark:text-light/40 italic py-4">
      No source connected.
    </div>
  );
}

function AnalyserPanelPlaceholder() {
  const analysers = ["TOF", "Quadrupole", "Orbitrap", "FT-ICR"];
  return (
    <div className="flex items-center gap-2 mb-4 pb-4 border-b-2 border-dark/10 dark:border-light/10">
      <span className="text-xs uppercase tracking-wider font-bold text-dark/60 dark:text-light/60 mr-2">
        Analyser:
      </span>
      {analysers.map((a, i) => (
        <button
          key={a}
          className={`px-3 py-1 text-xs rounded border-2 font-medium transition-colors
            ${
              i === 0
                ? "border-primary bg-primary/10 text-primary dark:border-primaryDark dark:bg-primaryDark/10 dark:text-primaryDark"
                : "border-dark/10 dark:border-light/10 hover:border-dark/30 dark:hover:border-light/30"
            }`}
        >
          {a}
        </button>
      ))}
      <span className="ml-auto text-xs font-mono text-dark/40 dark:text-light/40">
        Partition Lagrangian: L_M = ½μ|ẋ|² + μẋ·A_M − M(x,t)
      </span>
    </div>
  );
}

function ShaderCanvasPlaceholder({ onReady }) {
  React.useEffect(() => {
    // When the real shader canvas comes online, it calls onReady
    // For now, just simulate
    const t = setTimeout(() => onReady(), 600);
    return () => clearTimeout(t);
  }, [onReady]);

  return (
    <div
      className="flex-1 rounded-lg border-2 border-dark/10 dark:border-light/10
        bg-gradient-to-br from-dark/5 to-primary/5 dark:from-light/5 dark:to-primaryDark/5
        flex items-center justify-center min-h-[400px]"
    >
      <div className="text-center">
        <div className="text-xs uppercase tracking-wider text-dark/40 dark:text-light/40 mb-2">
          Shader Canvas
        </div>
        <div className="text-sm text-dark/60 dark:text-light/60">
          S-entropy space [0,1]³ viewer — coming in GPU layer
        </div>
      </div>
    </div>
  );
}

function ResultsTablePlaceholder() {
  return (
    <div className="text-xs text-dark/40 dark:text-light/40 italic py-4">
      No observations yet.
    </div>
  );
}

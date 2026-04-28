import React from "react";
import { useStore } from "@/lib/state/store";
import { ternary } from "@/lib/partition";

/**
 * DetailPanel — shows full information about the selected categorical state.
 *
 * Includes nearest-neighbour suggestions from the trie (resonance scores).
 */
export default function DetailPanel() {
  const states = useStore((s) => s.states);
  const trie = useStore((s) => s.trie);
  const selectedAddress = useStore((s) => s.selectedAddress);
  const selectAddress = useStore((s) => s.selectAddress);

  if (!selectedAddress) {
    return (
      <div className="rounded-lg border border-dashed border-dark/10 dark:border-light/10 p-3
        text-xs text-dark/40 dark:text-light/40 italic text-center">
        Click a row or a 3D point to inspect.
      </div>
    );
  }

  const target = states.find((s) => s.address === selectedAddress);
  if (!target) return null;

  const neighbours = trie.nearest(selectedAddress, 6).filter(
    ({ entry }) => entry.address !== selectedAddress
  );

  return (
    <div className="rounded-lg border-2 border-primary/30 dark:border-primaryDark/30
      bg-primary/5 dark:bg-primaryDark/5 p-3 text-xs space-y-3">
      <div>
        <div className="text-[10px] uppercase tracking-wider text-primary dark:text-primaryDark font-bold">
          Selected
        </div>
        <div className="font-mono break-all mt-1 text-[11px]">{target.address}</div>
      </div>

      <div className="grid grid-cols-3 gap-2">
        <KV k="Sk" v={target.sentropy.sk.toFixed(4)} />
        <KV k="St" v={target.sentropy.st.toFixed(4)} />
        <KV k="Se" v={target.sentropy.se.toFixed(4)} />
        <KV k="m/z" v={target.basePeakMz.toFixed(4)} />
        <KV k="MS" v={target.msLevel} />
        <KV k="z" v={target.charge || "—"} />
        <KV k="RT" v={target.retentionTime.toFixed(2)} />
        <KV k="peaks" v={target.nPeaks} />
        <KV k="pol" v={target.polarity[0]?.toUpperCase() || "—"} />
      </div>

      {target.observables && (
        <div className="pt-2 border-t border-dark/10 dark:border-light/10">
          <div className="text-[10px] uppercase tracking-wider text-dark/60 dark:text-light/60 mb-1">
            {target.observables.observable}
          </div>
          <div className="font-mono text-[11px]">
            {pickObservableValue(target.observables)}
          </div>
        </div>
      )}

      {target.hierarchy && target.hierarchy.oscillators?.length > 0 && (
        <HierarchyView h={target.hierarchy} />
      )}

      {neighbours.length > 0 && (
        <div className="pt-2 border-t border-dark/10 dark:border-light/10">
          <div className="text-[10px] uppercase tracking-wider text-dark/60 dark:text-light/60 mb-1">
            Resonant neighbours
          </div>
          <div className="space-y-1">
            {neighbours.map(({ entry, prefixLength }) => {
              const score = ternary.resonanceScore(entry.address, selectedAddress);
              return (
                <button
                  key={entry.address}
                  onClick={() => selectAddress(entry.address)}
                  className="w-full text-left rounded p-1 hover:bg-dark/10 dark:hover:bg-light/10
                    flex items-center justify-between gap-2"
                >
                  <span className="font-mono text-[11px] truncate flex-1">
                    {entry.address.substring(0, 14)}…
                  </span>
                  <span className="text-[10px] font-mono text-dark/60 dark:text-light/60 flex-shrink-0">
                    {prefixLength} trits · {(score * 100).toFixed(0)}%
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function KV({ k, v }) {
  return (
    <div className="flex flex-col">
      <span className="text-[10px] text-dark/50 dark:text-light/50 uppercase tracking-wider">{k}</span>
      <span className="font-mono text-[11px]">{v}</span>
    </div>
  );
}

function pickObservableValue(obs) {
  if (obs.T != null) return `T = ${(obs.T * 1e6).toFixed(3)} μs`;
  if (obs.frequencyHz != null) return `ω = ${(obs.frequencyHz / 1000).toFixed(2)} kHz`;
  if (obs.q != null) return `a = ${obs.a?.toFixed(4)}, q = ${obs.q.toFixed(4)} (${obs.stable ? "stable" : "unstable"})`;
  return JSON.stringify(obs);
}

function HierarchyView({ h }) {
  const top = h.oscillators.slice(0, 8);
  const fillRatio = h.totalCells > 0 ? h.occupiedCells / h.totalCells : 0;
  return (
    <div className="pt-2 border-t border-dark/10 dark:border-light/10 space-y-2">
      <div className="flex items-center justify-between">
        <div className="text-[10px] uppercase tracking-wider text-dark/60 dark:text-light/60">
          Oscillator hierarchy
        </div>
        <div className="text-[10px] font-mono text-dark/50 dark:text-light/50">
          n_max={h.nMax} · {h.occupiedCells}/{h.totalCells} cells
          {" · "}H={h.entropyNats.toFixed(2)}
        </div>
      </div>

      {h.shells && h.shells.length > 0 && (
        <div className="flex items-end gap-0.5 h-8">
          {h.shells.map((p, i) => (
            <div
              key={i}
              className="flex-1 bg-primary/40 dark:bg-primaryDark/40 rounded-sm relative"
              style={{ height: `${Math.max(2, p * 100)}%` }}
              title={`shell n=${i + 1} : ${(p * 100).toFixed(1)}%`}
            >
              <span className="absolute -bottom-3 left-0 right-0 text-center text-[8px] text-dark/40 dark:text-light/40">
                {i + 1}
              </span>
            </div>
          ))}
        </div>
      )}

      <div className="pt-3">
        <div className="grid grid-cols-[auto_auto_auto_auto_auto_1fr] gap-x-2 text-[10px] font-mono">
          <span className="text-dark/40 dark:text-light/40">n</span>
          <span className="text-dark/40 dark:text-light/40">l</span>
          <span className="text-dark/40 dark:text-light/40">m</span>
          <span className="text-dark/40 dark:text-light/40">s</span>
          <span className="text-dark/40 dark:text-light/40">m/z</span>
          <span className="text-dark/40 dark:text-light/40">w</span>
          {top.map((osc, i) => (
            <React.Fragment key={i}>
              <span>{osc.n}</span>
              <span>{osc.l}</span>
              <span>{osc.m >= 0 ? `+${osc.m}` : osc.m}</span>
              <span>{osc.s > 0 ? "+½" : "−½"}</span>
              <span className="text-dark/70 dark:text-light/70">{osc.mz.toFixed(3)}</span>
              <span className="text-primary dark:text-primaryDark">
                {(osc.weight * 100).toFixed(1)}%
              </span>
            </React.Fragment>
          ))}
        </div>
      </div>

      {h.address && (
        <div>
          <div className="text-[10px] uppercase tracking-wider text-dark/40 dark:text-light/40">
            burst address
          </div>
          <div className="font-mono text-[10px] break-all text-dark/70 dark:text-light/70">
            {h.address}
          </div>
        </div>
      )}

      <div className="text-[9px] text-dark/40 dark:text-light/40 italic">
        partition fill {(fillRatio * 100).toFixed(1)}% — each peak occupies its
        own (n,l,m,s) cell rather than being collapsed into 3 statistics.
      </div>
    </div>
  );
}

import React, { useMemo } from "react";
import { useStore } from "@/lib/state/store";
import D3SpectrumChart from "./charts/D3SpectrumChart";
import D3IsotopeEnvelope from "./charts/D3IsotopeEnvelope";
import D3FragmentationTree from "./charts/D3FragmentationTree";
import { classColor } from "./charts/palette";

export default function RecordDetail() {
  const records = useStore((s) => s.experimentRecords);
  const selId = useStore((s) => s.selectedRecordId);

  const record = useMemo(() => {
    if (!selId) return records[0];
    return records.find((r) => `${r.analyte}${r.adduct}` === selId) || records[0];
  }, [records, selId]);

  if (!record) return null;

  return (
    <section className="rounded-md border border-dark/10 dark:border-light/10 p-4 space-y-3">
      <div className="flex items-baseline gap-3 flex-wrap">
        <span
          className="px-2 py-0.5 rounded text-[10px] text-light"
          style={{ background: classColor(record.analyteClass) }}
        >
          {record.analyteClass}
        </span>
        <h3 className="text-base font-bold font-mono">
          {record.analyte} {record.adduct}
        </h3>
        <span className="text-[11px] text-dark/60 dark:text-light/60">
          m/z {record.precursorMz.toFixed(4)} · z {record.z} · I {record.intensity.toExponential(2)}
        </span>
      </div>

      <div className="grid grid-cols-4 gap-3 text-[11px]">
        <Stat label="partition (n,ℓ,m,s)" value={`${record.n}, ${record.l}, ${record.m}, ${record.s.toFixed(1)}`} />
        <Stat label="S-entropy" value={
          `Sₖ ${record.sentropy.sk.toFixed(2)} Sₜ ${record.sentropy.st.toFixed(2)} Sₑ ${record.sentropy.se.toFixed(2)}`
        } />
        <Stat label="bits / record" value={record.bitsTotal.toFixed(1)} />
        <Stat label="fragments" value={record.ms2.length} />
      </div>

      <div className="grid grid-cols-3 gap-3">
        <div className="col-span-2">
          <SectionTitle>Predicted MS/MS spectrum</SectionTitle>
          <D3SpectrumChart peaks={record.peaksAll} width={620} height={240} />
        </div>
        <div>
          <SectionTitle>Isotope envelope</SectionTitle>
          <D3IsotopeEnvelope record={record} width={300} height={200} />
        </div>
      </div>

      <div>
        <SectionTitle>Fragmentation tree</SectionTitle>
        <D3FragmentationTree record={record} width={920} height={300} />
      </div>
    </section>
  );
}

function Stat({ label, value }) {
  return (
    <div className="rounded bg-dark/5 dark:bg-light/5 px-2 py-1.5">
      <div className="text-[9px] uppercase tracking-wider text-dark/50 dark:text-light/50">{label}</div>
      <div className="font-mono">{value}</div>
    </div>
  );
}

function SectionTitle({ children }) {
  return (
    <div className="text-[10px] uppercase tracking-wider text-dark/60 dark:text-light/60 mb-1">
      {children}
    </div>
  );
}

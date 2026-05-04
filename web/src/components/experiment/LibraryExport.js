import React from "react";
import { useStore } from "@/lib/state/store";
import { FORMATS, downloadAs } from "@/lib/experiment/library";

export default function LibraryExport() {
  const records = useStore((s) => s.experimentRecords);
  const design = useStore((s) => s.experimentDesign);

  if (records.length === 0) return null;

  const stamp = new Date().toISOString().slice(0, 10);

  const onDownload = (fmt) => {
    const content = fmt.builder(records, design);
    const fname = `lavoisier_lib_${stamp}.${fmt.ext}`;
    downloadAs(fname, content, fmt.mime);
  };

  return (
    <div className="space-y-2">
      <div className="text-xs uppercase tracking-wider font-bold text-dark/60 dark:text-light/60">
        Export library
      </div>
      <div className="grid grid-cols-2 gap-2">
        {FORMATS.map((fmt) => (
          <button
            key={fmt.key}
            onClick={() => onDownload(fmt)}
            className="text-[11px] px-2 py-1.5 rounded border border-dark/15 dark:border-light/15
              hover:bg-dark/5 dark:hover:bg-light/5 text-left"
          >
            <div className="font-bold">{fmt.label}</div>
            <div className="text-dark/50 dark:text-light/50">.{fmt.ext}</div>
          </button>
        ))}
      </div>
    </div>
  );
}

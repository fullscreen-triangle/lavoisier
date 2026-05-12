import React from "react";
import { useStore } from "@/lib/state/store";
import {
  ADDUCTS,
  ADDUCTS_POSITIVE,
  ADDUCTS_NEGATIVE,
  ADDUCTS_PROTEOMICS_POSITIVE,
  ADDUCTS_PROTEOMICS_NEGATIVE,
} from "@/lib/experiment/adducts";

export default function IonizationConfig() {
  const design          = useStore((s) => s.experimentDesign);
  const setDesign       = useStore((s) => s.setExperimentDesign);
  const experimentType  = design.experimentType || "lipidomics";
  const polarity        = design.polarity;
  const adducts         = design.adductsAllowed;

  // Show a different adduct set depending on experiment type
  const candidates =
    experimentType === "proteomics"
      ? (polarity === "+" ? ADDUCTS_PROTEOMICS_POSITIVE : ADDUCTS_PROTEOMICS_NEGATIVE)
      : (polarity === "+" ? ADDUCTS_POSITIVE             : ADDUCTS_NEGATIVE);

  const togglePolarity = () => {
    const next = polarity === "+" ? "-" : "+";
    const defaults =
      experimentType === "proteomics"
        ? (next === "+" ? ["[M+2H]2+", "[M+3H]3+", "[M+H]+"] : ["[M-2H]2-", "[M-H]-"])
        : (next === "+" ? ["[M+H]+", "[M+Na]+", "[M+NH4]+"]  : ["[M-H]-", "[M+HCOO]-"]);
    setDesign({ polarity: next, adductsAllowed: defaults });
  };

  const toggleAdduct = (a) => {
    const next = adducts.includes(a)
      ? adducts.filter((x) => x !== a)
      : [...adducts, a];
    setDesign({ adductsAllowed: next });
  };

  return (
    <div className="space-y-3">
      <div className="text-xs uppercase tracking-wider font-bold text-dark/60 dark:text-light/60">
        Ionisation
      </div>

      <div className="flex items-center gap-2">
        <span className="text-[11px] text-dark/60 dark:text-light/60">Polarity</span>
        <button
          onClick={togglePolarity}
          className="px-3 py-1 rounded text-xs font-bold bg-dark text-light dark:bg-light dark:text-dark"
        >
          ESI {polarity}
        </button>
      </div>

      <div>
        <div className="text-[11px] text-dark/60 dark:text-light/60 mb-1">Allowed adducts</div>
        <div className="flex flex-wrap gap-1">
          {candidates.map((a) => {
            const active = adducts.includes(a);
            const info   = ADDUCTS[a];
            return (
              <button
                key={a}
                onClick={() => toggleAdduct(a)}
                className={`text-[10px] px-2 py-0.5 rounded border ${
                  active
                    ? "bg-dark text-light dark:bg-light dark:text-dark border-dark dark:border-light"
                    : "border-dark/20 dark:border-light/20 hover:bg-dark/5 dark:hover:bg-light/5"
                }`}
                title={info ? `z = ${info.z > 0 ? "+" : ""}${info.z}` : ""}
              >
                {a}
              </button>
            );
          })}
        </div>
        {experimentType === "proteomics" && (
          <p className="text-[9px] text-dark/40 dark:text-light/40 mt-1.5 leading-relaxed">
            Proteomics peptides typically form multiply-charged ions.
            Select two or more charge states for realistic coverage.
          </p>
        )}
      </div>
    </div>
  );
}

import React from "react";
import { useStore } from "@/lib/state/store";
import { ADDUCTS, ADDUCTS_POSITIVE, ADDUCTS_NEGATIVE } from "@/lib/experiment/adducts";

export default function IonizationConfig() {
  const design = useStore((s) => s.experimentDesign);
  const setDesign = useStore((s) => s.setExperimentDesign);

  const polarity = design.polarity;
  const adducts = design.adductsAllowed;
  const candidates = polarity === "+" ? ADDUCTS_POSITIVE : ADDUCTS_NEGATIVE;

  const togglePolarity = () => {
    const next = polarity === "+" ? "-" : "+";
    const defaults = next === "+" ? ["[M+H]+", "[M+Na]+", "[M+NH4]+"] : ["[M-H]-", "[M+HCOO]-"];
    setDesign({ polarity: next, adductsAllowed: defaults });
  };

  const toggleAdduct = (a) => {
    const next = adducts.includes(a) ? adducts.filter((x) => x !== a) : [...adducts, a];
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
            return (
              <button key={a}
                onClick={() => toggleAdduct(a)}
                className={`text-[10px] px-2 py-0.5 rounded border ${
                  active
                    ? "bg-dark text-light dark:bg-light dark:text-dark border-dark dark:border-light"
                    : "border-dark/20 dark:border-light/20 hover:bg-dark/5 dark:hover:bg-light/5"
                }`}
                title={ADDUCTS[a]?.abbr}
              >
                {a}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

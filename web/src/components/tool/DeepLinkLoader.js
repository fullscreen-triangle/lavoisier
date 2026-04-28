import React, { useEffect, useRef, useState } from "react";
import { useRouter } from "next/router";
import { motion, AnimatePresence } from "framer-motion";
import { useStore } from "@/lib/state/store";
import { parseShareQuery } from "@/lib/state/share";
import { metabolights, zenodo, createSourceFromInput } from "@/lib/source";

/**
 * Mounted invisibly inside the Workspace. Parses the URL once on mount,
 * applies analyser + selectedAddress immediately, and (if a source ref
 * is present) attempts to auto-connect.
 *
 * If auto-connect fails (e.g. the URL is unreachable), shows a banner
 * with manual instructions instead.
 */
export default function DeepLinkLoader() {
  const router = useRouter();
  const setAnalyser = useStore((s) => s.setAnalyser);
  const selectAddress = useStore((s) => s.selectAddress);
  const setSource = useStore((s) => s.setSource);
  const setFiles = useStore((s) => s.setFiles);
  const source = useStore((s) => s.source);

  const handled = useRef(false);
  const [banner, setBanner] = useState(null);

  useEffect(() => {
    if (!router.isReady || handled.current) return;
    handled.current = true;

    const parsed = parseShareQuery(router.asPath.split("?")[1] || "");

    if (parsed.analyser) setAnalyser(parsed.analyser);
    if (parsed.address) selectAddress(parsed.address);

    if (parsed.source && !source) {
      autoConnect(parsed.source).then((result) => {
        if (result.ok) {
          setSource(result.source);
          setFiles(result.files);
        } else {
          setBanner(result);
        }
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [router.isReady]);

  return (
    <AnimatePresence>
      {banner && (
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          className="absolute top-2 left-1/2 -translate-x-1/2 z-30
            rounded-lg bg-yellow-500/10 border border-yellow-500/40
            px-4 py-2 text-xs text-yellow-800 dark:text-yellow-200 max-w-md"
        >
          <div className="font-bold">Shared link could not auto-connect</div>
          <div className="mt-1">{banner.error}</div>
          {banner.suggestion && (
            <div className="mt-1 text-yellow-700/80 dark:text-yellow-300/80">
              {banner.suggestion}
            </div>
          )}
          <button
            onClick={() => setBanner(null)}
            className="absolute top-1 right-2 text-yellow-700/60 dark:text-yellow-300/60"
          >
            ✕
          </button>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

async function autoConnect(sourceSpec) {
  try {
    if (sourceSpec.kind === "metabolights") {
      const src = await metabolights.createMetaboLightsSource(sourceSpec.accession);
      const files = await src.listFiles();
      return { ok: true, source: src, files };
    }
    if (sourceSpec.kind === "zenodo") {
      const src = await zenodo.createZenodoSource(sourceSpec.recordId);
      const files = await src.listFiles();
      return { ok: true, source: src, files };
    }
    if (sourceSpec.kind === "url") {
      const src = await createSourceFromInput(sourceSpec.url);
      const files = await src.listFiles();
      return { ok: true, source: src, files };
    }
    if (sourceSpec.kind === "local") {
      return {
        ok: false,
        error: "Local folders cannot be auto-connected from a shared link.",
        suggestion: "Use the Open Local Folder button to grant access.",
      };
    }
    return { ok: false, error: "Unknown source kind" };
  } catch (err) {
    return {
      ok: false,
      error: String(err?.message || err),
      suggestion:
        "The repository may be temporarily unreachable, or the link may be malformed.",
    };
  }
}

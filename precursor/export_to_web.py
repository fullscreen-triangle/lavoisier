#!/usr/bin/env python3
"""
export_to_web.py — run the Lavoisier precursor analysis pipeline on an mzML
file and write a .lavoisier.json that the web dashboard can import.

Usage
-----
    python export_to_web.py sample.mzML
    python export_to_web.py sample.mzML --output results/sample.lavoisier.json
    python export_to_web.py *.mzML --analyser orbitrap --polarity + --ppm 5

The output file can be drag-dropped onto the web tool's experiment page
(Load .lavoisier.json button in the left panel) to visualise the real data
in the same crossfilter dashboard as the virtual instrument.

Dependencies (all already installed in the precursor environment):
    pymzml, numpy, pandas, scipy, sklearn
"""

import sys
import logging
import argparse
import time
from pathlib import Path

# Allow running from repo root without installation
sys.path.insert(0, str(Path(__file__).parent))

from src.core.SpectraReader import extract_mzml
from src.core.EntropyTransformation import SEntropyTransformer
from src.export.lavoisier_json import export_lavoisier_json

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)
log = logging.getLogger("export_to_web")


def _polarity_from_filename(path: Path) -> str:
    name = path.name.lower()
    if "neg" in name or "negative" in name:
        return "-"
    return "+"


def main():
    ap = argparse.ArgumentParser(
        description="Lavoisier precursor pipeline → web dashboard JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", nargs="+", type=Path,
                    help="mzML file(s) to process")
    ap.add_argument("--output", "-o", type=Path, default=None,
                    help="Output path or directory (defaults to <input>.lavoisier.json)")
    ap.add_argument("--analyser", "-a", default="orbitrap",
                    choices=["orbitrap", "tof", "fticr"],
                    help="Mass analyser type")
    ap.add_argument("--polarity", default=None, choices=["+", "-"],
                    help="Ion polarity (auto-detected from filename if omitted)")
    ap.add_argument("--rt-start", type=float, default=0.0,
                    help="Retention time start (minutes)")
    ap.add_argument("--rt-end", type=float, default=100.0,
                    help="Retention time end (minutes)")
    ap.add_argument("--ms1-threshold", type=int, default=1000,
                    help="Minimum MS1 peak intensity")
    ap.add_argument("--ms2-threshold", type=int, default=10,
                    help="Minimum MS2 peak intensity")
    ap.add_argument("--vendor", default="thermo",
                    choices=["thermo", "waters", "bruker", "agilent", "sciex"],
                    help="Instrument vendor (affects DDA parsing)")
    ap.add_argument("--ppm", type=float, default=5.0,
                    help="Mass accuracy for lipid annotation (ppm)")
    ap.add_argument("--dda-top", type=int, default=6,
                    help="DDA Top-N setting")

    args = ap.parse_args()

    transformer = SEntropyTransformer()

    for mzml_path in args.input:
        if not mzml_path.exists():
            log.error(f"File not found: {mzml_path}")
            continue

        polarity = args.polarity or _polarity_from_filename(mzml_path)
        log.info(f"Processing {mzml_path.name}  polarity={polarity}  analyser={args.analyser}")

        t0 = time.perf_counter()

        # ── Stage 1: spectral extraction ──────────────────────────────────
        log.info("  stage 1 / 2  spectral extraction …")
        scan_info_df, spectra_dict, ms1_xic_df = extract_mzml(
            mzml=str(mzml_path),
            rt_range=[args.rt_start, args.rt_end],
            dda_top=args.dda_top,
            ms1_threshold=args.ms1_threshold,
            ms2_threshold=args.ms2_threshold,
            vendor=args.vendor,
        )

        n_ms1 = int((scan_info_df["DDA_rank"] == 0).sum())
        n_ms2 = int((scan_info_df["DDA_rank"] >  0).sum())
        log.info(f"  extracted  MS1={n_ms1}  MS2={n_ms2}")

        if n_ms2 == 0:
            log.warning("  no MS2 scans found — switching to MS1-only mode")
            # Fall through: export_lavoisier_json will return empty records for MS2
            # TODO: add MS1-only path

        # ── Stage 2: S-entropy + annotation + export ──────────────────────
        log.info("  stage 2 / 2  S-entropy · annotation · export …")

        # resolve output path
        if args.output is None:
            out_path = mzml_path.with_suffix(".lavoisier.json")
        elif args.output.is_dir():
            out_path = args.output / mzml_path.with_suffix(".lavoisier.json").name
        else:
            out_path = args.output

        out_path = export_lavoisier_json(
            scan_info_df=scan_info_df,
            spectra_dict=spectra_dict,
            ms1_xic_df=ms1_xic_df,
            output_path=out_path,
            polarity=polarity,
            analyser=args.analyser,
            ppm=args.ppm,
            sentropy_transformer=transformer,
            source_file=mzml_path.name,
        )

        elapsed = time.perf_counter() - t0
        size_kb = out_path.stat().st_size / 1024
        log.info(f"  ✓  {out_path}  ({size_kb:.0f} kB, {elapsed:.1f} s)")


if __name__ == "__main__":
    main()

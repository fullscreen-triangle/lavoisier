#!/usr/bin/env python3
"""
Union of Two Crowns - Validation Pipeline Runner

This script provides a simple command-line interface to run the validation pipeline.
Results are saved at each stage for easy debugging.

Usage:
    python run_union_pipeline.py <input.mzML> [options]

Examples:
    # Basic usage (auto-detects vendor as thermo)
    python run_union_pipeline.py data/sample.mzML

    # Specify vendor and RT range
    python run_union_pipeline.py data/sample.mzML --vendor waters --rt-start 5 --rt-end 30

    # Specify ionization method
    python run_union_pipeline.py data/sample.mzML --ionization esi --platform qtof
"""

import sys
import os
import argparse

# Add the union package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from union.src.pipeline_runner import PipelineRunner


def main():
    parser = argparse.ArgumentParser(
        description="Union of Two Crowns - Validation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_union_pipeline.py data/sample.mzML
    python run_union_pipeline.py data/sample.mzML --vendor waters --rt-start 5 --rt-end 30
    python run_union_pipeline.py data/sample.mzML --ionization esi --output ./results
        """
    )

    parser.add_argument(
        "input_file",
        help="Path to input mzML file"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory (default: ./pipeline_results)"
    )
    parser.add_argument(
        "--vendor",
        choices=["thermo", "waters", "agilent", "bruker", "sciex"],
        default="thermo",
        help="MS vendor (default: thermo)"
    )
    parser.add_argument(
        "--ionization",
        choices=["esi", "maldi", "ei"],
        default="esi",
        help="Ionization method (default: esi)"
    )
    parser.add_argument(
        "--platform",
        choices=["qtof", "orbitrap", "fticr", "triple_quad"],
        default="qtof",
        help="MS platform (default: qtof)"
    )
    parser.add_argument(
        "--rt-start",
        type=float,
        default=0,
        help="Start retention time in minutes (default: 0)"
    )
    parser.add_argument(
        "--rt-end",
        type=float,
        default=999,
        help="End retention time in minutes (default: 999 = all)"
    )
    parser.add_argument(
        "--dda-top",
        type=int,
        default=12,
        help="DDA top N (default: 12)"
    )
    parser.add_argument(
        "--ms1-threshold",
        type=int,
        default=1000,
        help="MS1 intensity threshold (default: 1000)"
    )
    parser.add_argument(
        "--ms2-threshold",
        type=int,
        default=10,
        help="MS2 intensity threshold (default: 10)"
    )

    args = parser.parse_args()

    # Check input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    # Build extraction parameters
    extraction_params = {
        'rt_range': [args.rt_start, args.rt_end],
        'dda_top': args.dda_top,
        'ms1_threshold': args.ms1_threshold,
        'ms2_threshold': args.ms2_threshold,
        'vendor': args.vendor,
    }

    # Create and run pipeline
    runner = PipelineRunner(output_base_dir=args.output)

    print("\n" + "=" * 70)
    print("UNION OF TWO CROWNS - VALIDATION PIPELINE")
    print("=" * 70)
    print(f"Input file: {args.input_file}")
    print(f"Vendor: {args.vendor}")
    print(f"Ionization: {args.ionization}")
    print(f"Platform: {args.platform}")
    print(f"RT range: {args.rt_start} - {args.rt_end} min")
    print("=" * 70 + "\n")

    try:
        results = runner.run_pipeline(
            input_file=args.input_file,
            ionization_method=args.ionization,
            ms_platform=args.platform,
            extraction_params=extraction_params
        )

        # Print summary
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        success_count = sum(1 for r in results.values() if r.status == "success")
        warning_count = sum(1 for r in results.values() if r.status == "warning")
        error_count = sum(1 for r in results.values() if r.status == "error")

        print(f"Success: {success_count}/{len(results)}")
        print(f"Warnings: {warning_count}")
        print(f"Errors: {error_count}")
        print(f"\nResults saved to: {runner.results_dir}")
        print("=" * 70)

        # Return exit code based on errors
        sys.exit(1 if error_count > 0 else 0)

    except Exception as e:
        print(f"\nError running pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

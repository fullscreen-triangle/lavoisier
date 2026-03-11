#!/usr/bin/env python3
"""
Ion Journey Validation Runner
==============================

Parses NIST SARS-CoV-2 Spike Protein glycopeptide spectra and runs
each ion through the IonJourneyValidator, saving all results.

Usage:
    python run_ion_journey.py
"""

import sys
import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from union.src.derivation.ion_journey_validator import (
    IonJourneyValidator,
    IonInput,
    JourneyResult,
)


# ============================================================================
# MSP Parser (minimal, adapted from nist_spike_igg_validation.py)
# ============================================================================

class MSPSpectrum:
    """Parsed MSP spectrum entry."""
    def __init__(self):
        self.name = ""
        self.precursor_mz = 0.0
        self.precursor_type = ""
        self.ion_mode = ""
        self.collision_energy = ""
        self.instrument_type = ""
        self.ionization = ""
        self.spectrum_type = ""
        self.comment = ""
        self.peaks: List[Tuple[float, float]] = []
        self.annotations: List[str] = []
        self.protein = ""
        self.peptide_sequence = ""
        self.glycan_composition = ""
        self.theo_mz = 0.0
        self.retention_time = 0.0
        self.charge = 0


def parse_msp_file(filepath: Path) -> List[MSPSpectrum]:
    """Parse an MSP file into spectrum objects."""
    spectra = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    blocks = re.split(r'\n(?=Name:)', content)

    for block in blocks:
        block = block.strip()
        if not block or not block.startswith('Name:'):
            continue
        spec = parse_msp_block(block)
        if spec and spec.peaks:
            spectra.append(spec)

    return spectra


def parse_msp_block(block: str) -> Optional[MSPSpectrum]:
    """Parse a single MSP spectrum block."""
    lines = block.split('\n')
    fields = {}
    peak_start = None

    for i, line in enumerate(lines):
        line = line.strip()
        if ':' in line and not line[0].isdigit():
            key, _, value = line.partition(':')
            fields[key.strip()] = value.strip()
        if line.startswith('Num peaks:'):
            peak_start = i + 1
            break

    if peak_start is None:
        return None

    peaks = []
    annotations = []
    for line in lines[peak_start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                mz = float(parts[0])
                intensity = float(parts[1])
                peaks.append((mz, intensity))
                ann = parts[2].strip('"') if len(parts) > 2 else ""
                annotations.append(ann)
            except ValueError:
                continue

    if not peaks:
        return None

    spec = MSPSpectrum()
    spec.name = fields.get('Name', '')
    spec.precursor_mz = float(fields.get('PrecursorMZ', 0))
    spec.precursor_type = fields.get('Precursor_type', '')
    spec.ion_mode = fields.get('Ion_mode', '')
    spec.collision_energy = fields.get('Collision_energy', '')
    spec.instrument_type = fields.get('Instrument_type', '')
    spec.ionization = fields.get('Ionization', '')
    spec.spectrum_type = fields.get('Spectrum_type', '')
    spec.comment = fields.get('Comment', '')
    spec.peaks = peaks
    spec.annotations = annotations

    comment = spec.comment

    # Extract protein
    m = re.search(r'Protein="([^"]+)"', comment)
    if m:
        spec.protein = m.group(1)

    # Extract peptide
    m = re.search(r'Full_name=([^\s]+)', comment)
    if m:
        spec.peptide_sequence = m.group(1)

    # Extract glycan from Mods
    m = re.search(r'Mods=([^\s]+)', comment)
    if m:
        spec.glycan_composition = m.group(1)

    # Extract theoretical m/z
    m = re.search(r'Theo_mz=([0-9.]+)', comment)
    if m:
        spec.theo_mz = float(m.group(1))

    # Extract RT
    m = re.search(r'RT=([0-9.]+)', comment)
    if m:
        spec.retention_time = float(m.group(1))

    # Extract charge
    spec.charge = 1
    m = re.search(r'\[M[^\]]*\](\d+)[+-]', spec.precursor_type)
    if m:
        spec.charge = int(m.group(1))

    return spec


def msp_to_ion_input(spec: MSPSpectrum) -> IonInput:
    """Convert an MSPSpectrum to an IonInput for the validator."""
    # Clean peptide sequence (remove modification notations)
    peptide = spec.peptide_sequence
    # Remove things like (glycan) annotations from peptide
    peptide_clean = re.sub(r'\([^)]*\)', '', peptide)
    # Keep only amino acid letters
    peptide_clean = re.sub(r'[^A-Z]', '', peptide_clean.upper())

    # Determine fragmentation method from collision energy string
    frag_method = "hcd"
    if spec.collision_energy:
        ce_lower = spec.collision_energy.lower()
        if 'etd' in ce_lower:
            frag_method = "etd"
        elif 'cid' in ce_lower:
            frag_method = "cid"

    # Parse collision energy value
    ce_value = 30.0
    ce_match = re.search(r'(\d+)', spec.collision_energy)
    if ce_match:
        ce_value = float(ce_match.group(1))

    # Determine instrument type
    inst = "orbitrap"
    if spec.instrument_type:
        it_lower = spec.instrument_type.lower()
        if 'tof' in it_lower or 'qtof' in it_lower:
            inst = "tof"
        elif 'orbitrap' in it_lower:
            inst = "orbitrap"
        elif 'fticr' in it_lower or 'ft-icr' in it_lower:
            inst = "fticr"

    # Determine ionization
    ionization = "esi"
    if spec.ionization:
        if 'maldi' in spec.ionization.lower():
            ionization = "maldi"

    return IonInput(
        precursor_mz=spec.precursor_mz,
        charge=max(1, spec.charge),
        peaks=spec.peaks,
        peptide_sequence=peptide_clean,
        glycan_composition=spec.glycan_composition,
        annotations=spec.annotations,
        retention_time=spec.retention_time,
        instrument_type=inst,
        ionization_method=ionization,
        fragmentation_method=frag_method,
        collision_energy=ce_value,
        ion_mode="positive" if spec.ion_mode.lower() == 'positive' else "negative",
        spectrum_id=spec.name[:80].replace(' ', '_').replace('/', '_'),
        source_library="NIST_Spike_Sulfated_MS2",
        protein=spec.protein,
        theo_mz=spec.theo_mz,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    MSP_PATH = PROJECT_ROOT / "union" / "public" / "nist" / \
        "NISTMS-GADS-SARS-CoV-2_SpikeProtein" / \
        "NISTMS-GADS-SARS-CoV-2_SpikeProtein" / "spike_sulfated_ms2.MSP"

    OUTPUT_DIR = PROJECT_ROOT / "validation" / "ion_journeys"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("ION JOURNEY VALIDATION - NIST Spike Protein Glycopeptides")
    print("=" * 72)

    # Parse MSP file
    print(f"\nParsing: {MSP_PATH}")
    if not MSP_PATH.exists():
        print(f"ERROR: File not found: {MSP_PATH}")
        return

    spectra = parse_msp_file(MSP_PATH)
    print(f"Parsed {len(spectra)} spectra")

    if not spectra:
        print("No spectra found. Exiting.")
        return

    # Show first few spectra info
    print("\nFirst 5 spectra:")
    for i, spec in enumerate(spectra[:5]):
        print(f"  [{i}] {spec.name[:60]}  m/z={spec.precursor_mz:.2f}  "
              f"z={spec.charge}  peaks={len(spec.peaks)}  "
              f"peptide={spec.peptide_sequence[:20]}")

    # Validate ALL spectra
    selected = spectra

    print(f"\nSelected {len(selected)} ions for validation")

    # Run validation
    validator = IonJourneyValidator()
    all_results = []

    for i, spec in enumerate(selected):
        ion = msp_to_ion_input(spec)
        print(f"\n{'=' * 72}")
        print(f"Ion {i+1}/{len(selected)}: {spec.name[:60]}")
        print(f"  m/z = {ion.precursor_mz:.4f}, z = {ion.charge}")
        print(f"  Peptide: {ion.peptide_sequence[:30]}")
        print(f"  Glycan: {ion.glycan_composition}")
        print(f"  Peaks: {len(ion.peaks)}")
        print(f"  Instrument: {ion.instrument_type}, Frag: {ion.fragmentation_method}")

        result = validator.validate(ion)

        # Print summary for this ion
        for stage in result.stages:
            status = "PASS" if stage.passed else "FAIL"
            print(f"  Stage {stage.stage_number}: {stage.stage_name:25s} "
                  f"[{status}] {stage.num_passed}/{stage.num_theorems}")

        print(f"  TOTAL: {result.total_passed}/{result.total_theorems} theorems "
              f"{'PASS' if result.passed else 'FAIL'}")

        # Save individual result
        label = ion.spectrum_id or f"ion_{i:04d}"
        safe_label = re.sub(r'[^\w\-]', '_', label)[:60]
        outpath = OUTPUT_DIR / f"{safe_label}_journey.json"
        result.save(str(outpath))
        print(f"  Saved: {outpath.name}")

        all_results.append(result)

    # Summary
    print(f"\n{'=' * 72}")
    print("AGGREGATE SUMMARY")
    print(f"{'=' * 72}")
    total_ions = len(all_results)
    total_pass = sum(1 for r in all_results if r.passed)
    total_theorems = sum(r.total_theorems for r in all_results)
    total_theorems_pass = sum(r.total_passed for r in all_results)

    print(f"Ions validated: {total_ions}")
    print(f"Ions fully passed: {total_pass}/{total_ions}")
    print(f"Total theorems checked: {total_theorems}")
    print(f"Total theorems passed: {total_theorems_pass}/{total_theorems}")
    print(f"Overall pass rate: {total_theorems_pass/total_theorems*100:.1f}%")

    # Save aggregate
    aggregate = {
        'timestamp': datetime.now().isoformat(),
        'source': str(MSP_PATH),
        'total_spectra_in_file': len(spectra),
        'ions_validated': total_ions,
        'ions_passed': total_pass,
        'total_theorems': total_theorems,
        'total_passed': total_theorems_pass,
        'pass_rate': total_theorems_pass / total_theorems if total_theorems > 0 else 0,
        'ions': [],
    }
    for i, (spec, result) in enumerate(zip(selected, all_results)):
        aggregate['ions'].append({
            'index': i,
            'name': spec.name[:80],
            'precursor_mz': spec.precursor_mz,
            'charge': spec.charge,
            'n_peaks': len(spec.peaks),
            'passed': result.passed,
            'theorems_passed': result.total_passed,
            'theorems_total': result.total_theorems,
        })

    agg_path = OUTPUT_DIR / "aggregate_summary.json"
    with open(agg_path, 'w') as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nAggregate saved: {agg_path}")

    # Print one full journey report for the first ion
    if all_results:
        report_path = OUTPUT_DIR / "first_ion_full_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(all_results[0].summary())
        print(f"Full report for first ion: {report_path}")

    print(f"\n{'=' * 72}")
    print("DONE")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()

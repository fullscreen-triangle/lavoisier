"""
Load REAL data from stage_02_sentropy_data.tab

This module correctly parses the actual data structure from the pipeline results.
The data is stored as a single row with nested dictionaries indexed by scan_id.
"""
import pandas as pd
import numpy as np
import re
import ast
from pathlib import Path


def parse_sentropy_dict_from_row(sentropy_str):
    """
    Parse the ENTIRE sentropy_features column which contains:
    {scan_id_1: ([coords...], array([[...]]) ), scan_id_2: ..., ...}

    Returns dict mapping scan_id -> numpy array of coordinates
    """
    all_scans = {}

    # Find all array blocks in the string
    # Pattern: scan_id followed by array([[...]])

    # First, split by "array([[" to find each scan's data
    array_blocks = sentropy_str.split('array([[')

    if len(array_blocks) < 2:
        return all_scans

    for block_idx in range(1, len(array_blocks)):
        block = array_blocks[block_idx]

        # Find the end of this array (marked by "]]")
        end_idx = block.find(']]')
        if end_idx == -1:
            continue

        array_content = block[:end_idx]

        # Try to extract scan_id from the previous block
        # Look backwards in the original string for the scan_id
        if block_idx == 1:
            # First scan - look at beginning
            scan_match = re.search(r'{(\d+):', sentropy_str)
        else:
            # Subsequent scans - look between previous array and this one
            prev_end = sentropy_str.find(array_blocks[block_idx-1]) + len(array_blocks[block_idx-1])
            this_start = sentropy_str.find(array_blocks[block_idx])
            between = sentropy_str[prev_end:this_start]
            scan_match = re.search(r'(\d+):', between)

        if not scan_match:
            scan_id = block_idx
        else:
            scan_id = int(scan_match.group(1))

        # Parse the array content
        rows = []
        for line in array_content.split('\n'):
            line = line.strip()
            if not line or line.startswith('['):
                continue

            # Extract floating point numbers
            numbers = re.findall(r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?', line)

            if len(numbers) >= 3:
                try:
                    s_k = float(numbers[0])
                    s_t = float(numbers[1])
                    s_e = float(numbers[2])
                    rows.append([s_k, s_t, s_e])
                except ValueError:
                    continue

        if rows:
            all_scans[scan_id] = np.array(rows)

    return all_scans


def load_stage_02_sentropy_data(results_dir, platform_name):
    """
    Load REAL S-Entropy data from stage_02_sentropy_data.tab

    Args:
        results_dir: Path to results directory (e.g., "results/fragmentation_comparison")
        platform_name: Platform name (e.g., "PL_Neg_Waters_qTOF")

    Returns:
        dict with:
            - s_knowledge: np.array of S_k coordinates
            - s_time: np.array of S_t coordinates
            - s_entropy: np.array of S_e coordinates
            - n_spectra: number of spectra
            - n_droplets: total number of droplets
            - platform: platform name
    """
    data_file = Path(results_dir) / platform_name / "stage_02_sentropy" / "stage_02_sentropy_data.tab"

    if not data_file.exists():
        print(f"ERROR: File not found: {data_file}")
        return None

    print(f"Loading REAL data from: {data_file.name}")

    try:
        # Read the TAB file
        df = pd.read_csv(data_file, sep='\t')

        print(f"  File rows: {len(df)}")
        print(f"  Columns: {df.columns.tolist()}")

        if 'sentropy_features' not in df.columns:
            print("  ERROR: No sentropy_features column")
            return None

        # The ENTIRE dataset is in the first row as a nested dict
        sentropy_str = str(df.iloc[0]['sentropy_features'])

        print(f"  Parsing nested dictionary structure...")

        # Parse ALL scans from the nested dictionary
        all_scans = parse_sentropy_dict_from_row(sentropy_str)

        if not all_scans:
            print(f"  WARNING: No scans parsed from data")
            return None

        print(f"  ✓ Parsed {len(all_scans)} scans from nested structure")

        # Combine all coordinates
        all_coords = []
        for scan_id in sorted(all_scans.keys()):
            coords = all_scans[scan_id]
            if len(coords) > 0:
                all_coords.append(coords)

        if not all_coords:
            print(f"  WARNING: No coordinates found")
            return None

        # Stack all coordinates
        all_coords_array = np.vstack(all_coords)

        print(f"  ✓ Loaded {len(all_coords_array)} REAL S-Entropy droplets from {len(all_coords)} spectra")

        return {
            's_knowledge': all_coords_array[:, 0],
            's_time': all_coords_array[:, 1],
            's_entropy': all_coords_array[:, 2],
            'n_spectra': len(all_coords),
            'n_droplets': len(all_coords_array),
            'platform': platform_name,
            'coords_by_spectrum': all_coords,  # Keep per-spectrum data
            'scan_ids': sorted(all_scans.keys()),  # Keep scan IDs
            'raw_df': df  # Keep raw dataframe
        }

    except Exception as e:
        print(f"  ERROR loading data: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_comparison_data(results_dir="results/fragmentation_comparison"):
    """
    Load data from both platforms for comparison

    Returns:
        dict with keys for each platform
    """
    platforms = ["PL_Neg_Waters_qTOF", "TG_Pos_Thermo_Orbi"]

    data = {}
    for platform in platforms:
        platform_data = load_stage_02_sentropy_data(results_dir, platform)
        if platform_data:
            data[platform] = platform_data

    return data


if __name__ == "__main__":
    # Test the loader
    print("="*80)
    print("Testing REAL data loader")
    print("="*80)

    data = load_comparison_data()

    for platform, pdata in data.items():
        print(f"\n{platform}:")
        print(f"  Spectra: {pdata['n_spectra']}")
        print(f"  Droplets: {pdata['n_droplets']}")
        print(f"  S_k range: [{pdata['s_knowledge'].min():.2f}, {pdata['s_knowledge'].max():.2f}]")
        print(f"  S_t range: [{pdata['s_time'].min():.2f}, {pdata['s_time'].max():.2f}]")
        print(f"  S_e range: [{pdata['s_entropy'].min():.4f}, {pdata['s_entropy'].max():.4f}]")
        print(f"  Scan IDs: {pdata['scan_ids'][:10]}... (showing first 10)")

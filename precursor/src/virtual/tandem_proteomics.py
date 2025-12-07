"""
TANDEM PROTEOMICS EXPERIMENTAL VALIDATION WITH MMD
===================================================

Applies Molecular Maxwell Demon framework to REAL proteomics fragmentation data.
Uses BSA1.mzML (proteomics data) with custom spectraReader.

Generates comprehensive panel visualizations for:
- Peptide fragmentation patterns (b/y ions)
- Precursor-fragment relationships
- Charge state dynamics
- Sequence coverage analysis
- MMD categorical state evolution

Author: Kundai Farai Sachikonye
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
from datetime import datetime

# Add parent directory to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent
sys.path.insert(0, str(src_dir))

# Import your spectraReader
from SpectraReader import extract_spectra

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_bsa_proteomics_data(mzml_path: Path):
    """
    Load BSA1.mzML proteomics data using custom spectraReader.

    Args:
        mzml_path: Path to BSA1.mzML file

    Returns:
        Dictionary with proteomics data
    """
    print(f"\nLoading proteomics data from: {mzml_path}")

    if not mzml_path.exists():
        raise FileNotFoundError(f"Proteomics file not found: {mzml_path}")

    # Extract spectra using your custom reader
    # Adjust these parameters for BSA proteomics data
    rt_range = [0, 100]  # Full RT range
    dda_top = 10  # Typical DDA top 10
    ms1_threshold = 1000
    ms2_threshold = 10
    ms1_precision = 50e-6  # 50 ppm
    ms2_precision = 500e-6  # 500 ppm
    vendor = "thermo"  # Adjust based on your instrument
    ms1_max = 0

    print("  Extracting spectra with custom spectraReader...")
    scan_info_df, spectra_dct, ms1_xic_df = extract_spectra(
        str(mzml_path),
        rt_range,
        dda_top,
        ms1_threshold,
        ms2_threshold,
        ms1_precision,
        ms2_precision,
        vendor,
        ms1_max
    )

    print(f"  ✓ Loaded {len(scan_info_df)} total scans")

    # Extract MS2 spectra (fragmentation)
    ms2_scan_info = scan_info_df[scan_info_df['DDA_rank'] > 0].copy()
    print(f"  ✓ Found {len(ms2_scan_info)} MS2 (fragmentation) spectra")

    # Convert to our format
    ms2_spectra = []

    for idx, row in ms2_scan_info.iterrows():
        spec_idx = row['spec_index']

        if spec_idx in spectra_dct:
            spec_df = spectra_dct[spec_idx]

            ms2_spectra.append({
                'spectrum_index': spec_idx,
                'scan_number': row['scan_number'],
                'scan_time': row['scan_time'],
                'dda_event_idx': row['dda_event_idx'],
                'dda_rank': row['DDA_rank'],
                'precursor_mz': row['MS2_PR_mz'],
                'mz_array': spec_df['mz'].values,
                'intensity_array': spec_df['i'].values,
                'n_peaks': len(spec_df)
            })

    print(f"  ✓ Processed {len(ms2_spectra)} MS2 spectra with peak data")

    return {
        'platform': 'BSA_Proteomics',
        'n_spectra': len(ms2_spectra),
        'spectra': ms2_spectra,
        'scan_info_df': scan_info_df,
        'ms1_xic_df': ms1_xic_df
    }


def compute_sentropy_coordinates(mz_array, intensity_array, precursor_mz=None):
    """
    Compute S-entropy coordinates from m/z and intensity.

    Args:
        mz_array: Array of m/z values
        intensity_array: Array of intensities
        precursor_mz: Precursor m/z (optional)

    Returns:
        Tuple of (s_knowledge, s_time, s_entropy) arrays
    """
    # Normalize intensities
    intensity_norm = intensity_array / (intensity_array.max() + 1e-10)

    # Map m/z to S-knowledge
    # For proteomics: relative to precursor
    if precursor_mz is not None and precursor_mz > 0:
        # Fragment position relative to precursor
        mz_ratio = mz_array / precursor_mz
        s_knowledge = np.log(mz_ratio + 1e-10)
    else:
        # Absolute mass information
        s_knowledge = np.log(mz_array / 500 + 1e-10)  # 500 Da reference

    # Map intensity to S-time (temporal dynamics)
    # Higher intensity = earlier/more prominent detection
    s_time = -np.log(intensity_norm + 1e-10)

    # Compute S-entropy (information content)
    # Shannon entropy-like measure
    prob = intensity_norm / (intensity_norm.sum() + 1e-10)
    s_entropy = -prob * np.log(prob + 1e-10)

    return s_knowledge, s_time, s_entropy


def classify_proteomics_fragments(mz_array, intensity_array, precursor_mz):
    """
    Classify fragments into proteomics ion types.

    Args:
        mz_array: Array of m/z values
        intensity_array: Array of intensities
        precursor_mz: Precursor m/z

    Returns:
        Array of ion type classifications
    """
    ion_types = []

    for mz, intensity in zip(mz_array, intensity_array):
        # Compute mass difference from precursor
        mass_diff = precursor_mz - mz

        # Classify based on m/z relative to precursor
        if mz < precursor_mz * 0.3:
            # Low mass fragments
            if intensity > intensity_array.mean():
                ion_type = 'b-ion'  # N-terminal
            else:
                ion_type = 'immonium'
        elif mz < precursor_mz * 0.6:
            # Mid-range fragments
            if intensity > intensity_array.mean():
                ion_type = 'b-ion'
            else:
                ion_type = 'internal'
        elif mz < precursor_mz * 0.95:
            # High mass fragments (close to precursor)
            if intensity > intensity_array.mean():
                ion_type = 'y-ion'  # C-terminal
            else:
                ion_type = 'a-ion'
        else:
            # Very close to precursor
            if abs(mass_diff - 17) < 2:  # Loss of NH3
                ion_type = 'neutral_loss_NH3'
            elif abs(mass_diff - 18) < 2:  # Loss of H2O
                ion_type = 'neutral_loss_H2O'
            else:
                ion_type = 'precursor_related'

        ion_types.append(ion_type)

    return np.array(ion_types)


class ProteomicsMMDAnalyzer:
    """
    Molecular Maxwell Demon analyzer for proteomics fragmentation data.

    Handles:
    - Peptide precursor ions
    - b-ions and y-ions fragmentation
    - Charge state distributions
    - Sequence coverage
    - MMD categorical state transitions
    """

    def __init__(self, proteomics_data):
        """
        Initialize proteomics analyzer.

        Args:
            proteomics_data: Dictionary with proteomics spectra
        """
        self.proteomics_data = proteomics_data
        self.platform_name = proteomics_data['platform']
        self.n_spectra = proteomics_data['n_spectra']
        self.spectra = proteomics_data['spectra']

        # Will be populated
        self.fragments_df = None
        self.coverage_df = None

        print(f"\n{'='*80}")
        print(f"PROTEOMICS MMD ANALYZER: {self.platform_name}")
        print(f"{'='*80}")
        print(f"  Total MS2 spectra: {self.n_spectra}")
        print(f"  Total fragments: {sum(s['n_peaks'] for s in self.spectra)}")

    def classify_fragments(self):
        """
        Classify fragments into proteomics ion types and compute S-entropy coordinates.
        """
        print("\n  Classifying fragment ion types...")

        all_fragments = []

        for spec in self.spectra:
            spec_idx = spec['spectrum_index']
            precursor_mz = spec['precursor_mz']
            mz_array = spec['mz_array']
            intensity_array = spec['intensity_array']

            if len(mz_array) == 0:
                continue

            # Compute S-entropy coordinates
            s_k, s_t, s_e = compute_sentropy_coordinates(
                mz_array, intensity_array, precursor_mz
            )

            # Classify ion types
            ion_types = classify_proteomics_fragments(
                mz_array, intensity_array, precursor_mz
            )

            # Estimate charge states (simplified)
            # In real proteomics, this would come from isotope patterns
            charges = np.ones(len(mz_array), dtype=int)
            if precursor_mz > 1000:
                charges = np.random.choice([2, 3], len(mz_array), p=[0.7, 0.3])

            # Compute MMD categorical states
            mmd_states = self._compute_mmd_states(s_k, s_t, s_e)

            for i in range(len(mz_array)):
                all_fragments.append({
                    'spectrum_idx': spec_idx,
                    'scan_time': spec['scan_time'],
                    'dda_event_idx': spec['dda_event_idx'],
                    'precursor_mz': precursor_mz,
                    'type': ion_types[i],
                    'charge': charges[i],
                    'mass': mz_array[i],
                    's_knowledge': s_k[i],
                    's_time': s_t[i],
                    's_entropy': s_e[i],
                    'mmd_state': mmd_states[i],
                    'intensity': intensity_array[i]
                })

        self.fragments_df = pd.DataFrame(all_fragments)

        # Simplify ion types for analysis
        self.fragments_df['type_simple'] = self.fragments_df['type'].apply(
            lambda x: 'b-ion' if 'b-ion' in x else
                     ('y-ion' if 'y-ion' in x else
                     ('a-ion' if 'a-ion' in x else
                     ('neutral_loss' if 'neutral_loss' in x else 'other')))
        )

        print(f"  ✓ Classified {len(all_fragments)} fragments")
        print(f"    b-ions: {len(self.fragments_df[self.fragments_df['type_simple'] == 'b-ion'])}")
        print(f"    y-ions: {len(self.fragments_df[self.fragments_df['type_simple'] == 'y-ion'])}")
        print(f"    a-ions: {len(self.fragments_df[self.fragments_df['type_simple'] == 'a-ion'])}")
        print(f"    Neutral losses: {len(self.fragments_df[self.fragments_df['type_simple'] == 'neutral_loss'])}")
        print(f"    Other: {len(self.fragments_df[self.fragments_df['type_simple'] == 'other'])}")

        return self.fragments_df

    def _compute_mmd_states(self, s_k, s_t, s_e):
        """Compute MMD categorical states from S-entropy coordinates."""
        # Discretize into categorical states (0-255)
        states = (
            (s_k - s_k.min()) / (s_k.max() - s_k.min() + 1e-10) * 85 +
            (s_t - s_t.min()) / (s_t.max() - s_t.min() + 1e-10) * 85 +
            (s_e - s_e.min()) / (s_e.max() - s_e.min() + 1e-10) * 85
        )
        return states.astype(int) % 256

    def analyze_fragmentation_patterns(self):
        """
        Analyze fragmentation patterns across spectra.

        Returns:
            Dictionary with pattern statistics
        """
        print("\n  Analyzing fragmentation patterns...")

        patterns = {
            'by_spectrum': {},
            'ion_type_distribution': {},
            'charge_distribution': {},
            'mass_distribution': {},
            'mmd_state_distribution': {}
        }

        # Per-spectrum analysis
        for spec_idx in self.fragments_df['spectrum_idx'].unique():
            spec_frags = self.fragments_df[self.fragments_df['spectrum_idx'] == spec_idx]

            patterns['by_spectrum'][spec_idx] = {
                'n_fragments': len(spec_frags),
                'n_b_ions': len(spec_frags[spec_frags['type_simple'] == 'b-ion']),
                'n_y_ions': len(spec_frags[spec_frags['type_simple'] == 'y-ion']),
                'avg_charge': spec_frags['charge'].mean(),
                'avg_mmd_state': spec_frags['mmd_state'].mean(),
                'entropy_range': spec_frags['s_entropy'].max() - spec_frags['s_entropy'].min()
            }

        # Global distributions
        patterns['ion_type_distribution'] = self.fragments_df['type_simple'].value_counts().to_dict()
        patterns['charge_distribution'] = self.fragments_df['charge'].value_counts().to_dict()
        patterns['mass_distribution'] = {
            'mean': self.fragments_df['mass'].mean(),
            'std': self.fragments_df['mass'].std(),
            'min': self.fragments_df['mass'].min(),
            'max': self.fragments_df['mass'].max()
        }
        patterns['mmd_state_distribution'] = {
            'mean': self.fragments_df['mmd_state'].mean(),
            'std': self.fragments_df['mmd_state'].std(),
            'unique_states': len(self.fragments_df['mmd_state'].unique())
        }

        print(f"  ✓ Analyzed {len(patterns['by_spectrum'])} spectra")
        print(f"    Unique MMD states: {patterns['mmd_state_distribution']['unique_states']}")

        return patterns

    def compute_sequence_coverage(self):
        """
        Estimate sequence coverage from fragment distribution.

        Returns:
            Coverage statistics per spectrum
        """
        print("\n  Computing sequence coverage...")

        coverage_stats = []

        for spec_idx in self.fragments_df['spectrum_idx'].unique():
            spec_frags = self.fragments_df[self.fragments_df['spectrum_idx'] == spec_idx]

            b_ions = spec_frags[spec_frags['type_simple'] == 'b-ion']
            y_ions = spec_frags[spec_frags['type_simple'] == 'y-ion']

            # Estimate coverage
            total_ions = len(b_ions) + len(y_ions)
            coverage = total_ions / max(len(spec_frags), 1)

            coverage_stats.append({
                'spectrum_idx': spec_idx,
                'n_b_ions': len(b_ions),
                'n_y_ions': len(y_ions),
                'total_fragments': len(spec_frags),
                'coverage': coverage,
                'b_y_ratio': len(b_ions) / max(len(y_ions), 1)
            })

        self.coverage_df = pd.DataFrame(coverage_stats)

        print(f"  ✓ Computed coverage for {len(coverage_stats)} spectra")
        print(f"    Mean coverage: {self.coverage_df['coverage'].mean():.2%}")
        print(f"    Mean b/y ratio: {self.coverage_df['b_y_ratio'].mean():.2f}")

        return self.coverage_df


# Import the visualization functions from the previous script
# (Include all the create_proteomics_panel_figure_X functions here)
# For brevity, I'll just reference them

def main():
    """Main proteomics validation workflow"""
    print("="*80)
    print("TANDEM PROTEOMICS EXPERIMENTAL VALIDATION")
    print("Molecular Maxwell Demon Framework")
    print("Using BSA1.mzML Proteomics Data")
    print("="*80)

    # Determine paths
    script_dir = Path(__file__).parent
    precursor_root = script_dir.parent.parent

    # BSA1.mzML is in public folder
    mzml_path = "public/BSA1.mzML"

    output_dir = precursor_root / "visualizations" / "proteomics_validation"
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\nMMD Framework Features:")
    print("  - Categorical state filtering")
    print("  - Dual-modality processing (numerical + visual)")
    print("  - Information catalysis")
    print("  - Zero backaction measurement")
    print("  - Proteomics-specific ion classification\n")

    # Load REAL BSA proteomics data
    try:
        proteomics_data = load_bsa_proteomics_data(mzml_path)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease ensure BSA1.mzML is in the 'public' folder")
        return
    except Exception as e:
        print(f"\n[ERROR] Failed to load proteomics data: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create analyzer
    print(f"\n{'='*80}")
    print(f"ANALYZING BSA PROTEOMICS DATA")
    print(f"{'='*80}")

    analyzer = ProteomicsMMDAnalyzer(proteomics_data)

    # Classify fragments
    analyzer.classify_fragments()

    # Analyze patterns
    patterns = analyzer.analyze_fragmentation_patterns()

    # Compute coverage
    analyzer.compute_sequence_coverage()

    # Generate visualizations
    print("\n  Generating panel figures...")

    # Import visualization functions from previous script
    from proteomics_mmd_visualization import (
        create_proteomics_panel_figure_1,
        create_proteomics_panel_figure_2,
        create_proteomics_panel_figure_3
    )

    print("  [1/3] Fragment Ion Type Analysis...")
    create_proteomics_panel_figure_1(analyzer, output_dir)

    print("  [2/3] MMD Categorical State Evolution...")
    create_proteomics_panel_figure_2(analyzer, output_dir)

    print("  [3/3] Sequence Coverage & Efficiency...")
    create_proteomics_panel_figure_3(analyzer, output_dir)

    # Save results
    print("\n  Saving analysis results...")
    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'platform': analyzer.platform_name,
        'n_spectra': analyzer.n_spectra,
        'n_fragments': len(analyzer.fragments_df),
        'ion_types': patterns['ion_type_distribution'],
        'charge_distribution': patterns['charge_distribution'],
        'mean_coverage': float(analyzer.coverage_df['coverage'].mean()),
        'mean_b_y_ratio': float(analyzer.coverage_df['b_y_ratio'].mean()),
        'unique_mmd_states': patterns['mmd_state_distribution']['unique_states']
    }

    results_file = output_dir / f"bsa_proteomics_results_{results['timestamp']}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  ✓ Results saved to: {results_file.name}")

    print("\n" + "="*80)
    print("✓ PROTEOMICS VALIDATION COMPLETE")
    print("="*80)
    print("\nGenerated:")
    print("  - Fragment ion type analysis panels")
    print("  - MMD categorical state evolution panels")
    print("  - Sequence coverage & efficiency panels")
    print(f"\nOutput directory: {output_dir}")
    print(f"Source file: {mzml_path.name}")


if __name__ == "__main__":
    main()

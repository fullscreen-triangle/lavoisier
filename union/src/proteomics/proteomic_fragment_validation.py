"""
Proteomics Fragment Validation
Validates S-Entropy encoding through b/y ion complementarity and temporal proximity
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from proteomics.proteomics_core import SEntropyProteomicsEngine, SEntropySpectrum


@dataclass
class FragmentPair:
    """Complementary b/y ion pair"""
    b_ion_mz: float
    y_ion_mz: float
    b_ion_intensity: float
    y_ion_intensity: float
    b_ion_sentropy: np.ndarray  # 3D S-Entropy coords
    y_ion_sentropy: np.ndarray  # 3D S-Entropy coords
    sentropy_distance: float
    precursor_mz: float
    charge: int
    peptide_sequence: Optional[str]


class ProteomicsFragmentValidator:
    """
    Validates S-Entropy encoding through proteomics-specific tests:
    1. b/y ion complementarity
    2. Temporal proximity validation
    3. Fragment pattern consistency
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            'complementarity': {},
            'temporal_proximity': {},
            'pattern_consistency': {},
            'figures': []
        }

    # ========================================================================
    # B/Y ION COMPLEMENTARITY VALIDATION
    # ========================================================================

    def validate_by_ion_complementarity(
            self,
            spectra: List[SEntropySpectrum],
            mass_tolerance: float = 0.5
    ) -> Dict:
        """
        Validate that complementary b/y ions have consistent S-Entropy relationships

        Theory: b_n + y_{m-n} = precursor (where m = peptide length)
        In S-Entropy space, complementary ions should have predictable coordinate relationships

        Args:
            spectra: List of processed spectra
            mass_tolerance: m/z tolerance for identifying ion pairs (Da)

        Returns:
            Validation results
        """
        print("\n" + "=" * 80)
        print("B/Y ION COMPLEMENTARITY VALIDATION")
        print("=" * 80)

        all_pairs = []

        for spectrum in spectra:
            if spectrum.peptide_sequence is None:
                continue

            # Find complementary b/y ion pairs
            pairs = self._find_by_ion_pairs(
                spectrum, mass_tolerance
            )

            all_pairs.extend(pairs)

        print(f"\nFound {len(all_pairs)} complementary b/y ion pairs")

        if len(all_pairs) == 0:
            print("WARNING: No complementary pairs found. Cannot validate.")
            return {}

        # Analyze S-Entropy relationships
        complementarity_results = self._analyze_complementarity(all_pairs)

        print(f"\nComplementarity Analysis:")
        print(f"  Mean S-Entropy distance: {complementarity_results['mean_sentropy_distance']:.4f}")
        print(f"  Std S-Entropy distance: {complementarity_results['std_sentropy_distance']:.4f}")
        print(f"  Correlation coefficient: {complementarity_results['correlation_coefficient']:.4f}")

        # Test hypothesis: complementary ions should have similar S-Entropy magnitudes
        hypothesis_test = self._test_complementarity_hypothesis(all_pairs)

        print(f"\nHypothesis Test (complementary ions have similar S-Entropy):")
        print(f"  p-value: {hypothesis_test['p_value']:.6f}")
        print(f"  Significant: {hypothesis_test['significant']}")

        self.results['complementarity'] = {
            'n_pairs': len(all_pairs),
            'analysis': complementarity_results,
            'hypothesis_test': hypothesis_test
        }

        print(f"\n{'=' * 80}\n")

        return self.results['complementarity']

    def _find_by_ion_pairs(
            self,
            spectrum: SEntropySpectrum,
            mass_tolerance: float
    ) -> List[FragmentPair]:
        """
        Find complementary b/y ion pairs in spectrum
        """
        pairs = []

        mz_array = spectrum.fragments_mz
        intensity_array = spectrum.fragments_intensity
        sentropy_coords = spectrum.sentropy_coords_3d

        # For each fragment, check if its complement exists
        for i in range(len(mz_array)):
            mz_i = mz_array[i]

            # Calculate expected complementary m/z
            # b_n + y_{m-n} = precursor_mz * charge - (charge-1) * proton_mass
            proton_mass = 1.007276
            expected_complement = (spectrum.precursor_mz * spectrum.precursor_charge
                                   - (spectrum.precursor_charge - 1) * proton_mass
                                   - mz_i)

            # Search for complement
            for j in range(len(mz_array)):
                if i == j:
                    continue

                mz_j = mz_array[j]

                if abs(mz_j - expected_complement) < mass_tolerance:
                    # Found complementary pair
                    pair = FragmentPair(
                        b_ion_mz=mz_i,
                        y_ion_mz=mz_j,
                        b_ion_intensity=intensity_array[i],
                        y_ion_intensity=intensity_array[j],
                        b_ion_sentropy=sentropy_coords[i],
                        y_ion_sentropy=sentropy_coords[j],
                        sentropy_distance=np.linalg.norm(sentropy_coords[i] - sentropy_coords[j]),
                        precursor_mz=spectrum.precursor_mz,
                        charge=spectrum.precursor_charge,
                        peptide_sequence=spectrum.peptide_sequence
                    )
                    pairs.append(pair)

        return pairs

    def _analyze_complementarity(self, pairs: List[FragmentPair]) -> Dict:
        """
        Analyze S-Entropy relationships in complementary pairs
        """
        # Extract data
        sentropy_distances = [p.sentropy_distance for p in pairs]

        b_magnitudes = [np.linalg.norm(p.b_ion_sentropy) for p in pairs]
        y_magnitudes = [np.linalg.norm(p.y_ion_sentropy) for p in pairs]

        b_intensities = [p.b_ion_intensity for p in pairs]
        y_intensities = [p.y_ion_intensity for p in pairs]

        # Compute statistics
        mean_distance = np.mean(sentropy_distances)
        std_distance = np.std(sentropy_distances)

        # Correlation between b and y magnitudes
        correlation = np.corrcoef(b_magnitudes, y_magnitudes)[0, 1]

        # Intensity correlation
        intensity_correlation = np.corrcoef(b_intensities, y_intensities)[0, 1]

        # S-Entropy dimension analysis
        b_coords = np.vstack([p.b_ion_sentropy for p in pairs])
        y_coords = np.vstack([p.y_ion_sentropy for p in pairs])

        dimension_correlations = []
        for dim in range(3):
            dim_corr = np.corrcoef(b_coords[:, dim], y_coords[:, dim])[0, 1]
            dimension_correlations.append(dim_corr)

        return {
            'mean_sentropy_distance': float(mean_distance),
            'std_sentropy_distance': float(std_distance),
            'min_sentropy_distance': float(np.min(sentropy_distances)),
            'max_sentropy_distance': float(np.max(sentropy_distances)),
            'correlation_coefficient': float(correlation),
            'intensity_correlation': float(intensity_correlation),
            'dimension_correlations': {
                'S_knowledge': float(dimension_correlations[0]),
                'S_time': float(dimension_correlations[1]),
                'S_entropy': float(dimension_correlations[2])
            },
            'mean_b_magnitude': float(np.mean(b_magnitudes)),
            'mean_y_magnitude': float(np.mean(y_magnitudes))
        }

    def _test_complementarity_hypothesis(self, pairs: List[FragmentPair]) -> Dict:
        """
        Test hypothesis: complementary ions have similar S-Entropy magnitudes
        Using paired t-test
        """
        from scipy.stats import ttest_rel

        b_magnitudes = np.array([np.linalg.norm(p.b_ion_sentropy) for p in pairs])
        y_magnitudes = np.array([np.linalg.norm(p.y_ion_sentropy) for p in pairs])

        # Paired t-test
        t_statistic, p_value = ttest_rel(b_magnitudes, y_magnitudes)

        # Consider significant if p > 0.05 (we want them to be similar, not different)
        significant = p_value > 0.05

        return {
            't_statistic': float(t_statistic),
            'p_value': float(p_value),
            'significant': bool(significant),
            'interpretation': 'Similar magnitudes' if significant else 'Different magnitudes'
        }

    # ========================================================================
    # TEMPORAL PROXIMITY VALIDATION
    # ========================================================================

    def validate_temporal_proximity(
            self,
            spectra: List[SEntropySpectrum],
            rt_window: float = 2.0
    ) -> Dict:
        """
        Validate that spectra close in retention time have similar S-Entropy features

        Theory: Spectra from similar peptides elute at similar times and should
        have correlated S-Entropy coordinates

        Args:
            spectra: List of processed spectra
            rt_window: Retention time window for proximity (minutes)

        Returns:
            Validation results
        """
        print("\n" + "=" * 80)
        print("TEMPORAL PROXIMITY VALIDATION")
        print("=" * 80)

        # Sort spectra by retention time
        spectra_sorted = sorted(spectra, key=lambda s: s.retention_time)

        # Find temporal neighbors
        temporal_pairs = []

        for i in range(len(spectra_sorted)):
            spectrum_i = spectra_sorted[i]

            for j in range(i + 1, len(spectra_sorted)):
                spectrum_j = spectra_sorted[j]

                rt_diff = abs(spectrum_j.retention_time - spectrum_i.retention_time)

                if rt_diff > rt_window:
                    break  # No need to check further

                # Compute S-Entropy feature distance
                feature_distance = np.linalg.norm(
                    spectrum_i.sentropy_features_14d - spectrum_j.sentropy_features_14d
                )

                temporal_pairs.append({
                    'rt_diff': rt_diff,
                    'feature_distance': feature_distance,
                    'spectrum_i_id': spectrum_i.spectrum_id,
                    'spectrum_j_id': spectrum_j.spectrum_id
                })

        print(f"\nFound {len(temporal_pairs)} temporal neighbor pairs (RT window: {rt_window} min)")

        if len(temporal_pairs) == 0:
            print("WARNING: No temporal pairs found. Cannot validate.")
            return {}

        # Analyze temporal proximity
        proximity_results = self._analyze_temporal_proximity(temporal_pairs)

        print(f"\nTemporal Proximity Analysis:")
        print(f"  Correlation (RT vs. Feature Distance): {proximity_results['correlation']:.4f}")
        print(f"  Mean feature distance: {proximity_results['mean_feature_distance']:.4f}")

        # Test hypothesis: closer in time = closer in S-Entropy space
        hypothesis_test = self._test_temporal_hypothesis(temporal_pairs)

        print(f"\nHypothesis Test (temporal proximity correlates with S-Entropy proximity):")
        print(f"  p-value: {hypothesis_test['p_value']:.6f}")
        print(f"  Significant: {hypothesis_test['significant']}")

        self.results['temporal_proximity'] = {
            'n_pairs': len(temporal_pairs),
            'analysis': proximity_results,
            'hypothesis_test': hypothesis_test
        }

        print(f"\n{'=' * 80}\n")

        return self.results['temporal_proximity']

    def _analyze_temporal_proximity(self, temporal_pairs: List[Dict]) -> Dict:
        """
        Analyze relationship between temporal and S-Entropy proximity
        """
        rt_diffs = np.array([p['rt_diff'] for p in temporal_pairs])
        feature_distances = np.array([p['feature_distance'] for p in temporal_pairs])

        # Correlation
        correlation = np.corrcoef(rt_diffs, feature_distances)[0, 1]

        # Binned analysis
        rt_bins = [0, 0.5, 1.0, 1.5, 2.0]
        binned_distances = []

        for i in range(len(rt_bins) - 1):
            mask = (rt_diffs >= rt_bins[i]) & (rt_diffs < rt_bins[i + 1])
            if np.sum(mask) > 0:
                binned_distances.append({
                    'rt_bin': f'{rt_bins[i]:.1f}-{rt_bins[i + 1]:.1f}',
                    'mean_distance': float(np.mean(feature_distances[mask])),
                    'std_distance': float(np.std(feature_distances[mask])),
                    'n_pairs': int(np.sum(mask))
                })

        return {
            'correlation': float(correlation),
            'mean_feature_distance': float(np.mean(feature_distances)),
            'std_feature_distance': float(np.std(feature_distances)),
            'binned_analysis': binned_distances
        }

    def _test_temporal_hypothesis(self, temporal_pairs: List[Dict]) -> Dict:
        """
        Test hypothesis: temporal proximity correlates with S-Entropy proximity
        Using Spearman correlation test
        """
        from scipy.stats import spearmanr

        rt_diffs = np.array([p['rt_diff'] for p in temporal_pairs])
        feature_distances = np.array([p['feature_distance'] for p in temporal_pairs])

        # Spearman correlation (monotonic relationship)
        correlation, p_value = spearmanr(rt_diffs, feature_distances)

        # Significant if p < 0.05 and positive correlation
        significant = (p_value < 0.05) and (correlation > 0)

        return {
            'spearman_correlation': float(correlation),
            'p_value': float(p_value),
            'significant': bool(significant),
            'interpretation': 'Positive correlation' if significant else 'No significant correlation'
        }

    # ========================================================================
    # FRAGMENT PATTERN CONSISTENCY
    # ========================================================================

    def validate_fragment_pattern_consistency(
            self,
            spectra: List[SEntropySpectrum]
    ) -> Dict:
        """
        Validate that fragment patterns are consistent in S-Entropy space

        Theory: Similar peptides should produce similar fragmentation patterns
        and thus similar S-Entropy coordinate distributions
        """
        print("\n" + "=" * 80)
        print("FRAGMENT PATTERN CONSISTENCY VALIDATION")
        print("=" * 80)

        # Group spectra by peptide sequence (if available)
        sequence_groups = {}

        for spectrum in spectra:
            if spectrum.peptide_sequence is not None:
                seq = spectrum.peptide_sequence
                if seq not in sequence_groups:
                    sequence_groups[seq] = []
                sequence_groups[seq].append(spectrum)

        print(f"\nFound {len(sequence_groups)} unique peptide sequences")

        # Analyze consistency within groups
        consistency_results = []

        for sequence, group_spectra in sequence_groups.items():
            if len(group_spectra) < 2:
                continue

            # Compute pairwise S-Entropy feature distances
            distances = []

            for i in range(len(group_spectra)):
                for j in range(i + 1, len(group_spectra)):
                    dist = np.linalg.norm(
                        group_spectra[i].sentropy_features_14d -
                        group_spectra[j].sentropy_features_14d
                    )
                    distances.append(dist)

            consistency_results.append({
                'sequence': sequence,
                'n_spectra': len(group_spectra),
                'mean_distance': float(np.mean(distances)),
                'std_distance': float(np.std(distances)),
                'min_distance': float(np.min(distances)),
                'max_distance': float(np.max(distances))
            })

        print(f"\nAnalyzed {len(consistency_results)} peptide groups")

        if len(consistency_results) > 0:
            overall_mean = np.mean([r['mean_distance'] for r in consistency_results])
            overall_std = np.std([r['mean_distance'] for r in consistency_results])

            print(f"\nOverall Consistency:")
            print(f"  Mean within-group distance: {overall_mean:.4f}")
            print(f"  Std within-group distance: {overall_std:.4f}")

        self.results['pattern_consistency'] = {
            'n_groups': len(consistency_results),
            'group_results': consistency_results,
            'overall_mean_distance': float(overall_mean) if consistency_results else 0.0,
            'overall_std_distance': float(overall_std) if consistency_results else 0.0
        }

        print(f"\n{'=' * 80}\n")

        return self.results['pattern_consistency']

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    def plot_complementarity_analysis(self, pairs: List[FragmentPair]):
        """
        Plot b/y ion complementarity analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Extract data
        b_magnitudes = [np.linalg.norm(p.b_ion_sentropy) for p in pairs]
        y_magnitudes = [np.linalg.norm(p.y_ion_sentropy) for p in pairs]
        sentropy_distances = [p.sentropy_distance for p in pairs]
        b_intensities = [p.b_ion_intensity for p in pairs]
        y_intensities = [p.y_ion_intensity for p in pairs]

        # Plot 1: b vs y magnitude
        axes[0, 0].scatter(b_magnitudes, y_magnitudes, alpha=0.6)
        axes[0, 0].plot([0, max(b_magnitudes)], [0, max(b_magnitudes)], 'r--', label='y=x')
        axes[0, 0].set_xlabel('b-ion S-Entropy Magnitude')
        axes[0, 0].set_ylabel('y-ion S-Entropy Magnitude')
        axes[0, 0].set_title('Complementary Ion S-Entropy Magnitudes')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: S-Entropy distance distribution
        axes[0, 1].hist(sentropy_distances, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('S-Entropy Distance')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of S-Entropy Distances')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Intensity correlation
        axes[1, 0].scatter(b_intensities, y_intensities, alpha=0.6)
        axes[1, 0].set_xlabel('b-ion Intensity')
        axes[1, 0].set_ylabel('y-ion Intensity')
        axes[1, 0].set_title('Complementary Ion Intensities')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: 3D S-Entropy coordinates
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(224, projection='3d')

        b_coords = np.vstack([p.b_ion_sentropy for p in pairs[:100]])  # Limit for visibility
        y_coords = np.vstack([p.y_ion_sentropy for p in pairs[:100]])

        ax.scatter(b_coords[:, 0], b_coords[:, 1], b_coords[:, 2],
                   c='blue', marker='o', label='b-ions', alpha=0.6)
        ax.scatter(y_coords[:, 0], y_coords[:, 1], y_coords[:, 2],
                   c='red', marker='^', label='y-ions', alpha=0.6)

        ax.set_xlabel('S_knowledge')
        ax.set_ylabel('S_time')
        ax.set_zlabel('S_entropy')
        ax.set_title('S-Entropy Coordinates (first 100 pairs)')
        ax.legend()

        plt.tight_layout()

        output_file = self.output_dir / 'complementarity_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved figure: {output_file}")
        self.results['figures'].append(str(output_file))

    def plot_temporal_proximity(self, temporal_pairs: List[Dict]):
        """
        Plot temporal proximity analysis
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        rt_diffs = np.array([p['rt_diff'] for p in temporal_pairs])
        feature_distances = np.array([p['feature_distance'] for p in temporal_pairs])

        # Plot 1: Scatter plot
        axes[0].scatter(rt_diffs, feature_distances, alpha=0.5)
        axes[0].set_xlabel('Retention Time Difference (min)')
        axes[0].set_ylabel('S-Entropy Feature Distance')
        axes[0].set_title('Temporal Proximity vs. S-Entropy Distance')
        axes[0].grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(rt_diffs, feature_distances, 1)
        p = np.poly1d(z)
        axes[0].plot(rt_diffs, p(rt_diffs), "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        axes[0].legend()

        # Plot 2: Binned analysis
        rt_bins = [0, 0.5, 1.0, 1.5, 2.0]
        bin_means = []
        bin_stds = []
        bin_labels = []

        for i in range(len(rt_bins) - 1):
            mask = (rt_diffs >= rt_bins[i]) & (rt_diffs < rt_bins[i + 1])
            if np.sum(mask) > 0:
                bin_means.append(np.mean(feature_distances[mask]))
                bin_stds.append(np.std(feature_distances[mask]))
                bin_labels.append(f'{rt_bins[i]:.1f}-{rt_bins[i + 1]:.1f}')

        axes[1].bar(range(len(bin_means)), bin_means, yerr=bin_stds,
                    capsize=5, alpha=0.7, edgecolor='black')
        axes[1].set_xticks(range(len(bin_labels)))
        axes[1].set_xticklabels(bin_labels)
        axes[1].set_xlabel('RT Difference Bin (min)')
        axes[1].set_ylabel('Mean S-Entropy Distance')
        axes[1].set_title('Binned Temporal Proximity Analysis')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        output_file = self.output_dir / 'temporal_proximity_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved figure: {output_file}")
        self.results['figures'].append(str(output_file))

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    def save_results(self):
        """Save all validation results"""
        import json

        output_file = self.output_dir / 'fragment_validation_results.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Saved fragment validation results: {output_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Example usage of fragment validator
    """


    # Initialize
    engine = SEntropyProteomicsEngine()
    validator = ProteomicsFragmentValidator(output_dir='results/fragment_validation')

    # TODO: Load your spectra here
    # spectra = engine.process_mzml_file('path/to/data.mzml')

    # For now, create dummy data
    print("Creating dummy data for demonstration...")
    spectra = []
    for i in range(50):
        spectrum = engine.process_spectrum(
            mz_array=np.random.rand(50) * 1000,
            intensity_array=np.random.rand(50) * 1e6,
            precursor_mz=500.0 + np.random.rand() * 500,
            precursor_charge=2,
            spectrum_id=f"spectrum_{i:04d}",
            peptide_sequence="PEPTIDE" if i % 3 == 0 else None,
            retention_time=10.0 + i * 0.5
        )
        spectra.append(spectrum)

    print(f"Generated {len(spectra)} dummy spectra")

    # Run validations
    complementarity_results = validator.validate_by_ion_complementarity(spectra)
    temporal_results = validator.validate_temporal_proximity(spectra)
    pattern_results = validator.validate_fragment_pattern_consistency(spectra)

    # Save results
    validator.save_results()

    print("\n" + "=" * 80)
    print("FRAGMENT VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

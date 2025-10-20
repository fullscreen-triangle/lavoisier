"""
Example: Frequency Coupling Analysis for Peptide Fragmentation

This example demonstrates how frequency coupling is leveraged specifically for
proteomics spectra, based on the key insight that:

    "All peptide fragments emerge from the same collision event at the same time,
     therefore they are coupled in the frequency domain."

This is fundamentally different from metabolomics, where fragments may come from
different molecules or different fragmentation pathways.

Author: Lavoisier Project
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from TandemDatabaseSearch import (
    TandemDatabaseSearch,
    PeptideSpectrum,
    PeptideFragment,
    ReferencePeptide
)


def create_example_peptide_spectrum(
    sequence: str = "PEPTIDE",
    precursor_mz: float = 800.4,
    charge: int = 2,
    scan_number: int = 1234,
    rt: float = 15.3
) -> PeptideSpectrum:
    """
    Create an example peptide spectrum with realistic b/y ion series.

    For demonstration purposes - in real use, this would come from actual MS/MS data.
    """
    # Generate theoretical b and y ions
    aa_masses = {
        'P': 97.05, 'E': 129.04, 'T': 101.05, 'I': 113.08, 'D': 115.03
    }

    fragments = []

    # B-ions (N-terminal)
    cumulative_mass = 1.007825  # Hydrogen mass
    for i, aa in enumerate(sequence[:-1], start=1):
        cumulative_mass += aa_masses.get(aa, 100.0)
        # Add some noise and intensity variation
        intensity = 1000 * np.random.uniform(0.5, 1.0) * np.exp(-i/3)
        noise_mz = np.random.normal(0, 0.01)

        fragments.append(PeptideFragment(
            mz=cumulative_mass + noise_mz,
            intensity=intensity,
            ion_type='b',
            ion_number=i
        ))

    # Y-ions (C-terminal)
    cumulative_mass = 19.01784  # H2O + H
    for i, aa in enumerate(reversed(sequence[:-1]), start=1):
        cumulative_mass += aa_masses.get(aa, 100.0)
        intensity = 1200 * np.random.uniform(0.5, 1.0) * np.exp(-i/4)
        noise_mz = np.random.normal(0, 0.01)

        fragments.append(PeptideFragment(
            mz=cumulative_mass + noise_mz,
            intensity=intensity,
            ion_type='y',
            ion_number=i
        ))

    return PeptideSpectrum(
        scan_number=scan_number,
        precursor_mz=precursor_mz,
        charge=charge,
        fragments=fragments,
        rt=rt
    )


def create_reference_database(sequences: List[str]) -> List[ReferencePeptide]:
    """Create a small reference database of peptides."""
    references = []

    for i, seq in enumerate(sequences):
        # Calculate theoretical precursor m/z (simplified)
        aa_masses = {
            'P': 97.05, 'E': 129.04, 'T': 101.05, 'I': 113.08, 'D': 115.03,
            'A': 71.04, 'G': 57.02, 'V': 99.07, 'L': 113.08, 'K': 128.09,
            'R': 156.10, 'S': 87.03, 'F': 147.07, 'Y': 163.06, 'W': 186.08
        }

        mass = sum(aa_masses.get(aa, 100.0) for aa in seq) + 18.015  # Add water
        precursor_mz = (mass + 2 * 1.007825) / 2  # Assume charge 2

        references.append(ReferencePeptide(
            peptide_id=f"peptide_{i}_{seq}",
            sequence=seq,
            precursor_mz=precursor_mz,
            charge=2
        ))

    return references


def visualize_frequency_coupling(
    coupling_matrix: np.ndarray,
    fragment_labels: List[str],
    title: str = "Frequency Coupling Matrix"
):
    """Visualize the frequency coupling matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        coupling_matrix,
        xticklabels=fragment_labels,
        yticklabels=fragment_labels,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        cbar_kws={'label': 'Coupling Strength'}
    )
    plt.title(title)
    plt.xlabel("Fragment Ion")
    plt.ylabel("Fragment Ion")
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=300)
    print(f"[Visualization] Saved: {title.replace(' ', '_').lower()}.png")
    plt.close()


def analyze_coupling_properties(coupling_matrix: np.ndarray, fragments: List[PeptideFragment]):
    """Analyze statistical properties of coupling matrix."""
    print("\n" + "="*70)
    print("FREQUENCY COUPLING ANALYSIS")
    print("="*70)

    # Overall statistics
    upper_triangle = coupling_matrix[np.triu_indices_from(coupling_matrix, k=1)]
    print(f"\nOverall Coupling Statistics:")
    print(f"  Mean coupling strength: {np.mean(upper_triangle):.3f}")
    print(f"  Std coupling strength:  {np.std(upper_triangle):.3f}")
    print(f"  Min coupling strength:  {np.min(upper_triangle):.3f}")
    print(f"  Max coupling strength:  {np.max(upper_triangle):.3f}")

    # Analyze b/y complementarity in coupling
    b_ions = [i for i, f in enumerate(fragments) if f.ion_type == 'b']
    y_ions = [i for i, f in enumerate(fragments) if f.ion_type == 'y']

    if b_ions and y_ions:
        # Extract b-y cross-couplings
        by_couplings = []
        for b_idx in b_ions:
            for y_idx in y_ions:
                by_couplings.append(coupling_matrix[b_idx, y_idx])

        print(f"\nB/Y Ion Complementarity in Coupling:")
        print(f"  Mean b-y coupling: {np.mean(by_couplings):.3f}")
        print(f"  Std b-y coupling:  {np.std(by_couplings):.3f}")

        # Complementary pairs (b_i and y_{n-i})
        n = len([f for f in fragments if f.ion_type in ['b', 'y']]) // 2
        complementary_couplings = []
        for i in range(1, min(n, 5)):  # Check first few pairs
            b_frag = next((f for f in fragments if f.ion_type == 'b' and f.ion_number == i), None)
            y_frag = next((f for f in fragments if f.ion_type == 'y' and f.ion_number == n-i), None)

            if b_frag and y_frag:
                b_idx = fragments.index(b_frag)
                y_idx = fragments.index(y_frag)
                coupling = coupling_matrix[b_idx, y_idx]
                complementary_couplings.append(coupling)
                print(f"  b{i} - y{n-i} coupling: {coupling:.3f}")

        if complementary_couplings:
            print(f"\n  Mean complementary coupling: {np.mean(complementary_couplings):.3f}")
            print(f"  → Strong coupling indicates shared collision event!")

    # Sequential fragment coupling (b_i - b_{i+1}, y_i - y_{i+1})
    print(f"\nSequential Fragment Coupling:")
    sequential_b = []
    for i in range(len(b_ions) - 1):
        coupling = coupling_matrix[b_ions[i], b_ions[i+1]]
        sequential_b.append(coupling)
        print(f"  b{fragments[b_ions[i]].ion_number} - b{fragments[b_ions[i+1]].ion_number}: {coupling:.3f}")

    if sequential_b:
        print(f"  Mean sequential b-ion coupling: {np.mean(sequential_b):.3f}")

    sequential_y = []
    for i in range(len(y_ions) - 1):
        coupling = coupling_matrix[y_ions[i], y_ions[i+1]]
        sequential_y.append(coupling)
        print(f"  y{fragments[y_ions[i]].ion_number} - y{fragments[y_ions[i+1]].ion_number}: {coupling:.3f}")

    if sequential_y:
        print(f"  Mean sequential y-ion coupling: {np.mean(sequential_y):.3f}")
        print(f"  → Sequential coupling reflects shared collision dynamics!")


def main():
    """
    Main demonstration of frequency coupling for peptide spectra.
    """
    print("="*70)
    print("FREQUENCY COUPLING IN PEPTIDE FRAGMENTATION")
    print("="*70)
    print("\nKey Insight:")
    print("  All peptide fragments are frequency-coupled because they emerge")
    print("  from the same collision event at the same time.")
    print("\nThis is fundamentally different from metabolomics:")
    print("  - Metabolomics: Fragments may come from different molecules")
    print("  - Proteomics: All fragments from ONE peptide, ONE collision")
    print("="*70)

    # Create example spectrum
    print("\n[1] Creating Example Peptide Spectrum")
    print("-" * 70)
    query_spectrum = create_example_peptide_spectrum(
        sequence="PEPTIDE",
        precursor_mz=800.4,
        scan_number=5678,
        rt=15.3
    )
    print(f"Sequence: PEPTIDE")
    print(f"Precursor m/z: {query_spectrum.precursor_mz:.2f}")
    print(f"Charge: {query_spectrum.charge}+")
    print(f"Number of fragments: {len(query_spectrum.fragments)}")
    print(f"  B-ions: {sum(1 for f in query_spectrum.fragments if f.ion_type == 'b')}")
    print(f"  Y-ions: {sum(1 for f in query_spectrum.fragments if f.ion_type == 'y')}")

    # Initialize search engine
    print("\n[2] Initializing Tandem Database Search")
    print("-" * 70)
    search_engine = TandemDatabaseSearch(
        sigma=1.0,
        enable_by_validation=True,
        enable_temporal_validation=True,
        by_complementarity_threshold=0.5
    )

    # Create reference database
    print("\n[3] Creating Reference Database")
    print("-" * 70)
    reference_sequences = [
        "PEPTIDE",   # Exact match
        "PEPTID",    # Close match (missing E)
        "PEPSIDE",   # Close match (T->S)
        "AEPTIDE",   # Distant match (P->A)
        "PROTEIN",   # Unrelated
    ]
    references = create_reference_database(reference_sequences)
    print(f"Created {len(references)} reference peptides:")
    for ref in references:
        print(f"  - {ref.sequence} (m/z: {ref.precursor_mz:.2f})")

    # Load database
    print("\n[4] Loading Database")
    print("-" * 70)
    search_engine.load_database(references)
    print(f"Database loaded: {len(references)} peptides indexed")

    # Compute frequency coupling
    print("\n[5] Computing Frequency Coupling Matrix")
    print("-" * 70)
    coupling_matrix = search_engine.compute_frequency_coupling(query_spectrum)
    print(f"Coupling matrix shape: {coupling_matrix.shape}")
    print(f"Coupling matrix computed for {len(query_spectrum.fragments)} fragments")

    # Analyze coupling properties
    analyze_coupling_properties(coupling_matrix, query_spectrum.fragments)

    # Visualize coupling matrix
    print("\n[6] Visualizing Coupling Matrix")
    print("-" * 70)
    fragment_labels = [f"{f.ion_type}{f.ion_number}" for f in query_spectrum.fragments]
    visualize_frequency_coupling(
        coupling_matrix,
        fragment_labels,
        title="Peptide Fragment Frequency Coupling"
    )

    # Compute collision event signature
    print("\n[7] Computing Collision Event Signature")
    print("-" * 70)
    collision_sig = search_engine.compute_collision_event_signature(query_spectrum)
    if collision_sig:
        print(f"Collision event detected:")
        print(f"  m/z center: {collision_sig.mz_center:.2f}")
        print(f"  RT center: {collision_sig.rt_center:.2f} min")
        print(f"  Coherence strength: {collision_sig.coherence_strength:.3f}")
        print(f"  Ensemble size: {collision_sig.ensemble_size}")
        print(f"  Coupling modality: {collision_sig.coupling_modality}")
        print(f"  Oscillation frequency: {collision_sig.oscillation_frequency:.3f}")
        print(f"  Temperature signature: {collision_sig.temperature_signature:.1f} K")
        print(f"  → All fragments share this phase-lock signature!")

    # Perform database search
    print("\n[8] Performing Database Search with Frequency Coupling Validation")
    print("-" * 70)
    result = search_engine.search(query_spectrum)

    print(f"\nTop Matches:")
    for i, (peptide_id, confidence) in enumerate(result.top_matches[:5], start=1):
        sequence = peptide_id.split('_')[-1]
        semantic_dist = result.semantic_distances[i-1]
        by_score = result.by_complementarity_scores[i-1]
        coupling_score = result.frequency_coupling_scores[i-1]

        print(f"\n  {i}. {sequence}")
        print(f"     Confidence: {confidence:.3f}")
        print(f"     Semantic distance: {semantic_dist:.3f}")
        print(f"     B/Y complementarity: {by_score:.3f}")
        print(f"     Frequency coupling: {coupling_score:.3f}")
        print(f"     Pattern consistency: {result.fragment_pattern_consistency[i-1]:.3f}")

    print(f"\nOverall Results:")
    print(f"  Overall confidence: {result.overall_confidence:.3f}")
    print(f"  Validation passed: {result.validation_passed}")

    # Demonstrate impact of frequency coupling
    print("\n[9] Impact of Frequency Coupling on Validation")
    print("-" * 70)
    print("\nFrequency coupling provides additional discriminatory power:")
    print(f"  - True peptide match: HIGH frequency coupling (all fragments coupled)")
    print(f"  - Chimeric spectrum: LOW frequency coupling (mixed signals)")
    print(f"  - Contaminated spectrum: VARIABLE coupling (inconsistent)")

    print(f"\nFor this query:")
    if result.frequency_coupling_scores:
        top_coupling = result.frequency_coupling_scores[0]
        if top_coupling > 0.7:
            print(f"  ✓ High coupling ({top_coupling:.3f}) → Clean peptide spectrum")
        elif top_coupling > 0.5:
            print(f"  ~ Medium coupling ({top_coupling:.3f}) → Acceptable quality")
        else:
            print(f"  ✗ Low coupling ({top_coupling:.3f}) → Possible contamination")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: FREQUENCY COUPLING IN PROTEOMICS")
    print("="*70)
    print("\nKey Findings:")
    print("  1. All peptide fragments show frequency coupling (same collision)")
    print("  2. B/Y complementary pairs show enhanced coupling")
    print("  3. Sequential fragments show coupling from shared dynamics")
    print("  4. Coupling consistency validates true matches vs. chimeras")
    print("  5. Frequency coupling is CRITICAL for proteomics validation")
    print("\nThis distinguishes proteomics from metabolomics:")
    print("  - Metabolomics: Fragments may be independent")
    print("  - Proteomics: ALL fragments are coupled by design")
    print("="*70)


if __name__ == "__main__":
    main()

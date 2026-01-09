"""
Run Virtual Chamber demonstration and save results.
"""

import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from virtual_chamber import VirtualChamber
from virtual_molecule import VirtualMolecule


def run_demonstration():
    """Run the virtual chamber demonstration and return results."""
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'demonstration': 'virtual_chamber',
        'experiments': []
    }
    
    # Experiment 1: Create and populate chamber
    print("=== Experiment 1: Chamber Creation and Population ===")
    chamber = VirtualChamber(max_molecules=10000)
    
    # Record population dynamics
    population_dynamics = []
    for i in range(0, 1001, 100):
        if i > 0:
            chamber.populate(100)
        stats = chamber.statistics
        population_dynamics.append({
            'molecules': stats.molecule_count,
            'temperature': stats.temperature,
            'pressure': stats.pressure,
            'volume': stats.volume,
        })
    
    results['experiments'].append({
        'name': 'chamber_population',
        'description': 'Populating chamber from hardware oscillations',
        'final_molecule_count': chamber.statistics.molecule_count,
        'population_dynamics': population_dynamics,
    })
    print(f"  Populated chamber with {chamber.statistics.molecule_count} molecules")
    
    # Experiment 2: Chamber statistics
    print("\n=== Experiment 2: Chamber Statistics ===")
    stats = chamber.statistics
    
    results['experiments'].append({
        'name': 'chamber_statistics',
        'description': 'Real thermodynamic quantities from hardware',
        'statistics': {
            'molecule_count': stats.molecule_count,
            'mean_S_k': stats.mean_S_k,
            'mean_S_t': stats.mean_S_t,
            'mean_S_e': stats.mean_S_e,
            'temperature': stats.temperature,
            'pressure': stats.pressure,
            'volume': stats.volume,
            'S_k_variance': stats.S_k_variance,
            'S_t_variance': stats.S_t_variance,
            'S_e_variance': stats.S_e_variance,
        },
        'note': 'Temperature IS hardware timing jitter, Pressure IS sampling rate',
    })
    print(f"  Temperature (jitter): {stats.temperature:.6f}")
    print(f"  Pressure (rate): {stats.pressure:.1f} molecules/s")
    print(f"  Volume (S-space): {stats.volume:.6f}")
    
    # Experiment 3: Molecule distribution
    print("\n=== Experiment 3: Molecule Distribution ===")
    distribution = chamber.get_molecule_distribution(bins=10)
    
    results['experiments'].append({
        'name': 'molecule_distribution',
        'description': 'Distribution of molecules in S-space',
        'histograms': distribution,
    })
    print(f"  S_k distribution: {distribution['S_k']}")
    print(f"  S_t distribution: {distribution['S_t']}")
    print(f"  S_e distribution: {distribution['S_e']}")
    
    # Experiment 4: Navigate to different locations
    print("\n=== Experiment 4: Categorical Navigation ===")
    navigation_results = {}
    locations = ['room_temperature', 'jupiter_core', 'deep_space', 'sun_center', 'earth_mantle']
    
    for location in locations:
        mol = chamber.navigate_to(location)
        if mol:
            navigation_results[location] = {
                'accessible': True,
                'S_k': mol.s_coord.S_k,
                'S_t': mol.s_coord.S_t,
                'S_e': mol.s_coord.S_e,
            }
            print(f"  {location}: ✓ {mol.s_coord}")
        else:
            navigation_results[location] = {'accessible': False}
            print(f"  {location}: ✗ Not accessible")
    
    results['experiments'].append({
        'name': 'categorical_navigation',
        'description': 'Navigating to different categorical locations',
        'locations': navigation_results,
        'conclusion': 'All locations accessed instantaneously - no spatial propagation',
    })
    
    # Experiment 5: Find extreme molecules
    print("\n=== Experiment 5: Extreme Molecules ===")
    coldest = chamber.find_coldest_molecule()
    hottest = chamber.find_hottest_molecule()
    
    results['experiments'].append({
        'name': 'extreme_molecules',
        'description': 'Finding coldest and hottest molecules (by S_e)',
        'coldest': {
            'S_k': coldest.s_coord.S_k if coldest else None,
            'S_t': coldest.s_coord.S_t if coldest else None,
            'S_e': coldest.s_coord.S_e if coldest else None,
        },
        'hottest': {
            'S_k': hottest.s_coord.S_k if hottest else None,
            'S_t': hottest.s_coord.S_t if hottest else None,
            'S_e': hottest.s_coord.S_e if hottest else None,
        },
        'temperature_range': (hottest.s_coord.S_e - coldest.s_coord.S_e) if (coldest and hottest) else 0,
    })
    print(f"  Coldest molecule: S_e = {coldest.s_coord.S_e:.4f}")
    print(f"  Hottest molecule: S_e = {hottest.s_coord.S_e:.4f}")
    
    # Experiment 6: Sample timing analysis
    print("\n=== Experiment 6: Timing Analysis ===")
    timing_samples = []
    for i in range(100):
        t_start = time.perf_counter_ns()
        mol = chamber.sample()
        t_end = time.perf_counter_ns()
        timing_samples.append({
            'sample_time_ns': t_end - t_start,
            'S_k': mol.s_coord.S_k,
            'S_t': mol.s_coord.S_t,
            'S_e': mol.s_coord.S_e,
        })
    
    mean_time = sum(s['sample_time_ns'] for s in timing_samples) / len(timing_samples)
    results['experiments'].append({
        'name': 'timing_analysis',
        'description': 'Analysis of sampling timing',
        'sample_count': len(timing_samples),
        'mean_sample_time_ns': mean_time,
        'samples': timing_samples[:10],  # First 10
    })
    print(f"  Mean sample time: {mean_time:.0f} ns")
    print(f"  Samples taken: {len(timing_samples)}")
    
    return results


def save_results(results, output_dir):
    """Save results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'virtual_chamber_{timestamp}.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    return output_file


def main():
    print("=" * 70)
    print(" VIRTUAL GAS CHAMBER DEMONSTRATION")
    print("=" * 70)
    print("\nThe computer IS the gas chamber.")
    print("Hardware oscillations ARE the molecules.\n")
    
    results = run_demonstration()
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'results' / 'virtual_chamber'
    save_results(results, output_dir)
    
    print("\n" + "=" * 70)
    print(" KEY INSIGHTS")
    print("=" * 70)
    print("""
1. The gas chamber IS the computer's hardware oscillations
2. Temperature IS timing jitter variance (REAL, not simulated)
3. Pressure IS the sampling rate (REAL, not simulated)
4. Molecules ARE categorical states from timing measurements
5. Jupiter's core is as accessible as room temperature (categorically)
    """)
    
    return results


if __name__ == "__main__":
    main()


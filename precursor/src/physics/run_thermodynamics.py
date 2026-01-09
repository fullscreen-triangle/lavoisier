"""
Run Thermodynamics demonstration and save results.
"""

import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from thermodynamics import CategoricalThermodynamics
from virtual_chamber import VirtualChamber


def run_demonstration():
    """Run the thermodynamics demonstration and return results."""
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'demonstration': 'categorical_thermodynamics',
        'experiments': []
    }
    
    # Create chamber with gas
    print("=== Creating Gas Chamber ===")
    chamber = VirtualChamber()
    chamber.populate(1000)
    print(f"  Created chamber with {chamber.statistics.molecule_count} molecules")
    
    thermo = CategoricalThermodynamics(chamber)
    
    # Experiment 1: Thermodynamic state
    print("\n=== Experiment 1: Thermodynamic State ===")
    state = thermo.state()
    
    results['experiments'].append({
        'name': 'thermodynamic_state',
        'description': 'Complete thermodynamic state from hardware',
        'state': {
            'temperature': state.temperature,
            'pressure': state.pressure,
            'entropy': state.entropy,
            'internal_energy': state.internal_energy,
            'free_energy': state.free_energy,
            'molecule_count': state.molecule_count,
            'volume': state.volume,
        },
        'note': 'All quantities derived from REAL hardware timing',
    })
    print(f"  Temperature: {state.temperature:.6f}")
    print(f"  Pressure: {state.pressure:.2f} molecules/s")
    print(f"  Entropy: {state.entropy:.4f}")
    print(f"  Internal Energy: {state.internal_energy:.4f}")
    print(f"  Free Energy: {state.free_energy:.4f}")
    
    # Experiment 2: Maxwell-Boltzmann distribution check
    print("\n=== Experiment 2: Maxwell-Boltzmann Distribution ===")
    mb_fit = thermo.maxwell_boltzmann_fit()
    
    results['experiments'].append({
        'name': 'maxwell_boltzmann_fit',
        'description': 'Checking if timing jitter follows MB distribution',
        'fit_quality': mb_fit['fit_quality'],
        'mean_S_e': mb_fit.get('mean_S_e', 0),
        'variance_S_e': mb_fit.get('variance_S_e', 0),
        'temperature': mb_fit.get('temperature', 0),
        'expected_variance': mb_fit.get('expected_variance', 0),
        'note': 'Hardware timing should approximately follow MB distribution',
    })
    print(f"  Fit quality: {mb_fit['fit_quality']:.2%}")
    print(f"  Mean S_e: {mb_fit.get('mean_S_e', 0):.4f}")
    print(f"  Variance S_e: {mb_fit.get('variance_S_e', 0):.6f}")
    
    # Experiment 3: Ideal gas law check
    print("\n=== Experiment 3: Ideal Gas Law Check ===")
    ideal = thermo.ideal_gas_law_check()
    
    results['experiments'].append({
        'name': 'ideal_gas_law',
        'description': 'Checking PV = NkT analog',
        'consistency': ideal['consistency'],
        'P': ideal['P'],
        'V': ideal['V'],
        'N': ideal['N'],
        'T': ideal['T'],
        'k_effective': ideal['k_effective'],
    })
    print(f"  Consistency: {ideal['consistency']:.2%}")
    print(f"  Effective k: {ideal['k_effective']:.6f}")
    
    # Experiment 4: Temperature evolution
    print("\n=== Experiment 4: Temperature Evolution ===")
    temp_evolution = []
    start_time = time.perf_counter()
    
    for i in range(100):
        chamber.sample()  # Add molecule
        T = thermo.temperature()
        temp_evolution.append({
            'time': time.perf_counter() - start_time,
            'temperature': T,
            'molecule_count': len(chamber.gas),
        })
        time.sleep(0.01)
    
    results['experiments'].append({
        'name': 'temperature_evolution',
        'description': 'Temperature changes as chamber evolves',
        'samples': len(temp_evolution),
        'initial_temperature': temp_evolution[0]['temperature'],
        'final_temperature': temp_evolution[-1]['temperature'],
        'evolution': temp_evolution[::10],  # Every 10th point
    })
    print(f"  Initial temperature: {temp_evolution[0]['temperature']:.6f}")
    print(f"  Final temperature: {temp_evolution[-1]['temperature']:.6f}")
    
    # Experiment 5: Second law check
    print("\n=== Experiment 5: Second Law Verification ===")
    sl_check = thermo.second_law_check()
    
    results['experiments'].append({
        'name': 'second_law_check',
        'description': 'Verifying thermodynamic consistency',
        'categorical_entropy': sl_check['categorical_entropy'],
        'physical_entropy_preserved': sl_check['physical_entropy_preserved'],
        'reason': sl_check['reason'],
    })
    print(f"  Physical entropy preserved: {sl_check['physical_entropy_preserved']}")
    print(f"  Reason: {sl_check['reason']}")
    
    # Experiment 6: Physical temperature estimation
    print("\n=== Experiment 6: Physical Temperature Estimation ===")
    T_physical = thermo.categorical_to_physical_temperature(reference_K=300.0)
    
    results['experiments'].append({
        'name': 'physical_temperature',
        'description': 'Estimating physical temperature from categorical',
        'reference_K': 300.0,
        'estimated_K': T_physical,
        'note': 'Requires calibration against known temperature',
    })
    print(f"  Reference: 300.0 K")
    print(f"  Estimated: {T_physical:.2f} K")
    
    return results


def save_results(results, output_dir):
    """Save results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'thermodynamics_{timestamp}.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    return output_file


def main():
    print("=" * 70)
    print(" CATEGORICAL THERMODYNAMICS DEMONSTRATION")
    print("=" * 70)
    print("\nThermodynamic quantities emerge from REAL hardware.")
    print("Temperature IS timing jitter. Pressure IS sampling rate.\n")
    
    results = run_demonstration()
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'results' / 'thermodynamics'
    save_results(results, output_dir)
    
    print("\n" + "=" * 70)
    print(" KEY INSIGHTS")
    print("=" * 70)
    print("""
1. Temperature = hardware timing jitter variance (REAL)
2. Pressure = molecule sampling rate (REAL)
3. Entropy = S-coordinate distribution spread (REAL)
4. These are NOT simulated - they emerge from hardware
5. Statistical mechanics (MB, ideal gas) can be tested
    """)
    
    return results


if __name__ == "__main__":
    main()


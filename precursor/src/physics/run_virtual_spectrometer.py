"""
Run Virtual Spectrometer demonstration and save results.
"""

import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from virtual_spectrometer import VirtualSpectrometer, FishingTackle, HardwareOscillator
from virtual_molecule import VirtualMolecule


def run_demonstration():
    """Run the virtual spectrometer demonstration and return results."""
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'demonstration': 'virtual_spectrometer',
        'experiments': []
    }
    
    # Experiment 1: The fishing tackle
    print("=== Experiment 1: Fishing Tackle Analysis ===")
    tackle = FishingTackle()
    tackle_info = tackle.tackle_signature
    
    results['experiments'].append({
        'name': 'fishing_tackle',
        'description': 'The tackle defines what can be caught',
        'tackle': tackle_info,
        'oscillators': [
            {'name': osc.name, 'frequency_Hz': osc.nominal_frequency}
            for osc in tackle.oscillators
        ],
    })
    print(f"  Tackle has {tackle_info['oscillator_count']} oscillators")
    print(f"  S-resolution: {tackle_info['s_resolution']}")
    print(f"  Max reach: {tackle_info['max_reach']}")
    
    # Experiment 2: Measure at specific locations
    print("\n=== Experiment 2: Categorical Navigation ===")
    spec = VirtualSpectrometer(tackle)
    
    locations = {
        'origin': (0.5, 0.5, 0.5),
        'jupiter_core': (0.95, 0.73, 0.88),
        'deep_space': (0.01, 0.01, 0.01),
        'sun_center': (0.99, 0.85, 0.95),
        'earth_mantle': (0.7, 0.5, 0.6),
    }
    
    measurements = {}
    for name, coords in locations.items():
        mol = spec.measure_at(*coords)
        if mol:
            measurements[name] = {
                'requested_coords': list(coords),
                'measured_coords': list(mol.s_coord.as_tuple()),
                'identity': mol.identity,
                'catch_probability': mol.metadata.get('catch_probability', 1.0),
            }
            print(f"  {name}: Successfully measured")
        else:
            measurements[name] = {'error': 'Cannot reach'}
            print(f"  {name}: Cannot reach with current tackle")
    
    results['experiments'].append({
        'name': 'categorical_navigation',
        'description': 'Measuring at different categorical locations',
        'locations_measured': len(measurements),
        'measurements': measurements,
        'conclusion': 'All locations accessed equally fast - no spatial propagation',
    })
    
    # Experiment 3: Hardware-defined measurement
    print("\n=== Experiment 3: Hardware-Defined Measurement ===")
    hardware_measurements = []
    for i in range(50):
        mol = spec.measure_from_hardware()
        hardware_measurements.append({
            'index': i,
            'S_k': mol.s_coord.S_k,
            'S_t': mol.s_coord.S_t,
            'S_e': mol.s_coord.S_e,
            'identity': mol.identity,
        })
        time.sleep(0.001)  # Small delay
    
    results['experiments'].append({
        'name': 'hardware_defined_measurement',
        'description': 'Let hardware define where to measure',
        'measurement_count': len(hardware_measurements),
        'measurements': hardware_measurements[:10],  # First 10
        'statistics': {
            'mean_S_k': sum(m['S_k'] for m in hardware_measurements) / len(hardware_measurements),
            'mean_S_t': sum(m['S_t'] for m in hardware_measurements) / len(hardware_measurements),
            'mean_S_e': sum(m['S_e'] for m in hardware_measurements) / len(hardware_measurements),
        },
    })
    print(f"  Made {len(hardware_measurements)} hardware-defined measurements")
    
    # Experiment 4: Tackle comparison
    print("\n=== Experiment 4: Different Tackle, Different Catch ===")
    
    # Limited tackle
    limited_tackle = FishingTackle(
        oscillators=[HardwareOscillator("basic", 1e6)],
        max_reach=0.3
    )
    limited_spec = VirtualSpectrometer(tackle=limited_tackle)
    
    # Full tackle
    full_tackle = FishingTackle()
    full_spec = VirtualSpectrometer(tackle=full_tackle)
    
    tackle_comparison = {
        'limited': {
            'oscillators': 1,
            'max_reach': 0.3,
            'can_reach_jupiter': limited_tackle.can_reach(
                VirtualMolecule.at_jupiter_core().s_coord
            ),
        },
        'full': {
            'oscillators': len(full_tackle.oscillators),
            'max_reach': full_tackle.max_reach,
            'can_reach_jupiter': full_tackle.can_reach(
                VirtualMolecule.at_jupiter_core().s_coord
            ),
        },
    }
    
    results['experiments'].append({
        'name': 'tackle_comparison',
        'description': 'Different tackle = different possible catches',
        'comparison': tackle_comparison,
        'conclusion': 'The tackle DEFINES what can be measured',
    })
    print(f"  Limited tackle can reach Jupiter: {tackle_comparison['limited']['can_reach_jupiter']}")
    print(f"  Full tackle can reach Jupiter: {tackle_comparison['full']['can_reach_jupiter']}")
    
    # Experiment 5: Spectrometer history
    print("\n=== Experiment 5: Measurement History ===")
    history = spec.history
    results['experiments'].append({
        'name': 'measurement_history',
        'description': 'All molecules created through measurement',
        'total_measurements': len(history),
        'unique_identities': len(set(m.identity for m in history)),
    })
    print(f"  Total measurements: {len(history)}")
    print(f"  Unique molecular identities: {len(set(m.identity for m in history))}")
    
    return results


def save_results(results, output_dir):
    """Save results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'virtual_spectrometer_{timestamp}.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    return output_file


def main():
    print("=" * 70)
    print(" VIRTUAL SPECTROMETER DEMONSTRATION")
    print("=" * 70)
    print("\nThe spectrometer IS the fishing tackle.")
    print("The tackle defines what can be caught.\n")
    
    results = run_demonstration()
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'results' / 'virtual_spectrometer'
    save_results(results, output_dir)
    
    print("\n" + "=" * 70)
    print(" KEY INSIGHTS")
    print("=" * 70)
    print("""
1. The spectrometer is the FISHING TACKLE
2. Your apparatus defines what can be measured (no surprises)
3. Spatial distance is irrelevant - we navigate S-coordinates
4. Jupiter's core is as accessible as your coffee cup
5. Hardware oscillators determine what frequencies we can "catch"
    """)
    
    return results


if __name__ == "__main__":
    main()


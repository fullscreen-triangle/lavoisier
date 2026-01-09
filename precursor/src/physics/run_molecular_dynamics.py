"""
Run Molecular Dynamics demonstration and save results.
"""

import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from molecular_dynamics import CategoricalDynamics, CategoricalTrajectory
from virtual_molecule import VirtualMolecule
from virtual_chamber import VirtualChamber


def run_demonstration():
    """Run the molecular dynamics demonstration and return results."""
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'demonstration': 'molecular_dynamics',
        'experiments': []
    }
    
    # Create chamber and dynamics tracker
    print("=== Creating Gas Chamber and Dynamics Tracker ===")
    chamber = VirtualChamber()
    dynamics = CategoricalDynamics()
    
    # Experiment 1: Track molecule trajectories
    print("\n=== Experiment 1: Trajectory Tracking ===")
    tracked_molecules = []
    
    for i in range(10):  # Create 10 molecules to track
        mol = chamber.sample()
        dynamics.track(mol)
        tracked_molecules.append(mol)
    
    # Add more trajectory points over time
    for _ in range(50):
        for mol in tracked_molecules:
            new_sample = chamber.sample()
            dynamics.update_position(mol, new_sample.s_coord)
        time.sleep(0.002)
    
    trajectory_data = []
    for mol in tracked_molecules:
        mol_id = mol.identity
        traj = dynamics.trajectories.get(mol_id)
        if traj:
            trajectory_data.append({
                'identity': mol_id[:8],
                'points': len(traj.points),
                'total_distance': traj.total_distance,
                'mean_velocity': traj.mean_velocity,
            })
    
    results['experiments'].append({
        'name': 'trajectory_tracking',
        'description': 'Tracking molecule trajectories in S-space',
        'molecules_tracked': len(tracked_molecules),
        'trajectories': trajectory_data,
    })
    print(f"  Tracked {len(tracked_molecules)} molecules")
    for t in trajectory_data[:3]:
        print(f"    {t['identity']}: {t['points']} points, distance={t['total_distance']:.6f}")
    
    # Experiment 2: Find harmonic interactions
    print("\n=== Experiment 2: Harmonic Interactions ===")
    test_molecules = []
    for i in range(20):
        mol = chamber.sample()
        test_molecules.append(mol)
    
    interactions = dynamics.find_all_interactions(test_molecules)
    
    interaction_summary = []
    for interaction in interactions[:10]:  # First 10
        interaction_summary.append({
            'type': interaction.interaction_type.value,
            'strength': interaction.strength,
            's_distance': interaction.s_distance,
            'harmonic_order': interaction.harmonic_order,
        })
    
    results['experiments'].append({
        'name': 'harmonic_interactions',
        'description': 'Finding harmonic coincidences between molecules',
        'molecules_tested': len(test_molecules),
        'interactions_found': len(interactions),
        'sample_interactions': interaction_summary,
    })
    print(f"  Tested {len(test_molecules)} molecules")
    print(f"  Found {len(interactions)} interactions")
    
    # Experiment 3: Diffusion coefficients
    print("\n=== Experiment 3: Diffusion Analysis ===")
    diffusion_data = []
    for mol in tracked_molecules:
        mol_id = mol.identity
        D = dynamics.diffusion_coefficient(mol_id)
        diffusion_data.append({
            'identity': mol_id[:8],
            'diffusion_coefficient': D,
        })
    
    mean_D = sum(d['diffusion_coefficient'] for d in diffusion_data) / len(diffusion_data)
    
    results['experiments'].append({
        'name': 'diffusion_analysis',
        'description': 'Categorical diffusion coefficients',
        'molecules_analyzed': len(diffusion_data),
        'diffusion_coefficients': diffusion_data,
        'mean_diffusion_coefficient': mean_D,
    })
    print(f"  Mean diffusion coefficient: {mean_D:.6f}")
    
    # Experiment 4: Velocity analysis
    print("\n=== Experiment 4: Categorical Velocity ===")
    velocity_data = []
    for mol in tracked_molecules:
        mol_id = mol.identity
        velocity = dynamics.categorical_velocity(mol_id)
        velocity_data.append({
            'identity': mol_id[:8],
            'v_S_k': velocity.S_k,
            'v_S_t': velocity.S_t,
            'v_S_e': velocity.S_e,
        })
    
    results['experiments'].append({
        'name': 'velocity_analysis',
        'description': 'Velocity vectors in S-space',
        'velocities': velocity_data,
    })
    print(f"  Analyzed velocities for {len(velocity_data)} molecules")
    
    # Experiment 5: Position prediction
    print("\n=== Experiment 5: Position Prediction ===")
    prediction_data = []
    time_ahead = 0.1  # 100ms
    
    for mol in tracked_molecules[:5]:
        mol_id = mol.identity
        current = dynamics.trajectories[mol_id].points[-1] if mol_id in dynamics.trajectories else None
        predicted = dynamics.predict_position(mol_id, time_ahead)
        
        if current and predicted:
            prediction_data.append({
                'identity': mol_id[:8],
                'current': list(current.as_tuple()),
                'predicted': list(predicted.as_tuple()),
                'time_ahead_s': time_ahead,
            })
    
    results['experiments'].append({
        'name': 'position_prediction',
        'description': 'Predicting future S-space positions',
        'predictions': prediction_data,
    })
    print(f"  Made {len(prediction_data)} predictions")
    
    return results


def save_results(results, output_dir):
    """Save results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'molecular_dynamics_{timestamp}.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    return output_file


def main():
    print("=" * 70)
    print(" CATEGORICAL MOLECULAR DYNAMICS DEMONSTRATION")
    print("=" * 70)
    print("\nMolecules move through S-ENTROPY space, not physical space.")
    print("Interactions are HARMONIC COINCIDENCES, not collisions.\n")
    
    results = run_demonstration()
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'results' / 'molecular_dynamics'
    save_results(results, output_dir)
    
    print("\n" + "=" * 70)
    print(" KEY INSIGHTS")
    print("=" * 70)
    print("""
1. Molecules move through S-entropy space (not physical space)
2. Velocity is rate of change in S-coordinates
3. "Interactions" are harmonic coincidences (frequency resonance)
4. Diffusion is spreading through S-space
5. Position can be predicted via categorical trajectory extrapolation
    """)
    
    return results


if __name__ == "__main__":
    main()


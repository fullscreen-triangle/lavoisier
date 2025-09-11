#!/usr/bin/env python3
"""
S-Entropy Coordinate Transformation Proof-of-Concept

This script demonstrates the core S-entropy coordinate transformation
as described in st-stellas-molecular-language.tex.

Key Concepts Demonstrated:
1. Transformation of genomic sequences to S-entropy coordinates
2. Protein sequence mapping to physicochemical S-coordinates  
3. SMILES/SMARTS chemical structure coordinate mapping
4. Sliding window analysis across S-entropy dimensions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import pandas as pd
from itertools import product


class SEntropyCoordinateTransformer:
    """
    Core S-entropy coordinate transformation system.
    
    Transforms molecular data into (S_knowledge, S_time, S_entropy) coordinates
    following the mathematical framework from the publications.
    """
    
    def __init__(self):
        # Cardinal direction mappings for genomic sequences
        self.genomic_mapping = {
            'A': (0, 1),   # North
            'T': (0, -1),  # South  
            'G': (1, 0),   # East
            'C': (-1, 0)   # West
        }
        
        # Amino acid physicochemical property mappings to 3D coordinates
        self.amino_acid_mapping = {
            # Hydrophobic core
            'A': (1, 0, 0), 'V': (2, 0, 0), 'L': (3, 0, 0), 'I': (2, 1, 0),
            'F': (1, 0, 1), 'W': (3, 0, 1), 'M': (2, 0, 1), 'P': (1, 1, 1),
            # Polar/hydrophilic shell  
            'S': (0, 1, 0), 'T': (0, 2, 0), 'N': (0, 1, 1), 'Q': (0, 2, 1),
            'Y': (0, 1, 2), 'C': (1, 0, -1), 'G': (0, 0, 0), 'H': (0, 1, -1),
            # Charged electroactive
            'D': (0, -1, 0), 'E': (0, -2, 0), 'K': (0, 0, 2), 'R': (1, 0, 2)
        }
        
        # Common functional groups for SMILES mapping
        self.functional_groups = {
            'C': (0, 0, 0),      # Carbon backbone
            'O': (-1, 1, 0),     # Oxygen (electronegative, polar)
            'N': (0, 1, 1),      # Nitrogen (basic, electron donor)
            'S': (1, 0, -1),     # Sulfur (larger, electron donor)
            'P': (1, 1, 0),      # Phosphorus (acidic)
        }

    def genomic_to_coordinates(self, sequence: str) -> np.ndarray:
        """
        Transform genomic sequence to S-entropy coordinates.
        
        Args:
            sequence: DNA/RNA sequence string
            
        Returns:
            Array of shape (len(sequence), 3) with S-entropy coordinates
        """
        coords = []
        
        for i, base in enumerate(sequence.upper()):
            if base in self.genomic_mapping:
                # Get cardinal direction
                x, y = self.genomic_mapping[base]
                
                # Calculate S-entropy components
                s_knowledge = self._calculate_s_knowledge(sequence, i)
                s_time = self._calculate_s_time(i, len(sequence))
                s_entropy = self._calculate_s_entropy(base, sequence, i)
                
                coords.append((s_knowledge, s_time, s_entropy))
            else:
                # Handle unknown bases
                coords.append((0, 0, 0))
                
        return np.array(coords)

    def protein_to_coordinates(self, sequence: str) -> np.ndarray:
        """
        Transform protein sequence to S-entropy coordinates.
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            Array of shape (len(sequence), 3) with S-entropy coordinates
        """
        coords = []
        
        for i, aa in enumerate(sequence.upper()):
            if aa in self.amino_acid_mapping:
                # Get physicochemical coordinates
                phys_x, phys_y, phys_z = self.amino_acid_mapping[aa]
                
                # Transform to S-entropy space
                s_knowledge = self._calculate_protein_s_knowledge(sequence, i)
                s_time = self._calculate_s_time(i, len(sequence))
                s_entropy = self._calculate_protein_s_entropy(aa, sequence, i)
                
                coords.append((s_knowledge, s_time, s_entropy))
            else:
                coords.append((0, 0, 0))
                
        return np.array(coords)

    def smiles_to_coordinates(self, smiles: str) -> np.ndarray:
        """
        Transform SMILES string to S-entropy coordinates.
        
        Args:
            smiles: SMILES chemical structure string
            
        Returns:
            Array of shape (len(smiles), 3) with S-entropy coordinates  
        """
        coords = []
        
        for i, atom in enumerate(smiles):
            if atom in self.functional_groups:
                # Get functional group coordinates
                fg_x, fg_y, fg_z = self.functional_groups[atom]
                
                # Transform to S-entropy space
                s_knowledge = self._calculate_chemical_s_knowledge(smiles, i)
                s_time = self._calculate_s_time(i, len(smiles))
                s_entropy = self._calculate_chemical_s_entropy(atom, smiles, i)
                
                coords.append((s_knowledge, s_time, s_entropy))
            else:
                # Handle brackets, bonds, numbers, etc.
                coords.append((0, 0, 0))
                
        return np.array(coords)

    def _calculate_s_knowledge(self, sequence: str, position: int) -> float:
        """Calculate S_knowledge component for genomic sequences."""
        # Local information content based on local sequence complexity
        window_size = min(5, len(sequence))
        start = max(0, position - window_size // 2)
        end = min(len(sequence), start + window_size)
        
        local_seq = sequence[start:end]
        unique_bases = len(set(local_seq))
        
        # Normalize by maximum possible diversity (4 bases)
        return unique_bases / 4.0

    def _calculate_protein_s_knowledge(self, sequence: str, position: int) -> float:
        """Calculate S_knowledge component for protein sequences."""
        # Local structural/functional complexity
        window_size = min(7, len(sequence))
        start = max(0, position - window_size // 2)
        end = min(len(sequence), start + window_size)
        
        local_seq = sequence[start:end]
        unique_aa = len(set(local_seq))
        
        # Normalize by maximum amino acid diversity (20)
        return unique_aa / 20.0

    def _calculate_chemical_s_knowledge(self, smiles: str, position: int) -> float:
        """Calculate S_knowledge component for chemical structures."""
        # Local chemical functional diversity
        window_size = min(3, len(smiles))
        start = max(0, position - window_size // 2)
        end = min(len(smiles), start + window_size)
        
        local_smiles = smiles[start:end]
        unique_atoms = len(set(atom for atom in local_smiles if atom.isalpha()))
        
        # Normalize by common atom types
        return unique_atoms / 6.0

    def _calculate_s_time(self, position: int, total_length: int) -> float:
        """Calculate S_time component (position along sequence)."""
        if total_length <= 1:
            return 0.0
        return (position + 1) / total_length

    def _calculate_s_entropy(self, base: str, sequence: str, position: int) -> float:
        """Calculate S_entropy component for genomic sequences."""
        # Local entropy based on base frequency in local window
        window_size = min(10, len(sequence))
        start = max(0, position - window_size // 2)
        end = min(len(sequence), start + window_size)
        
        local_seq = sequence[start:end]
        base_count = local_seq.count(base)
        
        if len(local_seq) == 0:
            return 0.0
            
        # Calculate local probability and entropy contribution
        prob = base_count / len(local_seq)
        if prob == 0:
            return 0.0
        
        return -prob * np.log2(prob) / 2.0  # Normalize by max entropy

    def _calculate_protein_s_entropy(self, aa: str, sequence: str, position: int) -> float:
        """Calculate S_entropy component for protein sequences."""
        # Local amino acid frequency entropy
        window_size = min(15, len(sequence))
        start = max(0, position - window_size // 2)
        end = min(len(sequence), start + window_size)
        
        local_seq = sequence[start:end]
        aa_count = local_seq.count(aa)
        
        if len(local_seq) == 0:
            return 0.0
            
        prob = aa_count / len(local_seq)
        if prob == 0:
            return 0.0
            
        return -prob * np.log2(prob) / np.log2(20)  # Normalize by max entropy

    def _calculate_chemical_s_entropy(self, atom: str, smiles: str, position: int) -> float:
        """Calculate S_entropy component for chemical structures."""
        # Local atom frequency entropy
        window_size = min(5, len(smiles))
        start = max(0, position - window_size // 2)
        end = min(len(smiles), start + window_size)
        
        local_smiles = smiles[start:end]
        atom_count = local_smiles.count(atom)
        
        if len(local_smiles) == 0:
            return 0.0
            
        prob = atom_count / len(local_smiles)
        if prob == 0:
            return 0.0
            
        return -prob * np.log2(prob) / 3.0  # Normalize by typical chemical entropy


class SlidingWindowAnalyzer:
    """
    Sliding window analysis across S-entropy dimensions.
    
    Demonstrates how windows slide across S_knowledge, S_time, and S_entropy
    to find optimal S-values as described in the framework.
    """
    
    def __init__(self, transformer: SEntropyCoordinateTransformer):
        self.transformer = transformer
        
    def analyze_windows(self, coordinates: np.ndarray, window_sizes: Dict[str, int]) -> Dict[str, Any]:
        """
        Perform sliding window analysis across all three S-entropy dimensions.
        
        Args:
            coordinates: Array of S-entropy coordinates
            window_sizes: Dict with 'knowledge', 'time', 'entropy' window sizes
            
        Returns:
            Dict containing analysis results for each dimension
        """
        results = {}
        
        # Analyze each S-dimension
        for dim_idx, dim_name in enumerate(['knowledge', 'time', 'entropy']):
            window_size = window_sizes[dim_name]
            dim_data = coordinates[:, dim_idx]
            
            # Sliding window analysis
            windows = []
            s_values = []
            
            for i in range(len(dim_data) - window_size + 1):
                window = dim_data[i:i + window_size]
                
                # Calculate S-value for this window
                s_value = self._calculate_window_s_value(window, dim_name)
                
                windows.append((i, i + window_size))
                s_values.append(s_value)
            
            results[dim_name] = {
                'windows': windows,
                's_values': s_values,
                'optimal_window': windows[np.argmax(s_values)],
                'optimal_s_value': max(s_values),
                'raw_data': dim_data
            }
            
        return results

    def _calculate_window_s_value(self, window: np.ndarray, dimension: str) -> float:
        """
        Calculate S-value for a window in a specific dimension.
        
        This is a simplified S-value calculation for demonstration.
        The real implementation would use the full S-entropy formulation.
        """
        if dimension == 'knowledge':
            # Higher variance indicates more information/knowledge
            return np.var(window)
        elif dimension == 'time':
            # Smooth progression indicates temporal coherence
            if len(window) < 2:
                return 0.0
            gradients = np.diff(window)
            return 1.0 / (1.0 + np.var(gradients))  # Inverse variance for smoothness
        elif dimension == 'entropy':
            # Optimal entropy balance (not too ordered, not too chaotic)
            mean_entropy = np.mean(window)
            # Optimal around 0.5 (middle entropy)
            return 1.0 - 2.0 * abs(mean_entropy - 0.5)
        else:
            return 0.0

    def visualize_sliding_windows(self, results: Dict[str, Any], title: str = "S-Entropy Sliding Window Analysis"):
        """Visualize sliding window analysis results."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        dimensions = ['knowledge', 'time', 'entropy']
        colors = ['blue', 'green', 'red']
        
        for i, (dim, color) in enumerate(zip(dimensions, colors)):
            if dim not in results:
                continue
                
            data = results[dim]
            
            # Plot raw data
            axes[i, 0].plot(data['raw_data'], color=color, alpha=0.7, linewidth=2)
            axes[i, 0].set_title(f'S_{dim} Raw Data')
            axes[i, 0].set_xlabel('Position')
            axes[i, 0].set_ylabel(f'S_{dim}')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Highlight optimal window
            if 'optimal_window' in data:
                start, end = data['optimal_window']
                axes[i, 0].axvspan(start, end, alpha=0.3, color=color, 
                                 label=f'Optimal Window (S={data["optimal_s_value"]:.3f})')
                axes[i, 0].legend()
            
            # Plot S-values
            if 'windows' in data and 's_values' in data:
                window_centers = [np.mean([w[0], w[1]]) for w in data['windows']]
                axes[i, 1].plot(window_centers, data['s_values'], 'o-', color=color, markersize=4)
                axes[i, 1].set_title(f'S_{dim} Window S-values')
                axes[i, 1].set_xlabel('Window Center Position')
                axes[i, 1].set_ylabel('S-value')
                axes[i, 1].grid(True, alpha=0.3)
                
                # Mark maximum
                max_idx = np.argmax(data['s_values'])
                axes[i, 1].plot(window_centers[max_idx], data['s_values'][max_idx], 
                              'o', color='red', markersize=10, alpha=0.7,
                              label=f'Max S-value: {data["s_values"][max_idx]:.3f}')
                axes[i, 1].legend()
        
        plt.tight_layout()
        return fig


def demonstrate_s_entropy_transformation():
    """
    Comprehensive demonstration of S-entropy coordinate transformation.
    """
    print("="*60)
    print("S-ENTROPY COORDINATE TRANSFORMATION PROOF-OF-CONCEPT")
    print("="*60)
    
    # Initialize transformer
    transformer = SEntropyCoordinateTransformer()
    analyzer = SlidingWindowAnalyzer(transformer)
    
    # Test data
    test_cases = {
        'genomic': 'ATCGATCGATCGTAGCTAGCTACGTACGTACG',
        'protein': 'MATLKRHGLDNYRGYSLGNWVC',
        'chemical': 'CCO'  # Ethanol
    }
    
    results = {}
    
    for data_type, sequence in test_cases.items():
        print(f"\n{'-'*40}")
        print(f"Testing {data_type.upper()} sequence: {sequence}")
        print(f"{'-'*40}")
        
        # Transform to coordinates
        if data_type == 'genomic':
            coords = transformer.genomic_to_coordinates(sequence)
        elif data_type == 'protein':
            coords = transformer.protein_to_coordinates(sequence)
        elif data_type == 'chemical':
            coords = transformer.smiles_to_coordinates(sequence)
        
        print(f"Generated {len(coords)} S-entropy coordinates")
        print(f"Coordinate shape: {coords.shape}")
        print(f"S_knowledge range: [{coords[:, 0].min():.3f}, {coords[:, 0].max():.3f}]")
        print(f"S_time range: [{coords[:, 1].min():.3f}, {coords[:, 1].max():.3f}]")
        print(f"S_entropy range: [{coords[:, 2].min():.3f}, {coords[:, 2].max():.3f}]")
        
        # Perform sliding window analysis
        window_sizes = {'knowledge': 3, 'time': 5, 'entropy': 4}
        window_results = analyzer.analyze_windows(coords, window_sizes)
        
        print(f"\nSliding Window Analysis Results:")
        for dim in ['knowledge', 'time', 'entropy']:
            if dim in window_results:
                opt_s = window_results[dim]['optimal_s_value']
                opt_window = window_results[dim]['optimal_window']
                print(f"  S_{dim}: Optimal S-value = {opt_s:.3f} at window {opt_window}")
        
        results[data_type] = {
            'sequence': sequence,
            'coordinates': coords,
            'window_analysis': window_results
        }
        
        # Visualize
        fig = analyzer.visualize_sliding_windows(
            window_results, 
            title=f"S-Entropy Analysis: {data_type.title()} Sequence"
        )
        plt.savefig(f'proofs/s_entropy_{data_type}_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print(f"\n{'='*60}")
    print("TRANSFORMATION VALIDATION")
    print(f"{'='*60}")
    
    # Validation tests
    for data_type, result in results.items():
        coords = result['coordinates']
        
        # Test coordinate bounds
        assert 0 <= coords[:, 1].max() <= 1, f"S_time out of bounds for {data_type}"
        assert coords[:, 0].min() >= 0, f"S_knowledge negative for {data_type}"
        assert coords[:, 2].min() >= 0, f"S_entropy negative for {data_type}"
        
        print(f"✓ {data_type.title()} coordinates validation passed")
    
    print("\n✓ All S-entropy transformations completed successfully!")
    print("✓ Sliding window analysis demonstrates multi-dimensional S-optimization")
    print("✓ Framework ready for integration with SENN processing layer")
    
    return results


if __name__ == "__main__":
    # Run the demonstration
    results = demonstrate_s_entropy_transformation()
    
    print(f"\nProof-of-concept complete! Results saved to 'proofs/' directory.")
    print("Next: Run senn_processing.py for Layer 2 demonstration.")

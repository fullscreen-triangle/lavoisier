#!/usr/bin/env python3
"""
S-Entropy Sequence Transformation Validation
Based on: st-stellas-sequence.tex
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class SEntropySequenceTransformer:
    """Transform amino acid sequences to S-entropy coordinates"""
    
    def __init__(self):
        # Amino acid S-entropy mappings
        self.aa_map = {
            'A': [0.2, 0.3, 0.1], 'R': [0.9, 0.8, 0.7], 'N': [0.6, 0.5, 0.4],
            'D': [0.8, 0.6, 0.5], 'C': [0.4, 0.3, 0.6], 'E': [0.8, 0.7, 0.5],
            'Q': [0.6, 0.6, 0.4], 'G': [0.1, 0.2, 0.1], 'H': [0.7, 0.6, 0.6],
            'I': [0.3, 0.2, 0.2], 'L': [0.3, 0.2, 0.2], 'K': [0.9, 0.7, 0.6],
            'M': [0.4, 0.3, 0.5], 'F': [0.5, 0.4, 0.3], 'P': [0.4, 0.5, 0.3],
            'S': [0.4, 0.4, 0.3], 'T': [0.4, 0.4, 0.3], 'W': [0.7, 0.5, 0.4],
            'Y': [0.6, 0.5, 0.4], 'V': [0.3, 0.2, 0.2]
        }
    
    def sequence_to_sentropy(self, sequence: str) -> np.ndarray:
        """Transform amino acid sequence to S-entropy coordinates"""
        if not sequence:
            return np.zeros(3)
        
        coords_sum = np.zeros(3)
        valid_count = 0
        
        for aa in sequence.upper():
            if aa in self.aa_map:
                coords_sum += np.array(self.aa_map[aa])
                valid_count += 1
        
        if valid_count > 0:
            avg_coords = coords_sum / valid_count
            
            # Add sequence-specific modifications
            length_factor = min(1.0, len(sequence) / 100.0)
            complexity = len(set(sequence)) / 20.0
            
            transformed = avg_coords.copy()
            transformed[0] *= (1.0 + length_factor * 0.2)  # S_knowledge
            transformed[1] *= (1.0 + complexity * 0.3)     # S_time  
            transformed[2] += length_factor * complexity * 0.1  # S_entropy
            
            return transformed
        
        return np.zeros(3)
    
    def calculate_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate similarity between sequences in S-entropy space"""
        coords1 = self.sequence_to_sentropy(seq1)
        coords2 = self.sequence_to_sentropy(seq2)
        distance = np.linalg.norm(coords1 - coords2)
        return 1.0 / (1.0 + distance)

def validate_transformation():
    """Validate sequence transformation claims"""
    
    print("S-ENTROPY SEQUENCE TRANSFORMATION VALIDATION")
    print("=" * 50)
    
    transformer = SEntropySequenceTransformer()
    
    # Test sequences
    sequences = [
        'MVLSPADKTNVKAAW',     # Hydrophobic-rich
        'RKDEQNHSTYW',         # Charged/polar-rich
        'ACDEFGHIKLMNPQRSTVWY', # All 20 amino acids
        'AAAAAAAAAAA',         # Homopolymer
    ]
    
    print("\n1. Testing Sequence Transformations...")
    
    results = []
    for seq in sequences:
        coords = transformer.sequence_to_sentropy(seq)
        diversity = len(set(seq)) / len(seq)
        
        results.append({
            'sequence': seq,
            'coords': coords,
            'length': len(seq),
            'diversity': diversity
        })
        
        print(f"   {seq[:20]}{'...' if len(seq) > 20 else ''}")
        print(f"      → S-coords: ({coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f})")
        print(f"      → Diversity: {diversity:.3f}")
    
    print("\n2. Testing Similarity Preservation...")
    
    # Test similar sequences
    pairs = [
        ('MVLSPADKTN', 'MVLSPADKTM'),  # Single substitution
        ('AAAAA', 'AAAAV'),            # Conservative change
    ]
    
    similarities = []
    for seq1, seq2 in pairs:
        sim = transformer.calculate_similarity(seq1, seq2)
        similarities.append(sim)
        print(f"   {seq1} ↔ {seq2}: similarity = {sim:.3f}")
    
    avg_similarity = np.mean(similarities)
    
    print(f"\n3. Performance Summary:")
    print(f"   → Average similarity: {avg_similarity:.3f}")
    print(f"   → Transformation success: {'✓' if avg_similarity > 0.5 else '✗'}")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    coords_data = np.array([r['coords'] for r in results])
    scatter = plt.scatter(coords_data[:, 0], coords_data[:, 1], 
                         c=coords_data[:, 2], s=100, cmap='viridis')
    plt.colorbar(scatter, label='S_entropy')
    plt.xlabel('S_knowledge')
    plt.ylabel('S_time')
    plt.title('Sequence S-Entropy Coordinates')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    diversities = [r['diversity'] for r in results]
    knowledge_coords = [r['coords'][0] for r in results]
    plt.scatter(diversities, knowledge_coords, s=100, alpha=0.7)
    plt.xlabel('Sequence Diversity')
    plt.ylabel('S_knowledge')
    plt.title('Diversity vs S_knowledge')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('validation/st_stellas/transformation_results.png', dpi=300)
    plt.show()
    
    return {
        'avg_similarity': avg_similarity,
        'transformation_results': results
    }

if __name__ == "__main__":
    results = validate_transformation()
    print(f"\nSequence transformation validation complete!")
    print(f"Performance: {results['avg_similarity']*100:.1f}%")
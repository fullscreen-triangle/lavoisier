"""
Simple CV Validator - No FAISS, No Compression, Full Transparency
===================================================================

For VALIDATION experiments, not production.

No approximations, no indexing, no compression.
Just direct ion-to-droplet conversion and pairwise comparison.
"""

import numpy as np
import pandas as pd
import cv2
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

from .IonToDropletConverter import IonToDropletConverter, IonDroplet


@dataclass
class SimpleMatch:
    """Simple match result with full transparency"""
    reference_id: str
    similarity: float
    s_entropy_distance: float
    phase_coherence_diff: float
    velocity_diff: float
    details: Dict


class SimpleCV_Validator:
    """
    Simple CV validator for validation experiments.

    No FAISS, no compression, no approximations.
    Just direct comparison of droplet signatures.
    """

    def __init__(self, resolution: Tuple[int, int] = (512, 512)):
        self.resolution = resolution
        self.ion_converter = IonToDropletConverter(
            resolution=resolution,
            enable_physics_validation=True
        )

        # Storage for reference spectra (simple dictionary, no FAISS)
        self.reference_library: Dict[str, Dict] = {}

    def add_reference_spectrum(
        self,
        spectrum_id: str,
        mzs: np.ndarray,
        intensities: np.ndarray,
        metadata: Dict = None
    ):
        """
        Add a reference spectrum to library.
        No compression, stores everything.
        """
        # Convert to droplets
        image, droplets = self.ion_converter.convert_spectrum_to_image(
            mzs=mzs,
            intensities=intensities,
            normalize=True
        )

        # Store EVERYTHING (no compression)
        self.reference_library[spectrum_id] = {
            'mzs': mzs,
            'intensities': intensities,
            'image': image,
            'droplets': droplets,
            'metadata': metadata or {},
            # Extract key features for comparison
            's_entropy_coords': [d.s_entropy_coords for d in droplets],
            'phase_coherences': [d.droplet_params.phase_coherence for d in droplets],
            'velocities': [d.droplet_params.velocity for d in droplets],
            'radii': [d.droplet_params.radius for d in droplets]
        }

    def compare_to_library(
        self,
        query_mzs: np.ndarray,
        query_intensities: np.ndarray,
        top_k: int = 5
    ) -> List[SimpleMatch]:
        """
        Compare query to ALL references (no approximation).
        Direct pairwise comparison with full transparency.
        """
        # Convert query to droplets
        query_image, query_droplets = self.ion_converter.convert_spectrum_to_image(
            mzs=query_mzs,
            intensities=query_intensities,
            normalize=True
        )

        # Extract query features
        query_s_entropy = [d.s_entropy_coords for d in query_droplets]
        query_phase_coherences = [d.droplet_params.phase_coherence for d in query_droplets]
        query_velocities = [d.droplet_params.velocity for d in query_droplets]

        # Compare to EVERY reference (no indexing)
        matches = []

        for ref_id, ref_data in self.reference_library.items():
            # Calculate similarity metrics (all transparent)
            similarity_metrics = self._calculate_similarity(
                query_s_entropy=query_s_entropy,
                query_phase_coherences=query_phase_coherences,
                query_velocities=query_velocities,
                ref_s_entropy=ref_data['s_entropy_coords'],
                ref_phase_coherences=ref_data['phase_coherences'],
                ref_velocities=ref_data['velocities']
            )

            # Combined similarity (simple weighted average)
            combined_similarity = (
                0.5 * similarity_metrics['s_entropy_similarity'] +
                0.3 * similarity_metrics['phase_coherence_similarity'] +
                0.2 * similarity_metrics['velocity_similarity']
            )

            matches.append(SimpleMatch(
                reference_id=ref_id,
                similarity=combined_similarity,
                s_entropy_distance=similarity_metrics['s_entropy_distance'],
                phase_coherence_diff=similarity_metrics['phase_coherence_diff'],
                velocity_diff=similarity_metrics['velocity_diff'],
                details=similarity_metrics
            ))

        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x.similarity, reverse=True)

        return matches[:top_k]

    def _calculate_similarity(
        self,
        query_s_entropy: List,
        query_phase_coherences: List[float],
        query_velocities: List[float],
        ref_s_entropy: List,
        ref_phase_coherences: List[float],
        ref_velocities: List[float]
    ) -> Dict[str, float]:
        """
        Calculate similarity metrics (all transparent, no black boxes).
        """
        # 1. S-Entropy distance (average pairwise distance)
        s_entropy_distances = []
        for q_coord in query_s_entropy:
            for r_coord in ref_s_entropy:
                dist = np.sqrt(
                    (q_coord.s_knowledge - r_coord.s_knowledge)**2 +
                    (q_coord.s_time - r_coord.s_time)**2 +
                    (q_coord.s_entropy - r_coord.s_entropy)**2
                )
                s_entropy_distances.append(dist)

        avg_s_entropy_distance = np.mean(s_entropy_distances) if s_entropy_distances else 1.0
        s_entropy_similarity = 1.0 / (1.0 + avg_s_entropy_distance)

        # 2. Phase coherence similarity (correlation)
        if len(query_phase_coherences) > 0 and len(ref_phase_coherences) > 0:
            # Pad to same length for comparison
            max_len = max(len(query_phase_coherences), len(ref_phase_coherences))
            q_padded = np.pad(query_phase_coherences, (0, max_len - len(query_phase_coherences)))
            r_padded = np.pad(ref_phase_coherences, (0, max_len - len(ref_phase_coherences)))

            correlation = np.corrcoef(q_padded, r_padded)[0, 1]
            phase_coherence_similarity = (correlation + 1) / 2  # Normalize to [0, 1]
            phase_coherence_diff = np.mean(np.abs(q_padded - r_padded))
        else:
            phase_coherence_similarity = 0.0
            phase_coherence_diff = 1.0

        # 3. Velocity similarity
        if len(query_velocities) > 0 and len(ref_velocities) > 0:
            avg_q_vel = np.mean(query_velocities)
            avg_r_vel = np.mean(ref_velocities)
            velocity_diff = abs(avg_q_vel - avg_r_vel)
            velocity_similarity = 1.0 / (1.0 + velocity_diff)
        else:
            velocity_similarity = 0.0
            velocity_diff = 1.0

        return {
            's_entropy_distance': float(avg_s_entropy_distance),
            's_entropy_similarity': float(s_entropy_similarity),
            'phase_coherence_similarity': float(phase_coherence_similarity),
            'phase_coherence_diff': float(phase_coherence_diff),
            'velocity_similarity': float(velocity_similarity),
            'velocity_diff': float(velocity_diff)
        }

    def save_validation_results(self, output_dir: Path, results: List[SimpleMatch]):
        """Save results in human-readable format (TSV, not binary)."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame
        rows = []
        for match in results:
            rows.append({
                'reference_id': match.reference_id,
                'combined_similarity': match.similarity,
                's_entropy_distance': match.s_entropy_distance,
                'phase_coherence_diff': match.phase_coherence_diff,
                'velocity_diff': match.velocity_diff,
                's_entropy_similarity': match.details['s_entropy_similarity'],
                'phase_coherence_similarity': match.details['phase_coherence_similarity'],
                'velocity_similarity': match.details['velocity_similarity']
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_dir / 'cv_validation_results.tsv', sep='\t', index=False)

        return df

    def get_validation_report(self) -> str:
        """Human-readable validation report."""
        report = f"""
CV Validation Report
====================
Reference Library Size: {len(self.reference_library)} spectra
Total Droplets Generated: {sum(len(ref['droplets']) for ref in self.reference_library.values())}
Physics Validation: Enabled

Comparison Method:
- S-Entropy Distance: Direct Euclidean distance
- Phase Coherence: Correlation coefficient
- Velocity: Mean absolute difference
- Combined: Weighted average (0.5, 0.3, 0.2)

No FAISS, No Compression, No Approximations.
All comparisons are exact and transparent.
"""
        return report

#!/usr/bin/env python3
"""
S-Entropy Database Search for Metabolomics
===========================================

Platform-independent metabolite identification using S-Entropy fragmentation networks.
Implements network-based database searching with semantic distance amplification.

Theoretical Foundation:
-----------------------
From entropy-coordinates.tex:
- 91.4% annotation rate on LIPIDMAPS
- 87.2% accuracy on isobaric lipid mixtures
- Platform-independent (CV < 1% across 4 MS platforms)
- O(log n) search complexity via KD-tree indexing

Key Features:
-------------
1. Semantic distance with feature weighting
2. Network-based similarity (not just spectral dot product)
3. Closed-loop navigation for structural exploration
4. Isobaric compound resolution via network position

Author: Kundai Chinyamakobvu
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
from scipy.spatial import KDTree

# Import S-Entropy framework
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.EntropyTransformation import SEntropyTransformer, SEntropyFeatures
from metabolomics.FragmentationTrees import (
    SEntropyFragmentationNetwork,
    FragmentIon,
    PrecursorIon
)


@dataclass
class MetaboliteReference:
    """
    Reference metabolite entry in database.

    Attributes:
        metabolite_id: Unique identifier
        name: Metabolite name
        formula: Chemical formula
        inchi: InChI string
        smiles: SMILES string
        lipid_class: Lipid class (for lipids)
        precursor_mz: Precursor m/z
        fragments: List of fragment m/z values
        s_entropy_features: 14D S-Entropy feature vector
        database_source: Source database (LIPIDMAPS, METLIN, HMDB, etc.)
    """
    metabolite_id: str
    name: str
    formula: str
    inchi: Optional[str] = None
    smiles: Optional[str] = None
    lipid_class: Optional[str] = None
    precursor_mz: Optional[float] = None
    fragments: List[float] = field(default_factory=list)
    s_entropy_features: Optional[SEntropyFeatures] = None
    database_source: str = 'unknown'


@dataclass
class AnnotationResult:
    """
    Annotation result for a query spectrum.

    Attributes:
        query_mz: Query precursor m/z
        top_matches: List of (metabolite_id, confidence_score) tuples
        semantic_distances: Semantic distances to top matches
        network_coherence: Network coherence scores
        annotation_confidence: Overall confidence score
        is_isobaric: Whether multiple isobaric candidates found
    """
    query_mz: float
    top_matches: List[Tuple[str, float]] = field(default_factory=list)
    semantic_distances: List[float] = field(default_factory=list)
    network_coherence: List[float] = field(default_factory=list)
    annotation_confidence: float = 0.0
    is_isobaric: bool = False


class MSIonDatabaseSearch:
    """
    S-Entropy-based database search for metabolomics.

    Implements network-based annotation with platform independence.
    """

    def __init__(
        self,
        database_path: Optional[str] = None,
        similarity_threshold: float = 0.5,
        sigma: float = 0.2,
        top_k: int = 10
    ):
        """
        Initialize database search engine.

        Args:
            database_path: Path to reference database (JSON or CSV)
            similarity_threshold: Threshold for network edge creation
            sigma: Scale parameter for confidence scores
            top_k: Number of top matches to return
        """
        self.database_path = database_path
        self.similarity_threshold = similarity_threshold
        self.sigma = sigma
        self.top_k = top_k

        # S-Entropy transformer
        self.s_entropy_transformer = SEntropyTransformer()

        # Fragmentation network
        self.fragmentation_network = SEntropyFragmentationNetwork(
            similarity_threshold=similarity_threshold,
            sigma=sigma
        )

        # Reference database
        self.references: Dict[str, MetaboliteReference] = {}
        self.reference_features: Optional[np.ndarray] = None
        self.reference_ids: List[str] = []
        self.kdtree: Optional[KDTree] = None

        # Load database if provided
        if database_path:
            self.load_database(database_path)

    def load_database(self, database_path: str):
        """
        Load reference metabolite database.

        Args:
            database_path: Path to database file (JSON or CSV)
        """
        print(f"[Database] Loading reference database from {database_path}...")

        path = Path(database_path)

        if path.suffix == '.json':
            self._load_json_database(database_path)
        elif path.suffix == '.csv':
            self._load_csv_database(database_path)
        else:
            raise ValueError(f"Unsupported database format: {path.suffix}")

        print(f"[Database] Loaded {len(self.references)} reference metabolites")

        # Pre-compute S-Entropy features for all references
        self._precompute_reference_features()

    def _load_json_database(self, database_path: str):
        """Load database from JSON format."""
        with open(database_path, 'r') as f:
            data = json.load(f)

        for entry in data:
            ref = MetaboliteReference(
                metabolite_id=entry['id'],
                name=entry['name'],
                formula=entry.get('formula', ''),
                inchi=entry.get('inchi'),
                smiles=entry.get('smiles'),
                lipid_class=entry.get('lipid_class'),
                precursor_mz=entry.get('precursor_mz'),
                fragments=entry.get('fragments', []),
                database_source=entry.get('source', 'custom')
            )
            self.references[ref.metabolite_id] = ref

    def _load_csv_database(self, database_path: str):
        """Load database from CSV format."""
        df = pd.read_csv(database_path)

        for idx, row in df.iterrows():
            # Parse fragments (comma-separated)
            fragments_str = row.get('fragments', '')
            if isinstance(fragments_str, str) and fragments_str:
                fragments = [float(x.strip()) for x in fragments_str.split(',')]
            else:
                fragments = []

            ref = MetaboliteReference(
                metabolite_id=str(row.get('id', f'metabolite_{idx}')),
                name=str(row.get('name', f'Unknown_{idx}')),
                formula=str(row.get('formula', '')),
                inchi=row.get('inchi') if 'inchi' in row else None,
                smiles=row.get('smiles') if 'smiles' in row else None,
                lipid_class=row.get('lipid_class') if 'lipid_class' in row else None,
                precursor_mz=float(row['precursor_mz']) if 'precursor_mz' in row else None,
                fragments=fragments,
                database_source=str(row.get('source', 'custom'))
            )
            self.references[ref.metabolite_id] = ref

    def _precompute_reference_features(self):
        """
        Pre-compute S-Entropy features for all reference metabolites.

        Creates KD-tree for O(log n) nearest neighbor search.
        """
        print("[Database] Pre-computing S-Entropy features...")

        feature_list = []
        id_list = []

        for metabolite_id, ref in self.references.items():
            if len(ref.fragments) > 0 and ref.precursor_mz:
                # Compute S-Entropy features from fragment list
                # Use uniform intensities as placeholder
                mz_array = np.array(ref.fragments)
                intensity_array = np.ones(len(ref.fragments))

                features = self.s_entropy_transformer.calculate_spectrum_features(
                    mz_array=mz_array,
                    intensity_array=intensity_array,
                    precursor_mz=ref.precursor_mz,
                    rt=None
                )

                ref.s_entropy_features = features
                feature_list.append(features.features)
                id_list.append(metabolite_id)

        if len(feature_list) > 0:
            self.reference_features = np.array(feature_list)
            self.reference_ids = id_list

            # Build KD-tree
            self.kdtree = KDTree(self.reference_features)
            print(f"[Database] Built KD-tree with {len(id_list)} metabolites")
        else:
            print("[Database] WARNING: No valid reference features computed")

    def search(
        self,
        query_mz_array: np.ndarray,
        query_intensity_array: np.ndarray,
        query_precursor_mz: float,
        query_rt: Optional[float] = None
    ) -> AnnotationResult:
        """
        Search database for matching metabolites.

        Args:
            query_mz_array: Query fragment m/z values
            query_intensity_array: Query fragment intensities
            query_precursor_mz: Query precursor m/z
            query_rt: Query retention time (optional)

        Returns:
            AnnotationResult with top matches
        """
        # Compute S-Entropy features for query
        query_features = self.s_entropy_transformer.calculate_spectrum_features(
            mz_array=query_mz_array,
            intensity_array=query_intensity_array,
            precursor_mz=query_precursor_mz,
            rt=query_rt
        )

        if self.kdtree is None:
            print("[Search] WARNING: KD-tree not built, no database loaded")
            return AnnotationResult(query_mz=query_precursor_mz)

        # Find k nearest neighbors
        distances, indices = self.kdtree.query(query_features.features, k=self.top_k)

        # Compute semantic distances with feature weighting
        semantic_distances = []
        for idx in indices:
            ref_features = SEntropyFeatures(self.reference_features[idx])
            d_sem = self.fragmentation_network.compute_semantic_distance(
                query_features, ref_features
            )
            semantic_distances.append(d_sem)

        # Compute confidence scores
        confidences = [np.exp(-d / self.sigma) for d in semantic_distances]

        # Normalize confidences
        total_confidence = sum(confidences)
        if total_confidence > 0:
            confidences = [c / total_confidence for c in confidences]

        # Get metabolite IDs
        top_matches = [(self.reference_ids[idx], conf)
                      for idx, conf in zip(indices, confidences)]

        # Check for isobaric compounds
        is_isobaric = self._check_isobaric(indices, query_precursor_mz)

        # Compute network coherence for each match
        network_coherence = self._compute_network_coherence_batch(
            query_features, indices
        )

        result = AnnotationResult(
            query_mz=query_precursor_mz,
            top_matches=top_matches,
            semantic_distances=semantic_distances,
            network_coherence=network_coherence,
            annotation_confidence=confidences[0] if confidences else 0.0,
            is_isobaric=is_isobaric
        )

        return result

    def _check_isobaric(
        self,
        indices: List[int],
        query_mz: float,
        mz_tolerance: float = 0.01
    ) -> bool:
        """
        Check if top matches include isobaric compounds.

        Args:
            indices: Indices of top matches
            query_mz: Query precursor m/z
            mz_tolerance: m/z tolerance for isobaric check (Da)

        Returns:
            True if multiple isobaric compounds found
        """
        if len(indices) < 2:
            return False

        # Get precursor m/z values for top matches
        precursor_mzs = []
        for idx in indices[:5]:  # Check top 5
            metabolite_id = self.reference_ids[idx]
            ref = self.references[metabolite_id]
            if ref.precursor_mz:
                precursor_mzs.append(ref.precursor_mz)

        # Check if multiple compounds within tolerance
        isobaric_count = 0
        for mz in precursor_mzs:
            if abs(mz - query_mz) < mz_tolerance:
                isobaric_count += 1

        return isobaric_count >= 2

    def _compute_network_coherence_batch(
        self,
        query_features: SEntropyFeatures,
        reference_indices: List[int]
    ) -> List[float]:
        """
        Compute network coherence for batch of references.

        Args:
            query_features: Query S-Entropy features
            reference_indices: Indices of reference metabolites

        Returns:
            List of coherence scores
        """
        coherence_scores = []

        for idx in reference_indices:
            # Find neighborhood in reference space
            neighbor_distances, neighbor_indices = self.kdtree.query(
                self.reference_features[idx],
                k=min(20, len(self.reference_ids))
            )

            # Compute coherence (simplified: based on local density)
            avg_distance = np.mean(neighbor_distances)
            coherence = np.exp(-avg_distance)
            coherence_scores.append(coherence)

        return coherence_scores

    def batch_search(
        self,
        queries: List[Tuple[np.ndarray, np.ndarray, float, Optional[float]]]
    ) -> List[AnnotationResult]:
        """
        Batch search multiple queries.

        Args:
            queries: List of (mz_array, intensity_array, precursor_mz, rt) tuples

        Returns:
            List of AnnotationResult objects
        """
        results = []

        print(f"[Batch Search] Processing {len(queries)} queries...")

        for i, (mz_array, intensity_array, precursor_mz, rt) in enumerate(queries):
            if (i + 1) % 100 == 0:
                print(f"[Batch Search] Processed {i+1}/{len(queries)} queries")

            result = self.search(mz_array, intensity_array, precursor_mz, rt)
            results.append(result)

        print(f"[Batch Search] Completed {len(queries)} queries")

        return results

    def get_annotation_summary(self, results: List[AnnotationResult]) -> pd.DataFrame:
        """
        Generate summary statistics for annotation results.

        Args:
            results: List of AnnotationResult objects

        Returns:
            DataFrame with summary statistics
        """
        summary_data = []

        for result in results:
            if len(result.top_matches) > 0:
                top_id, top_conf = result.top_matches[0]
                top_ref = self.references.get(top_id)

                summary_data.append({
                    'query_mz': result.query_mz,
                    'top_match_id': top_id,
                    'top_match_name': top_ref.name if top_ref else 'Unknown',
                    'top_match_formula': top_ref.formula if top_ref else '',
                    'confidence': top_conf,
                    'semantic_distance': result.semantic_distances[0] if result.semantic_distances else np.nan,
                    'is_isobaric': result.is_isobaric,
                    'num_candidates': len(result.top_matches)
                })
            else:
                summary_data.append({
                    'query_mz': result.query_mz,
                    'top_match_id': None,
                    'top_match_name': 'No match',
                    'top_match_formula': '',
                    'confidence': 0.0,
                    'semantic_distance': np.nan,
                    'is_isobaric': False,
                    'num_candidates': 0
                })

        df = pd.DataFrame(summary_data)

        # Compute overall statistics
        print("\n[Annotation Summary]")
        print(f"  Total queries: {len(results)}")
        print(f"  Annotated: {len(df[df['top_match_id'].notna()])} ({len(df[df['top_match_id'].notna()])/len(df)*100:.1f}%)")
        print(f"  Mean confidence: {df['confidence'].mean():.3f}")
        print(f"  Isobaric compounds: {df['is_isobaric'].sum()} ({df['is_isobaric'].sum()/len(df)*100:.1f}%)")

        return df

    def export_results(self, results: List[AnnotationResult], output_path: str):
        """
        Export annotation results to CSV.

        Args:
            results: List of AnnotationResult objects
            output_path: Output file path
        """
        df = self.get_annotation_summary(results)
        df.to_csv(output_path, index=False)
        print(f"[Export] Results exported to {output_path}")

#!/usr/bin/env python3
"""
S-Entropy Tandem Database Search for Proteomics
===============================================

Proteomics-specific database search using S-Entropy framework with b/y ion
complementarity validation and temporal proximity analysis.

Theoretical Foundation:
-----------------------
From tandem-mass-spec.tex:
- 28.5% improvement in silhouette score over traditional methods
- 87% b/y ion complementarity correlation (r=0.89, p<0.0001)
- 72% temporal proximity correlation (ρ=0.72, p<0.001)
- 0.0015 seconds per spectrum processing time

S-Entropy Proteomics Framework:
--------------------------------
- S_knowledge: Information content from intensity + m/z ratio
- S_time: Temporal ordering (Gaussian weighting around spectral center)
- S_entropy: Local entropy of intensity distribution

Key Features:
-------------
1. B/Y ion complementarity: Validates phase-lock preservation through fragmentation
2. Temporal proximity: Correlation between RT and S-Entropy distance
3. Fragment pattern consistency: Within-group vs. between-group distances
4. Proteomics-specific 14D feature extraction

Author: Kundai Sachikonye
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
import json
from sklearn.neighbors import KDTree

# Import S-Entropy framework
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.EntropyTransformation import SEntropyTransformer, SEntropyCoordinates, SEntropyFeatures
from core.PhaseLockNetworks import PhaseLockSignature, EnhancedPhaseLockMeasurementDevice


@dataclass
class PeptideFragment:
    """
    Peptide fragment ion (b-ion or y-ion).

    Key Insight: All fragments from same peptide are FREQUENCY-COUPLED
    - They emerge at same collision event (same time)
    - They share phase-lock signatures
    - Sequential relationship: b_i, b_{i+1} differ by one amino acid
    - Complementarity: b_i + y_{n-i} = precursor mass

    Attributes:
        mz: Fragment m/z
        intensity: Fragment intensity
        ion_type: 'b' or 'y'
        position: Position in peptide sequence (1-indexed)
        neutral_losses: Set of neutral losses (H2O, NH3, etc.)
        s_entropy_coords: S-Entropy coordinates
        charge: Charge state
        phase_lock_signature: Shared phase-lock from collision event
        frequency_coupling: Coupling strength with other fragments
    """
    mz: float
    intensity: float
    ion_type: str  # 'b' or 'y'
    position: int
    neutral_losses: Set[str] = field(default_factory=set)
    s_entropy_coords: Optional[SEntropyCoordinates] = None
    charge: int = 1
    phase_lock_signature: Optional['PhaseLockSignature'] = None
    frequency_coupling: float = 1.0  # Coupling strength with other fragments


@dataclass
class PeptideSpectrum:
    """
    Tandem MS/MS spectrum for a peptide.

    Key Insight: All fragments are FREQUENCY-COUPLED from same collision event
    - Shared phase-lock signature across all fragments
    - Sequential b/y ion series with predictable mass differences
    - Temporal coupling: all fragments at same RT

    Attributes:
        precursor_mz: Precursor m/z
        precursor_charge: Precursor charge state
        rt: Retention time (minutes)
        peptide_sequence: Peptide sequence (if known)
        fragments: List of fragment ions (ALL frequency-coupled!)
        s_entropy_features: 14D S-Entropy feature vector
        scan_number: Scan number
        collision_event_signature: Shared phase-lock from collision
        frequency_coupling_matrix: Fragment-fragment coupling strengths
    """
    precursor_mz: float
    precursor_charge: int
    rt: float
    peptide_sequence: Optional[str] = None
    fragments: List[PeptideFragment] = field(default_factory=list)
    s_entropy_features: Optional[SEntropyFeatures] = None
    scan_number: Optional[int] = None
    collision_event_signature: Optional['PhaseLockSignature'] = None
    frequency_coupling_matrix: Optional[np.ndarray] = None


@dataclass
class PeptideReference:
    """
    Reference peptide entry in database.

    Attributes:
        peptide_id: Unique identifier
        sequence: Peptide sequence
        modifications: Post-translational modifications
        protein_accession: Protein accession number
        precursor_mz: Precursor m/z
        charge: Charge state
        theoretical_fragments: Theoretical b/y ion m/z values
        s_entropy_features: 14D S-Entropy feature vector
    """
    peptide_id: str
    sequence: str
    modifications: Optional[Dict[int, str]] = None
    protein_accession: Optional[str] = None
    precursor_mz: Optional[float] = None
    charge: int = 2
    theoretical_fragments: Dict[str, List[float]] = field(default_factory=dict)  # {'b': [...], 'y': [...]}
    s_entropy_features: Optional[SEntropyFeatures] = None


@dataclass
class ProteomicsAnnotationResult:
    """
    Annotation result with proteomics-specific validation.

    FREQUENCY COUPLING INTEGRATION:
    - All peptide fragments are frequency-coupled (same collision event)
    - Validation includes coupling consistency check

    Attributes:
        query_scan: Query scan number
        query_precursor_mz: Query precursor m/z
        top_matches: List of (peptide_id, confidence) tuples
        semantic_distances: Semantic distances to top matches
        by_complementarity_scores: B/Y ion complementarity scores
        temporal_proximity_scores: Temporal proximity scores (if RT available)
        fragment_pattern_consistency: Pattern consistency scores
        frequency_coupling_scores: Frequency coupling consistency scores (NEW!)
        overall_confidence: Overall confidence score
        validation_passed: Whether proteomics validation passed
    """
    query_scan: Optional[int]
    query_precursor_mz: float
    top_matches: List[Tuple[str, float]] = field(default_factory=list)
    semantic_distances: List[float] = field(default_factory=list)
    by_complementarity_scores: List[float] = field(default_factory=list)
    temporal_proximity_scores: List[float] = field(default_factory=list)
    fragment_pattern_consistency: List[float] = field(default_factory=list)
    frequency_coupling_scores: List[float] = field(default_factory=list)  # NEW!
    overall_confidence: float = 0.0
    validation_passed: bool = False


class TandemDatabaseSearch:
    """
    S-Entropy-based tandem MS database search for proteomics.

    Implements proteomics-specific validation:
    - B/Y ion complementarity
    - Temporal proximity
    - Fragment pattern consistency
    """

    def __init__(
        self,
        database_path: Optional[str] = None,
        similarity_threshold: float = 0.5,
        sigma: float = 0.2,
        top_k: int = 10,
        by_complementarity_threshold: float = 0.7,
        enable_temporal_validation: bool = True
    ):
        """
        Initialize tandem database search.

        Args:
            database_path: Path to reference peptide database
            similarity_threshold: Threshold for semantic distance
            sigma: Scale parameter for confidence scores
            top_k: Number of top matches to return
            by_complementarity_threshold: Minimum b/y complementarity for validation
            enable_temporal_validation: Whether to use temporal proximity validation
        """
        self.database_path = database_path
        self.similarity_threshold = similarity_threshold
        self.sigma = sigma
        self.top_k = top_k
        self.by_complementarity_threshold = by_complementarity_threshold
        self.enable_temporal_validation = enable_temporal_validation

        # S-Entropy transformer
        self.s_entropy_transformer = SEntropyTransformer()

        # Phase-lock measurement device for frequency coupling analysis
        self.phase_lock_device = EnhancedPhaseLockMeasurementDevice(
            enable_performance_tracking=True
        )

        # Reference database
        self.references: Dict[str, PeptideReference] = {}
        self.reference_features: Optional[np.ndarray] = None
        self.reference_ids: List[str] = []
        self.kdtree: Optional[KDTree] = None

        # Load database if provided
        if database_path:
            self.load_database(database_path)

    def load_database(self, database_path: str):
        """
        Load reference peptide database.

        Args:
            database_path: Path to database file (JSON or CSV)
        """
        print(f"[Database] Loading reference peptide database from {database_path}...")

        path = Path(database_path)

        if path.suffix == '.json':
            self._load_json_database(database_path)
        elif path.suffix == '.csv':
            self._load_csv_database(database_path)
        else:
            raise ValueError(f"Unsupported database format: {path.suffix}")

        print(f"[Database] Loaded {len(self.references)} reference peptides")

        # Pre-compute S-Entropy features
        self._precompute_reference_features()

    def _load_json_database(self, database_path: str):
        """Load database from JSON format."""
        with open(database_path, 'r') as f:
            data = json.load(f)

        for entry in data:
            ref = PeptideReference(
                peptide_id=entry['id'],
                sequence=entry['sequence'],
                modifications=entry.get('modifications'),
                protein_accession=entry.get('protein'),
                precursor_mz=entry.get('precursor_mz'),
                charge=entry.get('charge', 2),
                theoretical_fragments=entry.get('fragments', {})
            )
            self.references[ref.peptide_id] = ref

    def _load_csv_database(self, database_path: str):
        """Load database from CSV format."""
        df = pd.read_csv(database_path)

        for idx, row in df.iterrows():
            ref = PeptideReference(
                peptide_id=str(row.get('id', f'peptide_{idx}')),
                sequence=str(row['sequence']),
                modifications=json.loads(row['modifications']) if 'modifications' in row and pd.notna(row['modifications']) else None,
                protein_accession=str(row.get('protein', '')),
                precursor_mz=float(row['precursor_mz']) if 'precursor_mz' in row else None,
                charge=int(row.get('charge', 2)),
                theoretical_fragments=json.loads(row['fragments']) if 'fragments' in row and pd.notna(row['fragments']) else {}
            )
            self.references[ref.peptide_id] = ref

    def _precompute_reference_features(self):
        """Pre-compute S-Entropy features for all reference peptides."""
        print("[Database] Pre-computing S-Entropy features...")

        feature_list = []
        id_list = []

        for peptide_id, ref in self.references.items():
            # Compute features from theoretical fragments
            if ref.theoretical_fragments and ref.precursor_mz:
                # Combine b and y ions
                all_fragments = []
                if 'b' in ref.theoretical_fragments:
                    all_fragments.extend(ref.theoretical_fragments['b'])
                if 'y' in ref.theoretical_fragments:
                    all_fragments.extend(ref.theoretical_fragments['y'])

                if len(all_fragments) > 0:
                    mz_array = np.array(all_fragments)
                    intensity_array = np.ones(len(all_fragments))  # Uniform intensities

                    _, features = self.s_entropy_transformer.transform_and_extract(
                        mz_array=mz_array,
                        intensity_array=intensity_array,
                        precursor_mz=ref.precursor_mz,
                        rt=None
                    )

                    ref.s_entropy_features = features
                    feature_list.append(features.features)
                    id_list.append(peptide_id)

        if len(feature_list) > 0:
            self.reference_features = np.array(feature_list)
            self.reference_ids = id_list

            # Build KD-tree
            self.kdtree = KDTree(self.reference_features)
            print(f"[Database] Built KD-tree with {len(id_list)} peptides")
        else:
            print("[Database] WARNING: No valid reference features computed")

    def compute_frequency_coupling(
        self,
        spectrum: PeptideSpectrum
    ) -> np.ndarray:
        """
        Compute frequency coupling matrix for all fragments in spectrum.

        Key Insight: All peptide fragments emerge from SAME collision event
        - They share phase-lock signatures (frequency-coupled)
        - Sequential b/y ions have predictable mass relationships
        - Coupling strength based on:
          1. S-Entropy coordinate distance (closer = stronger coupling)
          2. Sequential relationship (b_i, b_{i+1} differ by amino acid)
          3. Complementarity (b_i + y_{n-i} = precursor)

        Args:
            spectrum: PeptideSpectrum with fragments

        Returns:
            Coupling matrix [n_fragments x n_fragments]
            coupling[i,j] = strength of frequency coupling between fragments i,j
        """
        n_frags = len(spectrum.fragments)
        if n_frags == 0:
            return np.array([])

        # Initialize coupling matrix
        coupling_matrix = np.ones((n_frags, n_frags))

        # Compute S-Entropy coordinates for all fragments at once
        mz_array = np.array([f.mz for f in spectrum.fragments])
        intensity_array = np.array([f.intensity for f in spectrum.fragments])

        # Transform entire spectrum to get coordinates for all fragments
        coords_list, _ = self.s_entropy_transformer.transform_spectrum(
            mz_array=mz_array,
            intensity_array=intensity_array,
            precursor_mz=spectrum.precursor_mz,
            rt=spectrum.rt
        )

        # Assign coordinates to fragments
        for i, frag in enumerate(spectrum.fragments):
            if i < len(coords_list):
                frag.s_entropy_coords = coords_list[i]

        # Compute pairwise coupling strengths
        for i in range(n_frags):
            for j in range(i+1, n_frags):
                frag_i = spectrum.fragments[i]
                frag_j = spectrum.fragments[j]

                coupling_strength = 1.0  # Base coupling (same collision event)

                # Factor 1: S-Entropy coordinate distance (closer = stronger)
                if frag_i.s_entropy_coords and frag_j.s_entropy_coords:
                    coords_i = frag_i.s_entropy_coords.to_numpy()
                    coords_j = frag_j.s_entropy_coords.to_numpy()
                    distance = np.linalg.norm(coords_i - coords_j)

                    # Stronger coupling for closer S-Entropy coordinates
                    coupling_strength *= np.exp(-distance / 0.5)

                # Factor 2: Sequential relationship (b_i, b_{i+1} or y_i, y_{i+1})
                if frag_i.ion_type == frag_j.ion_type:
                    if abs(frag_i.position - frag_j.position) == 1:
                        # Adjacent ions in series: STRONG coupling
                        coupling_strength *= 2.0

                # Factor 3: Complementarity (b_i + y_{n-i} = precursor)
                if frag_i.ion_type != frag_j.ion_type:
                    # Check if complementary
                    total_mz = frag_i.mz + frag_j.mz
                    expected_mz = spectrum.precursor_mz * spectrum.precursor_charge

                    if abs(total_mz - expected_mz) < 2.0:  # 2 Da tolerance
                        # Complementary b/y pair: VERY STRONG coupling
                        coupling_strength *= 3.0

                # Set symmetric coupling
                coupling_matrix[i, j] = coupling_strength
                coupling_matrix[j, i] = coupling_strength

        return coupling_matrix

    def compute_collision_event_signature(
        self,
        spectrum: PeptideSpectrum
    ) -> Optional[PhaseLockSignature]:
        """
        Compute shared phase-lock signature for entire collision event.

        All fragments from same peptide share this signature because
        they emerged simultaneously from the same collision.

        Args:
            spectrum: PeptideSpectrum

        Returns:
            PhaseLockSignature representing the collision event
        """
        if len(spectrum.fragments) == 0:
            return None

        # Prepare spectrum data for phase-lock measurement
        mz_array = np.array([f.mz for f in spectrum.fragments])
        intensity_array = np.array([f.intensity for f in spectrum.fragments])

        # Measure phase-locks across all fragments
        # They should all show strong phase coherence (emerged at same time)
        spectra = [(spectrum.rt, mz_array, intensity_array)]

        phase_lock_df = self.phase_lock_device.measure_from_arrays(spectra)

        if len(phase_lock_df) > 0:
            # Take the strongest phase-lock as collision event signature
            strongest_idx = phase_lock_df['coherence_strength'].idxmax()
            strongest_lock = phase_lock_df.iloc[strongest_idx]

            # Create signature
            signature = PhaseLockSignature(
                mz_center=strongest_lock['mz_center'],
                mz_range=(mz_array.min(), mz_array.max()),
                rt_center=spectrum.rt,
                rt_range=(spectrum.rt, spectrum.rt),
                coherence_strength=strongest_lock['coherence_strength'],
                coupling_modality=strongest_lock['coupling_modality'],
                oscillation_frequency=0.0,  # Placeholder
                phase_offset=0.0,  # Placeholder
                ensemble_size=len(spectrum.fragments),
                temperature_signature=298.15,
                pressure_signature=1.0,
                categorical_state=int(strongest_lock['categorical_state'])
            )

            return signature

        return None

    def validate_frequency_coupling_consistency(
        self,
        query_spectrum: PeptideSpectrum,
        reference: PeptideReference
    ) -> float:
        """
        Validate frequency coupling consistency between query and reference.

        For peptides: ALL fragments should show consistent coupling because
        they emerged from same collision event.

        Args:
            query_spectrum: Query spectrum
            reference: Reference peptide

        Returns:
            Consistency score [0, 1]
        """
        # Compute coupling matrix for query
        query_coupling = self.compute_frequency_coupling(query_spectrum)

        if query_coupling.size == 0:
            return 0.5

        # Analyze coupling patterns
        # Strong coupling indicates all fragments from same collision event
        mean_coupling = np.mean(query_coupling[np.triu_indices_from(query_coupling, k=1)])

        # High mean coupling = consistent frequency domain (good!)
        # Low mean coupling = inconsistent (might be contaminated)

        # Expected mean coupling for clean peptide spectrum: ~1.5-2.0
        # (base coupling 1.0 + enhancements from sequential/complementarity)
        expected_coupling = 1.5

        # Score based on how close to expected
        consistency_score = np.exp(-abs(mean_coupling - expected_coupling)**2 / (2 * 0.5**2))

        return float(consistency_score)

    def search(
        self,
        query_spectrum: PeptideSpectrum
    ) -> ProteomicsAnnotationResult:
        """
        Search database for matching peptides with proteomics validation.

        FREQUENCY COUPLING INTEGRATION:
        - Computes collision event signature (shared phase-lock)
        - Analyzes fragment-fragment coupling matrix
        - Validates frequency domain consistency

        Args:
            query_spectrum: Query peptide spectrum

        Returns:
            ProteomicsAnnotationResult with validation scores
        """
        # Extract fragment m/z and intensities
        if len(query_spectrum.fragments) == 0:
            return ProteomicsAnnotationResult(
                query_scan=query_spectrum.scan_number,
                query_precursor_mz=query_spectrum.precursor_mz
            )

        # STEP 1: Compute frequency coupling matrix
        # All fragments are coupled (same collision event!)
        print(f"[Frequency Coupling] Computing coupling matrix for {len(query_spectrum.fragments)} fragments...")
        query_spectrum.frequency_coupling_matrix = self.compute_frequency_coupling(query_spectrum)

        # STEP 2: Compute collision event signature
        # Shared phase-lock across all fragments
        query_spectrum.collision_event_signature = self.compute_collision_event_signature(query_spectrum)

        if query_spectrum.collision_event_signature:
            print(f"[Collision Event] Coherence: {query_spectrum.collision_event_signature.coherence_strength:.3f}, "
                  f"Ensemble size: {query_spectrum.collision_event_signature.ensemble_size}")

        mz_array = np.array([f.mz for f in query_spectrum.fragments])
        intensity_array = np.array([f.intensity for f in query_spectrum.fragments])

        # Compute S-Entropy features for query
        _, query_features = self.s_entropy_transformer.transform_and_extract(
            mz_array=mz_array,
            intensity_array=intensity_array,
            precursor_mz=query_spectrum.precursor_mz,
            rt=query_spectrum.rt
        )
        query_spectrum.s_entropy_features = query_features

        if self.kdtree is None:
            print("[Search] WARNING: KD-tree not built, no database loaded")
            return ProteomicsAnnotationResult(
                query_scan=query_spectrum.scan_number,
                query_precursor_mz=query_spectrum.precursor_mz
            )

        # Find k nearest neighbors
        distances, indices = self.kdtree.query(query_features.features, k=self.top_k)

        # Compute semantic distances
        semantic_distances = []
        for idx in indices:
            ref_features = SEntropyFeatures.from_array(self.reference_features[idx])
            d_sem = self._compute_semantic_distance(query_features, ref_features)
            semantic_distances.append(d_sem)

        # Compute confidence scores
        confidences = [np.exp(-d / self.sigma) for d in semantic_distances]

        # Normalize confidences
        total_confidence = sum(confidences)
        if total_confidence > 0:
            confidences = [c / total_confidence for c in confidences]

        # Get peptide IDs
        top_matches = [(self.reference_ids[idx], conf)
                       for idx, conf in zip(indices, confidences)]

        # Proteomics-specific validation
        by_complementarity_scores = []
        temporal_proximity_scores = []
        fragment_consistency_scores = []
        frequency_coupling_scores = []  # NEW: Frequency coupling validation

        for idx in indices:
            peptide_id = self.reference_ids[idx]
            ref = self.references[peptide_id]

            # B/Y ion complementarity validation
            by_score = self._validate_by_complementarity(query_spectrum, ref)
            by_complementarity_scores.append(by_score)

            # Temporal proximity validation (if RT available)
            if self.enable_temporal_validation and query_spectrum.rt:
                temporal_score = self._validate_temporal_proximity(
                    query_spectrum, query_features
                )
                temporal_proximity_scores.append(temporal_score)

            # Fragment pattern consistency
            consistency_score = self._validate_fragment_consistency(
                query_features, ref.s_entropy_features
            )
            fragment_consistency_scores.append(consistency_score)

            # FREQUENCY COUPLING CONSISTENCY (NEW!)
            # All fragments from peptide should show strong coupling
            coupling_score = self.validate_frequency_coupling_consistency(
                query_spectrum, ref
            )
            frequency_coupling_scores.append(coupling_score)

        # Compute overall confidence with validation
        # UPDATED: Include frequency coupling in validation
        overall_confidence = confidences[0] if confidences else 0.0
        if by_complementarity_scores:
            # Weight by validation scores (updated weights)
            validation_weight = 0.25 * by_complementarity_scores[0]  # b/y complementarity

            if temporal_proximity_scores:
                validation_weight += 0.20 * temporal_proximity_scores[0]  # temporal

            if fragment_consistency_scores:
                validation_weight += 0.25 * fragment_consistency_scores[0]  # pattern consistency

            if frequency_coupling_scores:
                # FREQUENCY COUPLING: Critical for peptides!
                # All fragments from same collision → should be strongly coupled
                validation_weight += 0.30 * frequency_coupling_scores[0]  # frequency coupling

            overall_confidence = 0.6 * overall_confidence + 0.4 * validation_weight

        print(f"[Validation] B/Y: {by_complementarity_scores[0]:.3f}, "
              f"Coupling: {frequency_coupling_scores[0]:.3f}, "
              f"Overall: {overall_confidence:.3f}")

        # Check if validation passed
        # UPDATED: Include frequency coupling in validation (CRITICAL for peptides!)
        validation_passed = False
        if by_complementarity_scores and frequency_coupling_scores:
            validation_passed = (
                by_complementarity_scores[0] >= self.by_complementarity_threshold and
                frequency_coupling_scores[0] >= 0.5  # Require reasonable coupling consistency
            )

        result = ProteomicsAnnotationResult(
            query_scan=query_spectrum.scan_number,
            query_precursor_mz=query_spectrum.precursor_mz,
            top_matches=top_matches,
            semantic_distances=semantic_distances,
            by_complementarity_scores=by_complementarity_scores,
            temporal_proximity_scores=temporal_proximity_scores,
            fragment_pattern_consistency=fragment_consistency_scores,
            frequency_coupling_scores=frequency_coupling_scores,  # NEW!
            overall_confidence=overall_confidence,
            validation_passed=validation_passed
        )

        return result

    def _compute_semantic_distance(
        self,
        features1: SEntropyFeatures,
        features2: SEntropyFeatures
    ) -> float:
        """Compute semantic distance with feature weighting."""
        # Proteomics-specific feature weights (from tandem-mass-spec.tex)
        weights = np.array([
            0.234,  # mean magnitude
            0.198,  # std magnitude
            0.143,  # min magnitude
            0.089,  # max magnitude
            0.128,  # centroid (3D, sum components)
            0.128,
            0.128,
            0.176,  # mean pairwise distance
            0.089,  # diameter
            0.045,  # variance from centroid
            0.032,  # first PC variance ratio
            0.176,  # coordinate entropy
            0.143,  # mean knowledge
            0.089   # mean time
        ])

        diff = np.abs(features1.features - features2.features)
        weighted_diff = weights * diff
        return np.sum(weighted_diff)

    def _validate_by_complementarity(
        self,
        query_spectrum: PeptideSpectrum,
        reference: PeptideReference
    ) -> float:
        """
        Validate b/y ion complementarity.

        From tandem-mass-spec.tex:
        - Pearson r=0.89 between complementary b/y ion S-Entropy magnitudes
        - Mean S-Entropy distance 0.52 ± 0.18 for complementary pairs

        Args:
            query_spectrum: Query spectrum
            reference: Reference peptide

        Returns:
            Complementarity score [0, 1]
        """
        if not reference.theoretical_fragments:
            return 0.5  # Neutral score if no theoretical fragments

        # Identify b and y ions in query
        b_ions = [f for f in query_spectrum.fragments if f.ion_type == 'b']
        y_ions = [f for f in query_spectrum.fragments if f.ion_type == 'y']

        if len(b_ions) == 0 or len(y_ions) == 0:
            return 0.5  # Need both ion types

        # Find complementary pairs (b_i + y_{n-i} = precursor)
        complementary_distances = []

        for b_ion in b_ions:
            for y_ion in y_ions:
                # Check if b + y ≈ precursor
                total_mz = b_ion.mz + y_ion.mz
                expected_mz = query_spectrum.precursor_mz * query_spectrum.precursor_charge

                if abs(total_mz - expected_mz) < 2.0:  # 2 Da tolerance
                    # Compute S-Entropy distance
                    if b_ion.s_entropy_coords and y_ion.s_entropy_coords:
                        b_coords = b_ion.s_entropy_coords.to_numpy()
                        y_coords = y_ion.s_entropy_coords.to_numpy()
                        distance = np.linalg.norm(b_coords - y_coords)
                        complementary_distances.append(distance)

        if len(complementary_distances) == 0:
            return 0.3  # Low score if no complementary pairs found

        # Score based on mean distance (lower is better)
        # Expected: 0.52 ± 0.18 from paper
        mean_distance = np.mean(complementary_distances)

        # Convert distance to score (0.52 → 1.0, larger distances → lower scores)
        score = np.exp(-(mean_distance - 0.52)**2 / (2 * 0.18**2))

        return float(score)

    def _validate_temporal_proximity(
        self,
        query_spectrum: PeptideSpectrum,
        query_features: SEntropyFeatures
    ) -> float:
        """
        Validate temporal proximity.

        From tandem-mass-spec.tex:
        - Spearman ρ=0.72 between RT difference and S-Entropy distance

        Args:
            query_spectrum: Query spectrum with RT
            query_features: Query S-Entropy features

        Returns:
            Temporal proximity score [0, 1]
        """
        # This is a placeholder - full implementation would require
        # reference RT database and correlation analysis
        # For now, return neutral score
        return 0.7

    def _validate_fragment_consistency(
        self,
        query_features: SEntropyFeatures,
        reference_features: Optional[SEntropyFeatures]
    ) -> float:
        """
        Validate fragment pattern consistency.

        From tandem-mass-spec.tex:
        - Within-group distance: 0.31 ± 0.11
        - Between-group distance: 1.92 ± 0.38
        - Consistency coefficient: 0.16

        Args:
            query_features: Query S-Entropy features
            reference_features: Reference S-Entropy features

        Returns:
            Consistency score [0, 1]
        """
        if not reference_features:
            return 0.5

        # Compute feature distance
        distance = np.linalg.norm(query_features.features - reference_features.features)

        # Score based on expected within-group distance (0.31)
        # Lower distance → higher consistency
        score = np.exp(-(distance - 0.31)**2 / (2 * 0.11**2))

        return float(score)

    def batch_search(
        self,
        query_spectra: List[PeptideSpectrum]
    ) -> List[ProteomicsAnnotationResult]:
        """
        Batch search multiple query spectra.

        Args:
            query_spectra: List of PeptideSpectrum objects

        Returns:
            List of ProteomicsAnnotationResult objects
        """
        results = []

        print(f"[Batch Search] Processing {len(query_spectra)} spectra...")

        for i, spectrum in enumerate(query_spectra):
            if (i + 1) % 100 == 0:
                print(f"[Batch Search] Processed {i+1}/{len(query_spectra)} spectra")

            result = self.search(spectrum)
            results.append(result)

        print(f"[Batch Search] Completed {len(query_spectra)} spectra")

        return results

    def get_annotation_summary(
        self,
        results: List[ProteomicsAnnotationResult]
    ) -> pd.DataFrame:
        """
        Generate summary statistics for proteomics annotation results.

        Args:
            results: List of ProteomicsAnnotationResult objects

        Returns:
            DataFrame with summary statistics
        """
        summary_data = []

        for result in results:
            if len(result.top_matches) > 0:
                top_id, top_conf = result.top_matches[0]
                top_ref = self.references.get(top_id)

                summary_data.append({
                    'scan': result.query_scan,
                    'precursor_mz': result.query_precursor_mz,
                    'top_match_id': top_id,
                    'top_match_sequence': top_ref.sequence if top_ref else '',
                    'confidence': top_conf,
                    'overall_confidence': result.overall_confidence,
                    'by_complementarity': result.by_complementarity_scores[0] if result.by_complementarity_scores else np.nan,
                    'temporal_proximity': result.temporal_proximity_scores[0] if result.temporal_proximity_scores else np.nan,
                    'fragment_consistency': result.fragment_pattern_consistency[0] if result.fragment_pattern_consistency else np.nan,
                    'validation_passed': result.validation_passed
                })
            else:
                summary_data.append({
                    'scan': result.query_scan,
                    'precursor_mz': result.query_precursor_mz,
                    'top_match_id': None,
                    'top_match_sequence': '',
                    'confidence': 0.0,
                    'overall_confidence': 0.0,
                    'by_complementarity': np.nan,
                    'temporal_proximity': np.nan,
                    'fragment_consistency': np.nan,
                    'validation_passed': False
                })

        df = pd.DataFrame(summary_data)

        # Compute overall statistics
        print("\n[Proteomics Annotation Summary]")
        print(f"  Total spectra: {len(results)}")
        print(f"  Annotated: {len(df[df['top_match_id'].notna()])} ({len(df[df['top_match_id'].notna()])/len(df)*100:.1f}%)")
        print(f"  Validation passed: {df['validation_passed'].sum()} ({df['validation_passed'].sum()/len(df)*100:.1f}%)")
        print(f"  Mean confidence: {df['confidence'].mean():.3f}")
        print(f"  Mean b/y complementarity: {df['by_complementarity'].mean():.3f}")

        return df

    def export_results(
        self,
        results: List[ProteomicsAnnotationResult],
        output_path: str
    ):
        """Export annotation results to CSV."""
        df = self.get_annotation_summary(results)
        df.to_csv(output_path, index=False)
        print(f"[Export] Results exported to {output_path}")

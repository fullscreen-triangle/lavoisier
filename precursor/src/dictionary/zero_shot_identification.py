#!/usr/bin/env python3
"""
Zero-Shot Identification
========================

From st-stellas-dictionary.tex Algorithm 3:
Identify molecular entities via nearest-neighbor S-Entropy lookup
WITHOUT requiring a traditional sequence database.

Author: Kundai Sachikonye
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from core.EntropyTransformation import SEntropyCoordinates

from dictionary.sentropy_dictionary import SEntropyDictionary
from dictionary.dictionary_entry import DictionaryEntry


@dataclass
class IdentificationResult:
    """
    Result of zero-shot identification.

    Attributes:
        query_coords: Query S-Entropy coordinates
        query_mass: Query mass (if available)
        top_match: Best matching dictionary entry
        confidence: Identification confidence [0, 1]
        distance: S-Entropy distance to match
        all_candidates: List of (entry, distance, score) for top candidates
        is_novel: Whether this appears to be a novel entity
    """
    query_coords: SEntropyCoordinates
    query_mass: Optional[float]
    top_match: Optional[DictionaryEntry]
    confidence: float
    distance: float
    all_candidates: List[Tuple[DictionaryEntry, float, float]]
    is_novel: bool = False

    def __str__(self) -> str:
        if self.top_match:
            return (f"Identified as {self.top_match.symbol} ({self.top_match.name}) "
                   f"with confidence {self.confidence:.3f} (distance {self.distance:.3f})")
        else:
            return "No match found (novel entity?)"


class ZeroShotIdentifier:
    """
    Zero-shot molecular identification via S-Entropy dictionary lookup.

    From st-stellas-dictionary.tex:
    "Zero-shot identification requires NO prior sequence database.
    Entities are identified purely by their S-Entropy coordinates."

    This is the KEY innovation for database-free proteomics!
    """

    def __init__(
        self,
        dictionary: SEntropyDictionary,
        distance_threshold: float = 0.15,
        mass_tolerance: float = 0.02,  # Da
        novelty_threshold: float = 0.30
    ):
        """
        Initialize zero-shot identifier.

        Args:
            dictionary: S-Entropy dictionary to search
            distance_threshold: Maximum S-Entropy distance for positive ID
            mass_tolerance: Mass tolerance for validation (Da)
            novelty_threshold: Distance above which we call it "novel"
        """
        self.dictionary = dictionary
        self.distance_threshold = distance_threshold
        self.mass_tolerance = mass_tolerance
        self.novelty_threshold = novelty_threshold

    def identify(
        self,
        query_coords: SEntropyCoordinates,
        query_mass: Optional[float] = None,
        top_k: int = 5
    ) -> IdentificationResult:
        """
        Identify molecular entity via zero-shot S-Entropy lookup.

        From st-stellas-dictionary.tex Algorithm 3:

        1. Query S-Entropy dictionary via nearest neighbor
        2. Filter by mass if available
        3. Compute confidence score
        4. Determine if novel entity

        Args:
            query_coords: Query S-Entropy coordinates
            query_mass: Query mass (optional, for validation)
            top_k: Number of candidates to consider

        Returns:
            IdentificationResult with top match and confidence
        """
        # Step 1: Query dictionary
        candidates = self.dictionary.lookup(
            query_coords,
            k=top_k,
            max_distance=None  # Get all candidates initially
        )

        if len(candidates) == 0:
            # Empty dictionary
            return IdentificationResult(
                query_coords=query_coords,
                query_mass=query_mass,
                top_match=None,
                confidence=0.0,
                distance=np.inf,
                all_candidates=[],
                is_novel=True
            )

        # Step 2: Filter by mass if available
        if query_mass is not None:
            mass_filtered = []
            for entry, dist in candidates:
                mass_diff = abs(entry.mass - query_mass)
                if mass_diff <= self.mass_tolerance:
                    mass_filtered.append((entry, dist))

            if len(mass_filtered) > 0:
                candidates = mass_filtered

        # Step 3: Compute confidence scores
        scored_candidates = []
        for entry, dist in candidates:
            # Confidence based on:
            # 1. S-Entropy distance (lower = higher confidence)
            # 2. Entry confidence (learned entries have lower confidence)
            # 3. Mass agreement (if available)

            # S-Entropy distance score [0, 1]
            dist_score = np.exp(-dist / 0.1)  # Exponential decay

            # Entry confidence
            entry_conf = entry.confidence

            # Mass score
            mass_score = 1.0
            if query_mass is not None:
                mass_diff = abs(entry.mass - query_mass)
                mass_score = np.exp(-mass_diff / self.mass_tolerance)

            # Combined confidence
            confidence = dist_score * entry_conf * mass_score

            scored_candidates.append((entry, dist, confidence))

        # Sort by confidence
        scored_candidates.sort(key=lambda x: x[2], reverse=True)

        # Top match
        top_entry, top_dist, top_conf = scored_candidates[0]

        # Step 4: Determine if novel
        is_novel = top_dist > self.novelty_threshold

        result = IdentificationResult(
            query_coords=query_coords,
            query_mass=query_mass,
            top_match=top_entry,
            confidence=top_conf,
            distance=top_dist,
            all_candidates=scored_candidates,
            is_novel=is_novel
        )

        return result

    def identify_and_learn(
        self,
        query_coords: SEntropyCoordinates,
        query_mass: Optional[float] = None,
        auto_learn_novel: bool = True
    ) -> IdentificationResult:
        """
        Identify entity and optionally learn if novel.

        From st-stellas-dictionary.tex Section 4:
        "Dynamic dictionary learning enables discovery of novel PTMs
        and non-standard amino acids."

        Args:
            query_coords: Query S-Entropy coordinates
            query_mass: Query mass
            auto_learn_novel: Automatically add novel entities to dictionary

        Returns:
            IdentificationResult
        """
        # Identify
        result = self.identify(query_coords, query_mass)

        # If novel and auto-learning enabled, add to dictionary
        if result.is_novel and auto_learn_novel and query_mass is not None:
            # Create new entry
            novel_entry = self.dictionary.learn_novel_entry(
                coords=query_coords,
                mass=query_mass,
                confidence=0.5,  # Medium confidence for auto-learned
                name_prefix="Novel_AA"
            )

            print(f"[Zero-Shot] Learned novel entity: {novel_entry.symbol} "
                  f"at S-Entropy ({query_coords.s_knowledge:.3f}, "
                  f"{query_coords.s_time:.3f}, {query_coords.s_entropy:.3f})")

            # Update result
            result.top_match = novel_entry
            result.confidence = 0.5
            result.distance = 0.0

        return result

    def batch_identify(
        self,
        queries: List[Tuple[SEntropyCoordinates, Optional[float]]],
        auto_learn: bool = False
    ) -> List[IdentificationResult]:
        """
        Batch identification of multiple entities.

        Args:
            queries: List of (coords, mass) tuples
            auto_learn: Enable dynamic learning

        Returns:
            List of IdentificationResults
        """
        results = []

        for coords, mass in queries:
            if auto_learn:
                result = self.identify_and_learn(coords, mass, auto_learn_novel=True)
            else:
                result = self.identify(coords, mass)

            results.append(result)

        # Statistics
        novel_count = sum(1 for r in results if r.is_novel)
        high_conf_count = sum(1 for r in results if r.confidence > 0.8)

        print(f"[Zero-Shot Batch] Identified {len(results)} entities: "
              f"{high_conf_count} high confidence, {novel_count} novel")

        return results

#!/usr/bin/env python3
"""
S-Entropy Dictionary
===================

Complete molecular dictionary with dynamic learning capability.

From st-stellas-dictionary.tex Section 3:
"The dictionary is not static but dynamically grows as new molecular
entities are encountered and validated."

Author: Kundai Sachikonye
"""

import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.neighbors import KDTree
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from core.EntropyTransformation import SEntropyCoordinates

from dictionary.dictionary_entry import DictionaryEntry, EquivalenceClass
from molecular_language.amino_acid_alphabet import STANDARD_AMINO_ACIDS


class SEntropyDictionary:
    """
    Dynamic S-Entropy dictionary for molecular entities.

    Provides:
    1. Fast nearest-neighbor lookup via KD-tree
    2. Dynamic learning of novel molecules
    3. Equivalence class management
    4. Persistence (save/load)
    """

    def __init__(self):
        """Initialize empty dictionary."""
        self.entries: Dict[str, DictionaryEntry] = {}
        self.equivalence_classes: Dict[str, EquivalenceClass] = {}

        # KD-tree for fast lookups (rebuilt when dictionary changes)
        self.kdtree: Optional[KDTree] = None
        self.kdtree_symbols: List[str] = []
        self.kdtree_dirty = True

    def add_entry(self, entry: DictionaryEntry):
        """
        Add entry to dictionary.

        Args:
            entry: Dictionary entry to add
        """
        self.entries[entry.symbol] = entry
        self.kdtree_dirty = True

        # Assign to equivalence class if not already assigned
        if not entry.equivalence_class:
            entry.equivalence_class = self._find_equivalence_class(entry.s_entropy_coords)

        print(f"[Dictionary] Added {entry.symbol} ({entry.name}) - {entry.discovery_method}")

    def _find_equivalence_class(self, coords: SEntropyCoordinates) -> str:
        """Find or create equivalence class for coordinates."""
        # Check existing classes
        for class_id, eq_class in self.equivalence_classes.items():
            if eq_class.contains(coords):
                # Don't add coords to members - just count membership
                return class_id

        # Create new equivalence class
        new_class_id = f"EC_{len(self.equivalence_classes) + 1}"
        new_class = EquivalenceClass(
            class_id=new_class_id,
            centroid=coords,
            radius=0.2,  # Default radius
            members=[]  # Empty list - we track via dictionary entries instead
        )
        self.equivalence_classes[new_class_id] = new_class

        return new_class_id

    def _rebuild_kdtree(self):
        """Rebuild KD-tree from current entries."""
        if not self.kdtree_dirty:
            return

        if len(self.entries) == 0:
            self.kdtree = None
            self.kdtree_symbols = []
            return

        # Build coordinate matrix
        coords_list = []
        symbols = []

        for symbol, entry in self.entries.items():
            coords_list.append(entry.s_entropy_coords.to_array())
            symbols.append(symbol)

        coords_matrix = np.array(coords_list)

        # Build KD-tree
        self.kdtree = KDTree(coords_matrix)
        self.kdtree_symbols = symbols
        self.kdtree_dirty = False

        print(f"[Dictionary] Rebuilt KD-tree with {len(symbols)} entries")

    def lookup(
        self,
        query_coords: SEntropyCoordinates,
        k: int = 1,
        max_distance: Optional[float] = None
    ) -> List[Tuple[DictionaryEntry, float]]:
        """
        Look up nearest entries to query coordinates.

        From st-stellas-dictionary.tex Algorithm 3:
        Dictionary Lookup via S-Entropy

        Args:
            query_coords: Query S-Entropy coordinates
            k: Number of nearest neighbors
            max_distance: Maximum allowed distance (None = no limit)

        Returns:
            List of (entry, distance) tuples
        """
        self._rebuild_kdtree()

        if self.kdtree is None:
            return []

        query_array = query_coords.to_array().reshape(1, -1)

        # Query KD-tree
        distances, indices = self.kdtree.query(query_array, k=min(k, len(self.kdtree_symbols)))

        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if max_distance and dist > max_distance:
                continue

            symbol = self.kdtree_symbols[idx]
            entry = self.entries[symbol]
            results.append((entry, float(dist)))

        return results

    def learn_novel_entry(
        self,
        coords: SEntropyCoordinates,
        mass: float,
        confidence: float = 0.5,
        name_prefix: str = "Novel"
    ) -> DictionaryEntry:
        """
        Learn a novel molecular entity from experimental observation.

        From st-stellas-dictionary.tex Section 4:
        "Dynamic Dictionary Learning"

        Args:
            coords: Observed S-Entropy coordinates
            mass: Observed mass
            confidence: Confidence in observation
            name_prefix: Prefix for generated name

        Returns:
            Newly created dictionary entry
        """
        # Generate unique symbol
        novel_id = len([e for e in self.entries.values() if e.discovery_method == 'learned']) + 1
        symbol = f"X{novel_id}"
        name = f"{name_prefix}_{novel_id}"

        # Create entry
        entry = DictionaryEntry(
            symbol=symbol,
            name=name,
            mass=mass,
            s_entropy_coords=coords,
            fragmentation_rules=[],
            confidence=confidence,
            discovery_method='learned'
        )

        self.add_entry(entry)

        return entry

    def get_entry(self, symbol: str) -> Optional[DictionaryEntry]:
        """Get dictionary entry by symbol."""
        return self.entries.get(symbol)

    def save(self, filepath: str):
        """Save dictionary to JSON file."""
        data = {
            'entries': [entry.to_dict() for entry in self.entries.values()],
            'equivalence_classes': {
                class_id: {
                    'centroid': [ec.centroid.s_knowledge, ec.centroid.s_time, ec.centroid.s_entropy],
                    'radius': ec.radius,
                    'member_count': len(ec.members)
                }
                for class_id, ec in self.equivalence_classes.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[Dictionary] Saved to {filepath}")

    def load(self, filepath: str):
        """Load dictionary from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Load entries
        for entry_data in data['entries']:
            entry = DictionaryEntry.from_dict(entry_data)
            self.entries[entry.symbol] = entry

        self.kdtree_dirty = True

        print(f"[Dictionary] Loaded {len(self.entries)} entries from {filepath}")


def create_standard_proteomics_dictionary() -> SEntropyDictionary:
    """
    Create dictionary with standard 20 amino acids.

    This is the base dictionary that grows dynamically during analysis.

    Returns:
        SEntropyDictionary with standard amino acids
    """
    dictionary = SEntropyDictionary()

    # Add standard amino acids
    for symbol, aa in STANDARD_AMINO_ACIDS.items():
        entry = DictionaryEntry(
            symbol=symbol,
            name=aa.name,
            mass=aa.mass,
            s_entropy_coords=aa.s_entropy_coords,
            fragmentation_rules=[],
            metadata={
                'hydrophobicity': aa.hydrophobicity,
                'volume': aa.volume,
                'charge': aa.charge,
                'polarity': aa.polarity
            },
            discovery_method='standard'
        )
        dictionary.add_entry(entry)

    print(f"[Dictionary] Created standard proteomics dictionary with {len(STANDARD_AMINO_ACIDS)} amino acids")

    return dictionary

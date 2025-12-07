#!/usr/bin/env python3
"""
Dictionary Entry Structure
==========================

From st-stellas-dictionary.tex Section 2:
Each molecular entity is defined by:
- Symbol
- Name
- Mass
- S-Entropy coordinates
- Fragmentation rules
- Equivalence class

Author: Kundai Sachikonye
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from core.EntropyTransformation import SEntropyCoordinates


@dataclass
class EquivalenceClass:
    """
    Equivalence class for molecular entities.

    Groups entities with similar S-Entropy coordinates.
    From st-stellas-dictionary.tex Section 2.5.
    """
    class_id: str
    centroid: SEntropyCoordinates
    radius: float  # Maximum distance from centroid
    members: List[str] = field(default_factory=list)  # Changed to list since SEntropyCoordinates not hashable

    def contains(self, coords: SEntropyCoordinates, tolerance: float = 0.1) -> bool:
        """Check if coordinates fall within this equivalence class."""
        centroid_array = self.centroid.to_array()
        coords_array = coords.to_array()
        distance = np.linalg.norm(centroid_array - coords_array)
        return distance <= (self.radius + tolerance)


@dataclass
class DictionaryEntry:
    """
    Single entry in the S-Entropy dictionary.

    From st-stellas-dictionary.tex Section 2:
    Dictionary Entry Format:
    {
        "symbol": "K",
        "name": "Lysine",
        "mass": 128.095,
        "s_entropy": [0.123, 0.456, 0.789],
        "fragmentation_rules": ["R1", "R2"],
        "equivalence_class": "BASIC_POLAR"
    }
    """
    symbol: str
    name: str
    mass: float
    s_entropy_coords: SEntropyCoordinates
    fragmentation_rules: List[str] = field(default_factory=list)
    equivalence_class: Optional[str] = None

    # Additional metadata
    metadata: Dict = field(default_factory=dict)
    confidence: float = 1.0  # Confidence in this entry [0, 1]
    discovery_method: str = "standard"  # 'standard' or 'learned'

    def to_dict(self) -> dict:
        """Convert to dictionary format for export."""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'mass': self.mass,
            's_entropy': [
                self.s_entropy_coords.s_knowledge,
                self.s_entropy_coords.s_time,
                self.s_entropy_coords.s_entropy
            ],
            'fragmentation_rules': self.fragmentation_rules,
            'equivalence_class': self.equivalence_class,
            'metadata': self.metadata,
            'confidence': self.confidence,
            'discovery_method': self.discovery_method
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DictionaryEntry':
        """Create entry from dictionary."""
        s_coords = SEntropyCoordinates(
            s_knowledge=data['s_entropy'][0],
            s_time=data['s_entropy'][1],
            s_entropy=data['s_entropy'][2]
        )

        return cls(
            symbol=data['symbol'],
            name=data['name'],
            mass=data['mass'],
            s_entropy_coords=s_coords,
            fragmentation_rules=data.get('fragmentation_rules', []),
            equivalence_class=data.get('equivalence_class'),
            metadata=data.get('metadata', {}),
            confidence=data.get('confidence', 1.0),
            discovery_method=data.get('discovery_method', 'standard')
        )

    def distance_to(self, other_coords: SEntropyCoordinates) -> float:
        """Calculate S-Entropy distance to another coordinate."""
        self_array = self.s_entropy_coords.to_array()
        other_array = other_coords.to_array()
        return float(np.linalg.norm(self_array - other_array))

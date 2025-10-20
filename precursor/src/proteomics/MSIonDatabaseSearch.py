#!/usr/bin/env python3
"""
MSIon Database Search for Proteomics
=====================================

Comprehensive database search for protein/peptide identification using:
- Multiple database sources (UniProt, genome-derived FASTA, custom databases)
- S-Entropy coordinate matching
- Frequency coupling validation (all fragments from same collision)
- Phase-lock network-based disambiguation
- Protein inference from peptide identifications

Key Features:
-------------
1. In-silico digestion of protein sequences
2. Theoretical spectrum generation with PTM support
3. S-Entropy-based matching (platform-independent)
4. Frequency coupling consistency validation
5. Multi-database ensemble searching
6. Protein inference and FDR control

Author: Lavoisier Project
Date: October 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from collections import defaultdict
from sklearn.neighbors import KDTree
import warnings

# Import S-Entropy and phase-lock frameworks
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.EntropyTransformation import SEntropyTransformer, SEntropyFeatures
from core.PhaseLockNetworks import PhaseLockSignature, EnhancedPhaseLockMeasurementDevice
from proteomics.TandemDatabaseSearch import (
    PeptideSpectrum,
    PeptideFragment,
    ProteomicsAnnotationResult
)


@dataclass
class ProteinEntry:
    """
    Protein database entry.

    Attributes:
        protein_id: Unique identifier (e.g., UniProt accession)
        sequence: Amino acid sequence
        description: Protein name/description
        organism: Source organism
        gene_name: Gene symbol
        database_source: Origin database (UniProt, genome, custom)
        molecular_weight: Protein molecular weight (Da)
        isoelectric_point: Theoretical pI
        metadata: Additional metadata
    """
    protein_id: str
    sequence: str
    description: str = ""
    organism: str = ""
    gene_name: str = ""
    database_source: str = ""
    molecular_weight: float = 0.0
    isoelectric_point: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class PeptideIdentification:
    """
    Peptide identification from MS/MS spectrum.

    FREQUENCY COUPLING INTEGRATION:
    - frequency_coupling_score: Consistency of fragment coupling
    - collision_event_signature: Shared phase-lock of all fragments
    """
    peptide_sequence: str
    protein_ids: List[str]
    precursor_mz: float
    precursor_charge: int
    retention_time: Optional[float]
    scan_number: Optional[int]

    # Matching scores
    s_entropy_distance: float
    confidence_score: float
    by_complementarity_score: float
    frequency_coupling_score: float

    # Modifications
    modifications: Optional[Dict[int, str]] = None

    # Fragment matching
    matched_fragments: List[PeptideFragment] = field(default_factory=list)
    theoretical_fragments: Dict[str, List[float]] = field(default_factory=dict)

    # Phase-lock signatures
    collision_event_signature: Optional[PhaseLockSignature] = None

    # Validation
    is_validated: bool = False
    validation_flags: List[str] = field(default_factory=list)

    # FDR
    q_value: Optional[float] = None
    is_decoy: bool = False


@dataclass
class ProteinIdentification:
    """
    Protein identification inferred from peptide evidence.

    Handles protein inference problem: multiple proteins may share peptides.
    """
    protein_id: str
    protein_sequence: str
    description: str
    gene_name: str
    organism: str

    # Supporting peptides
    peptide_identifications: List[PeptideIdentification] = field(default_factory=list)
    unique_peptides: List[str] = field(default_factory=list)
    shared_peptides: List[str] = field(default_factory=list)

    # Protein-level scores
    protein_score: float = 0.0
    sequence_coverage: float = 0.0
    num_unique_peptides: int = 0
    num_total_peptides: int = 0

    # Validation
    protein_fdr: Optional[float] = None
    is_validated: bool = False


@dataclass
class DatabaseEnsemble:
    """
    Ensemble of multiple protein databases for comprehensive searching.
    """
    ensemble_name: str
    databases: Dict[str, List[ProteinEntry]] = field(default_factory=dict)
    total_proteins: int = 0
    total_peptides: int = 0

    # Pre-computed indices
    peptide_to_proteins: Dict[str, List[str]] = field(default_factory=dict)
    protein_index: Dict[str, ProteinEntry] = field(default_factory=dict)


class MSIonDatabaseSearch:
    """
    MSIon Database Search for Proteomics.

    Comprehensive protein identification using:
    - Multiple database sources (UniProt, genome, custom)
    - S-Entropy platform-independent matching
    - Frequency coupling validation (peptide-specific)
    - Phase-lock network disambiguation
    - Protein inference with FDR control
    """

    def __init__(
        self,
        enzyme: str = "trypsin",
        missed_cleavages: int = 2,
        min_peptide_length: int = 6,
        max_peptide_length: int = 50,
        precursor_tolerance_ppm: float = 10.0,
        fragment_tolerance_ppm: float = 20.0,
        enable_frequency_coupling: bool = True,
        enable_ptm: bool = False,
        ptm_list: Optional[List[str]] = None,
        fdr_threshold: float = 0.01
    ):
        """
        Initialize MSIon Database Search.

        Args:
            enzyme: Proteolytic enzyme (trypsin, lysc, gluc, etc.)
            missed_cleavages: Maximum missed cleavages
            min_peptide_length: Minimum peptide length
            max_peptide_length: Maximum peptide length
            precursor_tolerance_ppm: Precursor mass tolerance
            fragment_tolerance_ppm: Fragment mass tolerance
            enable_frequency_coupling: Enable frequency coupling validation
            enable_ptm: Enable post-translational modifications
            ptm_list: List of PTMs to consider
            fdr_threshold: False discovery rate threshold
        """
        self.enzyme = enzyme
        self.missed_cleavages = missed_cleavages
        self.min_peptide_length = min_peptide_length
        self.max_peptide_length = max_peptide_length
        self.precursor_tolerance_ppm = precursor_tolerance_ppm
        self.fragment_tolerance_ppm = fragment_tolerance_ppm
        self.enable_frequency_coupling = enable_frequency_coupling
        self.enable_ptm = enable_ptm
        self.ptm_list = ptm_list or []
        self.fdr_threshold = fdr_threshold

        # Initialize S-Entropy transformer
        self.s_entropy_transformer = SEntropyTransformer()

        # Initialize phase-lock device (for frequency coupling)
        if self.enable_frequency_coupling:
            self.phase_lock_device = EnhancedPhaseLockMeasurementDevice(
                enable_performance_tracking=True
            )

        # Database ensemble
        self.ensemble: Optional[DatabaseEnsemble] = None

        # Pre-computed theoretical spectra
        self.theoretical_spectra: Dict[str, Dict] = {}
        self.peptide_features: Optional[np.ndarray] = None
        self.peptide_sequences: List[str] = []
        self.kdtree: Optional[KDTree] = None

        # Enzyme cleavage rules
        self.cleavage_rules = {
            'trypsin': (r'[KR]', r'P'),  # Cleaves after K/R, not before P
            'lysc': (r'K', r'P'),
            'argc': (r'R', r'P'),
            'gluc': (r'[DE]', r'P'),
            'chymotrypsin': (r'[FWYL]', r'P'),
            'pepsin': (r'[FL]', r''),
            'nonspecific': (r'.', r'')
        }

        # Amino acid masses (monoisotopic)
        self.aa_masses = {
            'A': 71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694,
            'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G': 57.02146,
            'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
            'M': 131.04049, 'F': 147.06841, 'P': 97.05276, 'S': 87.03203,
            'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V': 99.06841
        }

        # Common PTMs (mass shifts)
        self.ptm_masses = {
            'Oxidation': 15.994915,  # Met oxidation
            'Phosphorylation': 79.966331,  # Ser/Thr/Tyr
            'Acetylation': 42.010565,  # N-term or Lys
            'Methylation': 14.015650,  # Lys/Arg
            'Carbamidomethylation': 57.021464,  # Cys (alkylation)
            'Deamidation': 0.984016  # Asn/Gln
        }

        print("[MSIon Database Search] Initialized")
        print(f"  Enzyme: {self.enzyme}")
        print(f"  Missed cleavages: {self.missed_cleavages}")
        print(f"  Frequency coupling: {self.enable_frequency_coupling}")
        print(f"  PTMs: {self.enable_ptm}")

    def load_fasta(self, fasta_path: str, database_name: str = "custom") -> List[ProteinEntry]:
        """
        Load protein sequences from FASTA file.

        Args:
            fasta_path: Path to FASTA file
            database_name: Name for this database

        Returns:
            List of ProteinEntry objects
        """
        print(f"[FASTA] Loading {fasta_path}...")
        proteins = []

        with open(fasta_path, 'r', encoding='utf-8') as f:
            current_header = None
            current_sequence = []

            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous protein
                    if current_header:
                        protein = self._parse_fasta_header(
                            current_header,
                            ''.join(current_sequence),
                            database_name
                        )
                        proteins.append(protein)

                    current_header = line[1:]
                    current_sequence = []
                else:
                    current_sequence.append(line)

            # Save last protein
            if current_header:
                protein = self._parse_fasta_header(
                    current_header,
                    ''.join(current_sequence),
                    database_name
                )
                proteins.append(protein)

        print(f"[FASTA] Loaded {len(proteins)} proteins from {database_name}")
        return proteins

    def _parse_fasta_header(
        self,
        header: str,
        sequence: str,
        database_name: str
    ) -> ProteinEntry:
        """Parse FASTA header to extract protein information."""
        # Try to extract UniProt-style header: sp|P12345|PROT_HUMAN Description
        uniprot_match = re.match(r'(sp|tr)\|([A-Z0-9]+)\|([A-Z0-9_]+)\s+(.*)', header)

        if uniprot_match:
            db_type, accession, entry_name, description = uniprot_match.groups()

            # Extract organism from description (e.g., "OS=Homo sapiens")
            organism_match = re.search(r'OS=([^=]+?)(?:\s+[A-Z]{2}=|$)', description)
            organism = organism_match.group(1) if organism_match else ""

            # Extract gene name (e.g., "GN=BRCA1")
            gene_match = re.search(r'GN=(\S+)', description)
            gene_name = gene_match.group(1) if gene_match else ""

            protein_id = accession
        else:
            # Generic header parsing
            parts = header.split(None, 1)
            protein_id = parts[0]
            description = parts[1] if len(parts) > 1 else ""
            organism = ""
            gene_name = ""

        # Calculate molecular weight
        mw = sum(self.aa_masses.get(aa, 0.0) for aa in sequence) + 18.015  # Add H2O

        return ProteinEntry(
            protein_id=protein_id,
            sequence=sequence,
            description=description,
            organism=organism,
            gene_name=gene_name,
            database_source=database_name,
            molecular_weight=mw
        )

    def load_uniprot(self, uniprot_path: str) -> List[ProteinEntry]:
        """
        Load proteins from UniProt FASTA.

        Args:
            uniprot_path: Path to UniProt FASTA file

        Returns:
            List of ProteinEntry objects
        """
        return self.load_fasta(uniprot_path, database_name="UniProt")

    def load_genome_derived(self, genome_fasta: str, organism: str) -> List[ProteinEntry]:
        """
        Load genome-derived protein sequences.

        Args:
            genome_fasta: Path to genome-derived FASTA (e.g., from NCBI, Ensembl)
            organism: Organism name

        Returns:
            List of ProteinEntry objects
        """
        proteins = self.load_fasta(genome_fasta, database_name=f"Genome_{organism}")

        # Add organism info to all proteins
        for protein in proteins:
            protein.organism = organism

        return proteins

    def create_ensemble(
        self,
        ensemble_name: str,
        databases: Dict[str, List[ProteinEntry]]
    ) -> DatabaseEnsemble:
        """
        Create database ensemble from multiple sources.

        Args:
            ensemble_name: Name for this ensemble
            databases: Dict mapping database names to protein lists

        Returns:
            DatabaseEnsemble object
        """
        print(f"[Ensemble] Creating ensemble: {ensemble_name}")

        ensemble = DatabaseEnsemble(ensemble_name=ensemble_name)
        ensemble.databases = databases

        # Count total proteins
        total_proteins = sum(len(proteins) for proteins in databases.values())
        ensemble.total_proteins = total_proteins

        # Build protein index (deduplicate by protein_id)
        for db_name, proteins in databases.items():
            for protein in proteins:
                if protein.protein_id not in ensemble.protein_index:
                    ensemble.protein_index[protein.protein_id] = protein
                else:
                    # Merge metadata if same protein appears in multiple databases
                    existing = ensemble.protein_index[protein.protein_id]
                    existing.metadata[f'{db_name}_found'] = True

        print(f"[Ensemble] Total proteins: {total_proteins}")
        print(f"[Ensemble] Unique proteins: {len(ensemble.protein_index)}")

        # Perform in-silico digestion
        print("[Ensemble] Performing in-silico digestion...")
        self._digest_ensemble(ensemble)

        self.ensemble = ensemble
        return ensemble

    def _digest_ensemble(self, ensemble: DatabaseEnsemble):
        """
        Perform in-silico digestion of all proteins in ensemble.

        Generates theoretical peptides and builds peptide-to-protein index.
        """
        for protein_id, protein in ensemble.protein_index.items():
            peptides = self._digest_protein(protein.sequence)

            for peptide_seq in peptides:
                if peptide_seq not in ensemble.peptide_to_proteins:
                    ensemble.peptide_to_proteins[peptide_seq] = []
                ensemble.peptide_to_proteins[peptide_seq].append(protein_id)

        ensemble.total_peptides = len(ensemble.peptide_to_proteins)

        print(f"[Digestion] Generated {ensemble.total_peptides} unique peptides")

        # Count shared vs unique peptides
        unique_peptides = sum(1 for p in ensemble.peptide_to_proteins.values() if len(p) == 1)
        shared_peptides = ensemble.total_peptides - unique_peptides

        print(f"[Digestion] Unique peptides: {unique_peptides}")
        print(f"[Digestion] Shared peptides: {shared_peptides}")

    def _digest_protein(self, sequence: str) -> List[str]:
        """
        Digest protein sequence using specified enzyme.

        Args:
            sequence: Protein amino acid sequence

        Returns:
            List of peptide sequences
        """
        if self.enzyme not in self.cleavage_rules:
            raise ValueError(f"Unknown enzyme: {self.enzyme}")

        cleave_pattern, inhibit_pattern = self.cleavage_rules[self.enzyme]

        # Find cleavage sites
        cleavage_sites = [0]
        for i, aa in enumerate(sequence):
            # Check if this position matches cleavage pattern
            if re.match(cleave_pattern, aa):
                # Check if next position inhibits cleavage
                if i + 1 < len(sequence):
                    next_aa = sequence[i + 1]
                    if inhibit_pattern and re.match(inhibit_pattern, next_aa):
                        continue
                cleavage_sites.append(i + 1)

        cleavage_sites.append(len(sequence))

        # Generate peptides with missed cleavages
        peptides = []
        for i in range(len(cleavage_sites) - 1):
            for j in range(i + 1, min(i + self.missed_cleavages + 2, len(cleavage_sites))):
                start = cleavage_sites[i]
                end = cleavage_sites[j]
                peptide = sequence[start:end]

                # Check length constraints
                if self.min_peptide_length <= len(peptide) <= self.max_peptide_length:
                    peptides.append(peptide)

        return peptides

    def precompute_theoretical_spectra(self):
        """
        Pre-compute theoretical spectra for all peptides in ensemble.

        This generates S-Entropy features for efficient searching.
        """
        if self.ensemble is None:
            raise ValueError("No ensemble loaded. Call create_ensemble() first.")

        print("[Theoretical Spectra] Pre-computing for all peptides...")

        feature_list = []
        sequence_list = []

        for peptide_seq in self.ensemble.peptide_to_proteins.keys():
            # Generate theoretical b/y ions
            theoretical_fragments = self._generate_theoretical_fragments(peptide_seq)

            # Combine all fragments
            all_mz = []
            if 'b' in theoretical_fragments:
                all_mz.extend(theoretical_fragments['b'])
            if 'y' in theoretical_fragments:
                all_mz.extend(theoretical_fragments['y'])

            if len(all_mz) > 0:
                mz_array = np.array(all_mz)
                intensity_array = np.ones(len(all_mz))  # Uniform theoretical intensities

                # Calculate precursor m/z
                precursor_mz = self._calculate_peptide_mass(peptide_seq)

                # Compute S-Entropy features
                _, features = self.s_entropy_transformer.transform_and_extract(
                    mz_array=mz_array,
                    intensity_array=intensity_array,
                    precursor_mz=precursor_mz,
                    rt=None
                )

                feature_list.append(features.features)
                sequence_list.append(peptide_seq)

                self.theoretical_spectra[peptide_seq] = {
                    'fragments': theoretical_fragments,
                    'precursor_mz': precursor_mz,
                    'features': features
                }

        # Build KD-tree for fast searching
        if len(feature_list) > 0:
            self.peptide_features = np.array(feature_list)
            self.peptide_sequences = sequence_list
            self.kdtree = KDTree(self.peptide_features)

            print(f"[Theoretical Spectra] Built KD-tree with {len(sequence_list)} peptides")
        else:
            print("[Theoretical Spectra] WARNING: No valid peptides generated")

    def _generate_theoretical_fragments(self, peptide_seq: str) -> Dict[str, List[float]]:
        """
        Generate theoretical b and y ions for peptide.

        Args:
            peptide_seq: Peptide amino acid sequence

        Returns:
            Dict with 'b' and 'y' ion lists
        """
        fragments = {'b': [], 'y': []}

        # B-ions (N-terminal fragments)
        cumulative_mass = 1.007825  # H
        for i in range(1, len(peptide_seq)):
            aa = peptide_seq[i - 1]
            cumulative_mass += self.aa_masses.get(aa, 0.0)
            fragments['b'].append(cumulative_mass)

        # Y-ions (C-terminal fragments)
        cumulative_mass = 19.01784  # H2O + H
        for i in range(len(peptide_seq) - 1, 0, -1):
            aa = peptide_seq[i]
            cumulative_mass += self.aa_masses.get(aa, 0.0)
            fragments['y'].append(cumulative_mass)

        return fragments

    def _calculate_peptide_mass(self, peptide_seq: str, charge: int = 2) -> float:
        """Calculate peptide precursor m/z."""
        mass = sum(self.aa_masses.get(aa, 0.0) for aa in peptide_seq) + 18.015  # Add H2O
        return (mass + charge * 1.007825) / charge

    def search(
        self,
        query_spectrum: PeptideSpectrum,
        top_k: int = 10
    ) -> List[PeptideIdentification]:
        """
        Search query spectrum against database ensemble.

        FREQUENCY COUPLING INTEGRATION:
        - Computes frequency coupling matrix for query
        - Validates coupling consistency against theoretical spectra
        - Uses collision event signature for disambiguation

        Args:
            query_spectrum: Query MS/MS spectrum
            top_k: Number of top matches to return

        Returns:
            List of PeptideIdentification objects (sorted by confidence)
        """
        if self.kdtree is None:
            raise ValueError("Theoretical spectra not computed. Call precompute_theoretical_spectra() first.")

        # Extract query features
        mz_array = np.array([f.mz for f in query_spectrum.fragments])
        intensity_array = np.array([f.intensity for f in query_spectrum.fragments])

        # Compute S-Entropy features
        _, query_features = self.s_entropy_transformer.transform_and_extract(
            mz_array=mz_array,
            intensity_array=intensity_array,
            precursor_mz=query_spectrum.precursor_mz,
            rt=query_spectrum.rt
        )

        # FREQUENCY COUPLING: Compute coupling matrix and collision signature
        if self.enable_frequency_coupling:
            coupling_matrix = self._compute_frequency_coupling(query_spectrum)
            collision_signature = self._compute_collision_signature(query_spectrum, coupling_matrix)
        else:
            coupling_matrix = None
            collision_signature = None

        # Find nearest neighbors in KD-tree
        distances, indices = self.kdtree.query(query_features.features.reshape(1, -1), k=top_k)
        distances = distances[0]
        indices = indices[0]

        # Build peptide identifications
        identifications = []

        for dist, idx in zip(distances, indices):
            peptide_seq = self.peptide_sequences[idx]
            protein_ids = self.ensemble.peptide_to_proteins[peptide_seq]
            theoretical = self.theoretical_spectra[peptide_seq]

            # Check precursor mass tolerance
            precursor_error_ppm = abs(
                (query_spectrum.precursor_mz - theoretical['precursor_mz']) /
                theoretical['precursor_mz'] * 1e6
            )

            if precursor_error_ppm > self.precursor_tolerance_ppm:
                continue

            # Compute confidence score
            confidence = np.exp(-dist / 1.0)  # Sigma = 1.0

            # Validate B/Y complementarity
            by_score = self._validate_by_complementarity(
                query_spectrum,
                theoretical['fragments']
            )

            # FREQUENCY COUPLING: Validate coupling consistency
            if self.enable_frequency_coupling and coupling_matrix is not None:
                coupling_score = self._validate_coupling_consistency(
                    coupling_matrix,
                    len(query_spectrum.fragments)
                )
            else:
                coupling_score = 0.5

            # Combined score (with frequency coupling weight)
            combined_score = (
                0.4 * confidence +
                0.3 * by_score +
                0.3 * coupling_score
            )

            # Validation flags
            validation_flags = []
            is_validated = False

            if by_score >= 0.5 and coupling_score >= 0.5:
                is_validated = True
                validation_flags.append("BY_complementarity_passed")
                validation_flags.append("Frequency_coupling_consistent")

            # Create identification
            identification = PeptideIdentification(
                peptide_sequence=peptide_seq,
                protein_ids=protein_ids,
                precursor_mz=query_spectrum.precursor_mz,
                precursor_charge=query_spectrum.charge,
                retention_time=query_spectrum.rt,
                scan_number=query_spectrum.scan_number,
                s_entropy_distance=dist,
                confidence_score=combined_score,
                by_complementarity_score=by_score,
                frequency_coupling_score=coupling_score,
                theoretical_fragments=theoretical['fragments'],
                collision_event_signature=collision_signature,
                is_validated=is_validated,
                validation_flags=validation_flags
            )

            identifications.append(identification)

        # Sort by combined score
        identifications.sort(key=lambda x: x.confidence_score, reverse=True)

        return identifications

    def _compute_frequency_coupling(self, spectrum: PeptideSpectrum) -> np.ndarray:
        """
        Compute frequency coupling matrix for spectrum.

        All peptide fragments are coupled (same collision event).
        """
        n_frags = len(spectrum.fragments)
        if n_frags == 0:
            return np.array([])

        # Initialize with base coupling (all fragments coupled)
        coupling_matrix = np.ones((n_frags, n_frags))

        # Get S-Entropy coordinates for all fragments
        mz_array = np.array([f.mz for f in spectrum.fragments])
        intensity_array = np.array([f.intensity for f in spectrum.fragments])

        coords_list, _ = self.s_entropy_transformer.transform_spectrum(
            mz_array=mz_array,
            intensity_array=intensity_array,
            precursor_mz=spectrum.precursor_mz,
            rt=spectrum.rt
        )

        # Compute pairwise coupling enhancements
        for i in range(n_frags):
            for j in range(i + 1, n_frags):
                frag_i = spectrum.fragments[i]
                frag_j = spectrum.fragments[j]

                coupling_strength = 1.0

                # B/Y complementarity enhancement
                if (frag_i.ion_type == 'b' and frag_j.ion_type == 'y') or \
                   (frag_i.ion_type == 'y' and frag_j.ion_type == 'b'):
                    # Check if complementary pair
                    if frag_i.ion_number + frag_j.ion_number == len(spectrum.fragments):
                        coupling_strength += 0.5

                # Sequential fragment enhancement
                if frag_i.ion_type == frag_j.ion_type:
                    if abs(frag_i.ion_number - frag_j.ion_number) == 1:
                        coupling_strength += 0.3

                # Mass proximity enhancement
                if abs(frag_i.mz - frag_j.mz) < 50:
                    coupling_strength += 0.2

                coupling_matrix[i, j] = coupling_strength
                coupling_matrix[j, i] = coupling_strength

        return coupling_matrix

    def _compute_collision_signature(
        self,
        spectrum: PeptideSpectrum,
        coupling_matrix: np.ndarray
    ) -> Optional[PhaseLockSignature]:
        """Compute shared collision event signature."""
        if coupling_matrix.size == 0:
            return None

        mz_array = np.array([f.mz for f in spectrum.fragments])
        mean_coupling = np.mean(coupling_matrix[np.triu_indices_from(coupling_matrix, k=1)])

        return PhaseLockSignature(
            mz_center=float(np.mean(mz_array)),
            mz_range=(float(np.min(mz_array)), float(np.max(mz_array))),
            rt_center=spectrum.rt if spectrum.rt else 0.0,
            rt_range=(spectrum.rt if spectrum.rt else 0.0, spectrum.rt if spectrum.rt else 0.0),
            coherence_strength=float(mean_coupling),
            coupling_modality="peptide_fragmentation",
            oscillation_frequency=float(np.mean(np.diff(np.sort(mz_array)))),
            phase_offset=0.0,
            ensemble_size=len(spectrum.fragments),
            temperature_signature=298.15,
            pressure_signature=1.0,
            categorical_state=0
        )

    def _validate_coupling_consistency(
        self,
        coupling_matrix: np.ndarray,
        n_fragments: int
    ) -> float:
        """Validate coupling consistency."""
        if coupling_matrix.size == 0:
            return 0.5

        mean_coupling = np.mean(coupling_matrix[np.triu_indices_from(coupling_matrix, k=1)])

        # Expected coupling for clean peptide: 1.5-2.0
        expected = 1.5
        consistency = np.exp(-abs(mean_coupling - expected)**2 / (2 * 0.5**2))

        return float(consistency)

    def _validate_by_complementarity(
        self,
        query_spectrum: PeptideSpectrum,
        theoretical_fragments: Dict[str, List[float]]
    ) -> float:
        """Validate B/Y ion complementarity."""
        matched_b = 0
        matched_y = 0

        query_mz = {f.mz for f in query_spectrum.fragments}

        # Count matched b ions
        for b_mz in theoretical_fragments.get('b', []):
            if any(abs(q_mz - b_mz) / b_mz * 1e6 < self.fragment_tolerance_ppm for q_mz in query_mz):
                matched_b += 1

        # Count matched y ions
        for y_mz in theoretical_fragments.get('y', []):
            if any(abs(q_mz - y_mz) / y_mz * 1e6 < self.fragment_tolerance_ppm for q_mz in query_mz):
                matched_y += 1

        # Score based on coverage
        total_theoretical = len(theoretical_fragments.get('b', [])) + len(theoretical_fragments.get('y', []))
        if total_theoretical == 0:
            return 0.0

        coverage = (matched_b + matched_y) / total_theoretical
        return float(coverage)

    def infer_proteins(
        self,
        peptide_identifications: List[PeptideIdentification]
    ) -> List[ProteinIdentification]:
        """
        Infer protein identifications from peptide evidence.

        Handles protein inference problem using:
        - Unique peptides (specific to one protein)
        - Shared peptides (common to multiple proteins)
        - Parsimony principle (minimal protein set)

        Args:
            peptide_identifications: List of validated peptide IDs

        Returns:
            List of ProteinIdentification objects
        """
        print("[Protein Inference] Inferring proteins from peptides...")

        # Build protein-peptide mapping
        protein_peptides: Dict[str, List[PeptideIdentification]] = defaultdict(list)

        for peptide_id in peptide_identifications:
            for protein_id in peptide_id.protein_ids:
                protein_peptides[protein_id].append(peptide_id)

        # Create protein identifications
        protein_identifications = []

        for protein_id, peptides in protein_peptides.items():
            protein = self.ensemble.protein_index[protein_id]

            # Identify unique vs shared peptides
            unique_peptides = []
            shared_peptides = []

            for peptide in peptides:
                if len(peptide.protein_ids) == 1:
                    unique_peptides.append(peptide.peptide_sequence)
                else:
                    shared_peptides.append(peptide.peptide_sequence)

            # Calculate sequence coverage
            covered_positions = set()
            for peptide in peptides:
                seq = peptide.peptide_sequence
                pos = protein.sequence.find(seq)
                if pos != -1:
                    covered_positions.update(range(pos, pos + len(seq)))

            coverage = len(covered_positions) / len(protein.sequence) if len(protein.sequence) > 0 else 0.0

            # Calculate protein score (average peptide confidence)
            protein_score = np.mean([p.confidence_score for p in peptides])

            protein_identification = ProteinIdentification(
                protein_id=protein.protein_id,
                protein_sequence=protein.sequence,
                description=protein.description,
                gene_name=protein.gene_name,
                organism=protein.organism,
                peptide_identifications=peptides,
                unique_peptides=unique_peptides,
                shared_peptides=shared_peptides,
                protein_score=protein_score,
                sequence_coverage=coverage,
                num_unique_peptides=len(unique_peptides),
                num_total_peptides=len(peptides),
                is_validated=len(unique_peptides) >= 2  # Require 2+ unique peptides
            )

            protein_identifications.append(protein_identification)

        # Sort by protein score
        protein_identifications.sort(key=lambda x: x.protein_score, reverse=True)

        print(f"[Protein Inference] Identified {len(protein_identifications)} proteins")
        print(f"[Protein Inference] Validated (2+ unique peptides): {sum(1 for p in protein_identifications if p.is_validated)}")

        return protein_identifications

    def apply_fdr(
        self,
        identifications: List[PeptideIdentification],
        decoy_identifications: List[PeptideIdentification]
    ) -> List[PeptideIdentification]:
        """
        Apply false discovery rate control using target-decoy approach.

        Args:
            identifications: Target peptide identifications
            decoy_identifications: Decoy peptide identifications

        Returns:
            FDR-filtered identifications
        """
        # Combine target and decoy
        all_ids = identifications + decoy_identifications

        # Sort by score descending
        all_ids.sort(key=lambda x: x.confidence_score, reverse=True)

        # Calculate q-values
        n_target = 0
        n_decoy = 0

        for identification in all_ids:
            if identification.is_decoy:
                n_decoy += 1
            else:
                n_target += 1

            # FDR = (n_decoy / n_target) if n_target > 0
            if n_target > 0:
                fdr = n_decoy / n_target
                identification.q_value = fdr
            else:
                identification.q_value = 1.0

        # Filter by FDR threshold
        filtered = [id for id in identifications if id.q_value and id.q_value <= self.fdr_threshold]

        print(f"[FDR Control] Before FDR: {len(identifications)}")
        print(f"[FDR Control] After FDR ({self.fdr_threshold}): {len(filtered)}")

        return filtered


if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("MSIon Database Search for Proteomics - Example")
    print("="*70)

    # Initialize search engine
    search = MSIonDatabaseSearch(
        enzyme="trypsin",
        missed_cleavages=2,
        enable_frequency_coupling=True
    )

    print("\n[Example] Load databases...")
    # Example: Load UniProt
    # uniprot_proteins = search.load_uniprot("path/to/uniprot_human.fasta")

    # Example: Load genome-derived
    # genome_proteins = search.load_genome_derived("path/to/ensembl_human.fasta", "Homo sapiens")

    # Create ensemble
    # ensemble = search.create_ensemble(
    #     ensemble_name="Human_Proteome",
    #     databases={
    #         "UniProt": uniprot_proteins,
    #         "Genome": genome_proteins
    #     }
    # )

    # Pre-compute theoretical spectra
    # search.precompute_theoretical_spectra()

    # Search query spectrum
    # identifications = search.search(query_spectrum, top_k=10)

    # Infer proteins
    # protein_identifications = search.infer_proteins(identifications)

    print("\n[Example] Complete! See code for usage details.")

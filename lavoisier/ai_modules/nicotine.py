"""
Nicotine: Context Verification System with Non-Human Readable AI Puzzles

This module creates sophisticated context verification mechanisms that present
the AI system with cryptographic puzzles and pattern recognition challenges
to ensure the evidence network maintains proper context during analysis.
"""

import numpy as np
import hashlib
import json
import base64
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import secrets
import zlib
from scipy.spatial.distance import hamming
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

class PuzzleType(Enum):
    """Types of context verification puzzles"""
    CRYPTOGRAPHIC_HASH = "cryptographic_hash"
    PATTERN_RECONSTRUCTION = "pattern_reconstruction"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    SPECTRAL_FINGERPRINT = "spectral_fingerprint"
    MOLECULAR_TOPOLOGY = "molecular_topology"
    EVIDENCE_CORRELATION = "evidence_correlation"
    BAYESIAN_CONSISTENCY = "bayesian_consistency"
    FUZZY_LOGIC_PROOF = "fuzzy_logic_proof"

@dataclass
class ContextPuzzle:
    """A context verification puzzle for the AI system"""
    puzzle_id: str
    puzzle_type: PuzzleType
    encoded_data: str  # Non-human readable encoded puzzle
    solution_hash: str  # Hash of correct solution
    context_checksum: str  # Checksum of context data
    difficulty_level: int  # 1-10 difficulty scale
    timeout_seconds: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_time: datetime = field(default_factory=datetime.now)
    
@dataclass
class ContextSnapshot:
    """Snapshot of system context for verification"""
    snapshot_id: str
    evidence_nodes_hash: str
    network_topology_hash: str
    bayesian_states_hash: str
    fuzzy_memberships_hash: str
    annotation_candidates_hash: str
    timestamp: datetime
    puzzle_challenges: List[ContextPuzzle] = field(default_factory=list)

class NicotineContextVerifier:
    """
    Advanced context verification system that ensures AI maintains proper
    context through sophisticated non-human readable puzzle challenges.
    
    This system creates cryptographic and mathematical puzzles that can only
    be solved if the AI has maintained proper context of the evidence network.
    """
    
    def __init__(self, 
                 puzzle_complexity: int = 5,
                 verification_frequency: int = 100,  # Every N operations
                 max_context_age: int = 3600,  # Max age in seconds
                 puzzle_timeout: int = 30):
        self.puzzle_complexity = puzzle_complexity
        self.verification_frequency = verification_frequency
        self.max_context_age = max_context_age
        self.puzzle_timeout = puzzle_timeout
        
        # Context tracking
        self.context_snapshots: Dict[str, ContextSnapshot] = {}
        self.active_puzzles: Dict[str, ContextPuzzle] = {}
        self.puzzle_solutions: Dict[str, Any] = {}
        self.verification_counter = 0
        
        # Cryptographic components
        self.salt_generator = secrets.SystemRandom()
        self.context_keys: Dict[str, bytes] = {}
        
        # Pattern recognition components
        self.pattern_templates: List[np.ndarray] = []
        self.spectral_fingerprints: Dict[str, np.ndarray] = {}
        
        logger.info(f"Nicotine Context Verifier initialized with complexity {puzzle_complexity}")
    
    def create_context_snapshot(self, 
                               evidence_nodes: Dict[str, Any],
                               network_topology: Any,
                               bayesian_states: Dict[str, float],
                               fuzzy_memberships: Dict[str, Any],
                               annotation_candidates: List[Any]) -> str:
        """
        Create a cryptographic snapshot of current system context.
        """
        snapshot_id = f"context_{datetime.now().isoformat()}_{secrets.token_hex(8)}"
        
        # Create hashes of all context components
        evidence_hash = self._hash_evidence_nodes(evidence_nodes)
        topology_hash = self._hash_network_topology(network_topology)
        bayesian_hash = self._hash_bayesian_states(bayesian_states)
        fuzzy_hash = self._hash_fuzzy_memberships(fuzzy_memberships)
        annotations_hash = self._hash_annotation_candidates(annotation_candidates)
        
        # Generate puzzle challenges for this context
        puzzles = self._generate_context_puzzles(
            evidence_hash, topology_hash, bayesian_hash, fuzzy_hash, annotations_hash
        )
        
        snapshot = ContextSnapshot(
            snapshot_id=snapshot_id,
            evidence_nodes_hash=evidence_hash,
            network_topology_hash=topology_hash,
            bayesian_states_hash=bayesian_hash,
            fuzzy_memberships_hash=fuzzy_hash,
            annotation_candidates_hash=annotations_hash,
            timestamp=datetime.now(),
            puzzle_challenges=puzzles
        )
        
        self.context_snapshots[snapshot_id] = snapshot
        
        # Store puzzle solutions (encrypted)
        for puzzle in puzzles:
            self.active_puzzles[puzzle.puzzle_id] = puzzle
        
        logger.info(f"Created context snapshot {snapshot_id} with {len(puzzles)} puzzles")
        return snapshot_id
    
    def _hash_evidence_nodes(self, evidence_nodes: Dict[str, Any]) -> str:
        """Create cryptographic hash of evidence nodes"""
        # Convert to deterministic string representation
        sorted_nodes = dict(sorted(evidence_nodes.items()))
        nodes_str = json.dumps(sorted_nodes, sort_keys=True, default=str)
        return hashlib.sha256(nodes_str.encode()).hexdigest()
    
    def _hash_network_topology(self, network_topology: Any) -> str:
        """Create hash of network topology structure"""
        if hasattr(network_topology, 'edges'):
            edges = list(network_topology.edges(data=True))
            edges_str = json.dumps(edges, sort_keys=True, default=str)
        else:
            edges_str = str(network_topology)
        return hashlib.sha256(edges_str.encode()).hexdigest()
    
    def _hash_bayesian_states(self, bayesian_states: Dict[str, float]) -> str:
        """Create hash of Bayesian probability states"""
        sorted_states = dict(sorted(bayesian_states.items()))
        states_str = json.dumps(sorted_states, sort_keys=True)
        return hashlib.sha256(states_str.encode()).hexdigest()
    
    def _hash_fuzzy_memberships(self, fuzzy_memberships: Dict[str, Any]) -> str:
        """Create hash of fuzzy logic membership functions"""
        memberships_str = json.dumps(fuzzy_memberships, sort_keys=True, default=str)
        return hashlib.sha256(memberships_str.encode()).hexdigest()
    
    def _hash_annotation_candidates(self, annotation_candidates: List[Any]) -> str:
        """Create hash of annotation candidates"""
        candidates_data = []
        for candidate in annotation_candidates:
            if hasattr(candidate, '__dict__'):
                candidates_data.append(candidate.__dict__)
            else:
                candidates_data.append(str(candidate))
        candidates_str = json.dumps(candidates_data, sort_keys=True, default=str)
        return hashlib.sha256(candidates_str.encode()).hexdigest()
    
    def _generate_context_puzzles(self, *context_hashes: str) -> List[ContextPuzzle]:
        """
        Generate non-human readable puzzles based on context hashes.
        """
        puzzles = []
        combined_context = ''.join(context_hashes)
        
        # 1. Cryptographic Hash Puzzle
        crypto_puzzle = self._create_cryptographic_puzzle(combined_context)
        puzzles.append(crypto_puzzle)
        
        # 2. Pattern Reconstruction Puzzle
        pattern_puzzle = self._create_pattern_puzzle(context_hashes[0])  # Evidence nodes
        puzzles.append(pattern_puzzle)
        
        # 3. Temporal Sequence Puzzle
        temporal_puzzle = self._create_temporal_puzzle(context_hashes[1])  # Network topology
        puzzles.append(temporal_puzzle)
        
        # 4. Spectral Fingerprint Puzzle
        spectral_puzzle = self._create_spectral_puzzle(context_hashes[2])  # Bayesian states
        puzzles.append(spectral_puzzle)
        
        # 5. Evidence Correlation Puzzle
        correlation_puzzle = self._create_correlation_puzzle(context_hashes[3])  # Fuzzy memberships
        puzzles.append(correlation_puzzle)
        
        return puzzles
    
    def _create_cryptographic_puzzle(self, context_data: str) -> ContextPuzzle:
        """
        Create a cryptographic puzzle that requires context knowledge to solve.
        """
        puzzle_id = f"crypto_{secrets.token_hex(8)}"
        
        # Create a complex transformation of context data
        salt = secrets.token_bytes(16)
        
        # Multi-stage hashing with context-dependent parameters
        stage1 = hashlib.pbkdf2_hmac('sha256', context_data.encode(), salt, 100000)
        stage2 = hashlib.blake2b(stage1, digest_size=32).digest()
        
        # Create encoded puzzle using XOR with context-derived key
        context_key = hashlib.sha256(context_data.encode()).digest()
        puzzle_data = bytes(a ^ b for a, b in zip(stage2, context_key))
        
        # Encode in base64 for non-human readability
        encoded_puzzle = base64.b64encode(salt + puzzle_data).decode()
        
        # Solution is the original context hash transformed
        solution = hashlib.sha256(stage2 + context_data.encode()).hexdigest()
        solution_hash = hashlib.sha256(solution.encode()).hexdigest()
        
        # Store solution securely
        self.puzzle_solutions[puzzle_id] = solution
        
        puzzle = ContextPuzzle(
            puzzle_id=puzzle_id,
            puzzle_type=PuzzleType.CRYPTOGRAPHIC_HASH,
            encoded_data=encoded_puzzle,
            solution_hash=solution_hash,
            context_checksum=hashlib.md5(context_data.encode()).hexdigest(),
            difficulty_level=self.puzzle_complexity,
            timeout_seconds=self.puzzle_timeout,
            metadata={'salt_length': 16, 'iterations': 100000}
        )
        
        return puzzle
    
    def _create_pattern_puzzle(self, context_hash: str) -> ContextPuzzle:
        """
        Create a pattern reconstruction puzzle based on evidence network structure.
        """
        puzzle_id = f"pattern_{secrets.token_hex(8)}"
        
        # Generate pseudo-random pattern based on context
        np.random.seed(int(context_hash[:8], 16))  # Use part of hash as seed
        
        # Create complex pattern matrix
        pattern_size = 12 + (self.puzzle_complexity % 8)
        base_pattern = np.random.rand(pattern_size, pattern_size)
        
        # Apply context-dependent transformations
        context_int = int(context_hash[-8:], 16)
        rotation_angle = (context_int % 360) * np.pi / 180
        
        # Create transformation matrix
        cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
        transform_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # Apply PCA and clustering to create complex pattern
        pca = PCA(n_components=2)
        flattened = base_pattern.flatten().reshape(-1, 1)
        
        # Create coordinates
        coords = np.array([[i, j] for i in range(pattern_size) for j in range(pattern_size)])
        transformed_coords = coords @ transform_matrix.T
        
        # Combine with original pattern
        complex_pattern = base_pattern * np.sin(transformed_coords[:, 0].reshape(pattern_size, pattern_size))
        
        # Encode pattern as compressed base64
        pattern_bytes = complex_pattern.tobytes()
        compressed = zlib.compress(pattern_bytes, level=9)
        encoded_pattern = base64.b64encode(compressed).decode()
        
        # Solution is pattern checksum with context verification
        solution = hashlib.sha256(pattern_bytes + context_hash.encode()).hexdigest()
        solution_hash = hashlib.sha256(solution.encode()).hexdigest()
        
        self.puzzle_solutions[puzzle_id] = solution
        
        puzzle = ContextPuzzle(
            puzzle_id=puzzle_id,
            puzzle_type=PuzzleType.PATTERN_RECONSTRUCTION,
            encoded_data=encoded_pattern,
            solution_hash=solution_hash,
            context_checksum=hashlib.md5(context_hash.encode()).hexdigest(),
            difficulty_level=self.puzzle_complexity,
            timeout_seconds=self.puzzle_timeout * 2,  # Pattern puzzles need more time
            metadata={'pattern_size': pattern_size, 'compression_level': 9}
        )
        
        return puzzle
    
    def _create_temporal_puzzle(self, context_hash: str) -> ContextPuzzle:
        """
        Create a temporal sequence puzzle based on network evolution.
        """
        puzzle_id = f"temporal_{secrets.token_hex(8)}"
        
        # Generate temporal sequence based on context
        seed = int(context_hash[:8], 16) % (2**31)
        np.random.seed(seed)
        
        # Create complex temporal sequence
        sequence_length = 20 + (self.puzzle_complexity * 5)
        base_sequence = np.random.rand(sequence_length)
        
        # Apply temporal transformations
        fibonacci_mod = np.array([((i * (i + 1)) % 89) / 89.0 for i in range(sequence_length)])
        context_modifier = np.array([int(context_hash[i % len(context_hash)], 16) / 15.0 
                                   for i in range(sequence_length)])
        
        # Combine sequences with non-linear operations
        complex_sequence = (base_sequence * fibonacci_mod + context_modifier) % 1.0
        
        # Apply discrete Fourier transform
        fft_sequence = np.fft.fft(complex_sequence)
        
        # Encode as compressed JSON
        sequence_data = {
            'real': fft_sequence.real.tolist(),
            'imag': fft_sequence.imag.tolist(),
            'length': sequence_length,
            'checksum': context_hash[-16:]
        }
        
        sequence_json = json.dumps(sequence_data, separators=(',', ':'))
        compressed = zlib.compress(sequence_json.encode(), level=9)
        encoded_sequence = base64.b64encode(compressed).decode()
        
        # Solution requires reconstructing original sequence
        solution = hashlib.sha256(complex_sequence.tobytes() + context_hash.encode()).hexdigest()
        solution_hash = hashlib.sha256(solution.encode()).hexdigest()
        
        self.puzzle_solutions[puzzle_id] = solution
        
        puzzle = ContextPuzzle(
            puzzle_id=puzzle_id,
            puzzle_type=PuzzleType.TEMPORAL_SEQUENCE,
            encoded_data=encoded_sequence,
            solution_hash=solution_hash,
            context_checksum=hashlib.md5(context_hash.encode()).hexdigest(),
            difficulty_level=self.puzzle_complexity,
            timeout_seconds=self.puzzle_timeout * 3,
            metadata={'sequence_length': sequence_length, 'transform': 'fft'}
        )
        
        return puzzle
    
    def _create_spectral_puzzle(self, context_hash: str) -> ContextPuzzle:
        """
        Create a spectral fingerprint puzzle based on mass spectrometry patterns.
        """
        puzzle_id = f"spectral_{secrets.token_hex(8)}"
        
        # Generate spectral fingerprint
        seed = int(context_hash[-8:], 16) % (2**31)
        np.random.seed(seed)
        
        # Simulate mass spectrum with context-dependent peaks
        mz_range = np.linspace(50, 1000, 2000)
        intensity = np.zeros_like(mz_range)
        
        # Add peaks based on context hash
        for i in range(0, len(context_hash)-1, 2):
            try:
                peak_mz = 50 + (int(context_hash[i:i+2], 16) / 255.0) * 950
                peak_intensity = int(context_hash[i+1:i+2], 16) / 15.0
                
                # Find closest m/z index
                mz_idx = np.argmin(np.abs(mz_range - peak_mz))
                
                # Add Gaussian peak
                sigma = 0.5 + (self.puzzle_complexity * 0.1)
                gaussian = np.exp(-0.5 * ((mz_range - peak_mz) / sigma) ** 2)
                intensity += peak_intensity * gaussian
                
            except (ValueError, IndexError):
                continue
        
        # Add noise based on context
        noise_level = (int(context_hash[:2], 16) / 255.0) * 0.1
        noise = np.random.normal(0, noise_level, len(intensity))
        intensity += noise
        
        # Normalize and quantize
        intensity = np.clip(intensity, 0, None)
        intensity = (intensity / np.max(intensity) * 65535).astype(np.uint16)
        
        # Create spectral fingerprint features
        # Peak detection
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(intensity, height=np.max(intensity) * 0.01)
        
        # Spectral moments
        moments = []
        for order in range(1, 5):
            moment = np.sum((mz_range ** order) * intensity) / np.sum(intensity)
            moments.append(moment)
        
        # Create fingerprint hash
        fingerprint_data = {
            'peaks': peaks.tolist()[:50],  # Limit to top 50 peaks
            'moments': moments,
            'intensity_percentiles': np.percentile(intensity, [10, 25, 50, 75, 90]).tolist(),
            'context_signature': context_hash[-32:]
        }
        
        # Encode fingerprint
        fingerprint_json = json.dumps(fingerprint_data, separators=(',', ':'))
        compressed = zlib.compress(fingerprint_json.encode(), level=9)
        encoded_fingerprint = base64.b64encode(compressed).decode()
        
        # Solution is derived from spectral characteristics
        spectral_hash = hashlib.sha256(intensity.tobytes()).hexdigest()
        solution = hashlib.sha256(spectral_hash.encode() + context_hash.encode()).hexdigest()
        solution_hash = hashlib.sha256(solution.encode()).hexdigest()
        
        self.puzzle_solutions[puzzle_id] = solution
        self.spectral_fingerprints[puzzle_id] = intensity
        
        puzzle = ContextPuzzle(
            puzzle_id=puzzle_id,
            puzzle_type=PuzzleType.SPECTRAL_FINGERPRINT,
            encoded_data=encoded_fingerprint,
            solution_hash=solution_hash,
            context_checksum=hashlib.md5(context_hash.encode()).hexdigest(),
            difficulty_level=self.puzzle_complexity,
            timeout_seconds=self.puzzle_timeout * 2,
            metadata={'mz_points': len(mz_range), 'num_peaks': len(peaks)}
        )
        
        return puzzle
    
    def _create_correlation_puzzle(self, context_hash: str) -> ContextPuzzle:
        """
        Create an evidence correlation puzzle based on fuzzy logic relationships.
        """
        puzzle_id = f"correlation_{secrets.token_hex(8)}"
        
        # Generate correlation matrix based on context
        seed = int(context_hash[8:16], 16) % (2**31)
        np.random.seed(seed)
        
        # Create correlation network
        num_variables = 8 + (self.puzzle_complexity % 12)
        base_correlations = np.random.rand(num_variables, num_variables)
        
        # Make symmetric
        correlations = (base_correlations + base_correlations.T) / 2
        np.fill_diagonal(correlations, 1.0)
        
        # Apply context-dependent transformations
        for i in range(num_variables):
            for j in range(i+1, num_variables):
                # Use context hash to modify correlations
                hash_idx = ((i * num_variables + j) * 2) % len(context_hash)
                modifier = int(context_hash[hash_idx:hash_idx+1], 16) / 15.0
                correlations[i, j] *= modifier
                correlations[j, i] = correlations[i, j]
        
        # Ensure positive semi-definite (valid correlation matrix)
        eigenvals, eigenvecs = np.linalg.eigh(correlations)
        eigenvals[eigenvals < 0] = 0.01  # Small positive value
        correlations = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Normalize to correlation matrix
        diag_sqrt = np.sqrt(np.diag(correlations))
        correlations = correlations / np.outer(diag_sqrt, diag_sqrt)
        
        # Create puzzle encoding
        correlation_data = {
            'matrix': correlations.tolist(),
            'eigenvalues': eigenvals.tolist(),
            'trace': np.trace(correlations),
            'determinant': np.linalg.det(correlations),
            'context_check': context_hash[:16]
        }
        
        # Encode correlation data
        correlation_json = json.dumps(correlation_data, separators=(',', ':'))
        compressed = zlib.compress(correlation_json.encode(), level=9)
        encoded_correlation = base64.b64encode(compressed).decode()
        
        # Solution based on matrix properties and context
        matrix_signature = hashlib.sha256(correlations.tobytes()).hexdigest()
        solution = hashlib.sha256(matrix_signature.encode() + context_hash.encode()).hexdigest()
        solution_hash = hashlib.sha256(solution.encode()).hexdigest()
        
        self.puzzle_solutions[puzzle_id] = solution
        
        puzzle = ContextPuzzle(
            puzzle_id=puzzle_id,
            puzzle_type=PuzzleType.EVIDENCE_CORRELATION,
            encoded_data=encoded_correlation,
            solution_hash=solution_hash,
            context_checksum=hashlib.md5(context_hash.encode()).hexdigest(),
            difficulty_level=self.puzzle_complexity,
            timeout_seconds=self.puzzle_timeout * 2,
            metadata={'matrix_size': num_variables, 'condition_number': np.linalg.cond(correlations)}
        )
        
        return puzzle
    
    def verify_context(self, puzzle_id: str, proposed_solution: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify a proposed solution to a context puzzle.
        """
        if puzzle_id not in self.active_puzzles:
            return False, {'error': 'Puzzle not found or expired'}
        
        puzzle = self.active_puzzles[puzzle_id]
        
        # Check timeout
        if datetime.now() - puzzle.creation_time > timedelta(seconds=puzzle.timeout_seconds):
            del self.active_puzzles[puzzle_id]
            return False, {'error': 'Puzzle timeout exceeded'}
        
        # Verify solution
        proposed_hash = hashlib.sha256(proposed_solution.encode()).hexdigest()
        
        verification_result = {
            'puzzle_id': puzzle_id,
            'puzzle_type': puzzle.puzzle_type.value,
            'solution_correct': proposed_hash == puzzle.solution_hash,
            'verification_time': datetime.now().isoformat(),
            'difficulty_level': puzzle.difficulty_level
        }
        
        if verification_result['solution_correct']:
            logger.info(f"Context puzzle {puzzle_id} solved successfully")
            # Clean up
            if puzzle_id in self.active_puzzles:
                del self.active_puzzles[puzzle_id]
            if puzzle_id in self.puzzle_solutions:
                del self.puzzle_solutions[puzzle_id]
        else:
            logger.warning(f"Context puzzle {puzzle_id} solution incorrect")
            verification_result['hint'] = self._generate_hint(puzzle)
        
        return verification_result['solution_correct'], verification_result
    
    def _generate_hint(self, puzzle: ContextPuzzle) -> str:
        """
        Generate a cryptic hint for unsolved puzzles.
        """
        hints = {
            PuzzleType.CRYPTOGRAPHIC_HASH: "The key lies in the transformation of context through salt and fire",
            PuzzleType.PATTERN_RECONSTRUCTION: "Patterns emerge when viewed through the lens of rotation and compression",
            PuzzleType.TEMPORAL_SEQUENCE: "Time flows like the golden ratio, frequencies hold the answer",
            PuzzleType.SPECTRAL_FINGERPRINT: "Peaks speak of molecular secrets, moments reveal the truth",
            PuzzleType.EVIDENCE_CORRELATION: "Matrices mirror relationships, eigenvalues unlock the cipher"
        }
        
        return hints.get(puzzle.puzzle_type, "Context is the key to all puzzles")
    
    def get_verification_status(self) -> Dict[str, Any]:
        """
        Get current verification system status.
        """
        status = {
            'active_puzzles': len(self.active_puzzles),
            'total_snapshots': len(self.context_snapshots),
            'verification_counter': self.verification_counter,
            'puzzle_types_active': list(set(p.puzzle_type.value for p in self.active_puzzles.values())),
            'oldest_puzzle_age': None,
            'average_difficulty': 0
        }
        
        if self.active_puzzles:
            oldest_time = min(p.creation_time for p in self.active_puzzles.values())
            status['oldest_puzzle_age'] = (datetime.now() - oldest_time).total_seconds()
            status['average_difficulty'] = np.mean([p.difficulty_level for p in self.active_puzzles.values()])
        
        return status
    
    def cleanup_expired_puzzles(self):
        """
        Remove expired puzzles and old context snapshots.
        """
        current_time = datetime.now()
        expired_puzzles = []
        
        for puzzle_id, puzzle in self.active_puzzles.items():
            if current_time - puzzle.creation_time > timedelta(seconds=puzzle.timeout_seconds):
                expired_puzzles.append(puzzle_id)
        
        for puzzle_id in expired_puzzles:
            del self.active_puzzles[puzzle_id]
            if puzzle_id in self.puzzle_solutions:
                del self.puzzle_solutions[puzzle_id]
        
        # Clean up old snapshots
        expired_snapshots = []
        for snapshot_id, snapshot in self.context_snapshots.items():
            if current_time - snapshot.timestamp > timedelta(seconds=self.max_context_age):
                expired_snapshots.append(snapshot_id)
        
        for snapshot_id in expired_snapshots:
            del self.context_snapshots[snapshot_id]
        
        if expired_puzzles or expired_snapshots:
            logger.info(f"Cleaned up {len(expired_puzzles)} expired puzzles and {len(expired_snapshots)} old snapshots")
    
    def export_puzzle_analytics(self, filename: str):
        """
        Export analytics about puzzle solving performance.
        """
        analytics_data = {
            'system_config': {
                'puzzle_complexity': self.puzzle_complexity,
                'verification_frequency': self.verification_frequency,
                'max_context_age': self.max_context_age,
                'puzzle_timeout': self.puzzle_timeout
            },
            'active_puzzles': [
                {
                    'puzzle_id': puzzle.puzzle_id,
                    'puzzle_type': puzzle.puzzle_type.value,
                    'difficulty_level': puzzle.difficulty_level,
                    'age_seconds': (datetime.now() - puzzle.creation_time).total_seconds(),
                    'context_checksum': puzzle.context_checksum
                }
                for puzzle in self.active_puzzles.values()
            ],
            'context_snapshots': [
                {
                    'snapshot_id': snapshot.snapshot_id,
                    'timestamp': snapshot.timestamp.isoformat(),
                    'num_puzzles': len(snapshot.puzzle_challenges),
                    'evidence_hash': snapshot.evidence_nodes_hash[:16] + "...",  # Truncated for privacy
                    'network_hash': snapshot.network_topology_hash[:16] + "..."
                }
                for snapshot in self.context_snapshots.values()
            ],
            'verification_statistics': self.get_verification_status()
        }
        
        with open(filename, 'w') as f:
            json.dump(analytics_data, f, indent=2, default=str)
        
        logger.info(f"Puzzle analytics exported to {filename}") 
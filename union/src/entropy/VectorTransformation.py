#!/usr/bin/env python3
"""
Vector Transformation for Mass Spectrometry Spectra
====================================================

Converts S-Entropy coordinates and features into vector embeddings for:
- Spectral similarity search
- LLM-style comparison
- Database annotation
- Dual-modality (numerical + visual) analysis

CRITICAL DESIGN PRINCIPLE:
--------------------------
ALL vector representations are derived FROM S-Entropy coordinates.
This ensures:
1. Platform independence
2. Theoretical consistency with phase-lock theory
3. Categorical state preservation
4. Bijective information preservation

The transformation pipeline:
Spectrum → S-Entropy (3D) → 14D Features → Vector Embedding

Author: Kundai Sachikonye
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine, euclidean

# Import S-Entropy framework
from .EntropyTransformation import (
    SEntropyTransformer,
    SEntropyCoordinates,
    SEntropyFeatures,
    PhaseLockSignatureComputer
)


@dataclass
class SpectrumEmbedding:
    """
    Vector embedding for a spectrum.

    Attributes:
        embedding: Vector representation (variable dimension)
        s_entropy_features: Source 14D S-Entropy features
        phase_lock_signature: Phase-lock signature (64D)
        categorical_state: Categorical completion state
        embedding_method: Method used for embedding
        metadata: Additional metadata
    """
    embedding: np.ndarray
    s_entropy_features: SEntropyFeatures
    phase_lock_signature: np.ndarray
    categorical_state: int
    embedding_method: str
    metadata: Optional[Dict] = None

    def similarity_to(
        self,
        other: 'SpectrumEmbedding',
        metric: str = 'cosine'
    ) -> float:
        """
        Calculate similarity to another embedding.

        Args:
            other: Another SpectrumEmbedding
            metric: Similarity metric ('cosine', 'euclidean', 'phase_lock')

        Returns:
            Similarity score [0, 1]
        """
        if metric == 'cosine':
            return 1.0 - cosine(self.embedding, other.embedding)
        elif metric == 'euclidean':
            dist = euclidean(self.embedding, other.embedding)
            return 1.0 / (1.0 + dist)
        elif metric == 'phase_lock':
            return 1.0 - cosine(self.phase_lock_signature, other.phase_lock_signature)
        elif metric == 'dual':
            # Dual-modality: combine embedding and phase-lock
            emb_sim = 1.0 - cosine(self.embedding, other.embedding)
            phase_sim = 1.0 - cosine(self.phase_lock_signature, other.phase_lock_signature)
            return 0.7 * emb_sim + 0.3 * phase_sim
        else:
            raise ValueError(f"Unknown metric: {metric}")


class VectorTransformer:
    """
    Transform spectra into vector embeddings via S-Entropy coordinates.

    Supports multiple embedding methods:
    1. Direct S-Entropy (14D features as embedding)
    2. Enhanced S-Entropy (expanded to higher dimension)
    3. Spec2Vec-like (but derived from S-Entropy)
    4. LLM-style (contextualized embeddings)
    """

    def __init__(
        self,
        embedding_method: str = 'enhanced_entropy',
        embedding_dim: int = 256,
        normalize: bool = True,
        include_phase_lock: bool = True
    ):
        """
        Initialize vector transformer.

        Args:
            embedding_method: Method for embedding
                - 'direct_entropy': Use 14D S-Entropy features directly
                - 'enhanced_entropy': Expand to higher dimension (default)
                - 'spec2vec_style': Spec2Vec-like from S-Entropy
                - 'llm_style': LLM-contextualized embedding
            embedding_dim: Target embedding dimension (default: 256)
            normalize: Whether to L2-normalize embeddings (default: True)
            include_phase_lock: Include phase-lock signature (default: True)
        """
        self.embedding_method = embedding_method
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        self.include_phase_lock = include_phase_lock

        # Initialize S-Entropy transformer
        self.s_entropy_transformer = SEntropyTransformer()

        # Initialize phase-lock computer
        if include_phase_lock:
            self.phase_lock_computer = PhaseLockSignatureComputer(signature_dim=64)
        else:
            self.phase_lock_computer = None

        # Scaler for normalization
        self.scaler = StandardScaler()
        self.scaler_fitted = False

    def transform_spectrum(
        self,
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        precursor_mz: Optional[float] = None,
        rt: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> SpectrumEmbedding:
        """
        Transform a single spectrum to vector embedding.

        Args:
            mz_array: Array of m/z values
            intensity_array: Array of intensity values
            precursor_mz: Precursor m/z (optional)
            rt: Retention time (optional)
            metadata: Additional metadata (optional)

        Returns:
            SpectrumEmbedding
        """
        # Step 1: Transform to S-Entropy coordinates and extract features
        coords_list, features = self.s_entropy_transformer.transform_and_extract(
            mz_array, intensity_array, precursor_mz, rt
        )

        # Step 2: Compute phase-lock signature
        if self.include_phase_lock:
            _, coord_matrix = self.s_entropy_transformer.transform_spectrum(
                mz_array, intensity_array, precursor_mz, rt
            )
            phase_lock_sig = self.phase_lock_computer.compute_signature(
                coords_list, coord_matrix, features
            )
            categorical_state = self.phase_lock_computer.compute_categorical_state(
                phase_lock_sig
            )
        else:
            phase_lock_sig = np.array([])
            categorical_state = 0

        # Step 3: Generate embedding based on method
        embedding = self._generate_embedding(features, coords_list, phase_lock_sig)

        # Step 4: Normalize if requested
        if self.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return SpectrumEmbedding(
            embedding=embedding,
            s_entropy_features=features,
            phase_lock_signature=phase_lock_sig,
            categorical_state=categorical_state,
            embedding_method=self.embedding_method,
            metadata=metadata
        )

    def _generate_embedding(
        self,
        features: SEntropyFeatures,
        coords_list: List[SEntropyCoordinates],
        phase_lock_sig: np.ndarray
    ) -> np.ndarray:
        """
        Generate embedding based on selected method.

        Args:
            features: 14D S-Entropy features
            coords_list: List of S-Entropy coordinates
            phase_lock_sig: Phase-lock signature

        Returns:
            Embedding vector
        """
        if self.embedding_method == 'direct_entropy':
            return self._direct_entropy_embedding(features)
        elif self.embedding_method == 'enhanced_entropy':
            return self._enhanced_entropy_embedding(features, coords_list, phase_lock_sig)
        elif self.embedding_method == 'spec2vec_style':
            return self._spec2vec_style_embedding(features, coords_list)
        elif self.embedding_method == 'llm_style':
            return self._llm_style_embedding(features, coords_list, phase_lock_sig)
        else:
            raise ValueError(f"Unknown embedding method: {self.embedding_method}")

    def _direct_entropy_embedding(
        self,
        features: SEntropyFeatures
    ) -> np.ndarray:
        """
        Direct S-Entropy embedding: Use 14D features as-is.

        This is the simplest method, directly using the bijective
        14D feature space.
        """
        embedding = features.to_array()

        # Pad to target dimension if needed
        if len(embedding) < self.embedding_dim:
            padded = np.zeros(self.embedding_dim)
            padded[:len(embedding)] = embedding
            return padded
        else:
            return embedding[:self.embedding_dim]

    def _enhanced_entropy_embedding(
        self,
        features: SEntropyFeatures,
        coords_list: List[SEntropyCoordinates],
        phase_lock_sig: np.ndarray
    ) -> np.ndarray:
        """
        Enhanced S-Entropy embedding: Expand to higher dimension.

        Combines:
        - 14D S-Entropy features
        - Coordinate distribution statistics
        - Phase-lock signature components
        - Derived geometric properties
        """
        components = []

        # 1. Base 14D features
        components.append(features.to_array())

        # 2. Coordinate distribution moments
        if len(coords_list) > 0:
            coord_array = np.array([c.to_array() for c in coords_list])

            # Moments for each dimension
            for dim in range(3):
                dim_values = coord_array[:, dim]
                components.append(np.array([
                    np.mean(dim_values),
                    np.std(dim_values),
                    np.min(dim_values),
                    np.max(dim_values),
                    np.median(dim_values)
                ]))

            # 3. Inter-dimensional correlations
            if len(coord_array) > 1:
                corr_01 = np.corrcoef(coord_array[:, 0], coord_array[:, 1])[0, 1]
                corr_02 = np.corrcoef(coord_array[:, 0], coord_array[:, 2])[0, 1]
                corr_12 = np.corrcoef(coord_array[:, 1], coord_array[:, 2])[0, 1]
                components.append(np.array([corr_01, corr_02, corr_12]))
            else:
                components.append(np.zeros(3))

            # 4. Geometric properties
            magnitudes = np.array([c.magnitude() for c in coords_list])
            components.append(np.array([
                np.percentile(magnitudes, 25),
                np.percentile(magnitudes, 50),
                np.percentile(magnitudes, 75),
                np.percentile(magnitudes, 90),
                np.percentile(magnitudes, 95)
            ]))
        else:
            # Empty spectrum
            components.extend([
                np.zeros(15),  # Moments
                np.zeros(3),   # Correlations
                np.zeros(5)    # Percentiles
            ])

        # 5. Phase-lock components (if available)
        if len(phase_lock_sig) > 0:
            # Take first N components
            n_phase = min(32, len(phase_lock_sig))
            components.append(phase_lock_sig[:n_phase])

        # Concatenate all components
        embedding = np.concatenate(components)

        # Project to target dimension
        if len(embedding) < self.embedding_dim:
            # Pad with zeros
            padded = np.zeros(self.embedding_dim)
            padded[:len(embedding)] = embedding
            return padded
        elif len(embedding) > self.embedding_dim:
            # Average pooling to reduce dimension
            chunk_size = len(embedding) / self.embedding_dim
            reduced = np.zeros(self.embedding_dim)
            for i in range(self.embedding_dim):
                start_idx = int(i * chunk_size)
                end_idx = int((i + 1) * chunk_size)
                reduced[i] = embedding[start_idx:end_idx].mean()
            return reduced
        else:
            return embedding

    def _spec2vec_style_embedding(
        self,
        features: SEntropyFeatures,
        coords_list: List[SEntropyCoordinates]
    ) -> np.ndarray:
        """
        Spec2Vec-style embedding: Word2Vec analogy for spectra.

        Instead of using peak m/z as "words", we use S-Entropy coordinates
        as the atomic units, maintaining platform independence.

        This is a simplified version - full Spec2Vec would require training
        on a corpus of spectra.
        """
        # Start with base features
        embedding = features.to_array()

        # Add coordinate histogram (discretize 3D space)
        if len(coords_list) > 0:
            coord_array = np.array([c.to_array() for c in coords_list])

            # Create 3D histogram (5 bins per dimension = 125 bins total)
            n_bins = 5
            hist_components = []

            for dim in range(3):
                dim_values = coord_array[:, dim]
                hist, _ = np.histogram(dim_values, bins=n_bins, density=True)
                hist_components.append(hist)

            hist_flat = np.concatenate(hist_components)  # 15 dimensions
            embedding = np.concatenate([embedding, hist_flat])
        else:
            embedding = np.concatenate([embedding, np.zeros(15)])

        # Project to target dimension
        if len(embedding) < self.embedding_dim:
            padded = np.zeros(self.embedding_dim)
            padded[:len(embedding)] = embedding
            return padded
        elif len(embedding) > self.embedding_dim:
            # PCA-style reduction (simplified)
            chunk_size = len(embedding) / self.embedding_dim
            reduced = np.zeros(self.embedding_dim)
            for i in range(self.embedding_dim):
                start_idx = int(i * chunk_size)
                end_idx = int((i + 1) * chunk_size)
                reduced[i] = embedding[start_idx:end_idx].mean()
            return reduced
        else:
            return embedding

    def _llm_style_embedding(
        self,
        features: SEntropyFeatures,
        coords_list: List[SEntropyCoordinates],
        phase_lock_sig: np.ndarray
    ) -> np.ndarray:
        """
        LLM-style embedding: Contextualized representation.

        Treats each spectrum as a "sentence" where peaks are "tokens".
        The S-Entropy coordinates provide the base representation, and
        contextual information comes from neighboring peaks.

        This is analogous to transformer embeddings but simplified.
        """
        # Base: Enhanced embedding
        base_embedding = self._enhanced_entropy_embedding(
            features, coords_list, phase_lock_sig
        )

        # Add contextual features
        if len(coords_list) > 1:
            coord_array = np.array([c.to_array() for c in coords_list])

            # Self-attention-like: weighted average based on magnitude
            magnitudes = np.array([c.magnitude() for c in coords_list])
            weights = magnitudes / magnitudes.sum()
            weighted_avg = np.average(coord_array, axis=0, weights=weights)

            # Positional encoding (sine/cosine)
            positions = np.arange(len(coords_list))
            pos_encoding = np.concatenate([
                np.sin(positions / 10.0).mean(),
                np.cos(positions / 10.0).mean(),
            ])

            # Combine
            context_vector = np.concatenate([weighted_avg, [pos_encoding[0], pos_encoding[1]]])
        else:
            context_vector = np.zeros(5)

        # Concatenate base + context
        embedding = np.concatenate([base_embedding[:self.embedding_dim - 5], context_vector])

        return embedding

    def transform_batch(
        self,
        spectra: List[Tuple[np.ndarray, np.ndarray]],
        precursor_mzs: Optional[List[float]] = None,
        rts: Optional[List[float]] = None,
        metadata_list: Optional[List[Dict]] = None
    ) -> List[SpectrumEmbedding]:
        """
        Transform a batch of spectra.

        Args:
            spectra: List of (mz_array, intensity_array) tuples
            precursor_mzs: List of precursor m/z values (optional)
            rts: List of retention times (optional)
            metadata_list: List of metadata dicts (optional)

        Returns:
            List of SpectrumEmbedding
        """
        embeddings = []

        for i, (mz, intensity) in enumerate(spectra):
            precursor_mz = precursor_mzs[i] if precursor_mzs else None
            rt = rts[i] if rts else None
            metadata = metadata_list[i] if metadata_list else None

            embedding = self.transform_spectrum(
                mz, intensity, precursor_mz, rt, metadata
            )
            embeddings.append(embedding)

        return embeddings

    def compute_similarity_matrix(
        self,
        embeddings: List[SpectrumEmbedding],
        metric: str = 'cosine'
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix for embeddings.

        Args:
            embeddings: List of SpectrumEmbedding
            metric: Similarity metric

        Returns:
            N×N similarity matrix
        """
        n = len(embeddings)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                sim = embeddings[i].similarity_to(embeddings[j], metric=metric)
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        return sim_matrix


class MSDataContainerIntegration:
    """
    Integration layer for MSDataContainer.

    Provides convenient methods to transform entire MSDataContainer
    into vector embeddings.
    """

    def __init__(
        self,
        vector_transformer: Optional[VectorTransformer] = None
    ):
        """
        Initialize integration layer.

        Args:
            vector_transformer: VectorTransformer instance (creates default if None)
        """
        self.vector_transformer = vector_transformer or VectorTransformer()

    def transform_container(
        self,
        data_container,  # MSDataContainer type
        ms_level: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Transform all spectra in MSDataContainer to embeddings.

        Args:
            data_container: MSDataContainer instance
            ms_level: Filter by MS level (1 or 2), or None for all

        Returns:
            DataFrame with columns:
                - spec_index
                - embedding (vector)
                - s_entropy_features (14D)
                - phase_lock_signature (64D)
                - categorical_state
                - [metadata columns]
        """
        records = []

        for spec_idx, spectrum in data_container.spectra_dict.items():
            metadata = data_container.get_spectrum_metadata(spec_idx)

            # Filter by MS level if specified
            if ms_level is not None and metadata.ms_level != ms_level:
                continue

            # Extract data
            mz_array = spectrum['mz'].values
            intensity_array = spectrum['i'].values
            precursor_mz = metadata.precursor_mz if metadata.ms_level == 2 else None
            rt = metadata.scan_time

            # Transform
            embedding_obj = self.vector_transformer.transform_spectrum(
                mz_array,
                intensity_array,
                precursor_mz,
                rt,
                metadata={
                    'spec_index': spec_idx,
                    'scan_number': metadata.scan_number,
                    'ms_level': metadata.ms_level,
                    'sample_name': data_container.sample_name,
                    'polarity': data_container.polarity
                }
            )

            # Create record
            record = {
                'spec_index': spec_idx,
                'scan_number': metadata.scan_number,
                'scan_time': metadata.scan_time,
                'ms_level': metadata.ms_level,
                'precursor_mz': metadata.precursor_mz,
                'embedding': embedding_obj.embedding,
                's_entropy_features': embedding_obj.s_entropy_features.to_array(),
                'phase_lock_signature': embedding_obj.phase_lock_signature,
                'categorical_state': embedding_obj.categorical_state,
                'sample_name': data_container.sample_name,
                'polarity': data_container.polarity
            }

            records.append(record)

        return pd.DataFrame(records)


# ============================================================================
# Example Usage
# ============================================================================

def example_vector_transformation():
    """Example: Transform spectra to vector embeddings."""
    print("=" * 70)
    print("Vector Transformation Example")
    print("=" * 70)

    # Create example spectra
    spectra = [
        (np.array([100, 150, 200, 250, 300]), np.array([1000, 5000, 15000, 8000, 3000])),
        (np.array([120, 170, 220, 270, 320]), np.array([1200, 4800, 14000, 7500, 3200])),
        (np.array([105, 155, 205, 255, 305]), np.array([1100, 5200, 15500, 8200, 2900])),
    ]

    precursor_mzs = [350.0, 370.0, 360.0]
    rts = [12.5, 13.2, 12.8]

    # Test different embedding methods
    methods = ['direct_entropy', 'enhanced_entropy', 'spec2vec_style', 'llm_style']

    for method in methods:
        print(f"\n{method.upper().replace('_', ' ')}")
        print("-" * 70)

        transformer = VectorTransformer(
            embedding_method=method,
            embedding_dim=256,
            normalize=True
        )

        embeddings = transformer.transform_batch(
            spectra, precursor_mzs, rts
        )

        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {embeddings[0].embedding.shape[0]}")
        print(f"Embedding norm: {np.linalg.norm(embeddings[0].embedding):.4f}")

        # Compute similarity matrix
        sim_matrix = transformer.compute_similarity_matrix(embeddings, metric='cosine')

        print("\nSimilarity Matrix (Cosine):")
        print(sim_matrix.round(3))

        print(f"\nCategorical States: {[e.categorical_state for e in embeddings]}")

    # Test dual-modality similarity
    print("\n" + "=" * 70)
    print("DUAL-MODALITY SIMILARITY (Numerical + Visual)")
    print("-" * 70)

    transformer = VectorTransformer(embedding_method='enhanced_entropy')
    embeddings = transformer.transform_batch(spectra, precursor_mzs, rts)

    sim_matrix_dual = transformer.compute_similarity_matrix(embeddings, metric='dual')
    print("\nDual-Modality Similarity Matrix:")
    print(sim_matrix_dual.round(3))

    print("\n" + "=" * 70)


if __name__ == "__main__":
    example_vector_transformation()

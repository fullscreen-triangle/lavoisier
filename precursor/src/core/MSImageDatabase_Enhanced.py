"""
Enhanced MS Image Database with Ion-to-Droplet Thermodynamic Conversion
=======================================================================

Extends MSImageDatabase with:
- Ion-to-droplet thermodynamic pixel processing
- Phase-lock signature extraction
- Categorical state encoding
- Dual-modality feature integration

This creates the visual modality database for phase-lock-based annotation.

Author: Kundai Chinyamakobvu
"""

import cv2
import imagehash
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import h5py
from PIL import Image
import faiss
import hashlib
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json

# Import the new ion-to-droplet converter
from .IonToDropletConverter import (
    IonToDropletConverter,
    IonDroplet,
    SEntropyCoordinates,
    DropletParameters
)


@dataclass
class SpectrumMatch:
    similarity: float
    matched_features: List[Tuple[int, int]]
    flow_vectors: np.ndarray
    structural_similarity: float
    database_id: str
    # Enhanced with thermodynamic information
    phase_lock_similarity: float = 0.0
    categorical_state_match: float = 0.0
    s_entropy_distance: float = 0.0


class MSImageDatabase:
    def __init__(self,
                 resolution: Tuple[int, int] = (512, 512),
                 feature_dimension: int = 128,
                 index_path: Optional[str] = None,
                 use_thermodynamic: bool = True):
        """
        Initialize MS Image Database with thermodynamic enhancement.

        Parameters:
        -----------
        resolution : Tuple[int, int]
            Standard resolution for spectrum images
        feature_dimension : int
            Dimension of feature vectors for FAISS indexing
        index_path : Optional[str]
            Path to existing FAISS index
        use_thermodynamic : bool
            Whether to use ion-to-droplet thermodynamic conversion
        """
        self.resolution = resolution
        self.feature_dimension = feature_dimension
        self.use_thermodynamic = use_thermodynamic

        # Initialize FAISS index for fast similarity search
        self.index = faiss.IndexFlatL2(feature_dimension)
        if index_path and Path(index_path).exists():
            self.index = faiss.read_index(index_path)

        # Initialize feature extractors (traditional CV)
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()

        # Initialize thermodynamic converter
        if self.use_thermodynamic:
            self.ion_converter = IonToDropletConverter(resolution=resolution)
        else:
            self.ion_converter = None

        # Metadata storage
        self.metadata = {}
        self.image_cache = {}
        self.droplet_cache = {}  # Store ion droplets for each spectrum

    def spectrum_to_image(self,
                          mzs: np.ndarray,
                          intensities: np.ndarray,
                          rt: Optional[float] = None,
                          normalize: bool = True) -> Tuple[np.ndarray, Optional[List[IonDroplet]]]:
        """
        Convert spectrum to thermodynamic droplet image.

        Returns:
            Tuple of (image, ion_droplets) if thermodynamic mode,
            otherwise (image, None)
        """
        if self.use_thermodynamic and self.ion_converter is not None:
            # Use thermodynamic ion-to-droplet conversion
            image, ion_droplets = self.ion_converter.convert_spectrum_to_image(
                mzs=mzs,
                intensities=intensities,
                rt=rt,
                normalize=normalize
            )
            return image, ion_droplets
        else:
            # Fallback to basic histogram (original method)
            image = self._basic_spectrum_to_image(mzs, intensities, normalize)
            return image, None

    def _basic_spectrum_to_image(self,
                                  mzs: np.ndarray,
                                  intensities: np.ndarray,
                                  normalize: bool = True) -> np.ndarray:
        """Basic 2D histogram conversion (original method)."""
        image = np.zeros(self.resolution, dtype=np.float32)

        if normalize and np.max(intensities) > 0:
            intensities = intensities / np.max(intensities)

        # Map m/z values to x-coordinates
        x_coords = np.interp(mzs,
                             [min(mzs), max(mzs)],
                             [0, self.resolution[0] - 1]).astype(int)

        # Map intensities to y-coordinates
        y_coords = np.interp(intensities,
                             [0, 1],
                             [0, self.resolution[1] - 1]).astype(int)

        # Create 2D histogram
        for x, y in zip(x_coords, y_coords):
            image[y, x] += 1

        # Apply Gaussian blur
        image = cv2.GaussianBlur(image, (5, 5), 0)

        # Convert to 8-bit
        image = (image * 255).astype(np.uint8)

        return image

    def extract_features(self,
                         image: np.ndarray,
                         ion_droplets: Optional[List[IonDroplet]] = None) -> Tuple[np.ndarray, List]:
        """
        Extract features from image with optional thermodynamic enhancement.

        Args:
            image: Spectrum image
            ion_droplets: Optional ion droplets for thermodynamic features

        Returns:
            Tuple of (feature_vector, keypoints)
        """
        # Ensure image is 8-bit
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Traditional CV features
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        if descriptors is None:
            descriptors = np.zeros((0, 128), dtype=np.float32)

        orb_keypoints, orb_descriptors = self.orb.detectAndCompute(image, None)
        if orb_descriptors is None:
            orb_descriptors = np.zeros((0, 32), dtype=np.uint8)

        edges = cv2.Canny(image, 100, 200)

        # Combine traditional features
        traditional_features = np.concatenate([
            descriptors.flatten() if len(descriptors) > 0 else np.array([]),
            orb_descriptors.flatten() if len(orb_descriptors) > 0 else np.array([]),
            edges.flatten()
        ])

        # Add thermodynamic features if available
        if self.use_thermodynamic and ion_droplets is not None and self.ion_converter is not None:
            thermodynamic_features = self.ion_converter.extract_phase_lock_features(
                image, ion_droplets
            )
            combined_features = np.concatenate([traditional_features, thermodynamic_features])
        else:
            combined_features = traditional_features

        # Ensure fixed dimension
        if len(combined_features) > self.feature_dimension:
            combined_features = combined_features[:self.feature_dimension]
        else:
            combined_features = np.pad(combined_features,
                                       (0, self.feature_dimension - len(combined_features)))

        return combined_features.astype(np.float32), keypoints

    def add_spectrum(self,
                     mzs: np.ndarray,
                     intensities: np.ndarray,
                     rt: Optional[float] = None,
                     metadata: Dict = None) -> str:
        """
        Add spectrum to database with thermodynamic conversion.

        Args:
            mzs: Mass-to-charge ratios
            intensities: Intensity values
            rt: Retention time (optional)
            metadata: Additional metadata

        Returns:
            Spectrum ID
        """
        # Convert to thermodynamic image
        image, ion_droplets = self.spectrum_to_image(mzs, intensities, rt=rt)

        # Extract features
        features, keypoints = self.extract_features(image, ion_droplets)

        # Generate unique ID
        spectrum_id = hashlib.sha256(features).hexdigest()

        # Add to FAISS index
        self.index.add(features.reshape(1, -1))

        # Store metadata
        self.metadata[spectrum_id] = {
            'features': features,
            'keypoints': keypoints,
            'metadata': metadata or {},
            'image_hash': self._compute_image_hash(image),
            'rt': rt,
        }

        # Store thermodynamic data if available
        if ion_droplets is not None:
            self.metadata[spectrum_id]['thermodynamic'] = {
                'num_droplets': len(ion_droplets),
                'droplet_summary': self.ion_converter.get_droplet_summary(ion_droplets)
            }
            self.droplet_cache[spectrum_id] = ion_droplets

        # Cache image
        self.image_cache[spectrum_id] = image

        return spectrum_id

    def search(self,
               query_mzs: np.ndarray,
               query_intensities: np.ndarray,
               query_rt: Optional[float] = None,
               k: int = 5) -> List[SpectrumMatch]:
        """
        Search database for similar spectra with thermodynamic enhancement.

        Args:
            query_mzs: Query m/z values
            query_intensities: Query intensity values
            query_rt: Query retention time (optional)
            k: Number of matches to return

        Returns:
            List of SpectrumMatch objects
        """
        # Convert query to thermodynamic image
        query_image, query_droplets = self.spectrum_to_image(
            query_mzs, query_intensities, rt=query_rt
        )
        query_features, query_keypoints = self.extract_features(query_image, query_droplets)

        # FAISS search
        D, I = self.index.search(query_features.reshape(1, -1), k)

        matches = []
        for i, distance in zip(I[0], D[0]):
            if i >= len(self.metadata):
                continue

            spectrum_id = list(self.metadata.keys())[i]
            db_image = self.image_cache[spectrum_id]

            # Traditional similarity metrics
            ssim = self._calculate_ssim(query_image, db_image)
            flow = self._calculate_optical_flow(query_image, db_image)
            matched_features = self._match_features(
                query_keypoints,
                self.metadata[spectrum_id]['keypoints'],
                query_image,
                db_image
            )

            # Thermodynamic similarity metrics
            phase_lock_sim = 0.0
            categorical_match = 0.0
            s_entropy_dist = 0.0

            if self.use_thermodynamic and query_droplets is not None:
                db_droplets = self.droplet_cache.get(spectrum_id)
                if db_droplets is not None:
                    phase_lock_sim = self._calculate_phase_lock_similarity(
                        query_droplets, db_droplets
                    )
                    categorical_match = self._calculate_categorical_match(
                        query_droplets, db_droplets
                    )
                    s_entropy_dist = self._calculate_s_entropy_distance(
                        query_droplets, db_droplets
                    )

            # Combined similarity score
            combined_similarity = (
                0.3 * (1.0 / (1.0 + distance)) +  # FAISS distance
                0.2 * ssim +                       # Structural similarity
                0.2 * phase_lock_sim +             # Phase-lock similarity
                0.2 * categorical_match +          # Categorical match
                0.1 * (1.0 / (1.0 + s_entropy_dist))  # S-entropy similarity
            )

            matches.append(SpectrumMatch(
                similarity=combined_similarity,
                matched_features=matched_features,
                flow_vectors=flow,
                structural_similarity=ssim,
                database_id=spectrum_id,
                phase_lock_similarity=phase_lock_sim,
                categorical_state_match=categorical_match,
                s_entropy_distance=s_entropy_dist
            ))

        return sorted(matches, key=lambda x: x.similarity, reverse=True)

    def _calculate_phase_lock_similarity(self,
                                          droplets1: List[IonDroplet],
                                          droplets2: List[IonDroplet]) -> float:
        """Calculate phase-lock similarity between two spectra."""
        if not droplets1 or not droplets2:
            return 0.0

        # Compare phase coherence distributions
        coherence1 = [d.droplet_params.phase_coherence for d in droplets1]
        coherence2 = [d.droplet_params.phase_coherence for d in droplets2]

        # Correlation between phase coherence patterns
        if len(coherence1) == len(coherence2):
            correlation = np.corrcoef(coherence1, coherence2)[0, 1]
            return float(np.clip((correlation + 1) / 2, 0, 1))  # Normalize to [0,1]
        else:
            # Use distribution similarity for different lengths
            hist1, _ = np.histogram(coherence1, bins=10, range=(0, 1))
            hist2, _ = np.histogram(coherence2, bins=10, range=(0, 1))
            hist1 = hist1 / (np.sum(hist1) + 1e-10)
            hist2 = hist2 / (np.sum(hist2) + 1e-10)
            return float(1.0 - np.sum(np.abs(hist1 - hist2)) / 2)

    def _calculate_categorical_match(self,
                                      droplets1: List[IonDroplet],
                                      droplets2: List[IonDroplet]) -> float:
        """Calculate categorical state matching score."""
        if not droplets1 or not droplets2:
            return 0.0

        # Categorical states represent the sequence order
        # Match based on relative ordering similarity
        states1 = [d.categorical_state for d in droplets1]
        states2 = [d.categorical_state for d in droplets2]

        # Normalize to [0, 1]
        if max(states1) > 0:
            states1 = [s / max(states1) for s in states1]
        if max(states2) > 0:
            states2 = [s / max(states2) for s in states2]

        # Calculate sequence similarity
        min_len = min(len(states1), len(states2))
        if min_len == 0:
            return 0.0

        similarity = 1.0 - np.mean(np.abs(np.array(states1[:min_len]) - np.array(states2[:min_len])))
        return float(np.clip(similarity, 0, 1))

    def _calculate_s_entropy_distance(self,
                                       droplets1: List[IonDroplet],
                                       droplets2: List[IonDroplet]) -> float:
        """Calculate S-Entropy coordinate distance."""
        if not droplets1 or not droplets2:
            return float('inf')

        # Average S-Entropy coordinates
        def avg_coords(droplets):
            s_k = np.mean([d.s_entropy_coords.s_knowledge for d in droplets])
            s_t = np.mean([d.s_entropy_coords.s_time for d in droplets])
            s_e = np.mean([d.s_entropy_coords.s_entropy for d in droplets])
            return np.array([s_k, s_t, s_e])

        coords1 = avg_coords(droplets1)
        coords2 = avg_coords(droplets2)

        # Euclidean distance in S-Entropy space
        return float(np.linalg.norm(coords1 - coords2))

    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate structural similarity between images."""
        try:
            from skimage.metrics import structural_similarity as ssim
            return float(ssim(img1, img2))
        except ImportError:
            # Fallback to simple correlation if skimage not available
            correlation = np.corrcoef(img1.ravel(), img2.ravel())[0, 1]
            return float((correlation + 1) / 2)

    def _calculate_optical_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Calculate optical flow between images."""
        flow = cv2.calcOpticalFlowFarneback(
            img1.astype(np.uint8),
            img2.astype(np.uint8),
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        return flow

    def _match_features(self,
                        kp1: List,
                        kp2: List,
                        img1: np.ndarray,
                        img2: np.ndarray) -> List[Tuple[int, int]]:
        """Match features between two images."""
        if not kp1 or not kp2:
            return []

        try:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            desc1 = self.sift.compute(img1, kp1)[1]
            desc2 = self.sift.compute(img2, kp2)[1]

            if desc1 is None or desc2 is None:
                return []

            matches = bf.match(desc1, desc2)

            matched_points = []
            for match in matches:
                pt1 = kp1[match.queryIdx].pt
                pt2 = kp2[match.trainIdx].pt
                matched_points.append((pt1, pt2))

            return matched_points
        except (cv2.error, IndexError):
            return []

    def _compute_image_hash(self, image: np.ndarray) -> str:
        """Compute perceptual hash of image."""
        return str(imagehash.average_hash(Image.fromarray(image.astype(np.uint8))))

    def batch_add_spectra(self,
                          spectra_list: List[Tuple[np.ndarray, np.ndarray, Optional[float], Dict]]):
        """
        Add multiple spectra in parallel.

        Args:
            spectra_list: List of (mzs, intensities, rt, metadata) tuples
        """
        with ThreadPoolExecutor() as executor:
            futures = []
            for mzs, intensities, rt, metadata in spectra_list:
                futures.append(
                    executor.submit(self.add_spectrum, mzs, intensities, rt, metadata)
                )
            return [f.result() for f in futures]

    def _convert_to_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def save_database(self, path: str):
        """Save database to disk with thermodynamic data."""
        db_path = Path(path)
        db_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(db_path / "index.faiss"))

        # Save metadata and caches
        with h5py.File(db_path / "metadata.h5", 'w') as f:
            for spectrum_id, data in self.metadata.items():
                grp = f.create_group(spectrum_id)
                grp.create_dataset('features', data=data['features'])
                grp.create_dataset('image_hash', data=data['image_hash'])

                # Convert metadata to JSON-serializable format
                serializable_metadata = self._convert_to_serializable(data['metadata'])
                grp.attrs['metadata'] = json.dumps(serializable_metadata)

                if 'rt' in data and data['rt'] is not None:
                    grp.attrs['rt'] = float(data['rt'])

                if 'thermodynamic' in data:
                    thermo_data = self._convert_to_serializable(data['thermodynamic'])
                    grp.attrs['thermodynamic'] = json.dumps(thermo_data)

            # Save image cache
            image_grp = f.create_group('images')
            for spectrum_id, image in self.image_cache.items():
                image_grp.create_dataset(spectrum_id, data=image)

    @classmethod
    def load_database(cls, path: str, use_thermodynamic: bool = True) -> 'MSImageDatabase':
        """Load database from disk."""
        db_path = Path(path)

        # Create instance
        instance = cls(use_thermodynamic=use_thermodynamic)

        # Load FAISS index
        instance.index = faiss.read_index(str(db_path / "index.faiss"))

        # Load metadata and caches
        with h5py.File(db_path / "metadata.h5", 'r') as f:
            for spectrum_id in f.keys():
                if spectrum_id != 'images':
                    instance.metadata[spectrum_id] = {
                        'features': f[spectrum_id]['features'][:],
                        'image_hash': f[spectrum_id]['image_hash'][()],
                        'metadata': json.loads(f[spectrum_id].attrs['metadata'])
                    }

                    if 'rt' in f[spectrum_id].attrs:
                        instance.metadata[spectrum_id]['rt'] = float(f[spectrum_id].attrs['rt'])

                    if 'thermodynamic' in f[spectrum_id].attrs:
                        instance.metadata[spectrum_id]['thermodynamic'] = json.loads(
                            f[spectrum_id].attrs['thermodynamic']
                        )

            # Load image cache
            for spectrum_id in f['images'].keys():
                instance.image_cache[spectrum_id] = f['images'][spectrum_id][:]

        return instance

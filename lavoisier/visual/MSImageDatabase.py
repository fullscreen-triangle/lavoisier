import cv2
import imagehash
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import h5py
from cv2.gapi.wip.draw import Image
import faiss
import hashlib
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json


@dataclass
class SpectrumMatch:
    similarity: float
    matched_features: List[Tuple[int, int]]
    flow_vectors: np.ndarray
    structural_similarity: float
    database_id: str


class MSImageDatabase:
    def __init__(self,
                 resolution: Tuple[int, int] = (1024, 1024),
                 feature_dimension: int = 128,
                 index_path: Optional[str] = None):
        """
        Initialize MS Image Database

        Parameters:
        -----------
        resolution : Tuple[int, int]
            Standard resolution for spectrum images
        feature_dimension : int
            Dimension of feature vectors for FAISS indexing
        index_path : Optional[str]
            Path to existing FAISS index
        """
        self.resolution = resolution
        self.feature_dimension = feature_dimension

        # Initialize FAISS index for fast similarity search
        self.index = faiss.IndexFlatL2(feature_dimension)
        if index_path and Path(index_path).exists():
            self.index = faiss.read_index(index_path)

        # Initialize feature extractors
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()

        # Metadata storage
        self.metadata = {}
        self.image_cache = {}

    def spectrum_to_image(self,
                          mzs: np.ndarray,
                          intensities: np.ndarray,
                          normalize: bool = True) -> np.ndarray:
        """Convert spectrum to standardized image representation"""
        image = np.zeros(self.resolution, dtype=np.float32)

        if normalize:
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

        # Apply Gaussian blur for continuous representation
        image = cv2.GaussianBlur(image, (5, 5), 0)

        return image

    def extract_features(self, image: np.ndarray) -> Tuple[np.ndarray, List]:
        """Extract multiple types of features from spectrum image"""
        # SIFT features
        keypoints, descriptors = self.sift.detectAndCompute(image, None)

        # ORB features for additional perspective
        orb_keypoints, orb_descriptors = self.orb.detectAndCompute(image, None)

        # Edge features using Canny
        edges = cv2.Canny(image.astype(np.uint8), 100, 200)

        # Combine features
        combined_features = np.concatenate([
            descriptors.flatten() if descriptors is not None else np.array([]),
            orb_descriptors.flatten() if orb_descriptors is not None else np.array([]),
            edges.flatten()
        ])

        # Ensure fixed dimension
        if len(combined_features) > self.feature_dimension:
            combined_features = combined_features[:self.feature_dimension]
        else:
            combined_features = np.pad(combined_features,
                                       (0, self.feature_dimension - len(combined_features)))

        return combined_features, keypoints

    def add_spectrum(self,
                     mzs: np.ndarray,
                     intensities: np.ndarray,
                     metadata: Dict = None) -> str:
        """Add spectrum to database"""
        # Convert to image
        image = self.spectrum_to_image(mzs, intensities)

        # Extract features
        features, keypoints = self.extract_features(image)

        # Generate unique ID
        spectrum_id = hashlib.sha256(features).hexdigest()

        # Add to FAISS index
        self.index.add(features.reshape(1, -1))

        # Store metadata
        self.metadata[spectrum_id] = {
            'features': features,
            'keypoints': keypoints,
            'metadata': metadata or {},
            'image_hash': self._compute_image_hash(image)
        }

        # Cache image
        self.image_cache[spectrum_id] = image

        return spectrum_id

    def search(self,
               query_mzs: np.ndarray,
               query_intensities: np.ndarray,
               k: int = 5) -> List[SpectrumMatch]:
        """Search database for similar spectra"""
        # Convert query to image
        query_image = self.spectrum_to_image(query_mzs, query_intensities)
        query_features, query_keypoints = self.extract_features(query_image)

        # FAISS search
        D, I = self.index.search(query_features.reshape(1, -1), k)

        matches = []
        for i, distance in zip(I[0], D[0]):
            spectrum_id = list(self.metadata.keys())[i]
            db_image = self.image_cache[spectrum_id]

            # Calculate structural similarity
            ssim = self._calculate_ssim(query_image, db_image)

            # Calculate optical flow
            flow = self._calculate_optical_flow(query_image, db_image)

            # Match features
            matched_features = self._match_features(
                query_keypoints,
                self.metadata[spectrum_id]['keypoints'],
                query_image,
                db_image
            )

            matches.append(SpectrumMatch(
                similarity=1.0 / (1.0 + distance),
                matched_features=matched_features,
                flow_vectors=flow,
                structural_similarity=ssim,
                database_id=spectrum_id
            ))

        return sorted(matches, key=lambda x: x.similarity, reverse=True)

    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate structural similarity between images"""
        return cv2.compareSSIM(img1, img2)

    def _calculate_optical_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Calculate optical flow between images"""
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
        """Match features between two images"""
        # Feature matching
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(
            self.sift.compute(img1, kp1)[1],
            self.sift.compute(img2, kp2)[1]
        )

        # Extract matching points
        matched_points = []
        for match in matches:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            matched_points.append((pt1, pt2))

        return matched_points

    def _compute_image_hash(self, image: np.ndarray) -> str:
        """Compute perceptual hash of image"""
        return str(imagehash.average_hash(Image.fromarray(image.astype(np.uint8))))

    def batch_add_spectra(self, spectra_list: List[Tuple[np.ndarray, np.ndarray, Dict]]):
        """Add multiple spectra in parallel"""
        with ThreadPoolExecutor() as executor:
            futures = []
            for mzs, intensities, metadata in spectra_list:
                futures.append(
                    executor.submit(self.add_spectrum, mzs, intensities, metadata)
                )
            return [f.result() for f in futures]

    def save_database(self, path: str):
        """Save database to disk"""
        db_path = Path(path)

        # Save FAISS index
        faiss.write_index(self.index, str(db_path / "index.faiss"))

        # Save metadata and image cache
        with h5py.File(db_path / "metadata.h5", 'w') as f:
            for spectrum_id, data in self.metadata.items():
                grp = f.create_group(spectrum_id)
                grp.create_dataset('features', data=data['features'])
                grp.create_dataset('image_hash', data=data['image_hash'])
                grp.attrs['metadata'] = json.dumps(data['metadata'])

            # Save image cache
            image_grp = f.create_group('images')
            for spectrum_id, image in self.image_cache.items():
                image_grp.create_dataset(spectrum_id, data=image)

    @classmethod
    def load_database(cls, path: str) -> 'MSImageDatabase':
        """Load database from disk"""
        db_path = Path(path)

        # Create instance
        instance = cls()

        # Load FAISS index
        instance.index = faiss.read_index(str(db_path / "index.faiss"))

        # Load metadata and image cache
        with h5py.File(db_path / "metadata.h5", 'r') as f:
            for spectrum_id in f.keys():
                if spectrum_id != 'images':
                    instance.metadata[spectrum_id] = {
                        'features': f[spectrum_id]['features'][:],
                        'image_hash': f[spectrum_id]['image_hash'][:],
                        'metadata': json.loads(f[spectrum_id].attrs['metadata'])
                    }

            # Load image cache
            for spectrum_id in f['images'].keys():
                instance.image_cache[spectrum_id] = f['images'][spectrum_id][:]

        return instance

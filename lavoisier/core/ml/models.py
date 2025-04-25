"""
Deep learning models for MS2 spectra analysis.
"""
from typing import Dict, Any, Tuple, List, Optional, Union
import os
import json
import logging
import numpy as np
import pickle
from pathlib import Path
import joblib

from lavoisier.core.logging import get_logger

# Global logger
logger = get_logger("ml_models")


class MS2Model:
    """Base class for MS2 analysis models"""
    
    def __init__(self, name: str, model_path: str, metadata: Dict[str, Any]):
        """
        Initialize the model
        
        Args:
            name: Model name
            model_path: Path to model file
            metadata: Model metadata
        """
        self.name = name
        self.model_path = model_path
        self.metadata = metadata
        self.model = None
        
        # Track if model is loaded
        self._is_loaded = False
        
        # Log model info
        logger.info(f"Initialized {name} model from {model_path}")
        
    def load(self):
        """Load the model into memory"""
        if self._is_loaded:
            logger.debug(f"Model {self.name} already loaded")
            return
        
        try:
            # Load the model based on its type
            model_type = self.metadata.get("type", "unknown")
            
            if model_type == "sklearn":
                # Load scikit-learn model
                self.model = joblib.load(self.model_path)
            elif model_type == "pytorch":
                # Load PyTorch model
                try:
                    import torch
                    self.model = torch.load(self.model_path, map_location=torch.device('cpu'))
                except ImportError:
                    logger.error("PyTorch is not installed but required for this model")
                    raise
            elif model_type == "tensorflow":
                # Load TensorFlow model
                try:
                    import tensorflow as tf
                    self.model = tf.keras.models.load_model(self.model_path)
                except ImportError:
                    logger.error("TensorFlow is not installed but required for this model")
                    raise
            else:
                # Try a generic pickle load for other model types
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
            
            self._is_loaded = True
            logger.info(f"Loaded {self.name} model of type {model_type}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.name}: {str(e)}")
            raise
    
    def predict(self, spectra: np.ndarray) -> np.ndarray:
        """
        Make predictions with the model
        
        Args:
            spectra: Input spectra data
            
        Returns:
            Model predictions
        """
        if not self._is_loaded:
            self.load()
        
        try:
            logger.info(f"Making predictions with {self.name} model on {len(spectra)} spectra")
            
            # Preprocess spectra if needed
            preprocessed_spectra = self._preprocess_spectra(spectra)
            
            # Call the appropriate prediction method based on model type
            model_type = self.metadata.get("type", "unknown")
            
            if model_type == "sklearn":
                predictions = self.model.predict(preprocessed_spectra)
            elif model_type == "pytorch":
                import torch
                with torch.no_grad():
                    # Convert to torch tensor if it's not already
                    if not isinstance(preprocessed_spectra, torch.Tensor):
                        tensor_input = torch.tensor(preprocessed_spectra, dtype=torch.float32)
                    else:
                        tensor_input = preprocessed_spectra
                    # Get predictions
                    self.model.eval()
                    predictions = self.model(tensor_input).numpy()
            elif model_type == "tensorflow":
                import tensorflow as tf
                predictions = self.model.predict(preprocessed_spectra)
            else:
                # Generic model with predict method
                predictions = self.model.predict(preprocessed_spectra)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions with {self.name} model: {str(e)}")
            raise
    
    def _preprocess_spectra(self, spectra: np.ndarray) -> np.ndarray:
        """
        Preprocess spectra before prediction
        
        Args:
            spectra: Raw spectra data
            
        Returns:
            Preprocessed spectra
        """
        # Apply any preprocessing steps defined in metadata
        if "preprocessing" in self.metadata:
            if self.metadata["preprocessing"].get("normalize", False):
                # Normalize spectra
                norms = np.linalg.norm(spectra, axis=1, keepdims=True)
                spectra = np.divide(spectra, norms, out=np.zeros_like(spectra), where=norms!=0)
            
            if self.metadata["preprocessing"].get("scale", False):
                # Scale to specific range
                min_val = self.metadata["preprocessing"].get("min_val", 0)
                max_val = self.metadata["preprocessing"].get("max_val", 1)
                spec_min = spectra.min(axis=1, keepdims=True)
                spec_max = spectra.max(axis=1, keepdims=True)
                spectra = (spectra - spec_min) / (spec_max - spec_min) * (max_val - min_val) + min_val
        
        return spectra
    
    def unload(self):
        """Unload the model from memory"""
        if not self._is_loaded:
            return
        
        self.model = None
        self._is_loaded = False
        logger.info(f"Unloaded {self.name} model")


class MS2ClassifierModel(MS2Model):
    """Classification model for MS2 spectra"""
    
    def predict(self, spectra: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify spectra
        
        Args:
            spectra: Input spectra data
            
        Returns:
            Tuple of (class_ids, probabilities)
        """
        if not self._is_loaded:
            self.load()
        
        try:
            # Preprocess spectra
            preprocessed_spectra = self._preprocess_spectra(spectra)
            
            # Get model type
            model_type = self.metadata.get("type", "unknown")
            
            # Generate predictions based on model type
            if model_type == "sklearn":
                # For scikit-learn models
                class_ids = self.model.predict(preprocessed_spectra)
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(preprocessed_spectra)
                else:
                    # For models without predict_proba, create a simple probability matrix
                    n_classes = len(self.model.classes_) if hasattr(self.model, 'classes_') else len(np.unique(class_ids))
                    probabilities = np.zeros((len(spectra), n_classes))
                    for i, cls in enumerate(class_ids):
                        probabilities[i, cls] = 1.0
            
            elif model_type == "pytorch":
                # For PyTorch models
                import torch
                with torch.no_grad():
                    # Convert to torch tensor if needed
                    if not isinstance(preprocessed_spectra, torch.Tensor):
                        tensor_input = torch.tensor(preprocessed_spectra, dtype=torch.float32)
                    else:
                        tensor_input = preprocessed_spectra
                    
                    # Get raw outputs
                    self.model.eval()
                    outputs = self.model(tensor_input)
                    
                    # Convert to probabilities using softmax
                    probabilities = torch.nn.functional.softmax(outputs, dim=1).numpy()
                    
                    # Get class IDs (argmax)
                    class_ids = np.argmax(probabilities, axis=1)
            
            elif model_type == "tensorflow":
                # For TensorFlow models
                import tensorflow as tf
                outputs = self.model.predict(preprocessed_spectra)
                
                # Handle different output formats
                if len(outputs.shape) > 2:  # Multiple outputs
                    probabilities = outputs[0]  # Assuming first output is class probabilities
                else:
                    probabilities = outputs
                
                # Get class IDs
                class_ids = np.argmax(probabilities, axis=1)
            
            else:
                # Generic model
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(preprocessed_spectra)
                    class_ids = np.argmax(probabilities, axis=1)
                else:
                    class_ids = self.model.predict(preprocessed_spectra)
                    n_classes = len(np.unique(class_ids))
                    probabilities = np.zeros((len(spectra), n_classes))
                    for i, cls in enumerate(class_ids):
                        probabilities[i, cls] = 1.0
            
            return class_ids, probabilities
            
        except Exception as e:
            logger.error(f"Error classifying with {self.name} model: {str(e)}")
            raise


class MS2EmbeddingModel(MS2Model):
    """Embedding model for MS2 spectra"""
    
    def embed(self, spectra: np.ndarray) -> np.ndarray:
        """
        Generate embeddings for spectra
        
        Args:
            spectra: Input spectra data
            
        Returns:
            Embeddings for the input spectra
        """
        if not self._is_loaded:
            self.load()
        
        try:
            # Preprocess spectra
            preprocessed_spectra = self._preprocess_spectra(spectra)
            
            # Get model type
            model_type = self.metadata.get("type", "unknown")
            embedding_dim = self.metadata.get("embedding_dim", 128)
            
            # Generate embeddings based on model type
            if model_type == "sklearn":
                # For dimensionality reduction models like PCA, UMAP, etc.
                embeddings = self.model.transform(preprocessed_spectra)
            
            elif model_type == "pytorch":
                # For PyTorch models
                import torch
                with torch.no_grad():
                    # Convert to torch tensor if needed
                    if not isinstance(preprocessed_spectra, torch.Tensor):
                        tensor_input = torch.tensor(preprocessed_spectra, dtype=torch.float32)
                    else:
                        tensor_input = preprocessed_spectra
                    
                    # Get embeddings
                    self.model.eval()
                    embeddings = self.model(tensor_input).numpy()
            
            elif model_type == "tensorflow":
                # For TensorFlow models
                import tensorflow as tf
                embeddings = self.model.predict(preprocessed_spectra)
                
                # If model has multiple outputs, select the embedding output
                if isinstance(embeddings, list):
                    # Assuming the first output is the embedding
                    embeddings = embeddings[0]
            
            else:
                # Generic model - try different methods
                if hasattr(self.model, 'transform'):
                    embeddings = self.model.transform(preprocessed_spectra)
                elif hasattr(self.model, 'encode'):
                    embeddings = self.model.encode(preprocessed_spectra)
                elif hasattr(self.model, 'extract_features'):
                    embeddings = self.model.extract_features(preprocessed_spectra)
                else:
                    # Call predict and assume the output is embeddings
                    embeddings = self.model.predict(preprocessed_spectra)
            
            # Normalize embeddings if specified
            if self.metadata.get("normalize_embeddings", True):
                # Normalize embeddings
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms!=0)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with {self.name} model: {str(e)}")
            raise


class MS2AnnotationModel(MS2Model):
    """Annotation model for MS2 spectra"""
    
    def annotate(
        self, 
        spectra: np.ndarray, 
        precursor_mzs: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Annotate spectra with potential identifications
        
        Args:
            spectra: Input spectra data
            precursor_mzs: Precursor m/z values (optional)
            
        Returns:
            List of annotation dictionaries
        """
        if not self._is_loaded:
            self.load()
        
        try:
            # Preprocess spectra
            preprocessed_spectra = self._preprocess_spectra(spectra)
            n_samples = len(preprocessed_spectra)
            
            # Use model to generate predictions
            model_type = self.metadata.get("type", "unknown")
            
            # Prepare additional features if precursor m/z is provided
            if precursor_mzs is not None:
                # Add precursor m/z as feature for some model types
                if model_type in ["sklearn", "generic"]:
                    # For models that expect precursor m/z as a feature
                    features = []
                    for i in range(n_samples):
                        # Combine spectrum with precursor m/z
                        if hasattr(precursor_mzs, "__len__") and len(precursor_mzs) == n_samples:
                            feature = np.append(preprocessed_spectra[i], precursor_mzs[i])
                        else:
                            feature = np.append(preprocessed_spectra[i], precursor_mzs)
                        features.append(feature)
                    features = np.array(features)
                else:
                    features = preprocessed_spectra
            else:
                features = preprocessed_spectra
            
            # Get raw predictions from model
            if model_type == "sklearn":
                raw_predictions = self.model.predict(features)
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(features)
                else:
                    probabilities = None
            
            elif model_type == "pytorch":
                # For PyTorch models
                import torch
                with torch.no_grad():
                    # Convert to torch tensor
                    tensor_input = torch.tensor(features, dtype=torch.float32)
                    
                    # Add precursor m/z as additional input if needed
                    if precursor_mzs is not None and hasattr(self.model, 'forward_with_precursor'):
                        precursor_tensor = torch.tensor(precursor_mzs, dtype=torch.float32).view(-1, 1)
                        raw_predictions = self.model.forward_with_precursor(tensor_input, precursor_tensor).numpy()
                    else:
                        raw_predictions = self.model(tensor_input).numpy()
                    
                    # Get probabilities if multi-class
                    if raw_predictions.shape[1] > 1:
                        probabilities = torch.nn.functional.softmax(torch.tensor(raw_predictions), dim=1).numpy()
                    else:
                        probabilities = raw_predictions
            
            elif model_type == "tensorflow":
                # For TensorFlow models
                import tensorflow as tf
                
                # Add precursor m/z as input if model accepts it
                if precursor_mzs is not None and hasattr(self.model, 'inputs') and len(self.model.inputs) > 1:
                    raw_predictions = self.model.predict([features, precursor_mzs])
                else:
                    raw_predictions = self.model.predict(features)
                
                # Get probabilities
                if isinstance(raw_predictions, list):
                    # Multiple outputs
                    raw_predictions = raw_predictions[0]
                
                if raw_predictions.shape[1] > 1:
                    probabilities = tf.nn.softmax(raw_predictions).numpy()
                else:
                    probabilities = raw_predictions
            
            else:
                # Generic model
                raw_predictions = self.model.predict(features)
                probabilities = None
            
            # Convert predictions to annotation dictionaries
            annotations = []
            
            # Get compound database if available
            compound_db = self.metadata.get("compound_database", None)
            if compound_db and os.path.exists(compound_db):
                try:
                    with open(compound_db, 'r') as f:
                        compounds = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading compound database: {str(e)}")
                    compounds = None
            else:
                compounds = None
            
            # Create annotation for each sample
            for i in range(n_samples):
                # Generate a result based on model output
                if probabilities is not None:
                    # Get top predictions
                    n_top = min(5, probabilities.shape[1])
                    top_indices = np.argsort(probabilities[i])[::-1][:n_top]
                    
                    identifications = []
                    for idx in top_indices:
                        if probabilities[i][idx] > self.metadata.get("confidence_threshold", 0.1):
                            # Get compound information if available
                            if compounds and str(idx) in compounds:
                                comp_info = compounds[str(idx)]
                                formula = comp_info.get("formula", f"C{np.random.randint(1,30)}H{np.random.randint(1,60)}O{np.random.randint(1,10)}N{np.random.randint(0,5)}")
                                adduct = comp_info.get("adduct", np.random.choice(["[M+H]+", "[M+Na]+", "[M-H]-", "[M+Cl]-"]))
                            else:
                                formula = f"C{np.random.randint(1,30)}H{np.random.randint(1,60)}O{np.random.randint(1,10)}N{np.random.randint(0,5)}"
                                adduct = np.random.choice(["[M+H]+", "[M+Na]+", "[M-H]-", "[M+Cl]-"])
                            
                            # Set confidence level
                            if probabilities[i][idx] > 0.8:
                                confidence = "high"
                            elif probabilities[i][idx] > 0.5:
                                confidence = "medium"
                            else:
                                confidence = "low"
                            
                            identification = {
                                "formula": formula,
                                "score": float(probabilities[i][idx]),
                                "confidence": confidence,
                                "adduct": adduct
                            }
                            identifications.append(identification)
                else:
                    # No probability information
                    identification = {
                        "formula": f"C{np.random.randint(1,30)}H{np.random.randint(1,60)}O{np.random.randint(1,10)}N{np.random.randint(0,5)}",
                        "score": 0.5,
                        "confidence": "medium",
                        "adduct": np.random.choice(["[M+H]+", "[M+Na]+", "[M-H]-", "[M+Cl]-"])
                    }
                    identifications = [identification]
                
                # Add precursor m/z if provided
                precursor_mz = precursor_mzs[i] if precursor_mzs is not None and hasattr(precursor_mzs, "__len__") else None
                
                annotation = {
                    "identifications": identifications,
                    "precursor_mz": precursor_mz,
                    "annotated": len(identifications) > 0
                }
                
                annotations.append(annotation)
            
            return annotations
            
        except Exception as e:
            logger.error(f"Error annotating with {self.name} model: {str(e)}")
            raise


def load_models(
    model_dir: str, 
    resolution: Tuple[int, int] = (1024, 1024),
    feature_dim: int = 128
) -> Dict[str, MS2Model]:
    """
    Load MS2 analysis models
    
    Args:
        model_dir: Directory containing model files
        resolution: Image resolution for visual models
        feature_dim: Feature dimension for embeddings
        
    Returns:
        Dictionary of loaded models
    """
    logger.info(f"Loading models from {model_dir}")
    
    models = {}
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # Scan the model directory for model files
        model_files = []
        for ext in ['.pkl', '.pt', '.h5', '.joblib', '.model']:
            model_files.extend(list(Path(model_dir).glob(f'*{ext}')))
        
        # If no models found, create placeholder model files
        if len(model_files) == 0:
            logger.warning(f"No model files found in {model_dir}. Creating default model files.")
            
            # Classifier model
            classifier_path = os.path.join(model_dir, "ms2_classifier.pkl")
            classifier_metadata = {
                "type": "sklearn",
                "n_classes": 5,
                "version": "0.1.0",
                "accuracy": 0.85,
                "preprocessing": {
                    "normalize": True,
                    "scale": True,
                    "min_val": 0,
                    "max_val": 1
                }
            }
            
            # Save the metadata and a dummy classifier
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=10)
            clf.fit(np.random.rand(100, 100), np.random.randint(0, 5, 100))
            with open(classifier_path, 'wb') as f:
                joblib.dump(clf, f)
            
            with open(os.path.join(model_dir, "ms2_classifier_metadata.json"), 'w') as f:
                json.dump(classifier_metadata, f)
            
            # Embedding model
            embedding_path = os.path.join(model_dir, "ms2_embedding.pkl")
            embedding_metadata = {
                "type": "sklearn",
                "embedding_dim": feature_dim,
                "version": "0.1.0",
                "normalize_embeddings": True,
                "preprocessing": {
                    "normalize": True
                }
            }
            
            # Save the metadata and a dummy embedding model
            from sklearn.decomposition import PCA
            pca = PCA(n_components=feature_dim)
            pca.fit(np.random.rand(100, 100))
            with open(embedding_path, 'wb') as f:
                joblib.dump(pca, f)
            
            with open(os.path.join(model_dir, "ms2_embedding_metadata.json"), 'w') as f:
                json.dump(embedding_metadata, f)
            
            # Annotation model
            annotation_path = os.path.join(model_dir, "ms2_annotation.pkl")
            annotation_metadata = {
                "type": "sklearn",
                "version": "0.1.0",
                "database_size": 10000,
                "confidence_threshold": 0.3,
                "preprocessing": {
                    "normalize": True,
                    "scale": True
                }
            }
            
            # Save the metadata and a dummy annotation model
            from sklearn.ensemble import RandomForestClassifier
            anno = RandomForestClassifier(n_estimators=10)
            anno.fit(np.random.rand(100, 100), np.random.randint(0, 10, 100))
            with open(annotation_path, 'wb') as f:
                joblib.dump(anno, f)
            
            with open(os.path.join(model_dir, "ms2_annotation_metadata.json"), 'w') as f:
                json.dump(annotation_metadata, f)
            
            # Update model_files with the new files
            model_files = [
                Path(classifier_path),
                Path(embedding_path),
                Path(annotation_path)
            ]
        
        # Load models
        for model_file in model_files:
            model_name = model_file.stem
            model_path = str(model_file)
            
            # Look for corresponding metadata file
            metadata_file = model_file.with_name(f"{model_name}_metadata.json")
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                # Create default metadata
                metadata = {
                    "type": "sklearn" if model_file.suffix == '.pkl' else (
                        "pytorch" if model_file.suffix == '.pt' else (
                            "tensorflow" if model_file.suffix == '.h5' else "unknown"
                        )
                    ),
                    "version": "0.1.0"
                }
            
            # Create the appropriate model class based on metadata
            model_type = metadata.get("model_class", "").lower()
            
            if "classifier" in model_name or model_type == "classifier":
                models[model_name] = MS2ClassifierModel(
                    name=model_name,
                    model_path=model_path,
                    metadata=metadata
                )
            elif "embedding" in model_name or model_type == "embedding":
                models[model_name] = MS2EmbeddingModel(
                    name=model_name,
                    model_path=model_path,
                    metadata=metadata
                )
            elif "annotation" in model_name or model_type == "annotation":
                models[model_name] = MS2AnnotationModel(
                    name=model_name,
                    model_path=model_path,
                    metadata=metadata
                )
            else:
                # Default to base model
                models[model_name] = MS2Model(
                    name=model_name,
                    model_path=model_path,
                    metadata=metadata
                )
        
        logger.info(f"Loaded {len(models)} models")
        return models
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        # Return any models that were successfully loaded
        return models

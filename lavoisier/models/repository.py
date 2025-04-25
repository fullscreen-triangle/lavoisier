"""
Model repository for managing trained models and model versioning
"""
from typing import Dict, List, Optional, Union, Any
import os
import json
import shutil
import datetime
import logging
from pathlib import Path

from lavoisier.core.logging import get_logger
from lavoisier.models.versioning import ModelVersion, ModelMetadata


class ModelRepository:
    """
    Repository for managing trained models, including versioning, metadata,
    and model lifecycle management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model repository
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger("model_repository")
        
        # Base directory for model storage
        self.base_dir = Path(config.get("model_dir", "models"))
        if not self.base_dir.is_absolute():
            # Use config paths if available
            if hasattr(config, "paths") and hasattr(config.paths, "base_dir"):
                self.base_dir = Path(config.paths.base_dir) / self.base_dir
        
        # Create base directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Model type directories
        self.types = {
            "academic": self.base_dir / "academic",
            "numeric": self.base_dir / "numeric",
            "visual": self.base_dir / "visual"
        }
        
        # Create model type directories
        for dir_path in self.types.values():
            dir_path.mkdir(exist_ok=True)
        
        # Load model index
        self.model_index = self._load_model_index()
        
        self.logger.info(f"Model repository initialized at {self.base_dir}")
    
    def _load_model_index(self) -> Dict[str, List[ModelVersion]]:
        """
        Load model index from disk
        
        Returns:
            Dictionary mapping model types to lists of model versions
        """
        index_path = self.base_dir / "model_index.json"
        
        if not index_path.exists():
            # Create empty index
            index = {model_type: [] for model_type in self.types.keys()}
            self._save_model_index(index)
            return index
        
        try:
            with open(index_path, "r") as f:
                data = json.load(f)
            
            # Convert dictionaries to ModelVersion objects
            index = {}
            for model_type, versions in data.items():
                index[model_type] = [
                    ModelVersion(
                        version=v["version"],
                        metadata=ModelMetadata(**v["metadata"]),
                        path=v["path"]
                    )
                    for v in versions
                ]
            
            return index
        
        except Exception as e:
            self.logger.error(f"Error loading model index: {str(e)}")
            return {model_type: [] for model_type in self.types.keys()}
    
    def _save_model_index(self, index: Dict[str, List[ModelVersion]]) -> None:
        """
        Save model index to disk
        
        Args:
            index: Dictionary mapping model types to lists of model versions
        """
        index_path = self.base_dir / "model_index.json"
        
        try:
            # Convert ModelVersion objects to dictionaries
            data = {}
            for model_type, versions in index.items():
                data[model_type] = [
                    {
                        "version": v.version,
                        "metadata": v.metadata.__dict__,
                        "path": v.path
                    }
                    for v in versions
                ]
            
            with open(index_path, "w") as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Model index saved to {index_path}")
        
        except Exception as e:
            self.logger.error(f"Error saving model index: {str(e)}")
    
    def register_model(self, 
                      model_type: str, 
                      model_path: str,
                      metadata: Optional[Dict[str, Any]] = None) -> ModelVersion:
        """
        Register a new model in the repository
        
        Args:
            model_type: Type of model (academic, numeric, visual)
            model_path: Path to the model file
            metadata: Optional metadata about the model
            
        Returns:
            Model version information
        """
        if model_type not in self.types:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create metadata
        if metadata is None:
            metadata = {}
        
        model_metadata = ModelMetadata(
            created_at=datetime.datetime.now().isoformat(),
            updated_at=datetime.datetime.now().isoformat(),
            metrics=metadata.get("metrics", {}),
            parameters=metadata.get("parameters", {}),
            description=metadata.get("description", ""),
            tags=metadata.get("tags", [])
        )
        
        # Determine version number
        versions = self.model_index.get(model_type, [])
        version = len(versions) + 1
        
        # Create model directory
        model_dir = self.types[model_type] / f"v{version}"
        model_dir.mkdir(exist_ok=True)
        
        # Copy model file to repository
        dest_path = model_dir / Path(model_path).name
        try:
            shutil.copy2(model_path, dest_path)
        except Exception as e:
            self.logger.error(f"Error copying model file: {str(e)}")
            raise
        
        # Save metadata
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(model_metadata.__dict__, f, indent=2)
        
        # Create model version
        model_version = ModelVersion(
            version=version,
            metadata=model_metadata,
            path=str(dest_path)
        )
        
        # Update index
        self.model_index.setdefault(model_type, []).append(model_version)
        self._save_model_index(self.model_index)
        
        self.logger.info(f"Registered {model_type} model v{version} at {dest_path}")
        return model_version
    
    def get_model(self, model_type: str, version: Optional[int] = None) -> Optional[ModelVersion]:
        """
        Get a specific model version
        
        Args:
            model_type: Type of model (academic, numeric, visual)
            version: Optional version number, latest if None
            
        Returns:
            Model version information or None if not found
        """
        if model_type not in self.model_index:
            self.logger.error(f"Unknown model type: {model_type}")
            return None
        
        versions = self.model_index[model_type]
        
        if not versions:
            self.logger.error(f"No models found for type: {model_type}")
            return None
        
        if version is None:
            # Return latest version
            return versions[-1]
        
        # Find specific version
        for v in versions:
            if v.version == version:
                return v
        
        self.logger.error(f"Model version {version} not found for type: {model_type}")
        return None
    
    def list_models(self, model_type: Optional[str] = None) -> Dict[str, List[ModelVersion]]:
        """
        List available models
        
        Args:
            model_type: Optional type to filter by
            
        Returns:
            Dictionary mapping model types to lists of versions
        """
        if model_type is not None:
            if model_type not in self.model_index:
                return {}
            return {model_type: self.model_index[model_type]}
        
        return self.model_index
    
    def delete_model(self, model_type: str, version: int) -> bool:
        """
        Delete a model from the repository
        
        Args:
            model_type: Type of model (academic, numeric, visual)
            version: Version number to delete
            
        Returns:
            True if successfully deleted, False otherwise
        """
        if model_type not in self.model_index:
            self.logger.error(f"Unknown model type: {model_type}")
            return False
        
        # Find model version
        versions = self.model_index[model_type]
        version_obj = None
        
        for i, v in enumerate(versions):
            if v.version == version:
                version_obj = v
                index = i
                break
        
        if version_obj is None:
            self.logger.error(f"Model version {version} not found for type: {model_type}")
            return False
        
        # Delete model directory
        try:
            model_dir = Path(version_obj.path).parent
            shutil.rmtree(model_dir)
        except Exception as e:
            self.logger.error(f"Error deleting model directory: {str(e)}")
            return False
        
        # Update index
        versions.pop(index)
        self._save_model_index(self.model_index)
        
        self.logger.info(f"Deleted {model_type} model v{version}")
        return True
    
    def update_metadata(self, 
                      model_type: str, 
                      version: int,
                      metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a model
        
        Args:
            model_type: Type of model (academic, numeric, visual)
            version: Version number
            metadata: New metadata fields to update
            
        Returns:
            True if successfully updated, False otherwise
        """
        if model_type not in self.model_index:
            self.logger.error(f"Unknown model type: {model_type}")
            return False
        
        # Find model version
        versions = self.model_index[model_type]
        version_obj = None
        
        for i, v in enumerate(versions):
            if v.version == version:
                version_obj = v
                index = i
                break
        
        if version_obj is None:
            self.logger.error(f"Model version {version} not found for type: {model_type}")
            return False
        
        # Update metadata
        model_metadata = version_obj.metadata
        
        # Update each field if provided
        if "metrics" in metadata:
            model_metadata.metrics.update(metadata["metrics"])
        
        if "parameters" in metadata:
            model_metadata.parameters.update(metadata["parameters"])
        
        if "description" in metadata:
            model_metadata.description = metadata["description"]
        
        if "tags" in metadata:
            model_metadata.tags = metadata["tags"]
        
        model_metadata.updated_at = datetime.datetime.now().isoformat()
        
        # Save metadata
        try:
            model_dir = Path(version_obj.path).parent
            with open(model_dir / "metadata.json", "w") as f:
                json.dump(model_metadata.__dict__, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metadata: {str(e)}")
            return False
        
        # Update index
        versions[index] = ModelVersion(
            version=version_obj.version,
            metadata=model_metadata,
            path=version_obj.path
        )
        
        self._save_model_index(self.model_index)
        
        self.logger.info(f"Updated metadata for {model_type} model v{version}")
        return True 
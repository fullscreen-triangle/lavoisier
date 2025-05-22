from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List, Optional
import json
from pathlib import Path
import os


@dataclass
class ProcessingConfig:
    resolution: Tuple[int, int] = (1024, 1024)
    mz_range: Tuple[float, float] = (100, 1000)
    rt_window: int = 30
    feature_dimension: int = 128
    batch_size: int = 10  # Default batch size for processing
    use_ml: bool = True  # Whether to use ML models

    # Quality control parameters
    mass_accuracy_threshold: float = 0.002  # Da
    intensity_threshold: float = 1000
    signal_to_noise_threshold: float = 3.0

    # Database parameters
    similarity_threshold: float = 0.8
    max_results: int = 10

    @classmethod
    def from_json(cls, path: str) -> 'ProcessingConfig':
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)


@dataclass
class MSParameters:
    """Parameters for MS data processing"""
    ms1_threshold: float = 1000.0
    ms2_threshold: float = 100.0
    mz_tolerance: float = 0.01
    rt_tolerance: float = 0.5
    min_intensity: float = 500.0
    output_dir: str = "output"
    n_workers: int = 0  # 0 means use all available cores


@dataclass
class DistributedConfig:
    """Configuration for distributed computing"""
    use_ray: bool = True
    use_dask: bool = True
    memory_fraction: float = 0.8  # Fraction of total memory to use
    threads_per_worker: int = 2
    n_workers: int = 0  # 0 means auto-detect based on CPU cores


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    file: str = "logs/lavoisier.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_output: bool = True


@dataclass
class PathConfig:
    """Configuration for file paths"""
    input_dir: str = "data/input"
    output_dir: str = "data/output"
    model_dir: str = "models"
    cache_dir: str = ".cache"
    temp_dir: str = "temp"


@dataclass
class LLMConfig:
    """Configuration for LLM integration"""
    enabled: bool = True
    local_model: Optional[str] = "ollama/mistral"
    provider: str = "openai"  # or "anthropic", "ollama"
    api_key: Optional[str] = None
    model_name: str = "gpt-4-turbo"  # or "claude-3-sonnet" for Anthropic
    temperature: float = 0.7
    max_tokens: int = 4000
    context_window: int = 8000
    embedding_model: str = "text-embedding-3-small"


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation"""
    video_fps: int = 30
    image_format: str = "png"  # or "jpg"
    colormap: str = "viridis"
    resolution: Tuple[int, int] = (1920, 1080)
    dpi: int = 300
    interactive: bool = True


@dataclass
class CachingConfig:
    """Cache configuration for intermediate results"""
    enabled: bool = True
    strategy: str = "memory_disk"  # Options: memory, disk, memory_disk, none
    max_memory_size_mb: int = 1024  # Max memory cache size in MB
    disk_cache_path: str = "./cache"
    ttl_minutes: int = 60  # Time-to-live for cached items
    compress: bool = True  # Whether to compress cached data
    levels: List[str] = field(default_factory=lambda: ["raw", "processed", "analyzed"])


@dataclass
class MetacognitiveConfig:
    """Configuration for metacognitive layer"""
    enable_continuous_learning: bool = True
    knowledge_distillation: bool = True
    query_complexity_levels: List[str] = field(default_factory=lambda: ["basic", "intermediate", "advanced"])
    feedback_threshold: float = 0.7
    max_training_iterations: int = 10
    model_update_frequency: str = "daily"  # "hourly", "daily", "weekly", "monthly"


class GlobalConfig:
    def __init__(self):
        self.processing = ProcessingConfig()
        self.ms_parameters = MSParameters()
        self.distributed = DistributedConfig()
        self.logging = LoggingConfig()
        self.paths = PathConfig()
        self.llm = LLMConfig()
        self.visualization = VisualizationConfig()
        self.caching = CachingConfig()
        self.metacognitive = MetacognitiveConfig()
        self.cache_dir = Path.home() / ".ms_image_analyzer"
        self.cache_dir.mkdir(exist_ok=True)

    def save(self, path: str):
        config_dict = {
            "processing": self.processing.__dict__,
            "ms_parameters": self.ms_parameters.__dict__,
            "distributed": self.distributed.__dict__,
            "logging": self.logging.__dict__,
            "paths": self.paths.__dict__,
            "llm": self.llm.__dict__,
            "visualization": self.visualization.__dict__,
            "caching": self.caching.__dict__,
            "metacognitive": self.metacognitive.__dict__,
            "cache_dir": str(self.cache_dir)
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def load(cls, path: str) -> 'GlobalConfig':
        instance = cls()
        
        # Check file extension and load JSON
        if not path.endswith(".json"):
            raise ValueError(f"Unsupported config file format: {path}. Only JSON files (.json) are supported.")
            
        with open(path, 'r') as f:
            config_dict = json.load(f)
            
        # Load each section if present
        if "processing" in config_dict:
            instance.processing = ProcessingConfig(**config_dict["processing"])
        
        if "ms_parameters" in config_dict:
            instance.ms_parameters = MSParameters(**config_dict["ms_parameters"])
        
        if "distributed" in config_dict:
            instance.distributed = DistributedConfig(**config_dict["distributed"])
        
        if "logging" in config_dict:
            instance.logging = LoggingConfig(**config_dict["logging"])
        
        if "paths" in config_dict:
            instance.paths = PathConfig(**config_dict["paths"])
        
        if "llm" in config_dict:
            instance.llm = LLMConfig(**config_dict["llm"])
        
        if "visualization" in config_dict:
            instance.visualization = VisualizationConfig(**config_dict["visualization"])
        
        if "caching" in config_dict:
            instance.caching = CachingConfig(**config_dict["caching"])
        
        if "metacognitive" in config_dict:
            instance.metacognitive = MetacognitiveConfig(**config_dict["metacognitive"])
        
        if "cache_dir" in config_dict:
            instance.cache_dir = Path(config_dict["cache_dir"])
            
        return instance
        
    def update_paths(self, base_dir: str) -> None:
        """Update relative paths to absolute paths based on a base directory"""
        def make_absolute(path_str):
            path = Path(path_str)
            if not path.is_absolute():
                return str(Path(base_dir) / path)
            return path_str
            
        # Make cache_dir absolute if it's relative
        if not self.cache_dir.is_absolute():
            self.cache_dir = Path(base_dir) / self.cache_dir
            
        # Make paths in PathConfig absolute
        self.paths.input_dir = make_absolute(self.paths.input_dir)
        self.paths.output_dir = make_absolute(self.paths.output_dir)
        self.paths.model_dir = make_absolute(self.paths.model_dir)
        self.paths.cache_dir = make_absolute(self.paths.cache_dir)
        self.paths.temp_dir = make_absolute(self.paths.temp_dir)
        
        # Make paths in other configs absolute
        self.logging.file = make_absolute(self.logging.file)
        self.caching.disk_cache_path = make_absolute(self.caching.disk_cache_path)
        self.ms_parameters.output_dir = make_absolute(self.ms_parameters.output_dir)
    
    # For backward compatibility
    @classmethod
    def from_yaml(cls, path: str) -> 'GlobalConfig':
        """This method is deprecated. Use load() instead."""
        raise NotImplementedError("YAML support has been deprecated. Please use JSON config files instead.")

# Default configuration
CONFIG = GlobalConfig()

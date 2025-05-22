from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List, Optional
import json
from pathlib import Path
import os


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
class ProcessingConfig:
    """Configuration for spectral processing"""
    resolution: Tuple[int, int] = (1024, 1024)
    mz_range: Tuple[float, float] = (100, 1000)
    rt_window: int = 30
    feature_dimension: int = 128

    # Quality control parameters
    mass_accuracy_threshold: float = 0.002  # Da
    intensity_threshold: float = 1000
    signal_to_noise_threshold: float = 3.0

    # Database parameters
    similarity_threshold: float = 0.8
    max_results: int = 10

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProcessingConfig':
        return cls(**config_dict)


@dataclass
class DistributedConfig:
    """Configuration for distributed computing"""
    use_ray: bool = True
    use_dask: bool = True
    memory_fraction: float = 0.8  # Fraction of total memory to use
    threads_per_worker: int = 2
    n_workers: int = 0  # 0 means auto-detect based on CPU cores


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
class MetacognitiveConfig:
    """Configuration for metacognitive layer"""
    enable_continuous_learning: bool = True
    knowledge_distillation: bool = True
    query_complexity_levels: List[str] = field(default_factory=lambda: ["basic", "intermediate", "advanced"])
    feedback_threshold: float = 0.7
    max_training_iterations: int = 10
    model_update_frequency: str = "daily"  # "hourly", "daily", "weekly", "monthly"


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
class LavoisierConfig:
    """Master configuration for Lavoisier"""
    ms_parameters: MSParameters = field(default_factory=MSParameters)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    metacognitive: MetacognitiveConfig = field(default_factory=MetacognitiveConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)

    @classmethod
    def from_json(cls, path: str) -> 'LavoisierConfig':
        """Load configuration from JSON file"""
        try:
            with open(path, 'r') as f:
                config_dict = json.load(f)
            
            instance = cls()
            
            # Load each section if present
            if "ms_parameters" in config_dict:
                instance.ms_parameters = MSParameters(**config_dict["ms_parameters"])
            
            if "processing" in config_dict:
                instance.processing = ProcessingConfig(**config_dict["processing"])
            
            if "distributed" in config_dict:
                instance.distributed = DistributedConfig(**config_dict["distributed"])
            
            if "llm" in config_dict:
                instance.llm = LLMConfig(**config_dict["llm"])
            
            if "visualization" in config_dict:
                instance.visualization = VisualizationConfig(**config_dict["visualization"])
            
            if "logging" in config_dict:
                instance.logging = LoggingConfig(**config_dict["logging"])
            
            if "paths" in config_dict:
                instance.paths = PathConfig(**config_dict["paths"])
            
            if "metacognitive" in config_dict:
                instance.metacognitive = MetacognitiveConfig(**config_dict["metacognitive"])
            
            if "caching" in config_dict:
                instance.caching = CachingConfig(**config_dict["caching"])
            
            return instance
            
        except Exception as e:
            raise RuntimeError(f"Error loading config file: {str(e)}")

    def to_json(self, path: str) -> None:
        """Save configuration to JSON file"""
        config_dict = {
            "ms_parameters": self.ms_parameters.__dict__,
            "processing": self.processing.__dict__,
            "distributed": self.distributed.__dict__,
            "llm": self.llm.__dict__,
            "visualization": self.visualization.__dict__,
            "logging": self.logging.__dict__,
            "paths": self.paths.__dict__,
            "metacognitive": self.metacognitive.__dict__,
            "caching": self.caching.__dict__,
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)

    # Deprecated methods for backward compatibility
    @classmethod
    def from_yaml(cls, path: str) -> 'LavoisierConfig':
        """This method is deprecated. Use from_json() instead."""
        raise NotImplementedError("YAML support has been deprecated. Please use JSON config files instead.")
        
    def to_yaml(self, path: str) -> None:
        """This method is deprecated. Use to_json() instead."""
        raise NotImplementedError("YAML support has been deprecated. Please use JSON config files instead.")

    def update_paths(self, base_dir: str) -> None:
        """Update relative paths to absolute paths based on a base directory"""
        def make_absolute(path_str):
            path = Path(path_str)
            if not path.is_absolute():
                return str(Path(base_dir) / path)
            return path_str
        
        self.paths.input_dir = make_absolute(self.paths.input_dir)
        self.paths.output_dir = make_absolute(self.paths.output_dir)
        self.paths.model_dir = make_absolute(self.paths.model_dir)
        self.paths.cache_dir = make_absolute(self.paths.cache_dir)
        self.paths.temp_dir = make_absolute(self.paths.temp_dir)
        self.logging.file = make_absolute(self.logging.file)
        self.caching.disk_cache_path = make_absolute(self.caching.disk_cache_path)


# Default configuration
CONFIG = LavoisierConfig()

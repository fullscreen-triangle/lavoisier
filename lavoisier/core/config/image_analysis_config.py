from dataclasses import dataclass
from typing import Tuple, Dict, Any
import json
from pathlib import Path


@dataclass
class ProcessingConfig:
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
    def from_json(cls, path: str) -> 'ProcessingConfig':
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)


class GlobalConfig:
    def __init__(self):
        self.processing = ProcessingConfig()
        self.cache_dir = Path.home() / ".ms_image_analyzer"
        self.cache_dir.mkdir(exist_ok=True)

    def save(self, path: str):
        config_dict = {
            "processing": self.processing.__dict__,
            "cache_dir": str(self.cache_dir)
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def load(cls, path: str) -> 'GlobalConfig':
        instance = cls()
        with open(path, 'r') as f:
            config_dict = json.load(f)
        instance.processing = ProcessingConfig(**config_dict["processing"])
        instance.cache_dir = Path(config_dict["cache_dir"])
        return instance


# Default configuration
CONFIG = GlobalConfig()

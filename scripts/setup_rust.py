#!/usr/bin/env python3
"""
Setup script for building Rust extensions for Lavoisier
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and check for errors"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    
    return result.stdout

def check_rust_installation():
    """Check if Rust is installed"""
    try:
        output = run_command(["cargo", "--version"])
        print(f"Found Rust: {output.strip()}")
    except FileNotFoundError:
        print("Error: Rust/Cargo not found. Please install Rust from https://rustup.rs/")
        sys.exit(1)

def build_rust_extensions():
    """Build all Rust extensions"""
    print("Building Rust extensions...")
    
    # Build workspace in release mode
    run_command(["cargo", "build", "--release"], cwd=".")
    
    # Build Python wheels for each crate
    crates = ["lavoisier-core", "lavoisier-io"]
    
    for crate in crates:
        if Path(crate).exists():
            print(f"\nBuilding Python extension for {crate}...")
            
            # Use maturin to build Python wheels
            try:
                run_command(["maturin", "build", "--release"], cwd=crate)
            except FileNotFoundError:
                print("Warning: maturin not found. Installing...")
                run_command([sys.executable, "-m", "pip", "install", "maturin"])
                run_command(["maturin", "build", "--release"], cwd=crate)

def install_extensions():
    """Install the built extensions"""
    print("\nInstalling Rust extensions...")
    
    # Find and install wheels
    target_dir = Path("target/wheels")
    if target_dir.exists():
        for wheel in target_dir.glob("*.whl"):
            print(f"Installing {wheel}")
            run_command([sys.executable, "-m", "pip", "install", str(wheel), "--force-reinstall"])
    
    # Alternative: use maturin develop for development
    crates = ["lavoisier-core", "lavoisier-io"]
    for crate in crates:
        if Path(crate).exists():
            print(f"Installing {crate} in development mode...")
            run_command(["maturin", "develop"], cwd=crate)

def run_tests():
    """Run Rust tests"""
    print("\nRunning Rust tests...")
    run_command(["cargo", "test", "--workspace"])

def create_python_bindings():
    """Create Python stub files for better IDE support"""
    print("\nCreating Python stub files...")
    
    stub_content = '''"""
Lavoisier Rust Extensions

High-performance mass spectrometry data processing using Rust.
"""

from typing import List, Optional, Dict, Any
import numpy as np

class PySpectrum:
    """High-performance spectrum data structure"""
    
    def __init__(self, mz: np.ndarray, intensity: np.ndarray, 
                 retention_time: float, ms_level: int, scan_id: str) -> None: ...
    
    @property
    def mz(self) -> np.ndarray: ...
    
    @property
    def intensity(self) -> np.ndarray: ...
    
    @property
    def retention_time(self) -> float: ...
    
    @property
    def ms_level(self) -> int: ...
    
    @property
    def scan_id(self) -> str: ...
    
    def filter_intensity(self, threshold: float) -> None: ...
    
    def filter_mz_range(self, min_mz: float, max_mz: float) -> None: ...
    
    def normalize_intensity(self, method: str) -> None: ...
    
    def find_peaks(self, min_intensity: float, window_size: int) -> List["PyPeak"]: ...

class PyPeak:
    """Peak data structure"""
    
    def __init__(self, mz: float, intensity: float, retention_time: float) -> None: ...
    
    @property
    def mz(self) -> float: ...
    
    @property
    def intensity(self) -> float: ...
    
    @property
    def retention_time(self) -> float: ...
    
    @property
    def peak_width(self) -> Optional[float]: ...
    
    @property
    def area(self) -> Optional[float]: ...
    
    @property
    def signal_to_noise(self) -> Optional[float]: ...

class PySpectrumCollection:
    """High-performance spectrum collection"""
    
    def __init__(self) -> None: ...
    
    def add_spectrum(self, spectrum: PySpectrum) -> None: ...
    
    def get_spectrum(self, scan_id: str) -> Optional[PySpectrum]: ...
    
    def get_spectra_in_rt_range(self, min_rt: float, max_rt: float) -> List[PySpectrum]: ...
    
    def len(self) -> int: ...

class PyMzMLReader:
    """High-performance mzML file reader"""
    
    def __init__(self, file_path: str) -> None: ...
    
    def build_index(self) -> None: ...
    
    def read_spectra(self) -> PySpectrumCollection: ...
    
    def get_spectrum(self, scan_id: str) -> Optional[PySpectrum]: ...
    
    def get_metadata(self) -> Dict[str, str]: ...

class PyPeakDetector:
    """High-performance peak detection"""
    
    def __init__(self, min_intensity: float, min_signal_to_noise: float, window_size: int) -> None: ...
    
    def detect_peaks(self, mz: List[float], intensity: List[float], retention_time: float) -> List[PyPeak]: ...

def batch_filter_intensity(spectra: List[PySpectrum], threshold: float) -> List[PySpectrum]: ...

def batch_normalize_spectra(spectra: List[PySpectrum], method: str) -> List[PySpectrum]: ...
'''
    
    # Write stub file
    stub_file = Path("lavoisier/rust_extensions.pyi")
    stub_file.parent.mkdir(exist_ok=True)
    stub_file.write_text(stub_content)
    
    # Create __init__.py if it doesn't exist
    init_file = Path("lavoisier/__init__.py")
    if not init_file.exists():
        init_content = '''"""
Lavoisier: High-performance mass spectrometry data analysis
"""

try:
    # Import Rust extensions if available
    from .rust_extensions import *
    RUST_EXTENSIONS_AVAILABLE = True
except ImportError:
    RUST_EXTENSIONS_AVAILABLE = False
    print("Warning: Rust extensions not available. Install with: python setup_rust.py")

__version__ = "0.1.0"
'''
        init_file.write_text(init_content)

def main():
    """Main setup function"""
    print("Setting up Lavoisier Rust extensions...")
    
    # Check prerequisites
    check_rust_installation()
    
    # Build extensions
    build_rust_extensions()
    
    # Install extensions
    install_extensions()
    
    # Run tests
    run_tests()
    
    # Create Python bindings
    create_python_bindings()
    
    print("\n✅ Rust extensions setup complete!")
    print("\nUsage example:")
    print("```python")
    print("from lavoisier.rust_extensions import PyMzMLReader, PySpectrum")
    print("reader = PyMzMLReader('data.mzML')")
    print("reader.build_index()")
    print("spectra = reader.read_spectra()")
    print("print(f'Loaded {spectra.len()} spectra')")
    print("```")

if __name__ == "__main__":
    main() 
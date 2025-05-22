# visual.py
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
from datetime import datetime

from ms_image_analyzer.MSImageDatabase import MSImageDatabase
from ms_image_analyzer.MSImageProcessor import MSImageProcessor
from ms_image_analyzer.MSVideoAnalyzer import MSVideoAnalyzer


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"ms_analysis_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> dict:
    """
    Load and validate configuration from JSON file
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Convert relative paths to absolute paths
        config['input_directory'] = os.path.abspath(os.path.expanduser(config['input_directory']))
        config['output_directory'] = os.path.abspath(os.path.expanduser(config['output_directory']))
        config['log_directory'] = os.path.abspath(os.path.expanduser(config['log_directory']))
        config['video_output_path'] = os.path.abspath(os.path.expanduser(config['video_output_path']))

        # Validate required fields
        required_fields = [
            'input_directory',
            'output_directory',
            'log_directory',
            'video_output_path',
            'file_types',
            'n_workers',
            'ms_parameters'
        ]

        for field in required_fields:
            if field not in config:
                raise KeyError(f"Missing required field in config: {field}")

        return config

    except Exception as e:
        raise RuntimeError(f"Error loading config file: {str(e)}")



def process_spectra(input_paths: List[Path], output_path: Path, n_workers: int = 4) -> List:
    logging.info(f"Starting spectra processing with {n_workers} workers")
    processor = MSImageProcessor()
    processed_spectra = []

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(processor.load_spectrum, path) for path in input_paths]

            for future in tqdm(futures, desc="Processing spectra"):
                processed_spectra.extend(future.result())

        output_path.parent.mkdir(parents=True, exist_ok=True)
        processor.save_processed_spectra(processed_spectra, output_path)
        logging.info(f"Successfully processed {len(processed_spectra)} spectra")
        return processed_spectra
    except Exception as e:
        logging.error(f"Error during spectra processing: {str(e)}")
        raise


def build_image_database(processed_spectra: List, db_path: Path) -> MSImageDatabase:
    logging.info("Building image database")
    database = MSImageDatabase()

    try:
        spectra_list = [(spectrum.mz_array, spectrum.intensity_array, spectrum.metadata)
                        for spectrum in processed_spectra]
        database.batch_add_spectra(spectra_list)

        db_path.parent.mkdir(parents=True, exist_ok=True)
        database.save_database(db_path)
        logging.info(f"Database successfully built and saved to {db_path}")
        return database
    except Exception as e:
        logging.error(f"Error building database: {str(e)}")
        raise


def analyze_video(input_data: List[Tuple[np.ndarray, np.ndarray]], output_video_path: str):
    logging.info("Starting video analysis")
    analyzer = MSVideoAnalyzer()

    try:
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
        analyzer.extract_spectra_as_video(input_data, output_video_path)
        logging.info(f"Video analysis completed. Output saved to {output_video_path}")
    except Exception as e:
        logging.error(f"Error during video analysis: {str(e)}")
        raise


def main(config_path: str):
    try:
        # Load configuration
        config = load_config(config_path)

        # Setup logging
        setup_logging(Path(config['log_directory']))

        # Log configuration details
        logging.info(f"Configuration loaded from: {config_path}")
        logging.info(f"Input directory: {config['input_directory']}")
        logging.info(f"Output directory: {config['output_directory']}")

        # Prepare paths
        input_dir = Path(config['input_directory'])
        output_dir = Path(config['output_directory'])

        # Verify input directory exists
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find input files
        input_paths = list(input_dir.glob(config['file_types'][0]))
        logging.info(f"Found {len(input_paths)} input files")

        # Process spectra and store results
        output_path = output_dir / "processed_spectra.h5"
        processed_spectra = process_spectra(input_paths, output_path, config['n_workers'])

        # Initialize new database for this analysis
        database = MSImageDatabase(
            resolution=(1024, 1024),  # Default resolution
            feature_dimension=128  # Default feature dimension
        )

        # Add processed spectra to database
        logging.info("Adding spectra to database...")
        spectra_list = [(spectrum.mz_array, spectrum.intensity_array, spectrum.metadata)
                        for spectrum in processed_spectra]
        database.batch_add_spectra(spectra_list)

        # Save database in the output directory
        db_path = output_dir / "spectrum_database"
        db_path.mkdir(exist_ok=True)
        logging.info(f"Saving database to {db_path}")
        database.save_database(str(db_path))

        # Create visualization video
        video_path = Path(config['video_output_path']) / "analysis_video.mp4"
        video_input_data = [(spectrum.mz_array, spectrum.intensity_array)
                            for spectrum in processed_spectra]
        analyze_video(video_input_data, str(video_path))

        logging.info("Analysis completed successfully")

    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "../config/visual_config.json")
    main(config_path)





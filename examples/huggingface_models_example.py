#!/usr/bin/env python
"""
Example script demonstrating the use of Hugging Face models in Lavoisier.
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Add parent directory to path to import lavoisier
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lavoisier.models import (
    MODEL_REGISTRY, 
    SpecTUSModel, 
    CMSSPModel, 
    ChemBERTaModel,
    create_spectus_model,
    create_cmssp_model,
    create_chemberta_model
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_example_spectrum() -> Tuple[np.ndarray, np.ndarray]:
    """Create an example mass spectrum for demonstration.
    
    Returns:
        Tuple of (mz_values, intensity_values) arrays.
    """
    # This is a simplified example spectrum, not a real compound
    mz_values = np.array([
        50, 51, 52, 55, 57, 65, 69, 77, 78, 79, 91, 92, 93, 105, 106, 107, 119, 120, 121, 134
    ])
    intensity_values = np.array([
        25, 5, 10, 15, 20, 30, 10, 45, 15, 5, 100, 10, 5, 90, 8, 5, 60, 5, 8, 50
    ])
    
    return mz_values, intensity_values

def plot_spectrum(mz_values: np.ndarray, intensity_values: np.ndarray, title: str = "Mass Spectrum"):
    """Plot a mass spectrum.
    
    Args:
        mz_values: m/z values of the spectrum.
        intensity_values: Intensity values of the spectrum.
        title: Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.stem(mz_values, intensity_values, basefmt=" ")
    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("example_spectrum.png")
    plt.close()

def example_spectus_model():
    """Example of using the SpecTUS model for structure prediction."""
    logger.info("Example: Using SpecTUS model to predict molecular structure from spectrum")
    
    # Create an example spectrum
    mz_values, intensity_values = create_example_spectrum()
    plot_spectrum(mz_values, intensity_values, "Example Mass Spectrum")
    
    # Initialize the SpecTUS model
    logger.info("Initializing SpecTUS model...")
    model = create_spectus_model(device="cpu")  # Use CPU for demonstration
    
    # Process the spectrum
    logger.info("Processing spectrum...")
    predicted_smiles = model.process_spectrum(
        mz_values, 
        intensity_values,
        num_beams=3,
        num_return_sequences=3
    )
    
    logger.info("Predicted SMILES structures:")
    for i, smiles in enumerate(predicted_smiles):
        logger.info(f"  Candidate {i+1}: {smiles}")

def example_cmssp_model():
    """Example of using the CMSSP model for embedding spectra and molecules."""
    logger.info("Example: Using CMSSP model for joint embedding of spectra and molecules")
    
    # Create an example spectrum
    mz_values, intensity_values = create_example_spectrum()
    
    # Example SMILES strings
    smiles_list = [
        "CC1=CC=CC=C1",  # Toluene
        "CC(=O)OC1=CC=CC=C1",  # Acetylsalicylic acid
        "C1=CC=C(C=C1)C=O"  # Benzaldehyde
    ]
    
    # Initialize the CMSSP model
    logger.info("Initializing CMSSP model...")
    model = create_cmssp_model(device="cpu")  # Use CPU for demonstration
    
    # Encode the spectrum
    logger.info("Encoding spectrum...")
    spectrum_embedding = model.encode_spectrum(mz_values, intensity_values)
    
    # Encode SMILES
    logger.info("Encoding SMILES...")
    smiles_embeddings = model.encode_smiles(smiles_list)
    
    # Compute similarities
    logger.info("Computing similarities between spectrum and molecules:")
    for i, smiles in enumerate(smiles_list):
        similarity = model.compute_similarity(spectrum_embedding, smiles_embeddings[i])
        logger.info(f"  Similarity to {smiles}: {similarity:.4f}")

def example_chemberta_model():
    """Example of using the ChemBERTa model for SMILES embedding."""
    logger.info("Example: Using ChemBERTa model for SMILES embedding")
    
    # Example SMILES strings
    smiles_list = [
        "CC1=CC=CC=C1",  # Toluene
        "CC(=O)OC1=CC=CC=C1",  # Acetylsalicylic acid
        "C1=CC=C(C=C1)C=O",  # Benzaldehyde
        "C1=CC=C(C=C1)C(=O)O"  # Benzoic acid
    ]
    
    # Initialize the ChemBERTa model
    logger.info("Initializing ChemBERTa model...")
    model = create_chemberta_model(device="cpu")  # Use CPU for demonstration
    
    # Encode SMILES
    logger.info("Encoding SMILES...")
    embeddings = model.encode_smiles(smiles_list)
    
    # Calculate pairwise similarities
    logger.info("Calculating pairwise similarities:")
    for i in range(len(smiles_list)):
        for j in range(i+1, len(smiles_list)):
            similarity = model.compute_similarity(embeddings[i], embeddings[j])
            logger.info(f"  Similarity between {smiles_list[i]} and {smiles_list[j]}: {similarity:.4f}")

def main():
    """Run all examples."""
    logger.info("Starting Hugging Face models examples")
    
    example_spectus_model()
    print("\n" + "-" * 80 + "\n")
    
    example_cmssp_model()
    print("\n" + "-" * 80 + "\n")
    
    example_chemberta_model()
    
    logger.info("All examples completed")

if __name__ == "__main__":
    main() 
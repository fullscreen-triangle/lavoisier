"""
Lavoisier Models - Repository for trained models and knowledge distillation
"""

from lavoisier.models.repository import ModelRepository
from lavoisier.models.distillation import KnowledgeDistiller
from lavoisier.models.versioning import ModelVersion, ModelMetadata
from lavoisier.models.registry import ModelRegistry, MODEL_REGISTRY, ModelType, HuggingFaceModelInfo

# Import Hugging Face model wrappers
from lavoisier.models.spectral_transformers import SpecTUSModel, create_spectus_model
from lavoisier.models.embedding_models import CMSSPModel, create_cmssp_model
from lavoisier.models.chemical_language_models import (
    ChemBERTaModel, 
    MoLFormerModel, 
    PubChemDeBERTaModel,
    create_chemberta_model,
    create_molformer_model,
    create_pubchem_deberta_model
)

__all__ = [
    'ModelRepository',
    'KnowledgeDistiller',
    'ModelVersion',
    'ModelMetadata',
    'ModelRegistry',
    'MODEL_REGISTRY',
    'ModelType',
    'HuggingFaceModelInfo',
    # Spectral models
    'SpecTUSModel',
    'create_spectus_model',
    # Embedding models
    'CMSSPModel',
    'create_cmssp_model',
    # Chemical language models
    'ChemBERTaModel',
    'MoLFormerModel',
    'PubChemDeBERTaModel',
    'create_chemberta_model',
    'create_molformer_model',
    'create_pubchem_deberta_model',
] 
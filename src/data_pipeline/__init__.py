"""
src/data_pipeline/__init__.py
Point d entree du module data_pipeline.
"""
from src.data_pipeline.cleaner import DataCleaner
from src.data_pipeline.domains.adapters import get_adapter
from src.data_pipeline.features import FeatureEngineer
from src.data_pipeline.loader import DataLoader
from src.data_pipeline.pipeline import DataPipeline, PipelineResult
from src.data_pipeline.sequences import DataNormalizer, SequenceBuilder
from src.data_pipeline.synthetic import SyntheticDataGenerator

__all__ = [
    "DataPipeline",
    "PipelineResult",
    "DataLoader",
    "DataCleaner",
    "FeatureEngineer",
    "DataNormalizer",
    "SequenceBuilder",
    "SyntheticDataGenerator",
    "get_adapter",
]

"""
src/data_pipeline/domains/__init__.py
"""
from src.data_pipeline.domains.adapters import (
    FinanceAdapter,
    IndustryAdapter,
    SyntheticAdapter,
    TelecomAdapter,
    get_adapter,
)

__all__ = [
    "TelecomAdapter",
    "FinanceAdapter",
    "IndustryAdapter",
    "SyntheticAdapter",
    "get_adapter",
]

"""
src/database/__init__.py
Point d'entrée du module database.
Exporte tous les modèles pour un import simplifié.
"""
from src.database.base import Base, TimestampMixin, UUIDMixin
from src.database.models import (
    AgentRun,
    AgentType,
    Alert,
    Anomaly,
    AnomalyStatus,
    DataPoint,
    DataSource,
    DomainType,
    FeedbackStatus,
    FeedbackSuggestion,
    MLModel,
    ModelStatus,
    Prediction,
    Report,
    TimeSeries,
    User,
    UserRole,
)
from src.database.session import AsyncSessionLocal, engine, get_db, get_db_context

__all__ = [
    # Base
    "Base", "TimestampMixin", "UUIDMixin",
    # Models
    "User", "UserRole",
    "DataSource", "DomainType",
    "TimeSeries",
    "DataPoint",
    "MLModel", "ModelStatus",
    "Prediction",
    "Anomaly", "AnomalyStatus",
    "Alert",
    "AgentRun", "AgentType",
    "Report",
    "FeedbackSuggestion", "FeedbackStatus",
    # Session
    "engine", "AsyncSessionLocal", "get_db", "get_db_context",
]

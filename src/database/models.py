"""
src/database/models.py
Les 11 modèles SQLAlchemy correspondant au MCD du PFE.
Chaque classe = une table dans PostgreSQL.
"""
import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import (
    JSON, Boolean, DateTime, Float, ForeignKey,
    Integer, String, Text, UniqueConstraint, Enum,
    Index, func
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base, TimestampMixin, UUIDMixin


# ══════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════

class UserRole(str, PyEnum):
    """Rôles utilisateurs pour le contrôle d'accès."""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


class DomainType(str, PyEnum):
    """Domaine applicatif de la source de données."""
    TELECOM = "telecom"
    FINANCE = "finance"
    INDUSTRY = "industry"
    SYNTHETIC = "synthetic"


class ModelStatus(str, PyEnum):
    """Statut d'un modèle ML dans le registre."""
    TRAINING = "training"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class AnomalyStatus(str, PyEnum):
    """Statut de validation d'une anomalie détectée."""
    DETECTED = "detected"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class FeedbackStatus(str, PyEnum):
    """Statut d'une suggestion de la boucle de feedback."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class AgentType(str, PyEnum):
    """Type d'agent LangGraph."""
    SUPERVISOR = "supervisor"
    ANALYST = "analyst"
    RESEARCHER = "researcher"
    REPORTER = "reporter"


# ══════════════════════════════════════════════════════
# TABLE 1 — USERS
# Utilisateurs de la plateforme avec leurs rôles
# ══════════════════════════════════════════════════════

class User(Base, UUIDMixin, TimestampMixin):
    """
    Utilisateurs de la plateforme.
    Gère l'authentification JWT et les permissions.
    """
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255), nullable=False
    )
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole), default=UserRole.VIEWER, nullable=False
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relations
    reports: Mapped[list["Report"]] = relationship(
        back_populates="created_by_user", lazy="select"
    )
    feedback_decisions: Mapped[list["FeedbackSuggestion"]] = relationship(
        back_populates="decided_by_user", lazy="select"
    )

    def __repr__(self) -> str:
        return f"<User {self.username} ({self.role})>"


# ══════════════════════════════════════════════════════
# TABLE 2 — DATA SOURCES
# Sources de données connectées à la plateforme
# ══════════════════════════════════════════════════════

class DataSource(Base, UUIDMixin, TimestampMixin):
    """
    Sources de données connectées à la plateforme.
    Peut être une API, un fichier CSV, un flux temps réel.
    """
    __tablename__ = "data_sources"

    name: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True
    )
    domain: Mapped[DomainType] = mapped_column(
        Enum(DomainType), nullable=False
    )
    source_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # csv, api, stream, database
    connection_config: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )  # URL, credentials (chiffrés), paramètres
    update_frequency_minutes: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )
    last_ingestion: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relations
    time_series: Mapped[list["TimeSeries"]] = relationship(
        back_populates="source", lazy="select"
    )

    def __repr__(self) -> str:
        return f"<DataSource {self.name} ({self.domain})>"


# ══════════════════════════════════════════════════════
# TABLE 3 — TIME SERIES
# Séries temporelles associées à chaque source
# ══════════════════════════════════════════════════════

class TimeSeries(Base, UUIDMixin, TimestampMixin):
    """
    Séries temporelles associées à une source de données.
    Une source peut avoir plusieurs séries
    (ex : débit_dl, débit_ul, latence pour télécom).
    """
    __tablename__ = "time_series"

    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("data_sources.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    name: Mapped[str] = mapped_column(
        String(255), nullable=False
    )  # ex: "debit_download_mbps"
    unit: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # ex: "Mbps", "EUR", "°C"
    description: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
    sampling_interval_seconds: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    metadata_: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, name="metadata"
    )

    # Relations
    source: Mapped["DataSource"] = relationship(
        back_populates="time_series"
    )
    data_points: Mapped[list["DataPoint"]] = relationship(
        back_populates="series", lazy="dynamic"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="series", lazy="select"
    )

    __table_args__ = (
        UniqueConstraint("source_id", "name", name="uq_series_source_name"),
    )

    def __repr__(self) -> str:
        return f"<TimeSeries {self.name}>"


# ══════════════════════════════════════════════════════
# TABLE 4 — DATA POINTS ⭐ HYPERTABLE TIMESCALEDB
# Observations individuelles — cœur de la plateforme
# ══════════════════════════════════════════════════════

class DataPoint(Base):
    """
    Observations individuelles de chaque série temporelle.

    ⭐ HYPERTABLE TIMESCALEDB : cette table est convertie
    en hypertable partitionnée par mois pour des performances
    optimales sur les requêtes temporelles (x10 plus rapide).

    Pas d'UUID ici — on utilise (series_id, timestamp)
    comme clé composite pour optimiser TimescaleDB.
    """
    __tablename__ = "data_points"

    series_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("time_series.id", ondelete="CASCADE"),
        nullable=False,
        primary_key=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        primary_key=True
    )
    value: Mapped[float] = mapped_column(
        Float, nullable=False
    )
    quality_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )  # 0-1 : qualité de la mesure
    is_imputed: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )  # True si valeur imputée (manquante à l'origine)
    raw_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )

    # Relations
    series: Mapped["TimeSeries"] = relationship(
        back_populates="data_points"
    )

    __table_args__ = (
        Index("idx_datapoints_series_time", "series_id", "timestamp"),
    )


# ══════════════════════════════════════════════════════
# TABLE 5 — ML MODELS
# Registre des modèles CNN+LSTM entraînés
# ══════════════════════════════════════════════════════

class MLModel(Base, UUIDMixin, TimestampMixin):
    """
    Registre des modèles ML entraînés.
    Synchronisé avec MLflow pour la gestion des artefacts.
    """
    __tablename__ = "ml_models"

    name: Mapped[str] = mapped_column(
        String(255), nullable=False
    )
    version: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # ex: "1.0.0", "1.1.0"
    status: Mapped[ModelStatus] = mapped_column(
        Enum(ModelStatus),
        default=ModelStatus.TRAINING,
        nullable=False,
        index=True
    )
    domain: Mapped[DomainType] = mapped_column(
        Enum(DomainType), nullable=False
    )

    # Hyperparamètres
    hyperparameters: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )  # window_size, lr, batch_size, etc.

    # Métriques de performance
    metrics: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )  # {"r2": 0.95, "rmse": 0.03, "f1": 0.88}

    # Référence MLflow
    mlflow_run_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    mlflow_model_uri: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True
    )

    # Chemin ONNX
    onnx_path: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True
    )

    trained_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relations
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="model", lazy="select"
    )
    feedback_suggestions: Mapped[list["FeedbackSuggestion"]] = relationship(
        back_populates="model", lazy="select"
    )

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_name_version"),
    )

    def __repr__(self) -> str:
        return f"<MLModel {self.name} v{self.version} ({self.status})>"


# ══════════════════════════════════════════════════════
# TABLE 6 — PREDICTIONS
# Sorties du modèle CNN+LSTM
# ══════════════════════════════════════════════════════

class Prediction(Base, UUIDMixin, TimestampMixin):
    """
    Sorties du modèle CNN+LSTM.
    Stocke la valeur prédite, réelle et l'erreur
    pour alimenter la boucle de feedback.
    """
    __tablename__ = "predictions"

    series_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("time_series.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ml_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    predicted_value: Mapped[float] = mapped_column(
        Float, nullable=False
    )
    actual_value: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )  # Rempli quand la valeur réelle est disponible
    error: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )  # actual - predicted
    absolute_error: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    window_features: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )  # Features utilisées pour cette prédiction

    # Relations
    series: Mapped["TimeSeries"] = relationship(
        back_populates="predictions"
    )
    model: Mapped["MLModel"] = relationship(
        back_populates="predictions"
    )
    anomalies: Mapped[list["Anomaly"]] = relationship(
        back_populates="prediction", lazy="select"
    )

    def __repr__(self) -> str:
        return f"<Prediction {self.predicted_value:.3f} @ {self.timestamp}>"


# ══════════════════════════════════════════════════════
# TABLE 7 — ANOMALIES
# Anomalies détectées par l'autoencodeur LSTM
# ══════════════════════════════════════════════════════

class Anomaly(Base, UUIDMixin, TimestampMixin):
    """
    Anomalies détectées par l'autoencodeur LSTM.
    Chaque anomalie peut générer une alerte
    transmise aux agents LangGraph.
    """
    __tablename__ = "anomalies"

    prediction_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("predictions.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    series_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("time_series.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    anomaly_score: Mapped[float] = mapped_column(
        Float, nullable=False
    )  # Erreur de reconstruction normalisée
    threshold: Mapped[float] = mapped_column(
        Float, nullable=False
    )  # Seuil utilisé pour la détection
    status: Mapped[AnomalyStatus] = mapped_column(
        Enum(AnomalyStatus),
        default=AnomalyStatus.DETECTED,
        nullable=False,
        index=True
    )
    context: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )  # Contexte additionnel (valeurs voisines, etc.)

    # Relations
    prediction: Mapped[Optional["Prediction"]] = relationship(
        back_populates="anomalies"
    )
    alerts: Mapped[list["Alert"]] = relationship(
        back_populates="anomaly", lazy="select"
    )

    def __repr__(self) -> str:
        return f"<Anomaly score={self.anomaly_score:.3f} ({self.status})>"


# ══════════════════════════════════════════════════════
# TABLE 8 — ALERTS
# Alertes générées et transmises aux agents
# ══════════════════════════════════════════════════════

class Alert(Base, UUIDMixin, TimestampMixin):
    """
    Alertes générées à partir des anomalies détectées.
    Déclenchent l'exécution de l'Agent Analyse.
    """
    __tablename__ = "alerts"

    anomaly_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("anomalies.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    severity: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # low, medium, high, critical
    message: Mapped[str] = mapped_column(
        Text, nullable=False
    )
    is_processed: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, index=True
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    agent_run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="SET NULL"),
        nullable=True
    )

    # Relations
    anomaly: Mapped["Anomaly"] = relationship(
        back_populates="alerts"
    )

    def __repr__(self) -> str:
        return f"<Alert {self.severity} processed={self.is_processed}>"


# ══════════════════════════════════════════════════════
# TABLE 9 — AGENT RUNS
# Historique des exécutions des agents LangGraph
# ══════════════════════════════════════════════════════

class AgentRun(Base, UUIDMixin, TimestampMixin):
    """
    Historique de chaque exécution d'agent LangGraph.
    Traçabilité complète pour l'audit et l'évaluation.
    """
    __tablename__ = "agent_runs"

    agent_type: Mapped[AgentType] = mapped_column(
        Enum(AgentType), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True
    )  # running, completed, failed
    input_data: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )
    output_data: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )

    # Métriques de performance
    latency_ms: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    tokens_used: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    api_cost_usd: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )

    # Référence LangSmith
    langsmith_run_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )

    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relations
    reports: Mapped[list["Report"]] = relationship(
        back_populates="agent_run", lazy="select"
    )

    def __repr__(self) -> str:
        return f"<AgentRun {self.agent_type} ({self.status})>"


# ══════════════════════════════════════════════════════
# TABLE 10 — REPORTS
# Rapports générés par l'Agent Reporter
# ══════════════════════════════════════════════════════

class Report(Base, UUIDMixin, TimestampMixin):
    """
    Rapports structurés générés par l'Agent Reporter.
    Évalués via LLM-as-a-judge avant diffusion.
    """
    __tablename__ = "reports"

    agent_run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )
    title: Mapped[str] = mapped_column(
        String(500), nullable=False
    )
    report_type: Mapped[str] = mapped_column(
        String(100), nullable=False
    )  # anomaly_analysis, performance, feedback_suggestion
    content: Mapped[dict] = mapped_column(
        JSON, nullable=False
    )  # Contenu structuré Pydantic sérialisé

    # Évaluation LLM-as-a-judge
    faithfulness_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    relevance_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    is_validated: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    # Export
    pdf_path: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True
    )

    # Relations
    agent_run: Mapped[Optional["AgentRun"]] = relationship(
        back_populates="reports"
    )
    created_by_user: Mapped[Optional["User"]] = relationship(
        back_populates="reports"
    )

    def __repr__(self) -> str:
        return f"<Report {self.title[:50]}>"


# ══════════════════════════════════════════════════════
# TABLE 11 — FEEDBACK SUGGESTIONS ⭐ CONTRIBUTION ORIGINALE
# Boucle de rétroaction bidirectionnelle ML↔Agents
# ══════════════════════════════════════════════════════

class FeedbackSuggestion(Base, UUIDMixin, TimestampMixin):
    """
    Suggestions de la boucle de rétroaction bidirectionnelle.

    ⭐ CONTRIBUTION ORIGINALE DU PFE :
    L'Agent Analyse génère des suggestions d'ajustement
    d'hyperparamètres basées sur les résidus d'erreur du
    modèle CNN+LSTM. Ces suggestions sont soumises à
    validation humaine (HITL) avant tout réentraînement.
    """
    __tablename__ = "feedback_suggestions"

    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ml_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    decided_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )

    # Détail de la suggestion
    hyperparameter: Mapped[str] = mapped_column(
        String(100), nullable=False
    )  # ex: "learning_rate", "window_size", "dropout_rate"
    current_value: Mapped[str] = mapped_column(
        String(100), nullable=False
    )
    suggested_value: Mapped[str] = mapped_column(
        String(100), nullable=False
    )
    justification: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # Justification générée par le LLM
    confidence_score: Mapped[float] = mapped_column(
        Float, nullable=False
    )  # 0-1 : confiance de l'agent dans sa suggestion

    # Métriques qui ont déclenché la suggestion
    trigger_metrics: Mapped[dict] = mapped_column(
        JSON, nullable=False
    )  # {"rmse": 0.18, "consecutive_periods": 6}

    # Décision humaine (HITL)
    status: Mapped[FeedbackStatus] = mapped_column(
        Enum(FeedbackStatus),
        default=FeedbackStatus.PENDING,
        nullable=False,
        index=True
    )
    decision_reason: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
    decided_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Résultat après réentraînement (si approuvé)
    retrain_metrics: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )  # Métriques après réentraînement

    # Relations
    model: Mapped["MLModel"] = relationship(
        back_populates="feedback_suggestions"
    )
    decided_by_user: Mapped[Optional["User"]] = relationship(
        back_populates="feedback_decisions"
    )

    def __repr__(self) -> str:
        return (
            f"<FeedbackSuggestion {self.hyperparameter}: "
            f"{self.current_value} → {self.suggested_value} "
            f"({self.status})>"
        )

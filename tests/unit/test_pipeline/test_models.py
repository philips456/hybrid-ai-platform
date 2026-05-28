"""
tests/unit/test_pipeline/test_models.py
Tests unitaires pour les modèles SQLAlchemy.
Vérifie la création, les relations et les contraintes.
"""
import uuid
from datetime import datetime, timezone

import pytest

from src.database.models import (
    AgentRun, AgentType, Alert, Anomaly, AnomalyStatus,
    DataPoint, DataSource, DomainType, FeedbackStatus,
    FeedbackSuggestion, MLModel, ModelStatus, Prediction,
    Report, TimeSeries, User, UserRole,
)


class TestUserModel:
    """Tests du modèle User."""

    def test_user_creation(self):
        """Vérifie qu'un utilisateur peut être créé avec les bons attributs."""
        user = User(
            username="philippe",
            email="philippe@esprit.tn",
            hashed_password="hashed_pwd",
            role=UserRole.ANALYST,
        )
        assert user.username == "philippe"
        assert user.role == UserRole.ANALYST
        assert user.is_active is True

    def test_user_roles(self):
        """Vérifie que tous les rôles sont valides."""
        assert UserRole.ADMIN == "admin"
        assert UserRole.ANALYST == "analyst"
        assert UserRole.VIEWER == "viewer"

    def test_user_repr(self):
        """Vérifie la représentation string."""
        user = User(username="test", role=UserRole.ADMIN)
        assert "test" in repr(user)
        assert "admin" in repr(user)


class TestDataSourceModel:
    """Tests du modèle DataSource."""

    def test_datasource_creation(self):
        """Vérifie la création d'une source de données."""
        source = DataSource(
            name="telecom_5g",
            domain=DomainType.TELECOM,
            source_type="csv",
        )
        assert source.name == "telecom_5g"
        assert source.domain == DomainType.TELECOM
        assert source.is_active is True

    def test_domain_types(self):
        """Vérifie que tous les domaines sont valides."""
        assert DomainType.TELECOM == "telecom"
        assert DomainType.FINANCE == "finance"
        assert DomainType.INDUSTRY == "industry"
        assert DomainType.SYNTHETIC == "synthetic"


class TestMLModelModel:
    """Tests du modèle MLModel."""

    def test_mlmodel_creation(self):
        """Vérifie la création d'un modèle ML."""
        model = MLModel(
            name="cnn_lstm_telecom",
            version="1.0.0",
            domain=DomainType.TELECOM,
            status=ModelStatus.STAGING,
            metrics={"r2": 0.95, "rmse": 0.03},
        )
        assert model.name == "cnn_lstm_telecom"
        assert model.status == ModelStatus.STAGING
        assert model.metrics["r2"] == 0.95

    def test_model_status_values(self):
        """Vérifie les statuts valides d'un modèle."""
        assert ModelStatus.TRAINING == "training"
        assert ModelStatus.STAGING == "staging"
        assert ModelStatus.PRODUCTION == "production"
        assert ModelStatus.ARCHIVED == "archived"


class TestFeedbackSuggestionModel:
    """
    Tests du modèle FeedbackSuggestion.
    Contribution originale du PFE — boucle de feedback ML↔Agents.
    """

    def test_feedback_creation(self):
        """Vérifie la création d'une suggestion de feedback."""
        suggestion = FeedbackSuggestion(
            model_id=uuid.uuid4(),
            hyperparameter="learning_rate",
            current_value="0.001",
            suggested_value="0.0005",
            justification="Les résidus montrent une sur-adaptation.",
            confidence_score=0.85,
            trigger_metrics={"rmse": 0.18, "consecutive_periods": 6},
        )
        assert suggestion.hyperparameter == "learning_rate"
        assert suggestion.status == FeedbackStatus.PENDING
        assert suggestion.confidence_score == 0.85

    def test_feedback_status_values(self):
        """Vérifie les statuts valides d'une suggestion."""
        assert FeedbackStatus.PENDING == "pending"
        assert FeedbackStatus.APPROVED == "approved"
        assert FeedbackStatus.REJECTED == "rejected"

    def test_feedback_repr(self):
        """Vérifie la représentation string."""
        suggestion = FeedbackSuggestion(
            model_id=uuid.uuid4(),
            hyperparameter="window_size",
            current_value="24",
            suggested_value="48",
            justification="Test.",
            confidence_score=0.9,
            trigger_metrics={},
        )
        repr_str = repr(suggestion)
        assert "window_size" in repr_str
        assert "24" in repr_str
        assert "48" in repr_str


class TestAnomalyModel:
    """Tests du modèle Anomaly."""

    def test_anomaly_creation(self):
        """Vérifie la création d'une anomalie détectée."""
        anomaly = Anomaly(
            series_id=uuid.uuid4(),
            timestamp=datetime.now(timezone.utc),
            anomaly_score=0.92,
            threshold=0.85,
        )
        assert anomaly.anomaly_score == 0.92
        assert anomaly.status == AnomalyStatus.DETECTED

    def test_anomaly_status_values(self):
        """Vérifie les statuts d'anomalie."""
        assert AnomalyStatus.DETECTED == "detected"
        assert AnomalyStatus.CONFIRMED == "confirmed"
        assert AnomalyStatus.REJECTED == "rejected"

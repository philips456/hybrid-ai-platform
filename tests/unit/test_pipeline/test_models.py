"""
tests/unit/test_pipeline/test_models.py
Tests unitaires pour les modèles SQLAlchemy.

NOTE : Les valeurs `default` SQLAlchemy s'appliquent uniquement
lors de l'insertion en base de données. En dehors d'une session DB,
les champs avec default retournent None. On passe donc les valeurs
explicitement dans les tests unitaires.
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

    def test_user_creation(self):
        user = User(
            username="philippe",
            email="philippe@esprit.tn",
            hashed_password="hashed_pwd",
            role=UserRole.ANALYST,
            is_active=True,
        )
        assert user.username == "philippe"
        assert user.role == UserRole.ANALYST
        assert user.is_active is True

    def test_user_roles(self):
        assert UserRole.ADMIN == "admin"
        assert UserRole.ANALYST == "analyst"
        assert UserRole.VIEWER == "viewer"

    def test_user_repr(self):
        user = User(username="test", role=UserRole.ADMIN)
        assert "test" in repr(user)
        assert "ADMIN" in repr(user)


class TestDataSourceModel:

    def test_datasource_creation(self):
        source = DataSource(
            name="telecom_5g",
            domain=DomainType.TELECOM,
            source_type="csv",
            is_active=True,
        )
        assert source.name == "telecom_5g"
        assert source.domain == DomainType.TELECOM
        assert source.is_active is True

    def test_domain_types(self):
        assert DomainType.TELECOM == "telecom"
        assert DomainType.FINANCE == "finance"
        assert DomainType.INDUSTRY == "industry"
        assert DomainType.SYNTHETIC == "synthetic"


class TestMLModelModel:

    def test_mlmodel_creation(self):
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
        assert ModelStatus.TRAINING == "training"
        assert ModelStatus.STAGING == "staging"
        assert ModelStatus.PRODUCTION == "production"
        assert ModelStatus.ARCHIVED == "archived"


class TestFeedbackSuggestionModel:

    def test_feedback_creation(self):
        suggestion = FeedbackSuggestion(
            model_id=uuid.uuid4(),
            hyperparameter="learning_rate",
            current_value="0.001",
            suggested_value="0.0005",
            justification="Les résidus montrent une sur-adaptation.",
            confidence_score=0.85,
            trigger_metrics={"rmse": 0.18, "consecutive_periods": 6},
            status=FeedbackStatus.PENDING,
        )
        assert suggestion.hyperparameter == "learning_rate"
        assert suggestion.status == FeedbackStatus.PENDING
        assert suggestion.confidence_score == 0.85

    def test_feedback_status_values(self):
        assert FeedbackStatus.PENDING == "pending"
        assert FeedbackStatus.APPROVED == "approved"
        assert FeedbackStatus.REJECTED == "rejected"

    def test_feedback_repr(self):
        suggestion = FeedbackSuggestion(
            model_id=uuid.uuid4(),
            hyperparameter="window_size",
            current_value="24",
            suggested_value="48",
            justification="Test.",
            confidence_score=0.9,
            trigger_metrics={},
            status=FeedbackStatus.PENDING,
        )
        repr_str = repr(suggestion)
        assert "window_size" in repr_str
        assert "24" in repr_str
        assert "48" in repr_str


class TestAnomalyModel:

    def test_anomaly_creation(self):
        anomaly = Anomaly(
            series_id=uuid.uuid4(),
            timestamp=datetime.now(timezone.utc),
            anomaly_score=0.92,
            threshold=0.85,
            status=AnomalyStatus.DETECTED,
        )
        assert anomaly.anomaly_score == 0.92
        assert anomaly.status == AnomalyStatus.DETECTED

    def test_anomaly_status_values(self):
        assert AnomalyStatus.DETECTED == "detected"
        assert AnomalyStatus.CONFIRMED == "confirmed"
        assert AnomalyStatus.REJECTED == "rejected"
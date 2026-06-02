"""
tests/unit/test_agents/test_feedback_loop.py
Unit tests for FeedbackLoop — the original PFE contribution.
"""
import pytest
from unittest.mock import patch, MagicMock
from src.agents.feedback.loop import FeedbackLoop, FeedbackTrigger


class TestFeedbackLoop:

    def _make_loop(self):
        with patch("src.agents.feedback.loop.AnalystAgent"):
            loop = FeedbackLoop(domain="synthetic")
            loop.analyst = MagicMock()
            loop.analyst.analyze.return_value = [
                {
                    "hyperparameter": "learning_rate",
                    "current_value": "0.001",
                    "suggested_value": "0.0005",
                    "justification": "RMSE exceeded threshold for 6 periods",
                    "confidence_score": 0.85,
                }
            ]
            return loop

    def test_no_trigger_below_threshold(self):
        loop = self._make_loop()
        result = loop.evaluate(
            metrics={"rmse": 0.10, "consecutive_periods": 3},
            model_config={},
        )
        assert result.triggered is False
        assert result.suggestions == []

    def test_trigger_above_threshold(self):
        loop = self._make_loop()
        result = loop.evaluate(
            metrics={"rmse": 0.18, "consecutive_periods": 6},
            model_config={},
        )
        assert result.triggered is True
        assert result.trigger_reason == FeedbackTrigger.CONSECUTIVE_ERRORS.value

    def test_force_trigger(self):
        loop = self._make_loop()
        result = loop.evaluate(
            metrics={"rmse": 0.05, "consecutive_periods": 1},
            model_config={},
            force_trigger=True,
        )
        assert result.triggered is True
        assert result.trigger_reason == FeedbackTrigger.MANUAL.value

    def test_suggestions_generated_on_trigger(self):
        loop = self._make_loop()
        result = loop.evaluate(
            metrics={"rmse": 0.20, "consecutive_periods": 6},
            model_config={},
        )
        assert result.triggered is True
        assert len(result.suggestions) > 0
        assert result.requires_hitl is True

    def test_suggestions_have_validation_metadata(self):
        """All suggestions must have validation metadata for HITL display."""
        loop = self._make_loop()
        result = loop.evaluate(
            metrics={"rmse": 0.20, "consecutive_periods": 6},
            model_config={},
        )
        for suggestion in result.suggestions:
            assert "validation_issues" in suggestion
            assert "validation_warnings" in suggestion
            assert "pre_validated" in suggestion
            assert "trigger_metrics" in suggestion

    def test_suggestions_signed_with_hmac(self):
        """All suggestions must be HMAC signed before HITL."""
        loop = self._make_loop()
        result = loop.evaluate(
            metrics={"rmse": 0.20, "consecutive_periods": 6},
            model_config={},
        )
        for suggestion in result.suggestions:
            assert "integrity_hash" in suggestion
            assert len(suggestion["integrity_hash"]) == 64

    def test_hitl_approved(self):
        loop = self._make_loop()
        suggestion = {
            "hyperparameter": "learning_rate",
            "current_value": "0.001",
            "suggested_value": "0.0005",
            "justification": "test",
            "confidence_score": 0.8,
        }
        signed = loop.hmac_guard.sign_suggestion(suggestion)
        result = loop.process_hitl_decision(
            suggestion=signed,
            approved=True,
            reason="Metrics clearly justify this change",
        )
        assert result["status"] == "APPROVED"
        assert result["decision_reason"] == "Metrics clearly justify this change"

    def test_hitl_rejected_stores_reason(self):
        """Rejection reason must be stored in LongTermMemory."""
        loop = self._make_loop()
        suggestion = {
            "hyperparameter": "window_size",
            "current_value": "24",
            "suggested_value": "48",
            "justification": "test",
            "confidence_score": 0.6,
        }
        signed = loop.hmac_guard.sign_suggestion(suggestion)
        result = loop.process_hitl_decision(
            suggestion=signed,
            approved=False,
            reason="Change too aggressive for current data",
        )
        assert result["status"] == "REJECTED"
        assert result["decision_reason"] == "Change too aggressive for current data"

    def test_hitl_tampered_suggestion_blocked(self):
        """Tampered suggestion must be blocked by HMAC verification."""
        loop = self._make_loop()
        suggestion = {
            "hyperparameter": "learning_rate",
            "current_value": "0.001",
            "suggested_value": "0.0005",
            "justification": "test",
            "confidence_score": 0.8,
        }
        signed = loop.hmac_guard.sign_suggestion(suggestion)
        # Tamper
        signed["suggested_value"] = "0.00001"
        result = loop.process_hitl_decision(suggestion=signed, approved=True)
        assert result["status"] == "INTEGRITY_ERROR"

    def test_extreme_rmse_triggers_immediately(self):
        loop = self._make_loop()
        result = loop.evaluate(
            metrics={"rmse": 0.30, "consecutive_periods": 1},
            model_config={},
        )
        assert result.triggered is True
        assert result.trigger_reason == FeedbackTrigger.RMSE_THRESHOLD.value

"""
tests/unit/test_agents/test_validator.py
Unit tests for SuggestionValidator.
"""
import pytest
from src.agents.analyst.validator import SuggestionValidator, ValidationResult


class TestSuggestionValidator:

    def setup_method(self):
        self.validator = SuggestionValidator(min_confidence=0.5)

    def _make_suggestion(self, **overrides):
        base = {
            "hyperparameter": "learning_rate",
            "current_value": "0.001",
            "suggested_value": "0.0005",
            "justification": "RMSE=0.18 exceeded threshold=0.15 for 6 periods",
            "confidence_score": 0.85,
        }
        base.update(overrides)
        return base

    def test_valid_suggestion_passes(self):
        result = self.validator.validate(self._make_suggestion())
        assert result.is_valid is True
        assert result.issues == []

    def test_low_confidence_rejected(self):
        result = self.validator.validate(
            self._make_suggestion(confidence_score=0.3)
        )
        assert result.is_valid is False
        assert any("confidence" in i.lower() for i in result.issues)

    def test_short_justification_rejected(self):
        result = self.validator.validate(
            self._make_suggestion(justification="Too short")
        )
        assert result.is_valid is False
        assert any("justification" in i.lower() for i in result.issues)

    def test_value_out_of_bounds_rejected(self):
        result = self.validator.validate(
            self._make_suggestion(suggested_value="0.5")  # learning_rate max is 0.1
        )
        assert result.is_valid is False
        assert any("bounds" in i.lower() for i in result.issues)

    def test_excessive_change_rejected(self):
        result = self.validator.validate(
            self._make_suggestion(
                current_value="0.001",
                suggested_value="0.00001",  # 99% change > 50% max
            )
        )
        assert result.is_valid is False
        assert any("%" in i for i in result.issues)

    def test_unknown_hyperparameter_warning(self):
        result = self.validator.validate(
            self._make_suggestion(hyperparameter="custom_param")
        )
        assert result.is_valid is True  # Not rejected in non-strict mode
        assert any("unknown" in w.lower() for w in result.warnings)

    def test_unknown_hyperparameter_strict_mode(self):
        strict_validator = SuggestionValidator(strict_mode=True)
        result = strict_validator.validate(
            self._make_suggestion(hyperparameter="custom_param")
        )
        assert result.is_valid is False

    def test_dropout_rate_valid(self):
        result = self.validator.validate({
            "hyperparameter": "dropout_rate",
            "current_value": "0.2",
            "suggested_value": "0.3",
            "justification": "RMSE sustained above threshold for 5 periods — overfitting suspected",
            "confidence_score": 0.72,
        })
        assert result.is_valid is True

    def test_window_size_valid(self):
        result = self.validator.validate({
            "hyperparameter": "window_size",
            "current_value": "24",
            "suggested_value": "48",
            "justification": "Anomaly pattern shows 48-hour cycle exceeding threshold periods",
            "confidence_score": 0.68,
        })
        assert result.is_valid is True

    def test_filter_valid_removes_invalid(self):
        suggestions = [
            self._make_suggestion(),  # valid
            self._make_suggestion(confidence_score=0.1),  # invalid
            self._make_suggestion(suggested_value="0.5"),  # invalid — out of bounds
        ]
        valid = self.validator.filter_valid(suggestions)
        assert len(valid) == 1

    def test_validate_all_returns_all_results(self):
        suggestions = [
            self._make_suggestion(),
            self._make_suggestion(confidence_score=0.1),
        ]
        results = self.validator.validate_all(suggestions)
        assert len(results) == 2
        assert results[0].is_valid is True
        assert results[1].is_valid is False

    def test_no_metric_reference_adds_warning(self):
        result = self.validator.validate(
            self._make_suggestion(
                justification="This parameter should be adjusted to improve performance"
            )
        )
        assert any("metric" in w.lower() for w in result.warnings)

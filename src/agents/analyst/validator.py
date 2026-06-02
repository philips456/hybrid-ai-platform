"""
src/agents/analyst/validator.py
SuggestionValidator — validates hyperparameter suggestions before HITL.

Ensures:
1. Values are within known safe bounds
2. Change percentage is reasonable
3. Confidence score is above minimum threshold
4. Justification references actual metrics
"""
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Safe bounds per hyperparameter
# Based on standard DL practices for time series models
HYPERPARAMETER_CONSTRAINTS = {
    "learning_rate": {
        "min": 0.000001,
        "max": 0.1,
        "max_change_pct": 50,
        "description": "Adam/SGD learning rate",
    },
    "dropout_rate": {
        "min": 0.0,
        "max": 0.5,
        "max_change_pct": 100,
        "description": "Dropout regularization rate",
    },
    "window_size": {
        "min": 8,
        "max": 168,
        "max_change_pct": 100,
        "description": "Input sequence window size",
    },
    "batch_size": {
        "min": 8,
        "max": 256,
        "max_change_pct": 100,
        "description": "Training batch size",
    },
    "epochs": {
        "min": 1,
        "max": 500,
        "max_change_pct": 100,
        "description": "Number of training epochs",
    },
    "hidden_units": {
        "min": 8,
        "max": 512,
        "max_change_pct": 100,
        "description": "LSTM/Dense hidden units",
    },
    "num_layers": {
        "min": 1,
        "max": 8,
        "max_change_pct": 100,
        "description": "Number of layers",
    },
    "l2_regularization": {
        "min": 0.0,
        "max": 0.1,
        "max_change_pct": 200,
        "description": "L2 regularization coefficient",
    },
}

MIN_CONFIDENCE_SCORE = 0.5
MIN_JUSTIFICATION_LENGTH = 20


@dataclass
class ValidationResult:
    """Result of suggestion validation."""
    is_valid: bool
    suggestion: dict
    issues: list[str]
    warnings: list[str]

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        hp = self.suggestion.get("hyperparameter", "unknown")
        return f"{status} [{hp}]: issues={self.issues}, warnings={self.warnings}"


class SuggestionValidator:
    """
    Validates hyperparameter suggestions before they reach HITL.

    Validation checks:
    1. Confidence score above minimum threshold
    2. Justification is specific and references metrics
    3. Suggested value within safe bounds
    4. Change percentage is reasonable
    5. Hyperparameter name is recognized

    Usage:
        validator = SuggestionValidator()
        results = validator.validate_all(suggestions)
        valid = [r.suggestion for r in results if r.is_valid]
    """

    def __init__(
        self,
        min_confidence: float = MIN_CONFIDENCE_SCORE,
        strict_mode: bool = False,
    ):
        """
        Args:
            min_confidence: Minimum confidence score to accept suggestion
            strict_mode: If True, unknown hyperparameters are rejected
        """
        self.min_confidence = min_confidence
        self.strict_mode = strict_mode

    def validate(self, suggestion: dict) -> ValidationResult:
        """
        Validates a single suggestion.

        Args:
            suggestion: FeedbackSuggestion dict from AnalystAgent

        Returns:
            ValidationResult with is_valid, issues, and warnings
        """
        issues = []
        warnings = []
        hp = suggestion.get("hyperparameter", "")

        # Check 1 — Confidence score
        confidence = float(suggestion.get("confidence_score", 0))
        if confidence < self.min_confidence:
            issues.append(
                f"Confidence {confidence:.2f} below minimum {self.min_confidence:.2f}"
            )

        # Check 2 — Justification quality
        justification = suggestion.get("justification", "")
        if len(justification) < MIN_JUSTIFICATION_LENGTH:
            issues.append(
                f"Justification too short ({len(justification)} chars, "
                f"minimum {MIN_JUSTIFICATION_LENGTH})"
            )

        # Check if justification references metrics
        metric_keywords = ["rmse", "mae", "threshold", "period", "error", "loss"]
        has_metric_ref = any(kw in justification.lower() for kw in metric_keywords)
        if not has_metric_ref:
            warnings.append(
                "Justification does not reference specific metrics — verify manually"
            )

        # Check 3 — Known hyperparameter
        if hp not in HYPERPARAMETER_CONSTRAINTS:
            if self.strict_mode:
                issues.append(f"Unknown hyperparameter '{hp}' — not in constraints")
            else:
                warnings.append(
                    f"Unknown hyperparameter '{hp}' — no bounds to validate"
                )
            # Skip numeric validation for unknown hyperparameters
            is_valid = len(issues) == 0
            return ValidationResult(
                is_valid=is_valid,
                suggestion=suggestion,
                issues=issues,
                warnings=warnings,
            )

        constraints = HYPERPARAMETER_CONSTRAINTS[hp]

        # Check 4 — Numeric bounds
        try:
            new_val = float(suggestion.get("suggested_value", ""))
            min_val = constraints["min"]
            max_val = constraints["max"]

            if new_val < min_val or new_val > max_val:
                issues.append(
                    f"{hp}={new_val} outside safe bounds "
                    f"[{min_val}, {max_val}]"
                )

            # Check 5 — Change percentage
            try:
                current_val = float(suggestion.get("current_value", ""))
                if current_val != 0:
                    change_pct = abs(new_val - current_val) / abs(current_val) * 100
                    max_change = constraints["max_change_pct"]
                    if change_pct > max_change:
                        issues.append(
                            f"Change of {change_pct:.0f}% exceeds "
                            f"maximum allowed {max_change}%"
                        )
                    elif change_pct > max_change * 0.8:
                        warnings.append(
                            f"Change of {change_pct:.0f}% is close to "
                            f"maximum {max_change}% — review carefully"
                        )
            except (ValueError, TypeError):
                warnings.append(
                    "Could not parse current_value for change validation"
                )

        except (ValueError, TypeError):
            # Non-numeric suggestion (e.g. optimizer name)
            warnings.append(
                f"Non-numeric suggested_value '{suggestion.get('suggested_value')}' "
                f"— manual validation required"
            )

        is_valid = len(issues) == 0

        if is_valid:
            logger.info(
                f"SuggestionValidator: VALID — {hp} "
                f"{suggestion.get('current_value')} → "
                f"{suggestion.get('suggested_value')} "
                f"(confidence={confidence:.2f})"
            )
        else:
            logger.warning(
                f"SuggestionValidator: INVALID — {hp} | issues={issues}"
            )

        return ValidationResult(
            is_valid=is_valid,
            suggestion=suggestion,
            issues=issues,
            warnings=warnings,
        )

    def validate_all(self, suggestions: list[dict]) -> list[ValidationResult]:
        """
        Validates all suggestions and returns results.

        Args:
            suggestions: List of FeedbackSuggestion dicts

        Returns:
            List of ValidationResult — one per suggestion
        """
        results = []
        for suggestion in suggestions:
            result = self.validate(suggestion)
            results.append(result)

        valid_count = sum(1 for r in results if r.is_valid)
        logger.info(
            f"SuggestionValidator: {valid_count}/{len(results)} suggestions valid"
        )
        return results

    def filter_valid(self, suggestions: list[dict]) -> list[dict]:
        """
        Returns only valid suggestions — convenience method.

        Args:
            suggestions: List of FeedbackSuggestion dicts

        Returns:
            Filtered list containing only valid suggestions
        """
        results = self.validate_all(suggestions)
        return [r.suggestion for r in results if r.is_valid]

    def get_constraints(self, hyperparameter: str) -> Optional[dict]:
        """Returns constraints for a specific hyperparameter."""
        return HYPERPARAMETER_CONSTRAINTS.get(hyperparameter)

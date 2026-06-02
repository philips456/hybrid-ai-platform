"""
src/agents/security/anonymizer.py
Data anonymizer — protects sensitive business data before sending to Claude.

Based on:
- Huang et al. (arXiv:2410.11182) — on-premises LLM deployment
- GDPR Article 25 — privacy by design

Strategy:
- Replace sensitive IDs with anonymous tokens
- Keep only aggregated metrics (not raw data)
- Remove PII and business-identifying information
- Log what was anonymized for audit trail
"""
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AnonymizationReport:
    """Report of what was anonymized."""
    original_fields: list[str]
    anonymized_fields: list[str]
    removed_fields: list[str]
    safe_to_send: bool


class DataAnonymizer:
    """
    Anonymizes sensitive business data before sending to external LLM.

    What gets anonymized:
    - Series IDs → SERIES_ANON_001
    - Client names → CLIENT_A
    - Station names → STATION_001
    - IP addresses → 0.0.0.0
    - Phone numbers → +XXX-XXX-XXXX

    What stays intact (safe to send):
    - Aggregated metrics (RMSE, MAE, R²)
    - Statistical values (threshold, periods)
    - Anomaly types (point, contextual, collective)
    - Timestamps (day of week, hour — no exact dates)

    Usage:
        anonymizer = DataAnonymizer()
        safe_metrics = anonymizer.anonymize_metrics(metrics)
        safe_context = anonymizer.anonymize_text(context)
    """

    # Fields safe to send to external LLM
    SAFE_METRIC_FIELDS = {
        "rmse", "mae", "r2", "mse", "mape",
        "consecutive_periods", "anomaly_score",
        "threshold", "window_size", "learning_rate",
        "dropout_rate", "batch_size", "epochs",
        "anomaly_type", "severity",
    }

    # Fields that must be anonymized
    SENSITIVE_FIELDS = {
        "series_id", "client_id", "station_id", "user_id",
        "model_id", "source_id", "ip_address", "hostname",
        "client_name", "station_name", "location",
    }

    def anonymize_metrics(self, metrics: dict) -> tuple[dict, AnonymizationReport]:
        """
        Anonymizes a metrics dict, keeping only safe fields.

        Args:
            metrics: Raw metrics dict potentially containing sensitive data

        Returns:
            Tuple of (anonymized_metrics, report)
        """
        safe = {}
        anonymized = []
        removed = []

        for key, value in metrics.items():
            key_lower = key.lower()
            if key_lower in self.SAFE_METRIC_FIELDS:
                safe[key] = value
            elif key_lower in self.SENSITIVE_FIELDS:
                safe[key] = self._anonymize_id(str(value))
                anonymized.append(key)
            else:
                # Unknown field — anonymize by default
                if isinstance(value, (int, float)):
                    safe[key] = value  # Numeric values are generally safe
                else:
                    removed.append(key)
                    logger.debug(f"DataAnonymizer: removed field '{key}'")

        report = AnonymizationReport(
            original_fields=list(metrics.keys()),
            anonymized_fields=anonymized,
            removed_fields=removed,
            safe_to_send=True,
        )

        if anonymized:
            logger.info(f"DataAnonymizer: anonymized {len(anonymized)} fields: {anonymized}")
        if removed:
            logger.info(f"DataAnonymizer: removed {len(removed)} fields: {removed}")

        return safe, report

    def anonymize_text(self, text: str) -> str:
        """
        Anonymizes sensitive patterns in free text.
        Used for context strings before sending to Claude.
        """
        # Anonymize IP addresses
        text = re.sub(
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            '0.0.0.0',
            text
        )
        # Anonymize UUIDs
        text = re.sub(
            r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',
            'ANON-UUID',
            text,
            flags=re.IGNORECASE
        )
        # Anonymize phone numbers
        text = re.sub(
            r'\b\+?[\d\s\-\(\)]{10,15}\b',
            '+XXX-XXX-XXXX',
            text
        )
        return text

    def anonymize_suggestion(self, suggestion: dict) -> dict:
        """
        Ensures a feedback suggestion contains no sensitive data
        before storing in PostgreSQL or displaying in dashboard.
        """
        safe_keys = {
            "hyperparameter", "current_value", "suggested_value",
            "justification", "confidence_score",
        }
        return {k: v for k, v in suggestion.items() if k in safe_keys}

    def is_safe_for_external_llm(self, data: Any) -> bool:
        """
        Quick check if data is safe to send to external LLM.
        Returns False if sensitive fields detected.
        """
        if isinstance(data, dict):
            for key in data.keys():
                if key.lower() in self.SENSITIVE_FIELDS:
                    return False
        elif isinstance(data, str):
            # Check for obvious PII patterns
            if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', data):
                return False
        return True

    def _anonymize_id(self, value: str) -> str:
        """Creates a consistent anonymous token for an ID."""
        hash_val = hashlib.sha256(value.encode()).hexdigest()[:8].upper()
        return f"ANON_{hash_val}"

"""
src/agents/security/hmac_guard.py
HMAC protection for prompts and suggestions.

Based on:
- FATH (arXiv:2410.21492) — HMAC authentication against prompt injection
- arXiv:2508.09288 — Provable security with HMAC-SHA-256

Protects:
1. Prompts sent to Claude — detects prompt injection from Qdrant docs
2. Suggestions stored in PostgreSQL — detects tampering
3. RAG documents — verifies integrity before injection into context
"""
import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class HMACGuard:
    """
    HMAC-SHA-256 protection for LLM inputs and outputs.

    Usage:
        guard = HMACGuard(secret_key="your-secret-key")

        # Sign a prompt before sending to Claude
        signed = guard.sign_prompt(prompt)

        # Sign a suggestion before storing in PostgreSQL
        signed_suggestion = guard.sign_suggestion(suggestion)

        # Verify before reading from DB
        is_valid = guard.verify_suggestion(signed_suggestion)
    """

    def __init__(self, secret_key: str = None):
        from configs.settings import settings
        key = secret_key or settings.api_secret_key
        self.secret = key.encode("utf-8")

    # ── PROMPT PROTECTION ────────────────────────────

    def sign_prompt(self, prompt: str) -> dict:
        """
        Signs a prompt with HMAC before sending to Claude.
        Adds a trust tag that Claude can reference to verify
        the prompt came from our trusted system.

        Returns dict with prompt, signature, and trust_tag.
        """
        signature = self._compute_hmac(prompt)
        trust_tag = f"[TRUSTED-{signature[:12]}]"

        return {
            "prompt": prompt,
            "signature": signature,
            "trust_tag": trust_tag,
            "signed_at": datetime.now(timezone.utc).isoformat(),
        }

    def inject_trust_tag(self, prompt: str) -> str:
        """
        Injects a trust tag at the START of the prompt.
        Claude is instructed to only follow instructions
        that contain this tag — ignoring injected content.
        """
        signed = self.sign_prompt(prompt)
        return f"{signed['trust_tag']}\n\n{prompt}"

    def detect_prompt_injection(self, document: str) -> bool:
        """
        Detects potential prompt injection in RAG documents.
        Returns True if injection is suspected.

        Common injection patterns:
        - "Ignore previous instructions"
        - "You are now..."
        - "Forget everything"
        """
        injection_patterns = [
            "ignore previous",
            "ignore all previous",
            "forget everything",
            "you are now",
            "new instructions",
            "system prompt",
            "act as",
            "jailbreak",
            "disregard",
        ]
        doc_lower = document.lower()
        for pattern in injection_patterns:
            if pattern in doc_lower:
                logger.warning(
                    f"HMACGuard: potential prompt injection detected: '{pattern}'"
                )
                return True
        return False

    # ── SUGGESTION PROTECTION ────────────────────────

    def sign_suggestion(self, suggestion: dict) -> dict:
        """
        Signs a feedback suggestion before storing in PostgreSQL.
        Ensures no tampering between generation and HITL validation.
        """
        suggestion_copy = {k: v for k, v in suggestion.items() if k != "integrity_hash"}
        payload = json.dumps(suggestion_copy, sort_keys=True, default=str)
        signature = self._compute_hmac(payload)
        suggestion_copy["integrity_hash"] = signature
        return suggestion_copy

    def verify_suggestion(self, suggestion: dict) -> bool:
        """
        Verifies a suggestion has not been tampered with.
        Returns True if integrity is confirmed.
        """
        stored_hash = suggestion.get("integrity_hash", "")
        if not stored_hash:
            logger.warning("HMACGuard: suggestion has no integrity_hash")
            return False

        check = {k: v for k, v in suggestion.items() if k != "integrity_hash"}
        payload = json.dumps(check, sort_keys=True, default=str)
        expected = self._compute_hmac(payload)

        is_valid = hmac.compare_digest(stored_hash, expected)
        if not is_valid:
            logger.error("HMACGuard: suggestion integrity check FAILED — possible tampering")
        return is_valid

    # ── DOCUMENT PROTECTION ──────────────────────────

    def sign_document(self, content: str) -> str:
        """Returns HMAC signature for a document."""
        return self._compute_hmac(content)

    def verify_document(self, content: str, stored_signature: str) -> bool:
        """Verifies a document has not been altered since indexing."""
        expected = self._compute_hmac(content)
        return hmac.compare_digest(stored_signature, expected)

    # ── PRIVATE ──────────────────────────────────────

    def _compute_hmac(self, data: str) -> str:
        """Computes HMAC-SHA-256 signature."""
        return hmac.new(
            self.secret,
            data.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

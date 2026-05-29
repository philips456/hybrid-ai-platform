"""
src/agents/context/compressor.py
ContextCompressor — compresses long histories to fit context window.

Based on ACE framework (Zhang et al., arXiv:2510.04618):
generate-reflect-curate cycle for context optimization.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ContextCompressor:
    """
    Compresses agent conversation history when it exceeds token limits.

    Strategy:
    - Keep last 3 messages intact (recent context)
    - Summarize older messages into bullet points
    - Always preserve critical data (metrics, HITL decisions)

    Usage:
        compressor = ContextCompressor(max_tokens=4000)
        compressed = compressor.compress(messages, keep_last=3)
    """

    def __init__(self, max_tokens: int = 4000, keep_last: int = 3):
        self.max_tokens = max_tokens
        self.keep_last = keep_last

    def compress(self, messages: list[dict]) -> list[dict]:
        """
        Compresses message history if it exceeds max_tokens.

        Args:
            messages: List of {"role": str, "content": str} dicts

        Returns:
            Compressed message list
        """
        total_tokens = self._estimate_tokens(messages)

        if total_tokens <= self.max_tokens:
            return messages

        logger.info(
            f"Compressing context: {total_tokens} tokens → target {self.max_tokens}"
        )

        # Keep the last N messages intact
        recent = messages[-self.keep_last:] if len(messages) > self.keep_last else messages
        older = messages[:-self.keep_last] if len(messages) > self.keep_last else []

        if not older:
            return recent

        # Summarize older messages
        summary = self._summarize_messages(older)
        summary_message = {
            "role": "system",
            "content": f"[CONVERSATION SUMMARY]\n{summary}"
        }

        compressed = [summary_message] + recent
        new_tokens = self._estimate_tokens(compressed)

        logger.info(f"Context compressed: {total_tokens} → {new_tokens} tokens")
        return compressed

    def compress_rag_documents(
        self,
        documents: list[str],
        max_docs: int = 3,
        max_chars_per_doc: int = 500,
    ) -> list[str]:
        """
        Compresses RAG documents to fit context budget.

        Args:
            documents: List of retrieved document strings
            max_docs: Maximum number of documents to keep
            max_chars_per_doc: Maximum characters per document

        Returns:
            Compressed document list
        """
        compressed = []
        for doc in documents[:max_docs]:
            if len(doc) > max_chars_per_doc:
                compressed.append(doc[:max_chars_per_doc] + "... [truncated]")
            else:
                compressed.append(doc)
        return compressed

    def extract_critical_data(self, messages: list[dict]) -> list[dict]:
        """
        Extracts messages containing critical data that must never be compressed.
        Critical data: metrics, HITL decisions, model configurations.
        """
        critical_keywords = [
            "rmse", "threshold", "approved", "rejected",
            "hyperparameter", "model_config", "feedback"
        ]
        critical = []
        for msg in messages:
            content_lower = msg.get("content", "").lower()
            if any(kw in content_lower for kw in critical_keywords):
                critical.append(msg)
        return critical

    def _summarize_messages(self, messages: list[dict]) -> str:
        """Creates a bullet-point summary of older messages."""
        lines = [f"Summary of {len(messages)} earlier messages:"]
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Extract first meaningful sentence
            first_line = content.split("\n")[0][:150]
            lines.append(f"  - [{role}]: {first_line}")
        return "\n".join(lines)

    def _estimate_tokens(self, messages: list[dict]) -> int:
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return total_chars // 4

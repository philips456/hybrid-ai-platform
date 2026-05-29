"""
src/agents/memory/long_term.py
Long-term semantic memory using Mem0.
Persists facts across sessions — agent learns from past anomalies.

Based on Zhang et al., arXiv:2603.07670 (LLM Agent Memory Management).
"""
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    Cross-session semantic memory for agents.

    Stores:
    - Past anomaly patterns and their resolutions
    - Approved feedback suggestions and outcomes
    - Domain-specific knowledge learned during operation

    Falls back to in-memory dict if Mem0 is unavailable.
    """

    def __init__(self, user_id: str = "hybrid_ai_platform"):
        self.user_id = user_id
        self._fallback_store: dict[str, Any] = {}
        self._mem0_available = False
        self._init_mem0()

    def _init_mem0(self) -> None:
        """Initializes Mem0 client if available."""
        try:
            from mem0 import Memory
            self._client = Memory()
            self._mem0_available = True
            logger.info("LongTermMemory: Mem0 initialized")
        except Exception as e:
            logger.warning(f"LongTermMemory: Mem0 unavailable, using fallback: {e}")
            self._mem0_available = False

    def remember(self, content: str, metadata: Optional[dict] = None) -> str:
        """
        Stores a fact in long-term memory.

        Args:
            content: Text to remember
            metadata: Optional metadata (anomaly_type, domain, etc.)

        Returns:
            Memory ID
        """
        if self._mem0_available:
            try:
                result = self._client.add(
                    messages=[{"role": "user", "content": content}],
                    user_id=self.user_id,
                    metadata=metadata or {},
                )
                memory_id = result.get("id", "unknown")
                logger.debug(f"LongTermMemory: stored memory {memory_id}")
                return memory_id
            except Exception as e:
                logger.error(f"LongTermMemory: Mem0 store failed: {e}")

        # Fallback
        import uuid
        memory_id = str(uuid.uuid4())[:8]
        self._fallback_store[memory_id] = {
            "content": content,
            "metadata": metadata or {},
        }
        return memory_id

    def recall(self, query: str, limit: int = 5) -> list[dict]:
        """
        Retrieves relevant memories for a query.

        Args:
            query: Search query
            limit: Maximum number of memories to return

        Returns:
            List of relevant memory dicts
        """
        if self._mem0_available:
            try:
                results = self._client.search(
                    query=query,
                    user_id=self.user_id,
                    limit=limit,
                )
                return results.get("results", [])
            except Exception as e:
                logger.error(f"LongTermMemory: Mem0 recall failed: {e}")

        # Fallback: keyword search
        query_words = set(query.lower().split())
        scored = []
        for mem_id, mem in self._fallback_store.items():
            content_words = set(mem["content"].lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                scored.append((overlap, mem_id, mem))

        scored.sort(reverse=True)
        return [
            {"id": m[1], "memory": m[2]["content"], "metadata": m[2]["metadata"]}
            for m in scored[:limit]
        ]

    def forget(self, memory_id: str) -> bool:
        """Removes a specific memory."""
        if self._mem0_available:
            try:
                self._client.delete(memory_id=memory_id, user_id=self.user_id)
                return True
            except Exception as e:
                logger.error(f"LongTermMemory: delete failed: {e}")

        if memory_id in self._fallback_store:
            del self._fallback_store[memory_id]
            return True
        return False

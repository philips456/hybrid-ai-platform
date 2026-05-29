"""
src/agents/memory/short_term.py
Short-term memory using LangGraph checkpointer.
Stores current session state — cleared between sessions.
"""
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """
    In-session memory for agent state.
    Stores the current run's conversation, intermediate results,
    and agent decisions within a single workflow execution.
    """

    def __init__(self):
        self._store: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value
        logger.debug(f"ShortTermMemory: set '{key}'")

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def update(self, key: str, value: Any) -> None:
        if key in self._store and isinstance(self._store[key], list):
            self._store[key].append(value)
        else:
            self._store[key] = value

    def clear(self) -> None:
        self._store.clear()
        logger.info("ShortTermMemory: cleared")

    def get_all(self) -> dict:
        return dict(self._store)

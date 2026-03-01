"""Common conversation schema and abstract base parser."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Turn:
    """A single turn in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime | None = None


@dataclass
class Conversation:
    """A normalized conversation from any chatbot platform."""

    source: str  # "claude", "gemini", "chatgpt"
    conversation_id: str
    title: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    turns: list[Turn] = field(default_factory=list)


class BaseParser(ABC):
    """Abstract base class for chatbot export parsers."""

    @abstractmethod
    def parse(self, path: Path) -> list[Conversation]:
        """Parse exported data and return a list of normalized Conversations.

        Args:
            path: Path to the raw export file or directory.

        Returns:
            A list of Conversation objects.
        """
        ...

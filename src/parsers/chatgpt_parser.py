"""Parser for ChatGPT conversation exports.

This parser is a stub — the ChatGPT export file has not been provided yet.
Once available, it will parse ``conversations.json`` from the ChatGPT data
export and normalise it into the common :class:`Conversation` schema.
"""

from __future__ import annotations

from pathlib import Path

from .base import BaseParser, Conversation


class ChatGPTParser(BaseParser):
    """Parse ChatGPT conversation export JSON (not yet implemented)."""

    def parse(self, path: Path) -> list[Conversation]:
        """Raise :class:`NotImplementedError` until the export is available."""
        raise NotImplementedError("ChatGPT export not yet available")

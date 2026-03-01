"""Parser for Claude conversation exports.

Claude exports as a JSON file (conversations.json) containing a list of
conversation objects.  Each conversation has ``chat_messages`` with ``sender``
values of ``"human"`` or ``"assistant"`` and a ``content`` list of typed blocks.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from .base import BaseParser, Conversation, Turn

logger = logging.getLogger(__name__)


class ClaudeParser(BaseParser):
    """Parse Claude conversation export JSON."""

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def parse(self, path: Path) -> list[Conversation]:
        """Parse a Claude ``conversations.json`` export file.

        Args:
            path: Path to ``conversations.json`` **or** the directory that
                  contains it.

        Returns:
            A list of normalised :class:`Conversation` objects.
        """
        file_path = self._resolve_path(path)
        raw = self._load_json(file_path)
        conversations: list[Conversation] = []

        for idx, raw_conv in enumerate(raw):
            try:
                conv = self._parse_conversation(raw_conv)
                if conv is not None:
                    conversations.append(conv)
            except Exception:
                logger.warning(
                    "Claude: skipping conversation %d (uuid=%s) — unexpected error",
                    idx,
                    raw_conv.get("uuid", "?"),
                    exc_info=True,
                )

        logger.info(
            "Claude: parsed %d conversations (%d skipped) from %s",
            len(conversations),
            len(raw) - len(conversations),
            file_path,
        )
        return conversations

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_path(path: Path) -> Path:
        """Accept either the JSON file itself or its parent directory."""
        if path.is_dir():
            candidate = path / "conversations.json"
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"No conversations.json found in {path}")
        return path

    @staticmethod
    def _load_json(file_path: Path) -> list[dict]:
        with open(file_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON array at top level, got {type(data).__name__}"
            )
        return data

    # ------------------------------------------------------------------ #
    #  Conversation-level                                                 #
    # ------------------------------------------------------------------ #

    def _parse_conversation(self, raw: dict) -> Conversation | None:
        conv_id: str = raw.get("uuid", "")
        title: str | None = raw.get("name") or None
        created_at = self._parse_timestamp(raw.get("created_at"))
        updated_at = self._parse_timestamp(raw.get("updated_at"))

        messages: list[dict] = raw.get("chat_messages", [])
        if not messages:
            logger.warning(
                "Claude: skipping conversation %s (%s) — no messages",
                conv_id,
                title,
            )
            return None

        turns: list[Turn] = []
        for msg in messages:
            turn = self._parse_message(msg, conv_id)
            if turn is not None:
                turns.append(turn)

        if not turns:
            logger.warning(
                "Claude: skipping conversation %s (%s) — all messages empty",
                conv_id,
                title,
            )
            return None

        has_user = any(t.role == "user" for t in turns)
        if not has_user:
            logger.warning(
                "Claude: skipping conversation %s (%s) — no user turns",
                conv_id,
                title,
            )
            return None

        return Conversation(
            source="claude",
            conversation_id=conv_id,
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            turns=turns,
        )

    # ------------------------------------------------------------------ #
    #  Message-level                                                      #
    # ------------------------------------------------------------------ #

    _ROLE_MAP: dict[str, str] = {"human": "user", "assistant": "assistant"}

    def _parse_message(self, msg: dict, conv_id: str) -> Turn | None:
        sender = msg.get("sender", "")
        role = self._ROLE_MAP.get(sender)
        if role is None:
            logger.warning(
                "Claude: skipping message in %s — unknown sender %r",
                conv_id,
                sender,
            )
            return None

        # Build content from typed content blocks.  Prefer the structured
        # ``content`` list because the top-level ``text`` field is sometimes
        # empty even when the blocks contain text.
        text = self._extract_text_from_blocks(msg.get("content", []))

        # Fallback to top-level ``text`` when blocks yield nothing.
        if not text.strip():
            text = msg.get("text", "") or ""

        if not text.strip():
            logger.debug(
                "Claude: empty turn (role=%s) in conversation %s — skipping",
                role,
                conv_id,
            )
            return None

        timestamp = self._parse_timestamp(msg.get("created_at"))
        return Turn(role=role, content=text.strip(), timestamp=timestamp)

    # ------------------------------------------------------------------ #
    #  Content-block extraction                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_text_from_blocks(blocks: list[dict]) -> str:
        """Concatenate the ``text`` field of all ``type == "text"`` blocks."""
        parts: list[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text")
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    #  Timestamp                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        """Parse an ISO-8601 timestamp string with optional trailing ``Z``."""
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            logger.warning("Claude: could not parse timestamp %r", value)
            return None

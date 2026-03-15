"""Parser for ChatGPT conversation exports.

ChatGPT exports as one or more JSON files (conversations-NNN.json) each
containing a list of conversation objects.  Each conversation stores its
messages in a tree structure via a mapping dict.  The canonical thread is
recovered by tracing from current_node back to the root through parent links.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .base import BaseParser, Conversation, Turn

logger = logging.getLogger(__name__)


class ChatGPTParser(BaseParser):
    """Parse ChatGPT conversation export JSON."""

    def parse(self, path: Path) -> list[Conversation]:
        """Parse ChatGPT export files."""
        files = self._resolve_files(path)
        conversations: list[Conversation] = []
        total_raw = 0

        for file_path in files:
            raw = self._load_json(file_path)
            total_raw += len(raw)
            for idx, raw_conv in enumerate(raw):
                try:
                    conv = self._parse_conversation(raw_conv)
                    if conv is not None:
                        conversations.append(conv)
                except Exception:
                    logger.warning(
                        "ChatGPT: skipping conversation %d in %s",
                        idx,
                        file_path.name,
                        exc_info=True,
                    )

        logger.info(
            "ChatGPT: parsed %d conversations (%d skipped) from %d files",
            len(conversations),
            total_raw - len(conversations),
            len(files),
        )
        return conversations

    @staticmethod
    def _resolve_files(path: Path) -> list[Path]:
        """Return a sorted list of ChatGPT export JSON files."""
        if path.is_file():
            return [path]
        files = sorted(path.glob("conversations-*.json"))
        if not files:
            candidate = path / "conversations.json"
            if candidate.exists():
                return [candidate]
            raise FileNotFoundError(f"No ChatGPT export files found in {path}")
        return files

    @staticmethod
    def _load_json(file_path: Path) -> list[dict]:
        with open(file_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(
                f"Expected JSON array in {file_path.name}, got {type(data).__name__}"
            )
        return data

    def _parse_conversation(self, raw: dict) -> Conversation | None:
        conv_id: str = raw.get("conversation_id") or raw.get("id", "")
        title: str | None = raw.get("title") or None
        created_at = self._parse_epoch(raw.get("create_time"))
        updated_at = self._parse_epoch(raw.get("update_time"))

        mapping: dict = raw.get("mapping", {})
        current_node: str | None = raw.get("current_node")
        if not mapping or not current_node:
            return None

        ordered_nodes = self._trace_main_thread(mapping, current_node)

        turns: list[Turn] = []
        for node in ordered_nodes:
            turn = self._parse_node(node, conv_id)
            if turn is not None:
                turns.append(turn)

        if not turns:
            return None

        has_user = any(t.role == "user" for t in turns)
        if not has_user:
            return None

        return Conversation(
            source="chatgpt",
            conversation_id=conv_id,
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            turns=turns,
        )

    @staticmethod
    def _trace_main_thread(mapping: dict, current_node: str) -> list[dict]:
        """Walk from current_node up to the root, then reverse."""
        path: list[dict] = []
        node_id: str | None = current_node
        visited: set[str] = set()
        while node_id and node_id not in visited:
            visited.add(node_id)
            node = mapping.get(node_id)
            if node is None:
                break
            path.append(node)
            node_id = node.get("parent")
        path.reverse()
        return path

    _ROLE_MAP: dict[str, str] = {"user": "user", "assistant": "assistant"}

    def _parse_node(self, node: dict, conv_id: str) -> Turn | None:
        msg = node.get("message")
        if msg is None:
            return None

        author = msg.get("author", {})
        role_raw = author.get("role", "")
        role = self._ROLE_MAP.get(role_raw)
        if role is None:
            return None

        text = self._extract_text(msg)
        if not text.strip():
            return None

        timestamp = self._parse_epoch(msg.get("create_time"))
        return Turn(role=role, content=text.strip(), timestamp=timestamp)

    @staticmethod
    def _extract_text(msg: dict) -> str:
        """Extract text content from a ChatGPT message."""
        content = msg.get("content", {})
        content_type = content.get("content_type", "")

        if content_type not in ("text", "multimodal_text"):
            return ""

        parts = content.get("parts", [])
        text_parts: list[str] = []
        for part in parts:
            if isinstance(part, str) and part.strip():
                text_parts.append(part)
        return "\n\n".join(text_parts)

    @staticmethod
    def _parse_epoch(value: float | int | None) -> datetime | None:
        """Convert a Unix epoch timestamp to a datetime."""
        if value is None:
            return None
        try:
            ts = float(value)
            # Detect millisecond timestamps (year > 2100 in seconds)
            if ts > 4_102_444_800:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (ValueError, TypeError, OSError):
            logger.warning("ChatGPT: could not parse timestamp %r", value)
            return None

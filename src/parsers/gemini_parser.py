"""Parser for Gemini (Google Takeout) conversation exports.

Gemini exports as ``MyActivity.html`` — a single HTML file containing one
``<div class="outer-cell …">`` per interaction.  Each entry has a user prompt
(prefixed by ``Prompted\\xa0``), a timestamp, and an HTML-formatted assistant
response.  There are no native conversation IDs or threading; this parser
groups consecutive entries by temporal proximity to reconstruct conversations.

Assistant responses are converted from HTML to Markdown so that formatting
(headers, bold, lists, tables, etc.) is preserved for downstream analysis.
"""

from __future__ import annotations

import hashlib
import html as html_mod
import logging
import re
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path

from markdownify import markdownify

from .base import BaseParser, Conversation, Turn

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Lightweight HTML → plain-text stripper  (used for prompt parsing)  #
# ------------------------------------------------------------------ #

class _HTMLTextExtractor(HTMLParser):
    """Strip HTML tags and return plain text with basic whitespace."""

    _BLOCK_TAGS = frozenset({
        "p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "tr", "blockquote", "pre", "hr", "table",
    })

    def __init__(self) -> None:
        super().__init__()
        self._pieces: list[str] = []
        self._in_pre: int = 0  # nesting depth for <pre> blocks

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._BLOCK_TAGS:
            self._pieces.append("\n")
        if tag == "pre":
            self._in_pre += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._BLOCK_TAGS:
            self._pieces.append("\n")
        if tag == "pre":
            self._in_pre = max(0, self._in_pre - 1)

    def handle_data(self, data: str) -> None:
        self._pieces.append(data)

    def handle_entityref(self, name: str) -> None:
        char = html_mod.unescape(f"&{name};")
        self._pieces.append(char)

    def handle_charref(self, name: str) -> None:
        char = html_mod.unescape(f"&#{name};")
        self._pieces.append(char)

    def get_text(self) -> str:
        raw = "".join(self._pieces)
        # Collapse runs of whitespace (but keep explicit newlines).
        lines = raw.splitlines()
        cleaned: list[str] = []
        for line in lines:
            stripped = " ".join(line.split())
            cleaned.append(stripped)
        # Remove excessive blank lines.
        text = "\n".join(cleaned)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def strip_html(html_str: str) -> str:
    """Convert an HTML snippet to readable plain text.

    Used for user-prompt extraction and timestamp detection where we
    need raw text, **not** for assistant responses (use
    :func:`html_to_markdown` for those).
    """
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(html_str)
    except Exception:
        # Gracefully degrade on malformed HTML.
        return re.sub(r"<[^>]+>", " ", html_str).strip()
    return extractor.get_text()


# ------------------------------------------------------------------ #
#  HTML → Markdown converter  (used for assistant responses)          #
# ------------------------------------------------------------------ #

def html_to_markdown(html_str: str) -> str:
    """Convert an HTML snippet to clean Markdown.

    Uses the ``markdownify`` library with settings tuned for Gemini's
    HTML output (ATX-style headers, ``-`` bullets, fenced code blocks).
    Any remaining HTML tags are stripped after conversion and excessive
    blank lines are collapsed.
    """
    try:
        md = markdownify(
            html_str,
            heading_style="ATX",
            bullets="-",
            strip=["br"],
        )
    except Exception:
        logger.warning("Gemini: markdownify failed — falling back to plain-text")
        return strip_html(html_str)

    # Strip any stray HTML tags that markdownify didn't handle.
    md = re.sub(r"<[^>]+>", "", md)

    # Collapse 3+ consecutive newlines → 2.
    md = re.sub(r"\n{3,}", "\n\n", md)

    return md.strip()


# ------------------------------------------------------------------ #
#  Timestamp parsing                                                  #
# ------------------------------------------------------------------ #

# "Feb 28, 2026, 5:29:30\u202fPM CST"
_TS_PATTERN = re.compile(
    r"^([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}:\d{2})"
    r"[\s\u202f\xa0]*(AM|PM)"
    r"\s+(\w+)$"
)

# Rough UTC offsets for common US timezone abbreviations found in Takeout.
_TZ_OFFSETS: dict[str, int] = {
    "EST": -5, "EDT": -4,
    "CST": -6, "CDT": -5,
    "MST": -7, "MDT": -6,
    "PST": -8, "PDT": -7,
    "UTC": 0,  "GMT": 0,
}


def _parse_gemini_timestamp(raw: str) -> datetime | None:
    """Parse a Gemini-style timestamp string to a tz-aware datetime."""
    raw = raw.strip()
    m = _TS_PATTERN.match(raw)
    if not m:
        logger.warning("Gemini: could not parse timestamp %r", raw)
        return None
    date_part, ampm, tz_abbr = m.group(1), m.group(2), m.group(3)
    try:
        dt = datetime.strptime(f"{date_part} {ampm}", "%b %d, %Y, %I:%M:%S %p")
    except ValueError:
        logger.warning("Gemini: could not parse date portion %r", date_part)
        return None
    offset_hours = _TZ_OFFSETS.get(tz_abbr.upper(), 0)
    tz = timezone(timedelta(hours=offset_hours))
    return dt.replace(tzinfo=tz)


# ------------------------------------------------------------------ #
#  Main parser                                                        #
# ------------------------------------------------------------------ #

# Regex that splits the file into outer-cell blocks.
_OUTER_CELL_RE = re.compile(
    r'<div\s+class="outer-cell\s+mdl-cell\s+mdl-cell--12-col\s+mdl-shadow--2dp">(.*?)</div>\s*(?=<div\s+class="outer-cell|</div>\s*</body>)',
    re.DOTALL,
)

# Regex to extract the content cell (left column, not text-right).
_CONTENT_CELL_RE = re.compile(
    r'<div\s+class="content-cell\s+mdl-cell\s+mdl-cell--6-col\s+mdl-typography--body-1">(.*?)</div>',
    re.DOTALL,
)

# Match timestamp line within a content cell.
_TS_LINE_RE = re.compile(
    r"([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}:\d{2}[\s\u202f\xa0]*(?:AM|PM)\s+\w+)"
)

# Default gap to split sessions into separate conversations.
_SESSION_GAP = timedelta(minutes=60)


class GeminiParser(BaseParser):
    """Parse a Gemini Google Takeout ``MyActivity.html`` export."""

    def __init__(self, session_gap_minutes: int = 60) -> None:
        self._session_gap = timedelta(minutes=session_gap_minutes)

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def parse(self, path: Path) -> list[Conversation]:
        """Parse a Gemini ``MyActivity.html`` export.

        Args:
            path: Path to ``MyActivity.html`` **or** the directory that
                  contains it.

        Returns:
            A list of normalised :class:`Conversation` objects, grouped
            into sessions by temporal proximity.
        """
        file_path = self._resolve_path(path)
        raw_html = file_path.read_text(encoding="utf-8", errors="replace")

        entries = self._extract_entries(raw_html)
        logger.info("Gemini: extracted %d raw entries from %s", len(entries), file_path)

        # Parse individual entries into (timestamp, user_turn, assistant_turn) tuples.
        parsed_entries: list[tuple[datetime, Turn, Turn]] = []
        skipped = 0
        for entry_html in entries:
            result = self._parse_entry(entry_html)
            if result is None:
                skipped += 1
                continue
            parsed_entries.append(result)

        if skipped:
            logger.info("Gemini: skipped %d non-conversation entries", skipped)

        # Entries come in reverse chronological order — sort chronologically.
        parsed_entries.sort(key=lambda e: e[0])

        # Group into conversations by session gap.
        conversations = self._group_into_conversations(parsed_entries)

        logger.info(
            "Gemini: parsed %d conversations from %d prompt-response pairs",
            len(conversations),
            len(parsed_entries),
        )
        return conversations

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_path(path: Path) -> Path:
        if path.is_dir():
            candidate = path / "MyActivity.html"
            if candidate.exists():
                return candidate
            # Also look for any .html file.
            html_files = list(path.glob("*.html"))
            if html_files:
                return html_files[0]
            raise FileNotFoundError(f"No HTML file found in {path}")
        return path

    @staticmethod
    def _extract_entries(raw_html: str) -> list[str]:
        """Split the monolithic HTML into per-entry strings."""
        return _OUTER_CELL_RE.findall(raw_html)

    def _parse_entry(self, entry_html: str) -> tuple[datetime, Turn, Turn] | None:
        """Parse a single outer-cell div into (timestamp, user_turn, assistant_turn).

        Returns ``None`` for non-conversation entries (Canvas, feedback, etc.).
        """
        # Extract the left content cell.
        m = _CONTENT_CELL_RE.search(entry_html)
        if not m:
            logger.debug("Gemini: no content cell found in entry — skipping")
            return None
        content_html = m.group(1)

        # The content cell text starts with "Prompted\xa0" for real Q&A entries.
        # Strip tags for the initial detection but keep raw HTML for later.
        content_text = strip_html(content_html)
        if not content_text.startswith("Prompted"):
            logger.debug(
                "Gemini: non-Prompted entry (%s) — skipping",
                content_text[:40].replace("\n", " "),
            )
            return None

        # ---- split on <br> to isolate logical lines ---- #
        # Replace <br> / <br/> with a unique sentinel, then strip HTML from
        # each part individually to avoid null-byte leakage.
        sentinel = "|||BREAK|||"
        parts_html = re.sub(r"<br\s*/?>", sentinel, content_html)

        # Split the HTML-with-sentinels on the sentinel itself, then strip
        # tags from each chunk independently.
        parts_text = [strip_html(p) for p in parts_html.split(sentinel)]

        # Walk lines looking for the "Prompted" prefix and timestamp.
        user_text = ""
        timestamp_str = ""
        ts_line_idx = -1

        for i, line in enumerate(parts_text):
            line_s = line.strip()
            if _TS_LINE_RE.fullmatch(line_s):
                timestamp_str = line_s
                ts_line_idx = i
                break

        if not timestamp_str or ts_line_idx < 0:
            logger.warning("Gemini: could not locate timestamp in entry — skipping")
            return None

        # User prompt = everything before the timestamp, after "Prompted\xa0",
        # excluding "Attached N files." and file-list lines.
        prompt_lines: list[str] = []
        for line in parts_text[:ts_line_idx]:
            line_s = line.strip()
            if not line_s:
                continue
            if line_s.startswith("Prompted"):
                # Strip the "Prompted " prefix (may use \xa0 or regular space).
                line_s = re.sub(r"^Prompted[\s\xa0]+", "", line_s)
                if line_s:
                    prompt_lines.append(line_s)
                continue
            if re.match(r"^Attached\s+\d+\s+files?", line_s):
                continue
            if re.match(r"^-\s*[\xa0\s]", line_s):
                continue
            # Sometimes the prompt wraps across multiple br-delimited lines.
            prompt_lines.append(line_s)

        user_text = " ".join(prompt_lines).strip()
        if not user_text:
            logger.warning("Gemini: empty user prompt in entry — skipping")
            return None

        # Assistant response = everything after the timestamp line in the raw HTML.
        # Re-split the *HTML* version on the sentinel to grab post-timestamp content.
        html_parts = parts_html.split(sentinel)
        # Find the part that contains the timestamp, then everything after is response.
        response_html_parts: list[str] = []
        found_ts = False
        for part in html_parts:
            if found_ts:
                response_html_parts.append(part)
            elif _TS_LINE_RE.search(strip_html(part)):
                found_ts = True
                continue

        response_html = "<br>".join(response_html_parts)
        assistant_text = html_to_markdown(response_html)

        if not assistant_text:
            logger.warning("Gemini: empty assistant response — skipping entry")
            return None

        ts = _parse_gemini_timestamp(timestamp_str)
        if ts is None:
            return None

        user_turn = Turn(role="user", content=user_text, timestamp=ts)
        assistant_turn = Turn(role="assistant", content=assistant_text, timestamp=ts)
        return (ts, user_turn, assistant_turn)

    def _group_into_conversations(
        self,
        entries: list[tuple[datetime, Turn, Turn]],
    ) -> list[Conversation]:
        """Group chronologically-sorted entries into conversations by time gap."""
        if not entries:
            return []

        conversations: list[Conversation] = []
        current_group: list[tuple[datetime, Turn, Turn]] = [entries[0]]

        for entry in entries[1:]:
            prev_ts = current_group[-1][0]
            curr_ts = entry[0]
            if (curr_ts - prev_ts) > self._session_gap:
                conversations.append(self._make_conversation(current_group))
                current_group = [entry]
            else:
                current_group.append(entry)

        # Flush the last group.
        conversations.append(self._make_conversation(current_group))
        return conversations

    @staticmethod
    def _make_conversation(
        group: list[tuple[datetime, Turn, Turn]],
    ) -> Conversation:
        """Build a :class:`Conversation` from a group of entries."""
        turns: list[Turn] = []
        for _ts, user_turn, asst_turn in group:
            turns.append(user_turn)
            turns.append(asst_turn)

        first_ts = group[0][0]
        last_ts = group[-1][0]

        # Derive a stable conversation ID from the first timestamp.
        id_seed = f"gemini-{first_ts.isoformat()}"
        conv_id = hashlib.sha256(id_seed.encode()).hexdigest()[:16]

        # Use the first user prompt (truncated) as the conversation title.
        first_prompt = turns[0].content if turns else ""
        title = first_prompt[:80] + ("…" if len(first_prompt) > 80 else "")

        return Conversation(
            source="gemini",
            conversation_id=conv_id,
            title=title or None,
            created_at=first_ts,
            updated_at=last_ts,
            turns=turns,
        )

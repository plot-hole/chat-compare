"""User language and behavior analysis.

Analyses user turns only, grouped by which platform (source) the user
was talking to.  Covers message complexity, prompt engineering patterns,
formality & tone, rephrasing detection, first-message analysis, and
topic-controlled routing awareness.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import textstat

from src.parsers.base import Conversation

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Phrase inventories                                                  #
# ------------------------------------------------------------------ #

INSTRUCTION_PATTERNS: dict[str, list[str]] = {
    "step_by_step": ["step by step", "step-by-step"],
    "conciseness": ["be concise", "keep it short", "briefly", "in brief"],
    "simplification": [
        "explain like", "eli5", "simple terms", "in layman",
    ],
    "detail": ["be specific", "be detailed", "in detail", "elaborate"],
    "formatting": ["list", "bullet points", "numbered list"],
    "examples": ["example", "for instance", "give me an example"],
}

ROLE_PATTERNS: list[str] = [
    "act as", "you are a", "pretend you're", "imagine you're",
    "as a", "role of",
]

CONSTRAINT_PATTERNS: list[str] = [
    "don't", "do not", "avoid", "never", "without",
    "only", "just", "must", "always", "make sure",
]

CONTEXT_PATTERNS: list[str] = [
    "here is", "here's", "attached", "below is", "the following",
    "context:", "background:", "for context", "fyi",
]

POLITENESS_MARKERS: list[str] = [
    "please", "thanks", "thank you", "sorry", "appreciate",
    "would you", "could you", "mind if",
]

CONTRACTION_RE = re.compile(
    r"\b(?:don't|can't|i'm|it's|that's|what's|won't|didn't|isn't|"
    r"aren't|wasn't|weren't|wouldn't|couldn't|shouldn't|hasn't|haven't|"
    r"hadn't|he's|she's|we're|they're|you're|i've|we've|they've|"
    r"i'll|we'll|they'll|you'll|he'll|she'll|let's|there's|who's)\b",
    re.IGNORECASE,
)

IMPERATIVE_STARTERS = re.compile(
    r"^(?:write|create|fix|show|make|build|generate|explain|tell|"
    r"give|find|list|help|add|remove|update|change|convert|translate|"
    r"summarize|describe|compare|analyze|design|implement|define|"
    r"suggest|recommend|provide|calculate|solve|debug|refactor|"
    r"optimize|review|check|test|run|set|use|try|do)\b",
    re.IGNORECASE,
)

REQUEST_STARTERS = re.compile(
    r"^(?:can you|could you|would you|how do i|how can i|"
    r"is it possible|what is|what are|what does|what do|"
    r"how does|how do|why does|why do|is there|are there|"
    r"do you|will you|shall we)\b",
    re.IGNORECASE,
)

EMOJI_RE = re.compile(
    r"[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff"
    r"\U0001f680-\U0001f6ff\U0001f1e0-\U0001f1ff"
    r"\U00002702-\U000027b0\U0001f900-\U0001f9ff"
    r"\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff"
    r"\U00002600-\U000026ff]",
)

EMOTICON_RE = re.compile(
    r"(?:[:;][-']?[)(DPp/\\|oO3><])|(?:<3)|(?:xD|XD)",
)


# ------------------------------------------------------------------ #
#  Public entry point                                                  #
# ------------------------------------------------------------------ #


def run(conversations: list[Conversation], config: dict) -> dict[str, Any]:
    """Run all user-behaviour analyses on *conversations*.

    Args:
        conversations: Normalised conversation objects (all sources).
        config: Loaded ``config.yaml`` dict.

    Returns:
        Nested dict with keys: ``message_complexity``,
        ``prompt_engineering``, ``formality``, ``rephrasing``,
        ``first_message``, ``topic_routing``, ``_meta``.
    """
    min_len = config.get("analysis", {}).get("min_turn_length", 10)
    processed_root = Path(
        config.get("paths", {}).get("processed_data", "data/processed")
    )
    outputs_root = Path(
        config.get("paths", {}).get("outputs", "data/outputs")
    )

    # ---- collect user turns grouped by source ---- #
    user_texts: dict[str, list[str]] = {}
    conv_by_source: dict[str, list[Conversation]] = {}

    for conv in conversations:
        conv_by_source.setdefault(conv.source, []).append(conv)
        for turn in conv.turns:
            if turn.role == "user" and len(turn.content) >= min_len:
                user_texts.setdefault(conv.source, []).append(turn.content)

    sources = sorted(user_texts.keys())
    logger.info(
        "user_behavior: sources=%s, user turns: %s",
        sources,
        {s: len(user_texts[s]) for s in sources},
    )

    word_counts: dict[str, int] = {
        s: sum(len(t.split()) for t in user_texts[s]) for s in sources
    }

    # ---- run each sub-analysis ---- #
    results: dict[str, Any] = {}
    results["message_complexity"] = _message_complexity(user_texts, word_counts)
    results["prompt_engineering"] = _prompt_engineering(user_texts, word_counts)
    results["formality"] = _formality(user_texts, word_counts)
    results["rephrasing"] = _rephrasing(
        conv_by_source, processed_root, min_len,
    )
    results["first_message"] = _first_message(conv_by_source, min_len)
    results["topic_routing"] = _topic_routing(
        conv_by_source, outputs_root, word_counts, min_len,
    )
    results["_meta"] = {"sources": sources}

    # ---- data quality notes ---- #
    results["data_notes"] = {
        "gemini_capitalization": (
            "Google Takeout auto-capitalizes ~94% of user prompts at the "
            "application/export level.  The lowercase_only_fraction for Gemini "
            "reflects this export artifact, NOT actual user typing behaviour.  "
            "Cross-platform comparisons of capitalization-based metrics are "
            "unreliable."
        ),
    }

    return results


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #


def _count_phrase(text_lower: str, phrase: str) -> int:
    """Count non-overlapping occurrences of *phrase* in *text_lower*."""
    if " " in phrase:
        return text_lower.count(phrase)
    return len(re.findall(rf"\b{re.escape(phrase)}\b", text_lower))


def _per_1k(count: int, total_words: int) -> float:
    return round(count * 1000 / total_words, 3) if total_words else 0.0


def _describe_dist(values: list[float | int]) -> dict[str, Any]:
    """Return mean / median / std / count."""
    if not values:
        return {"mean": 0, "median": 0, "std": 0, "count": 0}
    n = len(values)
    mean = sum(values) / n
    sv = sorted(values)
    median = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2
    var = sum((x - mean) ** 2 for x in values) / n
    return {
        "mean": round(mean, 2),
        "median": round(float(median), 2),
        "std": round(math.sqrt(var), 2),
        "count": n,
    }


def _mattr(tokens: list[str], window: int = 200) -> float:
    """Moving-Average Type-Token Ratio."""
    if len(tokens) < window:
        if not tokens:
            return 0.0
        return round(len(set(tokens)) / len(tokens), 4)
    ttrs: list[float] = []
    for i in range(len(tokens) - window + 1):
        chunk = tokens[i : i + window]
        ttrs.append(len(set(chunk)) / window)
    return round(sum(ttrs) / len(ttrs), 4) if ttrs else 0.0


def _sentence_lengths(text: str) -> list[int]:
    """Split text into sentences and return word counts per sentence."""
    # Simple sentence splitter — split on .!? followed by space or end.
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    lengths = []
    for s in sents:
        words = s.split()
        if words:
            lengths.append(len(words))
    return lengths


# ------------------------------------------------------------------ #
#  1. Message complexity                                               #
# ------------------------------------------------------------------ #


def _message_complexity(
    user_texts: dict[str, list[str]],
    word_counts: dict[str, int],
) -> dict[str, Any]:
    """Per-source message length, sentence length, readability, vocab."""
    results: dict[str, Any] = {}

    for src in sorted(user_texts):
        texts = user_texts[src]

        # Message length (words)
        msg_lengths = [len(t.split()) for t in texts]

        # Sentence lengths
        all_sent_lengths: list[int] = []
        for t in texts:
            all_sent_lengths.extend(_sentence_lengths(t))

        # Readability (Flesch-Kincaid)
        fk_scores: list[float] = []
        for t in texts:
            if len(t.split()) >= 5:
                try:
                    fk_scores.append(textstat.flesch_kincaid_grade(t))
                except Exception:
                    pass

        # Vocabulary richness (MATTR window=200)
        all_tokens = []
        for t in texts:
            all_tokens.extend(
                w.lower() for w in re.findall(r"\b\w+\b", t)
            )
        mattr_val = _mattr(all_tokens, window=200)
        unique_vocab = len(set(all_tokens))

        results[src] = {
            "message_length": _describe_dist(msg_lengths),
            "sentence_length": _describe_dist(all_sent_lengths),
            "readability_fk": {
                "mean": round(sum(fk_scores) / len(fk_scores), 2) if fk_scores else 0,
                "std": round(
                    math.sqrt(
                        sum((x - sum(fk_scores) / len(fk_scores)) ** 2 for x in fk_scores)
                        / len(fk_scores)
                    ),
                    2,
                ) if len(fk_scores) > 1 else 0,
                "count": len(fk_scores),
            },
            "mattr_200": mattr_val,
            "unique_vocabulary": unique_vocab,
            "total_tokens": len(all_tokens),
            "total_words": word_counts[src],
        }

    return results


# ------------------------------------------------------------------ #
#  2. Prompt engineering patterns                                      #
# ------------------------------------------------------------------ #


def _prompt_engineering(
    user_texts: dict[str, list[str]],
    word_counts: dict[str, int],
) -> dict[str, Any]:
    """Count prompt-engineering patterns per source and category."""
    by_source: dict[str, Any] = {}
    by_category: dict[str, Any] = {}
    total_density: dict[str, float] = {}

    for src in sorted(user_texts):
        texts = user_texts[src]
        wc = word_counts[src]
        combined = "\n".join(t.lower() for t in texts)

        cat_counts: dict[str, int] = {}
        phrase_detail: dict[str, dict[str, int]] = {}
        grand_total = 0

        # Instruction patterns (by sub-category)
        for subcat, phrases in INSTRUCTION_PATTERNS.items():
            cat_total = 0
            for phrase in phrases:
                c = _count_phrase(combined, phrase)
                phrase_detail.setdefault(subcat, {})[phrase] = c
                cat_total += c
            cat_counts[subcat] = cat_total
            grand_total += cat_total

        # Role / persona patterns
        role_total = 0
        phrase_detail["role_persona"] = {}
        for phrase in ROLE_PATTERNS:
            c = _count_phrase(combined, phrase)
            phrase_detail["role_persona"][phrase] = c
            role_total += c
        cat_counts["role_persona"] = role_total
        grand_total += role_total

        # Constraint setting
        constraint_total = 0
        phrase_detail["constraints"] = {}
        for phrase in CONSTRAINT_PATTERNS:
            c = _count_phrase(combined, phrase)
            phrase_detail["constraints"][phrase] = c
            constraint_total += c
        cat_counts["constraints"] = constraint_total
        grand_total += constraint_total

        # Context providing
        context_total = 0
        phrase_detail["context_providing"] = {}
        for phrase in CONTEXT_PATTERNS:
            c = _count_phrase(combined, phrase)
            phrase_detail["context_providing"][phrase] = c
            context_total += c
        cat_counts["context_providing"] = context_total
        grand_total += context_total

        by_source[src] = {
            "category_counts": cat_counts,
            "category_per_1k": {k: _per_1k(v, wc) for k, v in cat_counts.items()},
            "phrase_detail": phrase_detail,
            "total_count": grand_total,
            "total_per_1k": _per_1k(grand_total, wc),
            "total_words": wc,
        }
        total_density[src] = _per_1k(grand_total, wc)

    # Aggregate by_category across sources
    all_cats = set()
    for src_data in by_source.values():
        all_cats.update(src_data["category_counts"].keys())
    for cat in sorted(all_cats):
        by_category[cat] = {
            src: by_source[src]["category_per_1k"].get(cat, 0)
            for src in sorted(by_source)
        }

    return {
        "by_source": by_source,
        "by_category": by_category,
        "total_density": total_density,
    }


# ------------------------------------------------------------------ #
#  3. Formality & tone                                                 #
# ------------------------------------------------------------------ #


def _formality(
    user_texts: dict[str, list[str]],
    word_counts: dict[str, int],
) -> dict[str, Any]:
    """Politeness, casual markers, imperative vs request framing."""
    politeness_results: dict[str, Any] = {}
    casual_results: dict[str, Any] = {}
    framing_results: dict[str, Any] = {}

    for src in sorted(user_texts):
        texts = user_texts[src]
        wc = word_counts[src]
        combined_lower = "\n".join(t.lower() for t in texts)
        n_msgs = len(texts)

        # ---- Politeness markers ---- #
        pol_counts: dict[str, int] = {}
        pol_total = 0
        for phrase in POLITENESS_MARKERS:
            c = _count_phrase(combined_lower, phrase)
            pol_counts[phrase] = c
            pol_total += c
        politeness_results[src] = {
            "phrase_counts": pol_counts,
            "phrase_per_1k": {k: _per_1k(v, wc) for k, v in pol_counts.items()},
            "total_count": pol_total,
            "total_per_1k": _per_1k(pol_total, wc),
        }

        # ---- Casual markers ---- #
        contraction_count = len(CONTRACTION_RE.findall(combined_lower))
        lowercase_only = sum(
            1 for t in texts if t == t.lower() and len(t.split()) >= 3
        )
        no_punctuation = sum(
            1 for t in texts
            if not re.search(r'[.!?,;:]', t) and len(t.split()) >= 3
        )
        emoji_count = len(EMOJI_RE.findall("\n".join(texts)))
        emoticon_count = len(EMOTICON_RE.findall("\n".join(texts)))

        casual_results[src] = {
            "contractions": contraction_count,
            "contractions_per_1k": _per_1k(contraction_count, wc),
            "lowercase_only_messages": lowercase_only,
            "lowercase_only_fraction": round(lowercase_only / max(n_msgs, 1), 3),
            "lowercase_only_reliable_for_comparison": src != "gemini",
            "no_punctuation_messages": no_punctuation,
            "no_punctuation_fraction": round(no_punctuation / max(n_msgs, 1), 3),
            "emoji_count": emoji_count,
            "emoticon_count": emoticon_count,
            "emoji_per_1k": _per_1k(emoji_count, wc),
        }

        # ---- Imperative vs request framing ---- #
        imperative_count = 0
        request_count = 0
        other_count = 0
        for t in texts:
            stripped = t.strip()
            if not stripped:
                other_count += 1
                continue
            if IMPERATIVE_STARTERS.match(stripped):
                imperative_count += 1
            elif REQUEST_STARTERS.match(stripped):
                request_count += 1
            else:
                other_count += 1

        framing_results[src] = {
            "imperative_count": imperative_count,
            "imperative_fraction": round(imperative_count / max(n_msgs, 1), 3),
            "request_count": request_count,
            "request_fraction": round(request_count / max(n_msgs, 1), 3),
            "other_count": other_count,
            "other_fraction": round(other_count / max(n_msgs, 1), 3),
            "n_messages": n_msgs,
        }

    # ---- Capitalization rate ---- #
    cap_rate: dict[str, float] = {}
    for src in sorted(user_texts):
        texts = user_texts[src]
        total_letters = 0
        upper_letters = 0
        for t in texts:
            for ch in t:
                if ch.isalpha():
                    total_letters += 1
                    if ch.isupper():
                        upper_letters += 1
        cap_rate[src] = round(upper_letters / max(total_letters, 1), 4)

    return {
        "politeness": politeness_results,
        "casual_markers": casual_results,
        "imperative_vs_request": framing_results,
        "capitalization_rate": cap_rate,
    }


# ------------------------------------------------------------------ #
#  4. Rephrasing detection                                             #
# ------------------------------------------------------------------ #


def _rephrasing(
    conv_by_source: dict[str, list[Conversation]],
    processed_root: Path,
    min_len: int,
) -> dict[str, Any]:
    """Detect consecutive user turns with high semantic similarity.

    Uses dual thresholds:
        - **strict** (0.85): high-confidence rephrases, min 10 words per turn.
        - **loose** (0.70): broader detection, no word-count filter.

    Primary metric is *per 100 user turns* to normalise across platforms
    with different conversation lengths.
    """
    embeddings_path = processed_root / "embeddings_user.npy"
    if not embeddings_path.exists():
        logger.warning(
            "user_behavior: embeddings_user.npy not found — skipping rephrasing."
        )
        return {"error": "embeddings_user.npy not found"}

    STRICT_THRESHOLD = 0.85
    LOOSE_THRESHOLD = 0.70
    MIN_WORDS_STRICT = 10  # both turns must have >= this many words

    # Load user embeddings and build an index mapping.
    all_embeddings = np.load(embeddings_path)

    # Rebuild the text list in the same order as semantic.py generates them.
    user_idx_map: list[tuple[str, str, int]] = []  # (source, conv_id, turn_idx)
    for source in sorted(conv_by_source):
        for conv in conv_by_source[source]:
            for i, turn in enumerate(conv.turns):
                if turn.role == "user" and len(turn.content) >= min_len:
                    user_idx_map.append((conv.source, conv.conversation_id, i))

    if len(user_idx_map) != all_embeddings.shape[0]:
        logger.warning(
            "user_behavior: embedding count mismatch (%d vs %d). "
            "Rebuilding index by conversation order.",
            all_embeddings.shape[0], len(user_idx_map),
        )
        n = min(len(user_idx_map), all_embeddings.shape[0])
        user_idx_map = user_idx_map[:n]
        all_embeddings = all_embeddings[:n]

    # Build lookup: (conv_id, turn_idx) -> embedding row
    lookup: dict[tuple[str, int], int] = {}
    for row, (_src, cid, tidx) in enumerate(user_idx_map):
        lookup[(cid, tidx)] = row

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = all_embeddings / norms

    # Detect rephrases at both thresholds
    rephrase_by_source: dict[str, dict[str, Any]] = {}
    top_conversations: list[dict[str, Any]] = []

    for source in sorted(conv_by_source):
        strict_total = 0
        loose_total = 0
        total_user_turns = 0
        total_conversations = len(conv_by_source[source])
        conv_strict_counts: list[tuple[str, str | None, int]] = []

        for conv in conv_by_source[source]:
            user_turn_indices = [
                i for i, t in enumerate(conv.turns)
                if t.role == "user" and len(t.content) >= min_len
            ]
            total_user_turns += len(user_turn_indices)

            conv_strict = 0
            conv_loose = 0
            for j in range(len(user_turn_indices) - 1):
                tidx_a = user_turn_indices[j]
                tidx_b = user_turn_indices[j + 1]
                row_a = lookup.get((conv.conversation_id, tidx_a))
                row_b = lookup.get((conv.conversation_id, tidx_b))
                if row_a is None or row_b is None:
                    continue

                sim = float(np.dot(normed[row_a], normed[row_b]))

                # Loose threshold — no word-count filter
                if sim > LOOSE_THRESHOLD:
                    conv_loose += 1

                # Strict threshold — both turns must have >= MIN_WORDS_STRICT
                if sim > STRICT_THRESHOLD:
                    text_a = conv.turns[tidx_a].content
                    text_b = conv.turns[tidx_b].content
                    if (len(text_a.split()) >= MIN_WORDS_STRICT
                            and len(text_b.split()) >= MIN_WORDS_STRICT):
                        conv_strict += 1

            strict_total += conv_strict
            loose_total += conv_loose
            if conv_strict > 0:
                conv_strict_counts.append(
                    (conv.conversation_id, conv.title, conv_strict)
                )

        conv_strict_counts.sort(key=lambda x: x[2], reverse=True)

        rephrase_by_source[source] = {
            # Primary metric: strict per 100 user turns
            "strict_total": strict_total,
            "strict_per_100_turns": round(
                strict_total * 100 / max(total_user_turns, 1), 3
            ),
            # Secondary metric: loose per 100 user turns
            "loose_total": loose_total,
            "loose_per_100_turns": round(
                loose_total * 100 / max(total_user_turns, 1), 3
            ),
            # Context
            "total_user_turns": total_user_turns,
            "total_conversations": total_conversations,
            "strict_per_conv": round(
                strict_total / max(total_conversations, 1), 3
            ),
            "loose_per_conv": round(
                loose_total / max(total_conversations, 1), 3
            ),
            "conversations_with_strict_rephrases": len(conv_strict_counts),
        }

        for cid, title, count in conv_strict_counts[:5]:
            top_conversations.append({
                "source": source,
                "conversation_id": cid,
                "title": title or "(untitled)",
                "rephrase_count": count,
            })

    top_conversations.sort(key=lambda x: x["rephrase_count"], reverse=True)
    top_conversations = top_conversations[:10]

    return {
        "by_source": rephrase_by_source,
        "top_conversations": top_conversations,
        "thresholds": {
            "strict": STRICT_THRESHOLD,
            "loose": LOOSE_THRESHOLD,
            "min_words_strict": MIN_WORDS_STRICT,
        },
    }


# ------------------------------------------------------------------ #
#  5. First message analysis                                           #
# ------------------------------------------------------------------ #


def _first_message(
    conv_by_source: dict[str, list[Conversation]],
    min_len: int,
) -> dict[str, Any]:
    """Analyse the first user message of each conversation."""
    results: dict[str, Any] = {}

    for source in sorted(conv_by_source):
        first_texts: list[str] = []
        for conv in conv_by_source[source]:
            for turn in conv.turns:
                if turn.role == "user":
                    if len(turn.content) >= min_len:
                        first_texts.append(turn.content)
                    break  # first user turn only

        if not first_texts:
            results[source] = {"n": 0}
            continue

        # Mean length (words)
        lengths = [len(t.split()) for t in first_texts]

        # Classification: question / command / context-dump
        question_count = 0
        command_count = 0
        context_count = 0
        other_count = 0
        for t in first_texts:
            stripped = t.strip()
            if not stripped:
                other_count += 1
                continue
            # Context dumps are long (>50 words) or start with context markers
            if len(stripped.split()) > 50 or re.match(
                r"(?:here is|here's|context:|background:|for context|the following)",
                stripped.lower(),
            ):
                context_count += 1
            elif stripped.rstrip().endswith("?") or REQUEST_STARTERS.match(stripped):
                question_count += 1
            elif IMPERATIVE_STARTERS.match(stripped):
                command_count += 1
            else:
                other_count += 1

        n = len(first_texts)

        # Top 15 opening words
        opening_words: list[str] = []
        opening_bigrams: list[str] = []
        for t in first_texts:
            words = t.split()[:10]
            if words:
                fw = words[0].strip("*#>-`\"'()[]")
                if fw:
                    opening_words.append(fw.lower())
            if len(words) >= 2:
                w1 = words[0].strip("*#>-`\"'()[]")
                w2 = words[1].strip("*#>-`\"'()[]")
                if w1 and w2:
                    opening_bigrams.append(f"{w1.lower()} {w2.lower()}")

        word_counter = Counter(opening_words)
        bigram_counter = Counter(opening_bigrams)

        results[source] = {
            "n": n,
            "length": _describe_dist(lengths),
            "classification": {
                "question": question_count,
                "question_fraction": round(question_count / n, 3),
                "command": command_count,
                "command_fraction": round(command_count / n, 3),
                "context_dump": context_count,
                "context_dump_fraction": round(context_count / n, 3),
                "other": other_count,
            },
            "top_opening_words": word_counter.most_common(15),
            "top_opening_bigrams": bigram_counter.most_common(15),
        }

    return results


# ------------------------------------------------------------------ #
#  6. Topic routing awareness                                          #
# ------------------------------------------------------------------ #


def _topic_routing(
    conv_by_source: dict[str, list[Conversation]],
    outputs_root: Path,
    word_counts: dict[str, int],
    min_len: int,
) -> dict[str, Any] | None:
    """Compare user behaviour on the same topic across platforms."""
    sem_path = outputs_root / "semantic_results.json"
    if not sem_path.exists():
        logger.warning(
            "user_behavior: semantic_results.json not found — "
            "skipping topic routing."
        )
        return None

    with open(sem_path, encoding="utf-8") as f:
        sem = json.load(f)

    user_topics = sem.get("user_topics", {})
    topics_list = user_topics.get("topics", [])

    if not topics_list:
        logger.info("user_behavior: no user topics found — skipping routing.")
        return None

    # Build topic -> source -> turn count mapping
    min_per_source = 10
    sources = sorted(conv_by_source.keys())
    qualifying_topics: list[dict] = []

    for topic in topics_list:
        if topic.get("topic_id", -1) == -1:
            continue  # skip outlier topic
        src_breakdown = topic.get("source_breakdown", {})
        # Check if at least 2 sources have >=min_per_source turns
        qualifying_sources = [
            s for s in sources if src_breakdown.get(s, 0) >= min_per_source
        ]
        if len(qualifying_sources) >= 2:
            qualifying_topics.append({
                "topic_id": topic["topic_id"],
                "words": topic.get("words", []),
                "sources": qualifying_sources,
                "counts": {s: src_breakdown.get(s, 0) for s in qualifying_sources},
            })

    if not qualifying_topics:
        logger.info(
            "user_behavior: no topics with >=%d turns on 2+ platforms — "
            "skipping routing.",
            min_per_source,
        )
        return {
            "skipped": True,
            "reason": f"No topics with >= {min_per_source} user turns on 2+ platforms.",
        }

    return {
        "skipped": False,
        "qualifying_topics": len(qualifying_topics),
        "topics": [
            {
                "topic_id": t["topic_id"],
                "words": t["words"][:5] if t["words"] else [],
                "source_counts": t["counts"],
            }
            for t in qualifying_topics
        ],
    }

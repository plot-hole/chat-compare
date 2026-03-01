"""Temporal trend analysis.

Tracks how bot and user metrics change over time — response length,
vocabulary richness, hedging density, formatting, readability, and
user behaviour patterns.  Detects inflection points where metrics
shift significantly.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import textstat

from src.parsers.base import Conversation

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Reuse pragmatic phrase inventories                                  #
# ------------------------------------------------------------------ #

from src.analysis.pragmatic import (
    HEDGE_PHRASES,
    _count_phrase,
    _per_1k,
)

# Formatting patterns (same as lexical.py)
_FMT_PATTERNS: dict[str, re.Pattern] = {
    "headers": re.compile(r"^#{1,6}\s", re.MULTILINE),
    "bullet_points": re.compile(r"^[\-\*]\s", re.MULTILINE),
    "numbered_lists": re.compile(r"^\d+\.\s", re.MULTILINE),
    "code_blocks": re.compile(r"```"),
    "bold": re.compile(r"\*\*[^*]+\*\*"),
    "italic": re.compile(r"(?<!\*)\*(?!\*)[^*]+\*(?!\*)"),
}


# ------------------------------------------------------------------ #
#  Public entry point                                                  #
# ------------------------------------------------------------------ #


def run(conversations: list[Conversation], config: dict) -> dict[str, Any]:
    """Run temporal analyses on *conversations*.

    Args:
        conversations: Normalised conversation objects (all sources).
        config: Loaded ``config.yaml`` dict.

    Returns:
        Nested dict with activity, bot metrics, user metrics,
        and inflection points over time.
    """
    min_len = config.get("analysis", {}).get("min_turn_length", 10)

    # ---- Step 1: Time bucketing ---- #
    activity, month_buckets = _build_activity(conversations, min_len)
    sources = sorted(activity["per_source"].keys())

    logger.info(
        "temporal: %d months of data across sources: %s",
        len(activity["months"]),
        sources,
    )

    # ---- Step 2: Bot metrics over time ---- #
    bot_metrics = _bot_metrics_over_time(month_buckets, sources)

    # ---- Step 3: User metrics over time ---- #
    user_metrics = _user_metrics_over_time(month_buckets, sources)

    # ---- Step 4: Topic shifts (best-effort) ---- #
    topic_shifts = _topic_shifts_over_time(config)

    # ---- Step 5: Inflection points ---- #
    inflection_points = _detect_inflection_points(bot_metrics, sources)

    results: dict[str, Any] = {
        "activity": activity,
        "bot_metrics_over_time": bot_metrics,
        "user_metrics_over_time": user_metrics,
        "topic_shifts": topic_shifts,
        "inflection_points": inflection_points,
        "_meta": {"sources": sources},
    }

    return results


# ------------------------------------------------------------------ #
#  Step 1: Time bucketing                                              #
# ------------------------------------------------------------------ #


def _month_key(dt: datetime) -> str:
    """Return YYYY-MM string for a datetime."""
    return dt.strftime("%Y-%m")


def _build_activity(
    conversations: list[Conversation],
    min_len: int,
) -> tuple[dict[str, Any], dict]:
    """Group conversations into monthly buckets and compute activity stats.

    Returns:
        Tuple of (activity summary dict, raw month buckets dict).
    """

    # month -> source -> {"assistant": [texts], "user": [texts], "conv_ids": [...]}
    buckets: dict[str, dict[str, dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: {"assistant": [], "user": [], "conv_ids": []})
    )

    skipped_no_date = 0

    for conv in conversations:
        # Determine timestamp for this conversation
        ts = conv.created_at
        if ts is None:
            # Try to use the first turn's timestamp
            for turn in conv.turns:
                if turn.timestamp is not None:
                    ts = turn.timestamp
                    break
        if ts is None:
            skipped_no_date += 1
            continue

        mk = _month_key(ts)
        src = conv.source
        buckets[mk][src]["conv_ids"].append(conv.conversation_id)

        for turn in conv.turns:
            if len(turn.content) < min_len:
                continue
            if turn.role in ("assistant", "user"):
                buckets[mk][src][turn.role].append(turn.content)

    if skipped_no_date:
        logger.warning(
            "temporal: skipped %d conversations with no timestamp",
            skipped_no_date,
        )

    # Sort months chronologically
    months = sorted(buckets.keys())

    # Build activity summary
    all_sources: set[str] = set()
    for mk in months:
        all_sources.update(buckets[mk].keys())
    sources = sorted(all_sources)

    per_source: dict[str, dict[str, list]] = {}
    for src in sources:
        per_source[src] = {
            "conversations": [],
            "turns": [],
            "words": [],
            "low_confidence": [],
        }
        for mk in months:
            bucket = buckets[mk].get(
                src, {"assistant": [], "user": [], "conv_ids": []}
            )
            n_convs = len(set(bucket["conv_ids"]))
            n_turns = len(bucket["assistant"]) + len(bucket["user"])
            n_words = sum(len(t.split()) for t in bucket["assistant"])
            is_low = n_turns < 20

            per_source[src]["conversations"].append(n_convs)
            per_source[src]["turns"].append(n_turns)
            per_source[src]["words"].append(n_words)
            per_source[src]["low_confidence"].append(is_low)

    # Determine source date ranges for overlap marking
    source_ranges: dict[str, dict[str, str]] = {}
    for src in sources:
        src_months = [
            mk
            for mk in months
            if buckets[mk].get(src, {}).get("conv_ids", [])
        ]
        if src_months:
            source_ranges[src] = {"start": src_months[0], "end": src_months[-1]}
        else:
            source_ranges[src] = {"start": "", "end": ""}

    # Find overlap period
    if len(sources) >= 2:
        starts = [
            source_ranges[s]["start"]
            for s in sources
            if source_ranges[s]["start"]
        ]
        ends = [
            source_ranges[s]["end"]
            for s in sources
            if source_ranges[s]["end"]
        ]
        overlap_start = max(starts) if starts else ""
        overlap_end = min(ends) if ends else ""
    else:
        overlap_start = ""
        overlap_end = ""

    activity = {
        "months": months,
        "per_source": per_source,
        "source_ranges": source_ranges,
        "overlap_period": {"start": overlap_start, "end": overlap_end},
    }

    return activity, dict(buckets)


# ------------------------------------------------------------------ #
#  Step 2: Bot metrics over time                                       #
# ------------------------------------------------------------------ #


def _mattr(tokens: list[str], window: int = 500) -> float | None:
    """Moving-Average Type-Token Ratio. Returns None if too few tokens."""
    n = len(tokens)
    if n < window:
        return None
    ttrs: list[float] = []
    for i in range(n - window + 1):
        segment = tokens[i : i + window]
        ttrs.append(len(set(segment)) / window)
    return round(sum(ttrs) / len(ttrs), 4)


def _count_formatting(texts: list[str]) -> int:
    """Count total formatting elements across texts."""
    total = 0
    for text in texts:
        for name, pat in _FMT_PATTERNS.items():
            count = len(pat.findall(text))
            if name == "code_blocks":
                count = count // 2  # Pairs
            total += count
    return total


def _count_all_hedges(texts: list[str]) -> int:
    """Count total hedge phrase occurrences across texts."""
    total = 0
    for text in texts:
        text_lower = text.lower()
        for _cat, phrases in HEDGE_PHRASES.items():
            for phrase in phrases:
                total += _count_phrase(text_lower, phrase)
    return total


def _count_questions(texts: list[str]) -> int:
    """Count question marks across texts."""
    return sum(text.count("?") for text in texts)


def _bot_metrics_over_time(
    month_buckets: dict,
    sources: list[str],
) -> dict[str, Any]:
    """Compute bot metrics for each month and source."""
    months = sorted(month_buckets.keys())

    result: dict[str, Any] = {"months": months}

    for src in sources:
        metrics: dict[str, list[float | None]] = {
            "response_length": [],
            "mattr": [],
            "readability": [],
            "hedge_density": [],
            "formatting_density": [],
            "question_rate": [],
        }

        for mk in months:
            bucket = month_buckets.get(mk, {}).get(src, {})
            asst_texts = bucket.get("assistant", [])

            if not asst_texts:
                for key in metrics:
                    metrics[key].append(None)
                continue

            # Response length (mean words)
            word_counts = [len(t.split()) for t in asst_texts]
            total_words = sum(word_counts)
            mean_words = total_words / len(word_counts)
            metrics["response_length"].append(round(mean_words, 1))

            # MATTR (window=500)
            all_tokens = []
            for t in asst_texts:
                all_tokens.extend(
                    w.lower() for w in t.split() if w.isalpha() and len(w) > 1
                )
            mattr_val = _mattr(all_tokens, window=500)
            metrics["mattr"].append(mattr_val)

            # Readability (mean Flesch-Kincaid)
            fk_scores: list[float] = []
            for text in asst_texts:
                if len(text.split()) < 5:
                    continue
                try:
                    fk = textstat.flesch_kincaid_grade(text)
                    fk_scores.append(fk)
                except Exception:
                    continue
            if fk_scores:
                metrics["readability"].append(
                    round(sum(fk_scores) / len(fk_scores), 2)
                )
            else:
                metrics["readability"].append(None)

            # Hedge density (per 1k words)
            total_hedges = _count_all_hedges(asst_texts)
            metrics["hedge_density"].append(
                _per_1k(total_hedges, total_words)
            )

            # Formatting density (per 1k words)
            total_fmt = _count_formatting(asst_texts)
            metrics["formatting_density"].append(
                _per_1k(total_fmt, total_words)
            )

            # Question rate (questions per turn)
            total_q = _count_questions(asst_texts)
            n_turns = len(asst_texts)
            metrics["question_rate"].append(
                round(total_q / max(n_turns, 1), 3)
            )

        result[src] = metrics

    return result


# ------------------------------------------------------------------ #
#  Step 3: User metrics over time                                      #
# ------------------------------------------------------------------ #


def _user_metrics_over_time(
    month_buckets: dict,
    sources: list[str],
) -> dict[str, Any]:
    """Compute user behaviour metrics for each month and source."""
    months = sorted(month_buckets.keys())

    result: dict[str, Any] = {"months": months}

    for src in sources:
        metrics: dict[str, list[float | None]] = {
            "message_length": [],
            "messages_per_convo": [],
            "question_rate": [],
        }

        for mk in months:
            bucket = month_buckets.get(mk, {}).get(src, {})
            user_texts = bucket.get("user", [])
            conv_ids = bucket.get("conv_ids", [])

            if not user_texts:
                for key in metrics:
                    metrics[key].append(None)
                continue

            # Mean message length (words)
            word_counts = [len(t.split()) for t in user_texts]
            mean_words = sum(word_counts) / len(word_counts)
            metrics["message_length"].append(round(mean_words, 1))

            # Messages per conversation
            n_unique_convs = len(set(conv_ids)) if conv_ids else 1
            msgs_per_conv = len(user_texts) / max(n_unique_convs, 1)
            metrics["messages_per_convo"].append(round(msgs_per_conv, 2))

            # Question rate (fraction of messages containing ?)
            msgs_with_q = sum(1 for t in user_texts if "?" in t)
            metrics["question_rate"].append(
                round(msgs_with_q / len(user_texts), 3)
            )

        result[src] = metrics

    return result


# ------------------------------------------------------------------ #
#  Step 4: Topic shifts over time                                      #
# ------------------------------------------------------------------ #


def _topic_shifts_over_time(config: dict) -> dict[str, Any]:
    """Attempt to load topic assignments from semantic results.

    If per-turn topic assignments aren't accessible from saved results,
    returns a note about the limitation.
    """
    outputs_root = Path(config.get("paths", {}).get("outputs", "data/outputs"))
    semantic_path = outputs_root / "semantic_results.json"

    if not semantic_path.exists():
        logger.info(
            "temporal: no semantic_results.json found, skipping topic shifts"
        )
        return {
            "status": "skipped",
            "reason": "semantic_results.json not found — run semantic analysis first",
        }

    try:
        with open(semantic_path, encoding="utf-8") as f:
            semantic = json.load(f)
    except Exception as exc:
        logger.warning("temporal: failed to load semantic results: %s", exc)
        return {"status": "skipped", "reason": str(exc)}

    # BERTopic results are saved with per-topic counts and source breakdowns,
    # but individual turn-level topic assignments are not persisted in the
    # current save format (stripped by _save_results_json).
    # We report overall topic structure but can't track per-month evolution
    # without re-running embeddings.

    asst_topics = semantic.get("assistant_topics", {})
    topics = asst_topics.get("topics", [])

    if not topics:
        return {
            "status": "skipped",
            "reason": "No topic data found in semantic results",
        }

    return {
        "status": "partial",
        "reason": (
            "Per-turn topic assignments are not persisted in the current "
            "save format. Monthly topic evolution requires re-embedding, "
            "which is deferred as a future enhancement. Overall topic "
            "structure is available in semantic_results.json."
        ),
        "n_topics": asst_topics.get("n_topics_found", 0),
        "top_topics": [
            {
                "topic_id": t["topic_id"],
                "words": [w["word"] for w in t.get("words", [])[:5]],
                "count": t["count"],
            }
            for t in topics[:10]
            if t.get("topic_id", -1) >= 0
        ],
    }


# ------------------------------------------------------------------ #
#  Step 5: Inflection point detection                                  #
# ------------------------------------------------------------------ #


def _detect_inflection_points(
    bot_metrics: dict[str, Any],
    sources: list[str],
    threshold: float = 1.5,
) -> list[dict[str, Any]]:
    """Find months where a metric deviates >threshold σ from running mean.

    Uses an expanding-window z-score: for each month t, compute the
    mean and std of months [0..t-1], then check if month t's value
    exceeds *threshold* standard deviations.
    """
    months = bot_metrics.get("months", [])
    if len(months) < 4:
        logger.info(
            "temporal: too few months (%d) for inflection detection",
            len(months),
        )
        return []

    metric_names = [
        "response_length",
        "mattr",
        "readability",
        "hedge_density",
        "formatting_density",
        "question_rate",
    ]

    inflections: list[dict[str, Any]] = []

    for src in sources:
        src_metrics = bot_metrics.get(src, {})

        for metric_name in metric_names:
            values = src_metrics.get(metric_name, [])

            # Filter to valid (non-None) entries with their month indices
            valid = [
                (i, v) for i, v in enumerate(values) if v is not None
            ]
            if len(valid) < 4:
                continue

            # Expanding window z-score
            for pos in range(3, len(valid)):
                prior_vals = [v for _, v in valid[:pos]]
                mean = sum(prior_vals) / len(prior_vals)
                var = sum((x - mean) ** 2 for x in prior_vals) / len(prior_vals)
                std = math.sqrt(var)

                if std < 1e-6:
                    continue  # No variance in prior data

                idx, current_val = valid[pos]
                zscore = (current_val - mean) / std

                if abs(zscore) >= threshold:
                    direction = "increase" if zscore > 0 else "decrease"
                    inflections.append({
                        "month": months[idx],
                        "source": src,
                        "metric": metric_name,
                        "value": round(current_val, 4),
                        "prior_mean": round(mean, 4),
                        "zscore": round(zscore, 2),
                        "direction": direction,
                    })

    # Sort by absolute z-score (most significant first)
    inflections.sort(key=lambda x: abs(x["zscore"]), reverse=True)

    logger.info("temporal: detected %d inflection points", len(inflections))
    return inflections

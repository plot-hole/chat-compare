"""Conversational behavior pattern analysis.

Analyses hedging language, question rate, disclaimers, verbosity ratio,
first-person usage, turn dynamics, and opening patterns.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any

from src.parsers.base import Conversation

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Phrase inventories                                                  #
# ------------------------------------------------------------------ #

HEDGE_PHRASES: dict[str, list[str]] = {
    "uncertainty": [
        "i think", "i believe", "it seems", "it appears", "arguably",
        "it depends", "in my opinion", "i'd suggest", "i'd recommend",
    ],
    "caveat_softener": [
        "however", "that said", "that being said", "on the other hand",
        "it's worth noting", "it's important to note", "keep in mind",
        "to be fair", "generally speaking", "in general",
    ],
    "epistemic": [
        "might", "could", "may", "perhaps", "possibly", "likely",
        "unlikely", "tends to",
    ],
}

DISCLAIMER_PHRASES: dict[str, list[str]] = {
    "ai_self_reference": [
        "i'm an ai", "as an ai", "i'm a language model",
        "i don't have personal", "i can't experience",
        "i don't have feelings", "i don't have opinions",
    ],
    "knowledge_disclaimer": [
        "as of my knowledge", "as of my last", "my training data",
        "i may not have the latest", "i don't have access to",
    ],
    "safety_professional": [
        "consult a professional", "consult a doctor",
        "this is not medical advice", "this is not financial advice",
        "not a substitute for professional", "seek professional",
    ],
    "hedge_disclaimer": [
        "i'm not sure", "i'm not certain", "i could be wrong",
        "don't quote me",
    ],
}

# Words whose first-person count should be whole-word only
_FIRST_PERSON_RE: dict[str, re.Pattern] = {
    "I": re.compile(r"\bI\b"),
    "we": re.compile(r"\bwe\b", re.IGNORECASE),
    "you": re.compile(r"\byou\b", re.IGNORECASE),
}

_OPENING_TRACK = [
    "I", "The", "Here", "Sure", "Yes", "No",
    "That", "This", "To", "It",
]


# ------------------------------------------------------------------ #
#  Public entry point                                                  #
# ------------------------------------------------------------------ #


def run(conversations: list[Conversation], config: dict) -> dict[str, Any]:
    """Run all pragmatic analyses on *conversations*.

    Args:
        conversations: Normalised conversation objects (all sources).
        config: Loaded ``config.yaml`` dict.

    Returns:
        Nested dict keyed by analysis name, each containing per-source
        results.
    """
    min_len = config.get("analysis", {}).get("min_turn_length", 10)

    # ---- collect assistant turns and conversation structures ---- #
    asst_texts: dict[str, list[str]] = {}      # source -> [text, ...]
    conv_by_source: dict[str, list[Conversation]] = {}

    for conv in conversations:
        conv_by_source.setdefault(conv.source, []).append(conv)
        for turn in conv.turns:
            if turn.role == "assistant" and len(turn.content) >= min_len:
                asst_texts.setdefault(conv.source, []).append(turn.content)

    sources = sorted(asst_texts.keys())
    logger.info(
        "pragmatic: sources=%s, assistant turns: %s",
        sources,
        {s: len(asst_texts[s]) for s in sources},
    )

    # ---- word counts (shared) ---- #
    word_counts: dict[str, int] = {
        s: sum(len(t.split()) for t in asst_texts[s]) for s in sources
    }

    # ---- run each sub-analysis ---- #
    results: dict[str, Any] = {}
    results["hedging"] = _hedging(asst_texts, word_counts)
    results["question_rate"] = _question_rate(asst_texts, word_counts)
    results["disclaimers"] = _disclaimers(asst_texts, word_counts)
    results["verbosity_ratio"] = _verbosity_ratio(conversations, min_len)
    results["first_person"] = _first_person(asst_texts, word_counts)
    results["turn_dynamics"] = _turn_dynamics(conv_by_source)
    results["opening_patterns"] = _opening_patterns(asst_texts)
    results["_meta"] = {"sources": sources}

    return results


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #


def _count_phrase(text_lower: str, phrase: str) -> int:
    """Count non-overlapping occurrences of *phrase* in *text_lower*.

    For single-word phrases, require word boundaries so that e.g. "may"
    doesn't match inside "mayor".
    """
    if " " in phrase:
        return text_lower.count(phrase)
    # Single word — use word-boundary regex.
    return len(re.findall(rf"\b{re.escape(phrase)}\b", text_lower))


def _per_1k(count: int, total_words: int) -> float:
    return round(count * 1000 / total_words, 3) if total_words else 0.0


def _describe_dist(values: list[float | int]) -> dict[str, Any]:
    """Return mean / median / std / count plus the raw distribution."""
    if not values:
        return {"mean": 0, "median": 0, "std": 0, "count": 0, "distribution": []}
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
        "distribution": values,
    }


# ------------------------------------------------------------------ #
#  1. Hedging language                                                 #
# ------------------------------------------------------------------ #


def _hedging(
    asst_texts: dict[str, list[str]],
    word_counts: dict[str, int],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for src in sorted(asst_texts):
        wc = word_counts[src]
        category_counts: dict[str, int] = {}
        phrase_counts: dict[str, int] = {}
        total = 0

        for cat, phrases in HEDGE_PHRASES.items():
            cat_total = 0
            for phrase in phrases:
                count = sum(
                    _count_phrase(t.lower(), phrase) for t in asst_texts[src]
                )
                phrase_counts[phrase] = count
                cat_total += count
            category_counts[cat] = cat_total
            total += cat_total

        results[src] = {
            "category_totals": {k: v for k, v in category_counts.items()},
            "category_per_1k": {k: _per_1k(v, wc) for k, v in category_counts.items()},
            "phrase_counts": {k: v for k, v in phrase_counts.items()},
            "phrase_per_1k": {k: _per_1k(v, wc) for k, v in phrase_counts.items()},
            "hedge_density": _per_1k(total, wc),
            "total_hedges": total,
            "total_words": wc,
        }
    return results


# ------------------------------------------------------------------ #
#  2. Question rate                                                    #
# ------------------------------------------------------------------ #


def _question_rate(
    asst_texts: dict[str, list[str]],
    word_counts: dict[str, int],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for src in sorted(asst_texts):
        texts = asst_texts[src]
        wc = word_counts[src]
        questions_per_turn: list[int] = []
        total_questions = 0

        for text in texts:
            # Count sentences ending with ?
            q_count = len(re.findall(r"\?\s*(?:\n|$|[A-Z\"'])", text + "\n"))
            # Also catch trailing ? at end of text
            if text.rstrip().endswith("?"):
                # Check if already counted
                q_count = max(q_count, text.count("?"))
                # Simple fallback: just count ? marks
                q_count = text.count("?")
            else:
                q_count = text.count("?")
            questions_per_turn.append(q_count)
            total_questions += q_count

        n_turns = len(texts)
        turns_with_q = sum(1 for q in questions_per_turn if q > 0)

        results[src] = {
            "mean_questions_per_turn": round(total_questions / max(n_turns, 1), 3),
            "fraction_turns_with_question": round(turns_with_q / max(n_turns, 1), 3),
            "total_questions": total_questions,
            "questions_per_1k_words": _per_1k(total_questions, wc),
            "n_turns": n_turns,
        }
    return results


# ------------------------------------------------------------------ #
#  3. Caveat / disclaimer patterns                                     #
# ------------------------------------------------------------------ #


def _disclaimers(
    asst_texts: dict[str, list[str]],
    word_counts: dict[str, int],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for src in sorted(asst_texts):
        wc = word_counts[src]
        category_counts: dict[str, int] = {}
        phrase_counts: dict[str, int] = {}
        total = 0

        for cat, phrases in DISCLAIMER_PHRASES.items():
            cat_total = 0
            for phrase in phrases:
                count = sum(
                    _count_phrase(t.lower(), phrase) for t in asst_texts[src]
                )
                phrase_counts[phrase] = count
                cat_total += count
            category_counts[cat] = cat_total
            total += cat_total

        results[src] = {
            "category_totals": {k: v for k, v in category_counts.items()},
            "category_per_1k": {k: _per_1k(v, wc) for k, v in category_counts.items()},
            "phrase_counts": {k: v for k, v in phrase_counts.items()},
            "phrase_per_1k": {k: _per_1k(v, wc) for k, v in phrase_counts.items()},
            "disclaimer_density": _per_1k(total, wc),
            "total_disclaimers": total,
        }
    return results


# ------------------------------------------------------------------ #
#  4. Verbosity ratio                                                  #
# ------------------------------------------------------------------ #


def _verbosity_ratio(
    conversations: list[Conversation],
    min_len: int,
) -> dict[str, Any]:
    """Compute assistant_words / user_words for each adjacent pair."""
    results: dict[str, Any] = {}
    source_ratios: dict[str, list[float]] = {}

    for conv in conversations:
        turns = conv.turns
        for i in range(len(turns) - 1):
            if turns[i].role == "user" and turns[i + 1].role == "assistant":
                u_words = len(turns[i].content.split())
                a_words = len(turns[i + 1].content.split())
                if u_words < 5:
                    continue
                if len(turns[i + 1].content) < min_len:
                    continue
                ratio = a_words / u_words
                source_ratios.setdefault(conv.source, []).append(ratio)

    for src in sorted(source_ratios):
        results[src] = _describe_dist(source_ratios[src])

    return results


# ------------------------------------------------------------------ #
#  5. First-person usage                                               #
# ------------------------------------------------------------------ #


def _first_person(
    asst_texts: dict[str, list[str]],
    word_counts: dict[str, int],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for src in sorted(asst_texts):
        wc = word_counts[src]
        counts: dict[str, int] = {}
        for label, pat in _FIRST_PERSON_RE.items():
            total = sum(len(pat.findall(t)) for t in asst_texts[src])
            counts[label] = total

        results[src] = {
            "raw_counts": counts,
            "per_1k_words": {k: _per_1k(v, wc) for k, v in counts.items()},
        }
    return results


# ------------------------------------------------------------------ #
#  6. Turn dynamics                                                    #
# ------------------------------------------------------------------ #


def _turn_dynamics(
    conv_by_source: dict[str, list[Conversation]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for src in sorted(conv_by_source):
        convs = conv_by_source[src]
        total_turns_list: list[int] = []
        user_turns_list: list[int] = []
        single_exchange = 0

        for conv in convs:
            n_total = len(conv.turns)
            n_user = sum(1 for t in conv.turns if t.role == "user")
            total_turns_list.append(n_total)
            user_turns_list.append(n_user)
            if n_user == 1 and (n_total - n_user) == 1:
                single_exchange += 1

        n_convs = len(convs)
        results[src] = {
            "n_conversations": n_convs,
            "turns_per_conv": _describe_dist(total_turns_list),
            "user_turns_per_conv": _describe_dist(user_turns_list),
            "single_exchange_fraction": round(
                single_exchange / max(n_convs, 1), 3
            ),
            "single_exchange_count": single_exchange,
        }
    return results


# ------------------------------------------------------------------ #
#  7. Opening patterns                                                 #
# ------------------------------------------------------------------ #


def _opening_patterns(
    asst_texts: dict[str, list[str]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for src in sorted(asst_texts):
        texts = asst_texts[src]
        n = len(texts)

        # ---- opening words ---- #
        first_words: list[str] = []
        bigrams: list[str] = []

        for text in texts:
            words = text.split()[:10]
            if words:
                # Normalize: strip leading punctuation/whitespace
                fw = words[0].strip("*#>-`\"'()")
                if fw:
                    first_words.append(fw)
            if len(words) >= 2:
                w1 = words[0].strip("*#>-`\"'()")
                w2 = words[1].strip("*#>-`\"'()")
                if w1 and w2:
                    bigrams.append(f"{w1} {w2}")

        # Top opening words
        word_counter = Counter(first_words)
        top_opening_words = word_counter.most_common(15)

        # Top opening bigrams
        bigram_counter = Counter(bigrams)
        top_bigrams = bigram_counter.most_common(15)

        # Tracked opening word rates
        tracked: dict[str, float] = {}
        for word in _OPENING_TRACK:
            count = sum(1 for fw in first_words if fw == word)
            tracked[word] = round(count / max(n, 1), 3)

        results[src] = {
            "top_opening_words": top_opening_words,
            "top_opening_bigrams": top_bigrams,
            "tracked_opening_rates": tracked,
            "n_turns": n,
        }
    return results

"""Surface-level language analysis.

Analyses assistant turns across sources for vocabulary richness, word
frequency, sentence/response statistics, readability, and formatting habits.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any

import spacy
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer

from src.parsers.base import Conversation

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Public entry point                                                  #
# ------------------------------------------------------------------ #


def run(conversations: list[Conversation], config: dict) -> dict[str, Any]:
    """Run all lexical analyses on *conversations*.

    Args:
        conversations: Normalised conversation objects (all sources).
        config: Loaded ``config.yaml`` dict.

    Returns:
        A nested dict keyed by analysis name, each containing per-source
        results plus any shared data (e.g. TF-IDF feature names).
    """
    min_len = config.get("analysis", {}).get("min_turn_length", 10)
    top_n = config.get("analysis", {}).get("top_n_terms", 30)

    # ---- collect assistant turns grouped by source ---- #
    source_texts: dict[str, list[str]] = {}
    for conv in conversations:
        for turn in conv.turns:
            if turn.role != "assistant":
                continue
            if len(turn.content) < min_len:
                continue
            source_texts.setdefault(conv.source, []).append(turn.content)

    sources = sorted(source_texts.keys())
    logger.info(
        "lexical: sources=%s, assistant turns per source: %s",
        sources,
        {s: len(source_texts[s]) for s in sources},
    )

    # ---- spaCy processing (batched) ---- #
    nlp = _load_spacy()
    stopwords = nlp.Defaults.stop_words | {
        "the", "is", "and", "of", "to", "a", "in", "for", "it", "that",
        "this", "with", "on", "as", "be", "at", "by", "an", "or", "not",
        "are", "was", "but", "from", "have", "has", "had", "do", "does",
        "did", "will", "would", "can", "could", "should", "may", "might",
        "i", "you", "we", "they", "he", "she", "my", "your", "our",
        "their", "its", "me", "us", "him", "her", "them",
    }

    # Build per-source tokenised data via spaCy
    logger.info("lexical: running spaCy tokenisation ...")
    source_docs: dict[str, list[spacy.tokens.Doc]] = {}
    for src in sources:
        docs = list(nlp.pipe(source_texts[src], batch_size=256))
        source_docs[src] = docs
    logger.info("lexical: spaCy done")

    # ---- run each sub-analysis ---- #
    results: dict[str, Any] = {}
    results["vocabulary"] = _vocabulary_richness(source_docs, stopwords)
    results["word_frequency"] = _word_frequency(source_docs, stopwords, top_50=50)
    results["distinctive_words"] = _distinctive_words(source_texts, top_n=top_n)
    results["sentence_stats"] = _sentence_stats(source_docs)
    results["response_length"] = _response_length_stats(source_docs)
    results["readability"] = _readability(source_texts)
    results["formatting"] = _formatting_habits(source_texts)
    results["_meta"] = {"sources": sources}

    return results


# ------------------------------------------------------------------ #
#  spaCy loader                                                        #
# ------------------------------------------------------------------ #

_NLP_CACHE: spacy.Language | None = None


def _load_spacy() -> spacy.Language:
    global _NLP_CACHE
    if _NLP_CACHE is None:
        _NLP_CACHE = spacy.load(
            "en_core_web_sm",
            disable=["ner", "lemmatizer"],  # keep tagger + parser for sentences
        )
        # Raise the max length for long assistant responses
        _NLP_CACHE.max_length = 2_000_000
    return _NLP_CACHE


# ------------------------------------------------------------------ #
#  1. Vocabulary richness                                              #
# ------------------------------------------------------------------ #


def _vocabulary_richness(
    source_docs: dict[str, list[spacy.tokens.Doc]],
    stopwords: set[str],
) -> dict[str, Any]:
    """TTR, MATTR (window=500), and unique-vocabulary size per source."""
    results: dict[str, Any] = {}
    for src, docs in source_docs.items():
        tokens = _alpha_tokens(docs, stopwords)
        n = len(tokens)
        types = set(tokens)
        ttr = len(types) / n if n else 0.0
        mattr = _mattr(tokens, window=500)
        results[src] = {
            "total_tokens": n,
            "unique_types": len(types),
            "ttr": round(ttr, 4),
            "mattr_500": round(mattr, 4),
        }
    return results


def _mattr(tokens: list[str], window: int = 500) -> float:
    """Moving-Average Type-Token Ratio."""
    n = len(tokens)
    if n < window:
        # Fallback to plain TTR when text is shorter than window.
        return len(set(tokens)) / n if n else 0.0
    ttrs: list[float] = []
    for i in range(n - window + 1):
        segment = tokens[i : i + window]
        ttrs.append(len(set(segment)) / window)
    return sum(ttrs) / len(ttrs)


# ------------------------------------------------------------------ #
#  2. Word frequency + TF-IDF distinctive words                        #
# ------------------------------------------------------------------ #


def _word_frequency(
    source_docs: dict[str, list[spacy.tokens.Doc]],
    stopwords: set[str],
    top_50: int = 50,
) -> dict[str, Any]:
    """Top-N most frequent non-stopword tokens per source."""
    results: dict[str, Any] = {}
    for src, docs in source_docs.items():
        tokens = _alpha_tokens(docs, stopwords)
        counter = Counter(tokens)
        results[src] = {
            "top_words": counter.most_common(top_50),
        }
    return results


def _distinctive_words(
    source_texts: dict[str, list[str]],
    top_n: int = 30,
) -> dict[str, Any]:
    """TF-IDF 'distinctive words' — one document per source."""
    sources = sorted(source_texts.keys())
    # Combine all assistant text per source into a single mega-document.
    corpus = [" ".join(source_texts[s]) for s in sources]

    vec = TfidfVectorizer(
        max_features=10_000,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",  # alpha only, >=2 chars
        sublinear_tf=True,
    )
    tfidf_matrix = vec.fit_transform(corpus)
    feature_names = vec.get_feature_names_out()

    results: dict[str, Any] = {}
    for idx, src in enumerate(sources):
        row = tfidf_matrix[idx].toarray().flatten()
        top_indices = row.argsort()[::-1][:top_n]
        results[src] = [
            {"word": feature_names[i], "tfidf": round(float(row[i]), 4)}
            for i in top_indices
            if row[i] > 0
        ]
    return results


# ------------------------------------------------------------------ #
#  3. Sentence statistics                                              #
# ------------------------------------------------------------------ #


def _sentence_stats(
    source_docs: dict[str, list[spacy.tokens.Doc]],
) -> dict[str, Any]:
    """Sentence-length (in words) statistics per source."""
    results: dict[str, Any] = {}
    for src, docs in source_docs.items():
        lengths: list[int] = []
        for doc in docs:
            for sent in doc.sents:
                # Count real words only (not pure punctuation / whitespace)
                word_count = sum(1 for t in sent if not t.is_punct and not t.is_space)
                if word_count > 0:
                    lengths.append(word_count)
        results[src] = _describe(lengths, "sentence_length")
    return results


def _response_length_stats(
    source_docs: dict[str, list[spacy.tokens.Doc]],
) -> dict[str, Any]:
    """Per-response word count statistics per source."""
    results: dict[str, Any] = {}
    for src, docs in source_docs.items():
        lengths: list[int] = []
        for doc in docs:
            wc = sum(1 for t in doc if not t.is_punct and not t.is_space)
            lengths.append(wc)
        results[src] = _describe(lengths, "response_words")
    return results


# ------------------------------------------------------------------ #
#  4. Readability                                                      #
# ------------------------------------------------------------------ #


def _readability(source_texts: dict[str, list[str]]) -> dict[str, Any]:
    """Flesch-Kincaid & Gunning Fog per response, aggregated per source."""
    results: dict[str, Any] = {}
    for src, texts in source_texts.items():
        fk_scores: list[float] = []
        gf_scores: list[float] = []
        for text in texts:
            # textstat needs at least a sentence; skip near-empty
            if len(text.split()) < 5:
                continue
            try:
                fk = textstat.flesch_kincaid_grade(text)
                gf = textstat.gunning_fog(text)
                fk_scores.append(fk)
                gf_scores.append(gf)
            except Exception:
                continue
        results[src] = {
            "flesch_kincaid": _stat_summary(fk_scores),
            "gunning_fog": _stat_summary(gf_scores),
        }
    return results


# ------------------------------------------------------------------ #
#  5. Formatting habits                                                #
# ------------------------------------------------------------------ #

# Patterns for formatting markers
_FMT_PATTERNS: dict[str, re.Pattern] = {
    "headers": re.compile(r"^#{1,6}\s", re.MULTILINE),
    "bullet_points": re.compile(r"^[\-\*]\s", re.MULTILINE),
    "numbered_lists": re.compile(r"^\d+\.\s", re.MULTILINE),
    "code_blocks": re.compile(r"```"),
    "bold": re.compile(r"\*\*[^*]+\*\*"),
    "italic": re.compile(r"(?<!\*)\*(?!\*)[^*]+\*(?!\*)"),
}


def _formatting_habits(source_texts: dict[str, list[str]]) -> dict[str, Any]:
    """Count markdown formatting elements, normalised per 1 000 words."""
    results: dict[str, Any] = {}
    for src, texts in source_texts.items():
        total_words = sum(len(t.split()) for t in texts)
        raw_counts: dict[str, int] = {k: 0 for k in _FMT_PATTERNS}
        for text in texts:
            for name, pat in _FMT_PATTERNS.items():
                raw_counts[name] += len(pat.findall(text))

        # Code-block markers come in pairs; count pairs as instances.
        raw_counts["code_blocks"] = raw_counts["code_blocks"] // 2

        norm = 1000 / total_words if total_words else 0
        results[src] = {
            "raw_counts": {k: v for k, v in raw_counts.items()},
            "per_1k_words": {
                k: round(v * norm, 2) for k, v in raw_counts.items()
            },
            "total_words": total_words,
        }
    return results


# ------------------------------------------------------------------ #
#  Shared helpers                                                      #
# ------------------------------------------------------------------ #


def _alpha_tokens(
    docs: list[spacy.tokens.Doc],
    stopwords: set[str],
) -> list[str]:
    """Extract lower-cased alphabetic non-stopword tokens from docs."""
    tokens: list[str] = []
    for doc in docs:
        for tok in doc:
            lower = tok.text.lower()
            if tok.is_alpha and lower not in stopwords and len(lower) > 1:
                tokens.append(lower)
    return tokens


def _describe(values: list[int | float], name: str) -> dict[str, Any]:
    """Return mean / median / std / count plus the raw distribution."""
    if not values:
        return {"mean": 0, "median": 0, "std": 0, "count": 0, "distribution": []}
    n = len(values)
    mean = sum(values) / n
    sorted_v = sorted(values)
    median = (
        sorted_v[n // 2]
        if n % 2
        else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
    )
    variance = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(variance)
    return {
        "mean": round(mean, 2),
        "median": round(median, 2),
        "std": round(std, 2),
        "count": n,
        "distribution": values,
    }


def _stat_summary(values: list[float]) -> dict[str, float]:
    """Compact mean/std summary for readability scores."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    return {
        "mean": round(mean, 2),
        "std": round(math.sqrt(var), 2),
        "count": n,
    }

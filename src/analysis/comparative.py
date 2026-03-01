"""Cross-platform comparison analysis.

Aggregates results from lexical, semantic, and pragmatic modules to build
a style fingerprint, compare user language across platforms, perform
topic-controlled comparisons, and produce a summary comparison table.
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
from sklearn.feature_extraction.text import TfidfVectorizer

from src.parsers.base import Conversation

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Public entry point                                                  #
# ------------------------------------------------------------------ #


def run(
    conversations: list[Conversation],
    config: dict,
    prior_results: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Run all comparative analyses.

    Args:
        conversations: Normalised conversation objects (all sources).
        config: Loaded ``config.yaml`` dict.
        prior_results: Dict mapping module name -> results dict from
            lexical, semantic, and pragmatic modules.  If *None*, they
            are loaded from ``data/outputs/``.

    Returns:
        Nested dict with keys: ``style_fingerprint``, ``user_language``,
        ``topic_controlled``, ``summary_table``, ``_meta``.
    """
    outputs_root = Path(config.get("paths", {}).get("outputs", "data/outputs"))

    # ---- load prior results ---- #
    if prior_results is None:
        prior_results = _load_prior_results(outputs_root)

    lexical = prior_results.get("lexical", {})
    semantic = prior_results.get("semantic", {})
    pragmatic = prior_results.get("pragmatic", {})

    sources = sorted(
        set(lexical.get("_meta", {}).get("sources", []))
        | set(pragmatic.get("_meta", {}).get("sources", []))
    )
    logger.info("comparative: sources=%s", sources)

    # ---- run sub-analyses ---- #
    results: dict[str, Any] = {}

    results["style_fingerprint"] = _style_fingerprint(
        lexical, semantic, pragmatic, sources,
    )

    results["user_language"] = _user_language(
        conversations, config, sources,
    )

    results["topic_controlled"] = _topic_controlled(
        conversations, semantic, config, sources,
    )

    results["summary_table"] = _summary_table(
        lexical, semantic, pragmatic, results, sources,
    )

    results["_meta"] = {"sources": sources}
    return results


# ------------------------------------------------------------------ #
#  Load prior module results                                           #
# ------------------------------------------------------------------ #


_REQUIRED_MODULES = ["lexical", "semantic", "pragmatic"]


def _load_prior_results(outputs_root: Path) -> dict[str, dict]:
    """Load results JSON files from prior analysis modules."""
    results: dict[str, dict] = {}
    missing: list[str] = []

    for mod_name in _REQUIRED_MODULES:
        path = outputs_root / f"{mod_name}_results.json"
        if not path.exists():
            missing.append(mod_name)
            continue
        with open(path, encoding="utf-8") as f:
            results[mod_name] = json.load(f)
        logger.info("comparative: loaded %s", path)

    if missing:
        raise FileNotFoundError(
            f"Missing prior results: {missing}. "
            f"Run these first:  "
            + "  ".join(f"python main.py analyze --module {m}" for m in missing)
        )

    return results


# ------------------------------------------------------------------ #
#  1. Style fingerprint                                                #
# ------------------------------------------------------------------ #


_FINGERPRINT_METRICS = [
    "vocabulary_mattr",
    "mean_response_length",
    "readability_fk",
    "hedge_density",
    "question_rate",
    "verbosity_median",
    "first_person_I",
    "formatting_density",
    "self_similarity",
]


def _style_fingerprint(
    lexical: dict,
    semantic: dict,
    pragmatic: dict,
    sources: list[str],
) -> dict[str, Any]:
    """Build a normalised style vector for each source."""
    raw: dict[str, list[float]] = {s: [] for s in sources}

    for s in sources:
        # 1. Vocabulary richness (MATTR)
        raw[s].append(
            lexical.get("vocabulary", {}).get(s, {}).get("mattr_500", 0.0)
        )
        # 2. Mean response length (words)
        raw[s].append(
            lexical.get("response_length", {}).get(s, {}).get("mean", 0.0)
        )
        # 3. Readability (Flesch-Kincaid mean)
        raw[s].append(
            lexical.get("readability", {}).get(s, {})
            .get("flesch_kincaid", {}).get("mean", 0.0)
        )
        # 4. Hedge density (total hedges per 1k words)
        raw[s].append(
            pragmatic.get("hedging", {}).get(s, {}).get("hedge_density", 0.0)
        )
        # 5. Question rate (questions per turn)
        raw[s].append(
            pragmatic.get("question_rate", {}).get(s, {})
            .get("mean_questions_per_turn", 0.0)
        )
        # 6. Verbosity ratio (median)
        raw[s].append(
            pragmatic.get("verbosity_ratio", {}).get(s, {}).get("median", 0.0)
        )
        # 7. First-person "I" usage (per 1k words)
        raw[s].append(
            pragmatic.get("first_person", {}).get(s, {})
            .get("per_1k_words", {}).get("I", 0.0)
        )
        # 8. Formatting density (sum of all per-1k elements)
        fmt = lexical.get("formatting", {}).get(s, {}).get("per_1k_words", {})
        raw[s].append(sum(fmt.values()))
        # 9. Self-similarity (mean cosine)
        raw[s].append(
            semantic.get("self_similarity", {}).get(s, {}).get("mean", 0.0)
        )

    # Min-max normalise each metric to [0, 1] across sources.
    n_metrics = len(_FINGERPRINT_METRICS)
    normalised: dict[str, list[float]] = {s: [0.0] * n_metrics for s in sources}

    for i in range(n_metrics):
        values = [raw[s][i] for s in sources]
        mn, mx = min(values), max(values)
        span = mx - mn
        for s in sources:
            if span > 0:
                normalised[s][i] = round((raw[s][i] - mn) / span, 4)
            else:
                normalised[s][i] = 0.5  # All same -> midpoint

    # Round raw values
    for s in sources:
        raw[s] = [round(v, 4) for v in raw[s]]

    return {
        "metrics": _FINGERPRINT_METRICS,
        "raw": raw,
        "normalized": normalised,
    }


# ------------------------------------------------------------------ #
#  2. User language comparison                                         #
# ------------------------------------------------------------------ #


def _user_language(
    conversations: list[Conversation],
    config: dict,
    sources: list[str],
) -> dict[str, Any]:
    """Analyse the USER's own language grouped by platform."""
    min_len = config.get("analysis", {}).get("min_turn_length", 10)

    # Collect user turns per source.
    user_texts: dict[str, list[str]] = {}
    for conv in conversations:
        for turn in conv.turns:
            if turn.role == "user" and len(turn.content) >= min_len:
                user_texts.setdefault(conv.source, []).append(turn.content)

    results: dict[str, Any] = {}

    # ---- Message length ---- #
    msg_length: dict[str, Any] = {}
    for s in sources:
        texts = user_texts.get(s, [])
        lengths = [len(t.split()) for t in texts]
        msg_length[s] = _describe_dist(lengths)
    results["message_length"] = msg_length

    # ---- Vocabulary richness (MATTR window=200) ---- #
    vocab: dict[str, Any] = {}
    nlp = _load_spacy()
    stopwords = nlp.Defaults.stop_words | {
        "the", "is", "and", "of", "to", "a", "in", "for", "it", "that",
        "this", "with", "on", "as", "be", "at", "by", "an", "or", "not",
        "are", "was", "but", "from", "have", "has", "had", "do", "does",
        "did", "will", "would", "can", "could", "should", "may", "might",
        "i", "you", "we", "they", "he", "she", "my", "your", "our",
        "their", "its", "me", "us", "him", "her", "them",
    }

    for s in sources:
        texts = user_texts.get(s, [])
        all_text = " ".join(texts)
        doc = nlp(all_text[:500_000])  # Cap for speed
        tokens = [
            tok.text.lower() for tok in doc
            if tok.is_alpha and tok.text.lower() not in stopwords
            and len(tok.text) > 1
        ]
        n = len(tokens)
        types = set(tokens)
        ttr = len(types) / n if n else 0.0
        mattr = _mattr(tokens, window=200)
        vocab[s] = {
            "total_tokens": n,
            "unique_types": len(types),
            "ttr": round(ttr, 4),
            "mattr_200": round(mattr, 4),
        }
    results["vocabulary"] = vocab

    # ---- Top 20 words per source ---- #
    word_freq: dict[str, list] = {}
    for s in sources:
        texts = user_texts.get(s, [])
        all_text = " ".join(texts)
        doc = nlp(all_text[:500_000])
        tokens = [
            tok.text.lower() for tok in doc
            if tok.is_alpha and tok.text.lower() not in stopwords
            and len(tok.text) > 1
        ]
        counter = Counter(tokens)
        word_freq[s] = counter.most_common(20)
    results["word_frequency"] = word_freq

    # ---- Distinctive words (TF-IDF) ---- #
    if len(sources) >= 2:
        corpus = [" ".join(user_texts.get(s, [])) for s in sources]
        vec = TfidfVectorizer(
            max_features=10_000,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
            sublinear_tf=True,
        )
        tfidf_matrix = vec.fit_transform(corpus)
        feature_names = vec.get_feature_names_out()

        distinctive: dict[str, list] = {}
        for idx, s in enumerate(sources):
            row = tfidf_matrix[idx].toarray().flatten()
            top_indices = row.argsort()[::-1][:20]
            distinctive[s] = [
                {"word": feature_names[i], "tfidf": round(float(row[i]), 4)}
                for i in top_indices
                if row[i] > 0
            ]
        results["distinctive_words"] = distinctive
    else:
        results["distinctive_words"] = {}

    # ---- Question rate ---- #
    q_rate: dict[str, Any] = {}
    for s in sources:
        texts = user_texts.get(s, [])
        n_turns = len(texts)
        total_q = sum(t.count("?") for t in texts)
        turns_with_q = sum(1 for t in texts if "?" in t)
        q_rate[s] = {
            "mean_questions_per_turn": round(total_q / max(n_turns, 1), 3),
            "fraction_turns_with_question": round(
                turns_with_q / max(n_turns, 1), 3
            ),
            "total_questions": total_q,
            "n_turns": n_turns,
        }
    results["question_rate"] = q_rate

    # ---- Sentence length ---- #
    sent_len: dict[str, Any] = {}
    for s in sources:
        texts = user_texts.get(s, [])
        all_text = " ".join(texts)
        doc = nlp(all_text[:500_000])
        lengths = [
            sum(1 for t in sent if not t.is_punct and not t.is_space)
            for sent in doc.sents
        ]
        lengths = [ln for ln in lengths if ln > 0]
        sent_len[s] = _describe_dist(lengths)
    results["sentence_length"] = sent_len

    # ---- Politeness markers ---- #
    politeness_phrases = [
        "please", "thanks", "thank you", "sorry",
        "could you", "would you",
    ]
    politeness: dict[str, Any] = {}
    for s in sources:
        texts = user_texts.get(s, [])
        total_words = sum(len(t.split()) for t in texts)
        counts: dict[str, int] = {}
        for phrase in politeness_phrases:
            count = sum(t.lower().count(phrase) for t in texts)
            counts[phrase] = count
        total_polite = sum(counts.values())
        norm = 1000 / total_words if total_words else 0
        politeness[s] = {
            "raw_counts": counts,
            "per_1k_words": {k: round(v * norm, 3) for k, v in counts.items()},
            "total_politeness_markers": total_polite,
            "politeness_per_1k": round(total_polite * norm, 3),
            "total_words": total_words,
        }
    results["politeness"] = politeness

    return results


# ------------------------------------------------------------------ #
#  3. Topic-controlled comparison                                      #
# ------------------------------------------------------------------ #


def _topic_controlled(
    conversations: list[Conversation],
    semantic: dict,
    config: dict,
    sources: list[str],
) -> dict[str, Any]:
    """Re-compute metrics on overlapping-topic turns only.

    Identifies the largest shared assistant topic (where both sources
    have >= 10% representation) and re-runs key metrics on those turns
    to approximate an apples-to-apples comparison.
    """
    min_len = config.get("analysis", {}).get("min_turn_length", 10)

    # Find best shared topic from semantic results.
    topics = semantic.get("assistant_topics", {}).get("topics", [])
    best_topic = None
    best_score = 0.0

    for t in topics:
        breakdown = t.get("source_breakdown", {})
        shares = [breakdown.get(s, 0) for s in sources]
        if all(sh >= 0.10 for sh in shares):
            balance = min(shares) / max(max(shares), 0.001)
            count = t.get("count", 0)
            score = count * balance
            if score > best_score:
                best_topic = t
                best_score = score

    if best_topic is None:
        logger.warning(
            "comparative: no shared topic with >=10%% from each source"
        )
        return {
            "topic_name": "none",
            "topic_id": -1,
            "n_turns_per_source": {},
            "metrics": {},
            "note": "No shared topic with >=10% from each source",
        }

    topic_id = best_topic["topic_id"]
    topic_words = [w["word"] for w in best_topic.get("words", [])[:5]]
    topic_name = ", ".join(topic_words) if topic_words else f"topic_{topic_id}"
    breakdown = best_topic.get("source_breakdown", {})

    logger.info(
        "comparative: shared topic #%d (%s) count=%d breakdown=%s",
        topic_id, topic_name, best_topic.get("count", 0), breakdown,
    )

    # Collect assistant turns and user-assistant pairs per source.
    asst_by_source: dict[str, list[str]] = {}
    conv_pairs: dict[str, list[tuple[str, str]]] = {}

    for conv in conversations:
        turns = conv.turns
        for turn in turns:
            if turn.role == "assistant" and len(turn.content) >= min_len:
                asst_by_source.setdefault(conv.source, []).append(turn.content)
        for i in range(len(turns) - 1):
            if turns[i].role == "user" and turns[i + 1].role == "assistant":
                if (len(turns[i + 1].content) >= min_len
                        and len(turns[i].content.split()) >= 5):
                    conv_pairs.setdefault(conv.source, []).append(
                        (turns[i].content, turns[i + 1].content)
                    )

    # For topic 0 (the broad catch-all), use balanced random sampling.
    # For specific topics, use keyword-based filtering.
    rng = np.random.RandomState(42)

    if topic_id == 0:
        min_count = min(len(asst_by_source.get(s, [])) for s in sources)
        sample_n = min(min_count, 500)

        sampled: dict[str, list[str]] = {}
        sampled_pairs: dict[str, list[tuple[str, str]]] = {}
        for s in sources:
            indices = rng.choice(
                len(asst_by_source.get(s, [])),
                size=sample_n,
                replace=False,
            )
            sampled[s] = [asst_by_source[s][i] for i in indices]
            pair_n = min(sample_n, len(conv_pairs.get(s, [])))
            pair_idx = rng.choice(
                len(conv_pairs.get(s, [])),
                size=pair_n,
                replace=False,
            )
            sampled_pairs[s] = [conv_pairs[s][i] for i in pair_idx]

        note = (
            f"Topic 0 is the broad catch-all "
            f"({best_topic.get('count', 0)} turns). "
            f"Used balanced random sample of {sample_n} turns per source."
        )
    else:
        filter_words = set(w.lower() for w in topic_words[:5] if len(w) > 2)
        sampled = {}
        sampled_pairs = {}
        for s in sources:
            sampled[s] = [
                t for t in asst_by_source.get(s, [])
                if any(w in t.lower() for w in filter_words)
            ]
            sampled_pairs[s] = [
                (u, a) for u, a in conv_pairs.get(s, [])
                if any(w in a.lower() for w in filter_words)
            ]
        note = f"Filtered by topic keywords: {filter_words}"

    # ---- Compute metrics on sampled turns ---- #
    metrics: dict[str, Any] = {}

    # Mean response length
    resp_len: dict[str, float] = {}
    for s in sources:
        if sampled[s]:
            resp_len[s] = round(
                sum(len(t.split()) for t in sampled[s]) / len(sampled[s]), 2
            )
        else:
            resp_len[s] = 0.0
    metrics["mean_response_length"] = resp_len

    # Hedge density
    from src.analysis.pragmatic import HEDGE_PHRASES, _count_phrase

    hedge: dict[str, float] = {}
    for s in sources:
        total_words = sum(len(t.split()) for t in sampled[s])
        total_hedges = 0
        for cat_phrases in HEDGE_PHRASES.values():
            for phrase in cat_phrases:
                total_hedges += sum(
                    _count_phrase(t.lower(), phrase) for t in sampled[s]
                )
        hedge[s] = round(
            total_hedges * 1000 / total_words if total_words else 0, 3
        )
    metrics["hedge_density_per_1k"] = hedge

    # Verbosity ratio
    verb: dict[str, Any] = {}
    for s in sources:
        ratios = []
        for u, a in sampled_pairs.get(s, []):
            u_words = len(u.split())
            a_words = len(a.split())
            if u_words >= 5:
                ratios.append(a_words / u_words)
        if ratios:
            sorted_r = sorted(ratios)
            verb[s] = {
                "mean": round(sum(ratios) / len(ratios), 2),
                "median": round(sorted_r[len(sorted_r) // 2], 2),
                "count": len(ratios),
            }
        else:
            verb[s] = {"mean": 0, "median": 0, "count": 0}
    metrics["verbosity_ratio"] = verb

    # Self-similarity on sampled turns (via cached embeddings)
    processed_root = Path(
        config.get("paths", {}).get("processed_data", "data/processed")
    )
    emb_path = processed_root / "embeddings_assistant.npy"
    if emb_path.exists():
        try:
            all_asst_texts: list[str] = []
            all_asst_sources: list[str] = []
            for conv in conversations:
                for turn in conv.turns:
                    if turn.role == "assistant" and len(turn.content) >= min_len:
                        all_asst_texts.append(turn.content)
                        all_asst_sources.append(conv.source)

            embeddings = np.load(emb_path)
            if embeddings.shape[0] == len(all_asst_texts):
                sim_results: dict[str, float] = {}
                for s in sources:
                    src_indices = [
                        i for i, src in enumerate(all_asst_sources) if src == s
                    ]
                    if len(src_indices) < 10:
                        sim_results[s] = 0.0
                        continue
                    sample_idx = rng.choice(
                        src_indices,
                        size=min(len(src_indices), 200),
                        replace=False,
                    )
                    src_emb = embeddings[sample_idx]
                    norms = np.linalg.norm(src_emb, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1, norms)
                    normed = src_emb / norms
                    sim_matrix = normed @ normed.T
                    n = len(sample_idx)
                    triu = np.triu_indices(n, k=1)
                    sims = sim_matrix[triu]
                    sim_results[s] = round(float(np.mean(sims)), 4)
                metrics["self_similarity"] = sim_results
            else:
                metrics["self_similarity"] = {
                    "note": "embedding count mismatch"
                }
        except Exception as e:
            logger.warning("comparative: self-similarity failed: %s", e)
            metrics["self_similarity"] = {"note": str(e)}
    else:
        metrics["self_similarity"] = {"note": "no cached embeddings"}

    n_per_source = {s: len(sampled.get(s, [])) for s in sources}

    return {
        "topic_name": topic_name,
        "topic_id": topic_id,
        "topic_count": best_topic.get("count", 0),
        "source_breakdown": breakdown,
        "n_turns_per_source": n_per_source,
        "metrics": metrics,
        "note": note,
    }


# ------------------------------------------------------------------ #
#  4. Summary comparison table                                         #
# ------------------------------------------------------------------ #


def _summary_table(
    lexical: dict,
    semantic: dict,
    pragmatic: dict,
    comparative_results: dict,
    sources: list[str],
) -> list[dict[str, Any]]:
    """Build a comprehensive comparison table with ~25 key metrics."""
    rows: list[dict[str, Any]] = []

    def _add(
        metric: str,
        vals: dict[str, float | int | str],
        note_fn=None,
    ):
        row: dict[str, Any] = {"metric": metric}
        for s in sources:
            row[s] = vals.get(s, "?")
        if len(sources) == 2:
            v0 = vals.get(sources[0])
            v1 = vals.get(sources[1])
            if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
                row["delta"] = round(v0 - v1, 4)
            else:
                row["delta"] = "n/a"
        if note_fn:
            try:
                row["note"] = note_fn(vals, sources)
            except Exception:
                row["note"] = ""
        else:
            row["note"] = ""
        rows.append(row)

    def _higher_wins(label_high, label_low):
        def fn(vals, srcs):
            v = [(vals.get(s, 0), s) for s in srcs]
            v.sort(reverse=True)
            return f"{v[0][1].capitalize()}: {label_high}"
        return fn

    def _lower_wins(label_low, label_high):
        def fn(vals, srcs):
            v = [(vals.get(s, 0), s) for s in srcs]
            v.sort()
            return f"{v[0][1].capitalize()}: {label_low}"
        return fn

    # ---- Lexical ---- #
    _add(
        "Vocabulary (MATTR-500)",
        {s: lexical.get("vocabulary", {}).get(s, {}).get("mattr_500", 0)
         for s in sources},
        _higher_wins("Richer vocabulary", "Less varied"),
    )
    _add(
        "Unique word types",
        {s: lexical.get("vocabulary", {}).get(s, {}).get("unique_types", 0)
         for s in sources},
    )
    _add(
        "Mean response length (words)",
        {s: lexical.get("response_length", {}).get(s, {}).get("mean", 0)
         for s in sources},
        _lower_wins("More concise", "More verbose"),
    )
    _add(
        "Median response length (words)",
        {s: lexical.get("response_length", {}).get(s, {}).get("median", 0)
         for s in sources},
    )
    _add(
        "Mean sentence length (words)",
        {s: lexical.get("sentence_stats", {}).get(s, {}).get("mean", 0)
         for s in sources},
    )
    _add(
        "Flesch-Kincaid grade",
        {s: lexical.get("readability", {}).get(s, {})
         .get("flesch_kincaid", {}).get("mean", 0)
         for s in sources},
        lambda v, srcs: (
            "Similar"
            if len(srcs) >= 2
            and abs(v.get(srcs[0], 0) - v.get(srcs[1], 0)) < 1
            else (
                f"{min(srcs, key=lambda s: v.get(s, 0)).capitalize()}: "
                "More readable"
            )
        ),
    )
    _add(
        "Gunning Fog index",
        {s: lexical.get("readability", {}).get(s, {})
         .get("gunning_fog", {}).get("mean", 0)
         for s in sources},
    )
    _add(
        "Formatting density (per 1k)",
        {s: round(sum(
            lexical.get("formatting", {}).get(s, {})
            .get("per_1k_words", {}).values()
        ), 2) for s in sources},
        _higher_wins("Heavy formatter", "Minimal formatting"),
    )
    _add(
        "Bold usage (per 1k)",
        {s: lexical.get("formatting", {}).get(s, {})
         .get("per_1k_words", {}).get("bold", 0)
         for s in sources},
    )
    _add(
        "Code blocks (per 1k)",
        {s: lexical.get("formatting", {}).get(s, {})
         .get("per_1k_words", {}).get("code_blocks", 0)
         for s in sources},
    )

    # ---- Pragmatic ---- #
    _add(
        "Hedge density (per 1k)",
        {s: pragmatic.get("hedging", {}).get(s, {}).get("hedge_density", 0)
         for s in sources},
        _higher_wins("Hedges more", "More direct"),
    )
    _add(
        "Uncertainty hedges (per 1k)",
        {s: pragmatic.get("hedging", {}).get(s, {})
         .get("category_per_1k", {}).get("uncertainty", 0)
         for s in sources},
        _higher_wins("More personally uncertain", "More confident"),
    )
    _add(
        "Questions per turn",
        {s: pragmatic.get("question_rate", {}).get(s, {})
         .get("mean_questions_per_turn", 0)
         for s in sources},
        _higher_wins("More interactive", "Less probing"),
    )
    _add(
        "% turns with question",
        {s: round(
            pragmatic.get("question_rate", {}).get(s, {})
            .get("fraction_turns_with_question", 0) * 100, 1
        ) for s in sources},
    )
    _add(
        "Disclaimer density (per 1k)",
        {s: pragmatic.get("disclaimers", {}).get(s, {})
         .get("disclaimer_density", 0)
         for s in sources},
    )
    _add(
        "Verbosity ratio (median)",
        {s: pragmatic.get("verbosity_ratio", {}).get(s, {}).get("median", 0)
         for s in sources},
        _lower_wins("More proportional", "More expansive"),
    )
    _add(
        "Verbosity ratio (mean)",
        {s: pragmatic.get("verbosity_ratio", {}).get(s, {}).get("mean", 0)
         for s in sources},
    )
    _add(
        "First-person 'I' (per 1k)",
        {s: pragmatic.get("first_person", {}).get(s, {})
         .get("per_1k_words", {}).get("I", 0)
         for s in sources},
        _higher_wins("More personal", "More impersonal"),
    )
    _add(
        "'you' usage (per 1k)",
        {s: pragmatic.get("first_person", {}).get(s, {})
         .get("per_1k_words", {}).get("you", 0)
         for s in sources},
    )
    _add(
        "Turns per conv (median)",
        {s: pragmatic.get("turn_dynamics", {}).get(s, {})
         .get("turns_per_conv", {}).get("median", 0)
         for s in sources},
    )
    _add(
        "Single-exchange fraction",
        {s: pragmatic.get("turn_dynamics", {}).get(s, {})
         .get("single_exchange_fraction", 0)
         for s in sources},
    )

    # ---- Semantic ---- #
    _add(
        "Self-similarity (cosine)",
        {s: semantic.get("self_similarity", {}).get(s, {}).get("mean", 0)
         for s in sources},
        _lower_wins("More varied responses", "More formulaic"),
    )
    _add(
        "Mean sentiment polarity",
        {s: semantic.get("sentiment", {}).get(s, {}).get("mean_polarity", 0)
         for s in sources},
    )

    # ---- Topic-controlled ---- #
    tc = comparative_results.get("topic_controlled", {})
    tc_metrics = tc.get("metrics", {})
    tc_topic = tc.get("topic_name", "?")

    if tc_metrics.get("mean_response_length"):
        _add(
            f"[Topic-ctrl] Response length",
            tc_metrics.get("mean_response_length", {}),
            note_fn=lambda v, s: f"Shared topic: {tc_topic}",
        )
    if tc_metrics.get("hedge_density_per_1k"):
        _add(
            f"[Topic-ctrl] Hedge density",
            tc_metrics.get("hedge_density_per_1k", {}),
        )
    if isinstance(tc_metrics.get("verbosity_ratio"), dict):
        vr = tc_metrics["verbosity_ratio"]
        # Only add if it has actual source data (not just notes)
        vr_vals = {}
        for s in sources:
            if isinstance(vr.get(s), dict):
                vr_vals[s] = vr[s].get("median", 0)
        if vr_vals:
            _add(
                f"[Topic-ctrl] Verbosity median",
                vr_vals,
            )

    return rows


# ------------------------------------------------------------------ #
#  Shared helpers                                                      #
# ------------------------------------------------------------------ #


_NLP_CACHE = None


def _load_spacy():
    """Load and cache spaCy model."""
    global _NLP_CACHE
    if _NLP_CACHE is None:
        import spacy
        _NLP_CACHE = spacy.load(
            "en_core_web_sm",
            disable=["ner", "lemmatizer"],
        )
        _NLP_CACHE.max_length = 2_000_000
    return _NLP_CACHE


def _mattr(tokens: list[str], window: int = 200) -> float:
    """Moving-Average Type-Token Ratio."""
    n = len(tokens)
    if n < window:
        return len(set(tokens)) / n if n else 0.0
    ttrs: list[float] = []
    for i in range(n - window + 1):
        segment = tokens[i: i + window]
        ttrs.append(len(set(segment)) / window)
    return sum(ttrs) / len(ttrs)


def _describe_dist(values: list[int | float]) -> dict[str, Any]:
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

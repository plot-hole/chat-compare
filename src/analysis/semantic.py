"""Embeddings, topic modeling, similarity, and sentiment analysis.

Uses BERTopic for topic discovery, sentence-transformers for embeddings,
and a lexicon-based approach for sentiment classification.
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.parsers.base import Conversation

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Public entry point                                                  #
# ------------------------------------------------------------------ #


def run(conversations: list[Conversation], config: dict) -> dict[str, Any]:
    """Run all semantic analyses on *conversations*.

    Args:
        conversations: Normalised conversation objects (all sources).
        config: Loaded ``config.yaml`` dict.

    Returns:
        Nested dict with keys: ``assistant_topics``, ``user_topics``,
        ``self_similarity``, ``sentiment``, ``_meta``.
    """
    min_len = config.get("analysis", {}).get("min_turn_length", 10)
    min_topic_size = config.get("analysis", {}).get("topic_model_min_topic_size", 5)
    embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
    processed_root = Path(config.get("paths", {}).get("processed_data", "data/processed"))

    # ---- collect turns by role, tracking source ---- #
    asst_texts: list[str] = []
    asst_sources: list[str] = []
    user_texts: list[str] = []
    user_sources: list[str] = []

    for conv in conversations:
        for turn in conv.turns:
            if len(turn.content) < min_len:
                continue
            if turn.role == "assistant":
                asst_texts.append(turn.content)
                asst_sources.append(conv.source)
            elif turn.role == "user":
                user_texts.append(turn.content)
                user_sources.append(conv.source)

    sources = sorted(set(asst_sources))
    logger.info(
        "semantic: %d assistant turns, %d user turns, sources=%s",
        len(asst_texts), len(user_texts), sources,
    )

    # ---- compute / load cached embeddings ---- #
    asst_embeddings = _get_embeddings(
        asst_texts, embedding_model_name, processed_root, "assistant",
    )
    user_embeddings = _get_embeddings(
        user_texts, embedding_model_name, processed_root, "user",
    )

    # ---- analyses ---- #
    results: dict[str, Any] = {}

    logger.info("semantic: running assistant topic model ...")
    results["assistant_topics"] = _run_topic_model(
        asst_texts, asst_sources, asst_embeddings,
        min_topic_size=min_topic_size, top_n_topics=20, label="assistant",
    )

    logger.info("semantic: running user topic model ...")
    results["user_topics"] = _run_topic_model(
        user_texts, user_sources, user_embeddings,
        min_topic_size=min_topic_size, top_n_topics=15, label="user",
    )

    logger.info("semantic: computing self-similarity ...")
    results["self_similarity"] = _self_similarity(
        asst_embeddings, asst_sources, n_pairs=500,
    )

    logger.info("semantic: computing sentiment ...")
    results["sentiment"] = _sentiment_analysis(asst_texts, asst_sources)

    results["_meta"] = {"sources": sources}
    return results


# ------------------------------------------------------------------ #
#  Embeddings (with caching)                                           #
# ------------------------------------------------------------------ #


def _get_embeddings(
    texts: list[str],
    model_name: str,
    cache_dir: Path,
    role_label: str,
) -> np.ndarray:
    """Compute or load cached sentence-transformer embeddings."""
    cache_path = cache_dir / f"embeddings_{role_label}.npy"

    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            logger.info("semantic: loaded cached embeddings from %s", cache_path)
            return cached
        logger.info(
            "semantic: cached embeddings stale (%d vs %d texts), recomputing",
            cached.shape[0], len(texts),
        )

    from sentence_transformers import SentenceTransformer

    logger.info(
        "semantic: encoding %d %s turns with %s ...",
        len(texts), role_label, model_name,
    )
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    logger.info("semantic: cached embeddings to %s", cache_path)
    return embeddings


# ------------------------------------------------------------------ #
#  1 & 2. Topic modeling                                               #
# ------------------------------------------------------------------ #


def _run_topic_model(
    texts: list[str],
    sources: list[str],
    embeddings: np.ndarray,
    *,
    min_topic_size: int = 5,
    top_n_topics: int = 20,
    label: str = "assistant",
) -> dict[str, Any]:
    """Fit BERTopic and return topic metadata with per-source breakdowns."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    # Filter out very short texts (< 20 words) to improve topic quality
    min_words = 20
    mask = [len(t.split()) >= min_words for t in texts]
    filtered_texts = [t for t, m in zip(texts, mask) if m]
    filtered_sources = [s for s, m in zip(sources, mask) if m]
    filtered_embeddings = embeddings[mask]

    logger.info(
        "semantic [%s]: %d/%d texts pass min-word filter (%d words)",
        label, len(filtered_texts), len(texts), min_words,
    )

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        prediction_data=True,
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=min_topic_size,
        nr_topics="auto",
        verbose=False,
    )

    topics, _probs = topic_model.fit_transform(filtered_texts, filtered_embeddings)

    # ---- extract topic info ---- #
    topic_info = topic_model.get_topic_info()
    # Drop the outlier topic (-1)
    topic_info = topic_info[topic_info["Topic"] != -1]

    all_sources_set = sorted(set(filtered_sources))

    # Build per-topic metadata
    topic_results: list[dict[str, Any]] = []
    for _, row in topic_info.head(top_n_topics).iterrows():
        tid = int(row["Topic"])
        count = int(row["Count"])

        # Get representative words
        topic_words_raw = topic_model.get_topic(tid)
        top_words = [
            {"word": w, "score": round(float(s), 4)}
            for w, s in (topic_words_raw[:10] if topic_words_raw else [])
        ]

        # Source breakdown for this topic
        topic_mask = [t == tid for t in topics]
        topic_sources = [s for s, m in zip(filtered_sources, topic_mask) if m]
        source_counter = Counter(topic_sources)
        source_breakdown = {
            src: round(source_counter.get(src, 0) / max(count, 1), 3)
            for src in all_sources_set
        }

        topic_results.append({
            "topic_id": tid,
            "count": count,
            "words": top_words,
            "source_breakdown": source_breakdown,
        })

    # ---- per-source topic distributions ---- #
    source_distributions: dict[str, dict[str, float]] = {}
    for src in all_sources_set:
        src_topics = [t for t, s in zip(topics, filtered_sources) if s == src]
        n_src = len(src_topics)
        if n_src == 0:
            continue
        counter = Counter(src_topics)
        dist: dict[str, float] = {}
        for tid in sorted(counter.keys()):
            if tid == -1:
                dist["outlier"] = round(counter[tid] / n_src, 3)
            else:
                dist[str(tid)] = round(counter[tid] / n_src, 3)
        source_distributions[src] = dist

    # Summary stats
    n_outliers = sum(1 for t in topics if t == -1)
    n_topics_found = len(set(topics) - {-1})

    return {
        "topics": topic_results,
        "source_distributions": source_distributions,
        "n_topics_found": n_topics_found,
        "n_outliers": n_outliers,
        "n_documents": len(filtered_texts),
    }


# ------------------------------------------------------------------ #
#  3. Self-similarity                                                  #
# ------------------------------------------------------------------ #


def _self_similarity(
    embeddings: np.ndarray,
    sources: list[str],
    n_pairs: int = 500,
) -> dict[str, Any]:
    """Compute mean pairwise cosine similarity within each source."""
    results: dict[str, Any] = {}
    all_sources = sorted(set(sources))

    for src in all_sources:
        indices = [i for i, s in enumerate(sources) if s == src]
        n = len(indices)

        if n < 2:
            results[src] = {"mean": 0.0, "std": 0.0, "n_pairs": 0}
            continue

        src_embeddings = embeddings[indices]

        # Normalise for cosine similarity via dot product
        norms = np.linalg.norm(src_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = src_embeddings / norms

        # Sample random pairs
        max_pairs = n * (n - 1) // 2
        actual_pairs = min(n_pairs, max_pairs)

        rng = random.Random(42)
        if max_pairs <= n_pairs:
            # Compute all pairs
            sim_matrix = normed @ normed.T
            # Extract upper triangle (excluding diagonal)
            triu_indices = np.triu_indices(n, k=1)
            sims = sim_matrix[triu_indices].tolist()
        else:
            sims: list[float] = []
            seen: set[tuple[int, int]] = set()
            while len(sims) < actual_pairs:
                i, j = rng.sample(range(n), 2)
                pair = (min(i, j), max(i, j))
                if pair in seen:
                    continue
                seen.add(pair)
                sim = float(np.dot(normed[i], normed[j]))
                sims.append(sim)

        mean_sim = float(np.mean(sims))
        std_sim = float(np.std(sims))

        results[src] = {
            "mean": round(mean_sim, 4),
            "std": round(std_sim, 4),
            "n_pairs": len(sims),
        }

    return results


# ------------------------------------------------------------------ #
#  4. Sentiment / Tone (lexicon-based)                                 #
# ------------------------------------------------------------------ #

# Curated positive/negative word lists (broad coverage, no external deps)
_POSITIVE_WORDS: set[str] = {
    "good", "great", "excellent", "wonderful", "fantastic", "amazing",
    "awesome", "perfect", "beautiful", "brilliant", "outstanding",
    "superb", "terrific", "delightful", "pleasant", "positive",
    "happy", "glad", "love", "enjoy", "appreciate", "helpful",
    "useful", "valuable", "effective", "successful", "benefit",
    "improve", "enhance", "strengthen", "support", "encourage",
    "exciting", "interesting", "impressive", "remarkable", "exceptional",
    "innovative", "creative", "elegant", "efficient", "reliable",
    "convenient", "flexible", "robust", "powerful", "insightful",
    "clear", "straightforward", "intuitive", "comprehensive",
    "thorough", "well", "best", "better", "advantage", "opportunity",
    "progress", "growth", "achievement", "accomplishment", "reward",
    "definitely", "absolutely", "certainly", "fortunately", "thankfully",
    "recommended", "ideal", "optimal", "preferable", "favorable",
}

_NEGATIVE_WORDS: set[str] = {
    "bad", "terrible", "awful", "horrible", "poor", "worst",
    "wrong", "fail", "failure", "error", "mistake", "problem",
    "issue", "bug", "broken", "crash", "damage", "harm",
    "dangerous", "risky", "threat", "vulnerability", "weakness",
    "difficult", "complex", "complicated", "confusing", "unclear",
    "unfortunately", "sadly", "however", "but", "although",
    "warning", "caution", "concern", "worry", "anxiety",
    "frustrating", "annoying", "disappointing", "inadequate",
    "insufficient", "lacking", "missing", "limited", "restrict",
    "impossible", "unable", "cannot", "never", "nothing",
    "expensive", "costly", "slow", "delay", "obstacle",
    "drawback", "disadvantage", "downside", "negative", "worse",
    "harder", "painful", "struggle", "conflict", "challenge",
    "deprecated", "obsolete", "outdated", "incompatible",
}

# Intensifiers and negators for simple adjustments
_NEGATORS: set[str] = {
    "not", "no", "never", "neither", "nor", "don't", "doesn't",
    "didn't", "won't", "wouldn't", "can't", "cannot", "shouldn't",
    "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
}


def _sentiment_score(text: str) -> float:
    """Compute a simple lexicon-based sentiment polarity in [-1, 1].

    Uses word counting with basic negation handling.  A preceding
    negator within a 3-word window flips the word's valence.
    """
    words = text.lower().split()
    score = 0.0
    n_sentiment_words = 0

    for i, word in enumerate(words):
        # Clean punctuation
        clean = word.strip(".,!?;:\"'()-[]{}#")
        if not clean:
            continue

        valence = 0.0
        if clean in _POSITIVE_WORDS:
            valence = 1.0
        elif clean in _NEGATIVE_WORDS:
            valence = -1.0
        else:
            continue

        # Check for negation in preceding 3 words
        window_start = max(0, i - 3)
        preceding = words[window_start:i]
        if any(w.strip(".,!?;:'\"") in _NEGATORS for w in preceding):
            valence *= -0.5  # Partial flip (negation rarely fully inverts)

        score += valence
        n_sentiment_words += 1

    if n_sentiment_words == 0:
        return 0.0

    # Normalise: average valence, clamp to [-1, 1]
    raw = score / n_sentiment_words
    return max(-1.0, min(1.0, raw))


def _classify_polarity(score: float) -> str:
    """Map a polarity score to positive / negative / neutral."""
    if score > 0.1:
        return "positive"
    elif score < -0.1:
        return "negative"
    return "neutral"


def _sentiment_analysis(
    texts: list[str],
    sources: list[str],
) -> dict[str, Any]:
    """Classify sentiment of each turn, aggregate by source."""
    all_sources = sorted(set(sources))
    results: dict[str, Any] = {}

    for src in all_sources:
        src_texts = [t for t, s in zip(texts, sources) if s == src]
        scores = [_sentiment_score(t) for t in src_texts]
        labels = [_classify_polarity(s) for s in scores]
        counter = Counter(labels)
        n = len(labels)

        results[src] = {
            "positive": round(counter.get("positive", 0) / max(n, 1), 3),
            "negative": round(counter.get("negative", 0) / max(n, 1), 3),
            "neutral": round(counter.get("neutral", 0) / max(n, 1), 3),
            "mean_polarity": round(float(np.mean(scores)), 4) if scores else 0.0,
            "n_turns": n,
        }

    return results

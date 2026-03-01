"""Conversation-level structural analysis.

Treats each conversation as a unit rather than analysing individual turns.
Computes shape metrics, rephrasing detection, depth classification,
resolution patterns, and optional clustering.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.parsers.base import Conversation

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Resolution-pattern phrase sets                                      #
# ------------------------------------------------------------------ #

_THANK_PHRASES = [
    "thanks", "thank you", "perfect", "great", "got it",
    "that works", "awesome", "appreciate", "wonderful",
    "excellent", "nice", "cheers",
]

_CORRECTION_PHRASES = [
    "actually", "that's wrong", "that's not right", "that's incorrect",
    "i meant", "no,", "no.", "nope", "that's not what",
]


# ------------------------------------------------------------------ #
#  Public entry point                                                  #
# ------------------------------------------------------------------ #


def run(conversations: list[Conversation], config: dict) -> dict[str, Any]:
    """Run conversation-level structural analyses.

    Args:
        conversations: Normalised conversation objects (all sources).
        config: Loaded ``config.yaml`` dict.

    Returns:
        Nested dict with shape_metrics, rephrasing, depth_classification,
        resolution_patterns, and clustering results.
    """
    min_len = config.get("analysis", {}).get("min_turn_length", 10)
    embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
    processed_root = Path(
        config.get("paths", {}).get("processed_data", "data/processed")
    )

    # Group conversations by source
    convs_by_source: dict[str, list[Conversation]] = {}
    for conv in conversations:
        convs_by_source.setdefault(conv.source, []).append(conv)
    sources = sorted(convs_by_source.keys())

    logger.info(
        "conversation_structure: %d conversations, sources=%s",
        len(conversations), sources,
    )

    # ---- Step 1: Shape metrics ---- #
    logger.info("conversation_structure: computing shape metrics ...")
    shape_metrics = _shape_metrics(convs_by_source, min_len)

    # ---- Step 2: Rephrasing detection ---- #
    logger.info("conversation_structure: detecting rephrases ...")
    rephrasing = _rephrasing_detection(
        conversations, min_len, embedding_model_name, processed_root,
    )

    # ---- Step 3: Depth classification ---- #
    logger.info("conversation_structure: classifying depth ...")
    depth_classification = _depth_classification(convs_by_source, min_len)

    # ---- Step 4: Resolution patterns ---- #
    logger.info("conversation_structure: analysing resolution patterns ...")
    resolution_patterns = _resolution_patterns(convs_by_source)

    # ---- Step 5: Clustering ---- #
    logger.info("conversation_structure: clustering conversations ...")
    clustering = _conversation_clustering(
        conversations, shape_metrics, min_len,
    )

    results: dict[str, Any] = {
        "shape_metrics": shape_metrics,
        "rephrasing": rephrasing,
        "depth_classification": depth_classification,
        "resolution_patterns": resolution_patterns,
        "clustering": clustering,
        "_meta": {"sources": sources},
    }
    return results


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #


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


def _spearman_rank_corr(x: list[float], y: list[float]) -> float | None:
    """Compute Spearman rank correlation between two lists.

    Returns None if fewer than 3 data points.
    """
    n = len(x)
    if n < 3:
        return None

    def _rank(arr: list[float]) -> list[float]:
        indexed = sorted(enumerate(arr), key=lambda p: p[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and indexed[j][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j - 1) / 2.0 + 1
            for k in range(i, j):
                ranks[indexed[k][0]] = avg_rank
            i = j
        return ranks

    rx = _rank(x)
    ry = _rank(y)

    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    denom = n * (n * n - 1)
    if denom == 0:
        return None
    rho = 1 - (6 * d_sq / denom)
    return round(rho, 4)


# ------------------------------------------------------------------ #
#  Step 1: Shape metrics                                               #
# ------------------------------------------------------------------ #


def _shape_metrics(
    convs_by_source: dict[str, list[Conversation]],
    min_len: int,
) -> dict[str, Any]:
    """Per-conversation shape metrics, aggregated per source."""
    per_source: dict[str, Any] = {}

    for src, convs in sorted(convs_by_source.items()):
        turn_counts: list[int] = []
        user_turns_list: list[int] = []
        asst_turns_list: list[int] = []
        user_words_list: list[int] = []
        asst_words_list: list[int] = []
        durations: list[float] = []
        mean_asst_lengths: list[float] = []
        verbosity_trajectories: list[float] = []
        user_effort_trajectories: list[float] = []

        for conv in convs:
            turns = conv.turns
            n_total = len(turns)
            n_user = sum(1 for t in turns if t.role == "user")
            n_asst = sum(1 for t in turns if t.role == "assistant")
            u_words = sum(
                len(t.content.split()) for t in turns if t.role == "user"
            )
            a_words = sum(
                len(t.content.split())
                for t in turns
                if t.role == "assistant" and len(t.content) >= min_len
            )

            turn_counts.append(n_total)
            user_turns_list.append(n_user)
            asst_turns_list.append(n_asst)
            user_words_list.append(u_words)
            asst_words_list.append(a_words)

            # Duration
            timestamps = [
                t.timestamp for t in turns if t.timestamp is not None
            ]
            if len(timestamps) >= 2:
                delta = (max(timestamps) - min(timestamps)).total_seconds()
                durations.append(delta / 60.0)  # minutes

            # Mean assistant response length
            asst_lens = [
                len(t.content.split())
                for t in turns
                if t.role == "assistant" and len(t.content) >= min_len
            ]
            if asst_lens:
                mean_asst_lengths.append(sum(asst_lens) / len(asst_lens))

            # Verbosity trajectory (Spearman: turn index vs response length)
            if len(asst_lens) >= 3:
                indices = list(range(len(asst_lens)))
                rho = _spearman_rank_corr(
                    [float(i) for i in indices],
                    [float(v) for v in asst_lens],
                )
                if rho is not None:
                    verbosity_trajectories.append(rho)

            # User effort trajectory
            user_lens = [
                len(t.content.split())
                for t in turns
                if t.role == "user" and len(t.content) >= min_len
            ]
            if len(user_lens) >= 3:
                indices = list(range(len(user_lens)))
                rho = _spearman_rank_corr(
                    [float(i) for i in indices],
                    [float(v) for v in user_lens],
                )
                if rho is not None:
                    user_effort_trajectories.append(rho)

        per_source[src] = {
            "turn_count": _describe_dist(turn_counts),
            "user_turns": _describe_dist(user_turns_list),
            "assistant_turns": _describe_dist(asst_turns_list),
            "user_words": _describe_dist(user_words_list),
            "assistant_words": _describe_dist(asst_words_list),
            "duration_minutes": _describe_dist(durations) if durations else {
                "mean": 0, "median": 0, "std": 0, "count": 0,
                "note": "no timestamp data available",
            },
            "mean_asst_response_length": _describe_dist(mean_asst_lengths),
            "verbosity_trajectory": _describe_dist(verbosity_trajectories),
            "user_effort_trajectory": _describe_dist(user_effort_trajectories),
        }

    return {"per_source": per_source}


# ------------------------------------------------------------------ #
#  Step 2: Rephrasing detection                                        #
# ------------------------------------------------------------------ #


def _rephrasing_detection(
    conversations: list[Conversation],
    min_len: int,
    model_name: str,
    processed_root: Path,
    similarity_threshold: float = 0.6,
) -> dict[str, Any]:
    """Detect rephrase events — consecutive similar user messages."""

    # Build user turns with conversation context, replicating the same
    # ordering as semantic.py so we can align with cached embeddings.
    user_entries: list[dict] = []  # {text, source, conv_id, turn_idx}
    flat_texts: list[str] = []

    for conv in conversations:
        for ti, turn in enumerate(conv.turns):
            if turn.role == "user" and len(turn.content) >= min_len:
                user_entries.append({
                    "text": turn.content,
                    "source": conv.source,
                    "conv_id": conv.conversation_id,
                    "turn_idx": ti,
                })
                flat_texts.append(turn.content)

    # Load or compute embeddings
    embeddings = _load_or_compute_embeddings(
        flat_texts, model_name, processed_root, "user",
    )

    # Group entries by conversation
    conv_groups: dict[str, list[int]] = defaultdict(list)
    for i, entry in enumerate(user_entries):
        conv_groups[entry["conv_id"]].append(i)

    # Detect rephrase events
    rephrase_events: dict[str, list[dict]] = defaultdict(list)

    for conv_id, indices in conv_groups.items():
        if len(indices) < 2:
            continue

        for pos in range(1, len(indices)):
            i_prev = indices[pos - 1]
            i_curr = indices[pos]

            # Check that an assistant turn existed between these user turns
            prev_turn_idx = user_entries[i_prev]["turn_idx"]
            curr_turn_idx = user_entries[i_curr]["turn_idx"]
            if curr_turn_idx - prev_turn_idx < 2:
                # No room for an assistant response between them
                continue

            # Cosine similarity
            e1 = embeddings[i_prev]
            e2 = embeddings[i_curr]
            norm1 = np.linalg.norm(e1)
            norm2 = np.linalg.norm(e2)
            if norm1 < 1e-9 or norm2 < 1e-9:
                continue
            sim = float(np.dot(e1, e2) / (norm1 * norm2))

            if sim >= similarity_threshold:
                src = user_entries[i_curr]["source"]
                rephrase_events[src].append({
                    "conv_id": conv_id,
                    "msg1": user_entries[i_prev]["text"][:200],
                    "msg2": user_entries[i_curr]["text"][:200],
                    "similarity": round(sim, 4),
                })

    # Compute per-source stats
    # Count total user turns per source for rate calculation
    user_turn_counts: dict[str, int] = Counter()
    conv_counts: dict[str, int] = Counter()
    for entry in user_entries:
        user_turn_counts[entry["source"]] += 1
    for conv in conversations:
        conv_counts[conv.source] += 1

    sources = sorted(set(e["source"] for e in user_entries))
    per_source: dict[str, Any] = {}

    for src in sources:
        events = rephrase_events.get(src, [])
        total = len(events)
        n_user_turns = user_turn_counts.get(src, 1)
        n_convs = conv_counts.get(src, 1)
        rate_per_100 = round(total * 100 / max(n_user_turns, 1), 2)

        # Count conversations that had at least one rephrase
        convs_with_rephrase = len(set(e["conv_id"] for e in events))
        rephrase_per_conv = round(total / max(n_convs, 1), 3)

        # Top examples by similarity (most similar = clearest rephrases)
        examples = sorted(events, key=lambda x: -x["similarity"])[:5]

        per_source[src] = {
            "total": total,
            "rate_per_100_user_turns": rate_per_100,
            "conversations_with_rephrase": convs_with_rephrase,
            "rephrase_per_conversation": rephrase_per_conv,
            "n_user_turns": n_user_turns,
            "n_conversations": n_convs,
            "examples": examples,
        }

    return {
        "per_source": per_source,
        "threshold": similarity_threshold,
    }


def _load_or_compute_embeddings(
    texts: list[str],
    model_name: str,
    cache_dir: Path,
    role_label: str,
) -> np.ndarray:
    """Load cached embeddings if they match, otherwise recompute."""
    cache_path = cache_dir / f"embeddings_{role_label}.npy"

    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            logger.info(
                "conversation_structure: loaded cached embeddings from %s",
                cache_path,
            )
            return cached
        logger.info(
            "conversation_structure: cached embeddings size mismatch "
            "(%d vs %d), recomputing",
            cached.shape[0], len(texts),
        )

    from sentence_transformers import SentenceTransformer

    logger.info(
        "conversation_structure: encoding %d %s turns with %s ...",
        len(texts), role_label, model_name,
    )
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    logger.info("conversation_structure: cached embeddings to %s", cache_path)
    return embeddings


# ------------------------------------------------------------------ #
#  Step 3: Depth classification                                        #
# ------------------------------------------------------------------ #

_DEPTH_BINS = [
    ("quick_exchange", 1, 2),
    ("short_session", 3, 6),
    ("working_session", 7, 20),
    ("deep_dive", 21, 50),
    ("marathon", 51, 999999),
]


def _count_turn_pairs(conv: Conversation) -> int:
    """Count user+assistant exchange pairs in a conversation."""
    pairs = 0
    turns = conv.turns
    for i in range(len(turns) - 1):
        if turns[i].role == "user" and turns[i + 1].role == "assistant":
            pairs += 1
    # Also count a trailing user turn as half a pair (round up)
    if not pairs:
        pairs = max(1, len(turns) // 2)
    return pairs


def _classify_depth(n_pairs: int) -> str:
    """Return the depth category name for a given number of turn pairs."""
    for name, lo, hi in _DEPTH_BINS:
        if lo <= n_pairs <= hi:
            return name
    return "marathon"


def _depth_classification(
    convs_by_source: dict[str, list[Conversation]],
    min_len: int,
) -> dict[str, Any]:
    """Classify conversations by depth and compute response length per depth."""
    per_source: dict[str, Any] = {}
    response_length_by_depth: dict[str, dict[str, Any]] = {}

    for src, convs in sorted(convs_by_source.items()):
        depth_counts: dict[str, int] = {name: 0 for name, _, _ in _DEPTH_BINS}
        depth_response_lengths: dict[str, list[float]] = {
            name: [] for name, _, _ in _DEPTH_BINS
        }

        for conv in convs:
            n_pairs = _count_turn_pairs(conv)
            category = _classify_depth(n_pairs)
            depth_counts[category] += 1

            # Mean assistant response length for this conversation
            asst_lens = [
                len(t.content.split())
                for t in conv.turns
                if t.role == "assistant" and len(t.content) >= min_len
            ]
            if asst_lens:
                mean_len = sum(asst_lens) / len(asst_lens)
                depth_response_lengths[category].append(mean_len)

        per_source[src] = depth_counts

        # Mean response length per depth category
        response_length_by_depth[src] = {
            cat: (
                round(sum(vals) / len(vals), 1) if vals else None
            )
            for cat, vals in depth_response_lengths.items()
        }

    return {
        "per_source": per_source,
        "response_length_by_depth": response_length_by_depth,
        "categories": [name for name, _, _ in _DEPTH_BINS],
    }


# ------------------------------------------------------------------ #
#  Step 4: Resolution patterns                                         #
# ------------------------------------------------------------------ #


def _resolution_patterns(
    convs_by_source: dict[str, list[Conversation]],
) -> dict[str, Any]:
    """Analyse the final user message in each conversation."""
    per_source: dict[str, Any] = {}

    for src, convs in sorted(convs_by_source.items()):
        n_total = len(convs)
        counts = {
            "thank_you": 0,
            "question": 0,
            "correction": 0,
            "short_close": 0,
            "other": 0,
            "no_user_turns": 0,
        }

        for conv in convs:
            # Find last user turn
            last_user = None
            for turn in reversed(conv.turns):
                if turn.role == "user":
                    last_user = turn
                    break

            if last_user is None:
                counts["no_user_turns"] += 1
                continue

            text = last_user.content.strip()
            text_lower = text.lower()
            word_count = len(text.split())

            # Classify (priority order: correction > thank_you > question > short)
            classified = False

            # Correction/pushback
            for phrase in _CORRECTION_PHRASES:
                if phrase in text_lower:
                    counts["correction"] += 1
                    classified = True
                    break
            if classified:
                continue

            # Thank you / acknowledgment
            for phrase in _THANK_PHRASES:
                if phrase in text_lower:
                    counts["thank_you"] += 1
                    classified = True
                    break
            if classified:
                continue

            # Ends with question
            if text.rstrip().endswith("?"):
                counts["question"] += 1
                continue

            # Short close
            if word_count < 5:
                counts["short_close"] += 1
                continue

            counts["other"] += 1

        # Convert to rates
        denom = max(n_total, 1)
        per_source[src] = {
            "raw_counts": counts,
            "rates": {
                k: round(v / denom, 3) for k, v in counts.items()
            },
            "n_conversations": n_total,
        }

    return {"per_source": per_source}


# ------------------------------------------------------------------ #
#  Step 5: Conversation clustering                                     #
# ------------------------------------------------------------------ #


def _conversation_clustering(
    conversations: list[Conversation],
    shape_metrics: dict[str, Any],
    min_len: int,
) -> dict[str, Any]:
    """Cluster conversations by structural features using K-means."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    # Build feature vectors per conversation
    features: list[list[float]] = []
    conv_sources: list[str] = []
    conv_ids: list[str] = []

    for conv in conversations:
        turns = conv.turns
        n_pairs = _count_turn_pairs(conv)

        asst_lens = [
            len(t.content.split())
            for t in turns
            if t.role == "assistant" and len(t.content) >= min_len
        ]
        user_lens = [
            len(t.content.split())
            for t in turns
            if t.role == "user" and len(t.content) >= min_len
        ]

        mean_asst_len = (
            sum(asst_lens) / len(asst_lens) if asst_lens else 0
        )

        # Verbosity trajectory
        verb_traj = 0.0
        if len(asst_lens) >= 3:
            rho = _spearman_rank_corr(
                [float(i) for i in range(len(asst_lens))],
                [float(v) for v in asst_lens],
            )
            if rho is not None:
                verb_traj = rho

        # User effort trajectory
        user_traj = 0.0
        if len(user_lens) >= 3:
            rho = _spearman_rank_corr(
                [float(i) for i in range(len(user_lens))],
                [float(v) for v in user_lens],
            )
            if rho is not None:
                user_traj = rho

        # Duration
        timestamps = [t.timestamp for t in turns if t.timestamp is not None]
        duration = 0.0
        if len(timestamps) >= 2:
            duration = (
                max(timestamps) - min(timestamps)
            ).total_seconds() / 60.0

        features.append([
            float(n_pairs),
            duration,
            verb_traj,
            user_traj,
            mean_asst_len,
        ])
        conv_sources.append(conv.source)
        conv_ids.append(conv.conversation_id)

    if len(features) < 10:
        return {
            "status": "skipped",
            "reason": f"Too few conversations ({len(features)}) for clustering",
        }

    X = np.array(features)
    feature_names = [
        "turn_pairs", "duration_min", "verbosity_trajectory",
        "user_effort_trajectory", "mean_asst_length",
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Try k=4 to k=6, pick best silhouette
    best_k = 4
    best_score = -1.0
    best_labels = None

    for k in range(4, min(7, len(features))):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)

            # Silhouette requires at least 2 clusters with >1 member
            unique_labels = set(labels)
            if len(unique_labels) < 2:
                continue

            score = silhouette_score(X_scaled, labels)
            logger.info(
                "conversation_structure: k=%d, silhouette=%.3f", k, score
            )

            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        except Exception as exc:
            logger.warning(
                "conversation_structure: k=%d failed: %s", k, exc
            )

    if best_labels is None or best_score < 0.15:
        return {
            "status": "low_quality",
            "silhouette_score": round(best_score, 4) if best_score > -1 else None,
            "reason": (
                f"Best silhouette score ({best_score:.3f}) below 0.15 threshold. "
                "Clusters are not well-separated."
            ),
        }

    # Build cluster descriptions
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km_final.fit_predict(X_scaled)

    # Unscale centroids for interpretability
    centroids_scaled = km_final.cluster_centers_
    centroids_raw = scaler.inverse_transform(centroids_scaled)

    clusters: list[dict[str, Any]] = []
    sources_set = sorted(set(conv_sources))

    for ci in range(best_k):
        mask = labels == ci
        size = int(mask.sum())

        # Source breakdown
        cluster_sources = [s for s, m in zip(conv_sources, mask) if m]
        source_breakdown = {
            s: round(cluster_sources.count(s) / max(size, 1), 3)
            for s in sources_set
        }

        centroid = {
            feature_names[fi]: round(float(centroids_raw[ci][fi]), 2)
            for fi in range(len(feature_names))
        }

        # Auto-label based on centroid characteristics
        label = _auto_label_cluster(centroid)

        clusters.append({
            "cluster_id": ci,
            "label": label,
            "size": size,
            "source_breakdown": source_breakdown,
            "centroid": centroid,
        })

    clusters.sort(key=lambda c: -c["size"])

    return {
        "status": "ok",
        "n_clusters": best_k,
        "silhouette_score": round(best_score, 4),
        "clusters": clusters,
        "feature_names": feature_names,
    }


def _auto_label_cluster(centroid: dict[str, float]) -> str:
    """Generate a descriptive label from centroid values."""
    pairs = centroid.get("turn_pairs", 0)
    duration = centroid.get("duration_min", 0)
    mean_len = centroid.get("mean_asst_length", 0)
    verb_traj = centroid.get("verbosity_trajectory", 0)

    parts: list[str] = []

    # Depth
    if pairs <= 2:
        parts.append("Quick")
    elif pairs <= 6:
        parts.append("Short")
    elif pairs <= 20:
        parts.append("Medium")
    else:
        parts.append("Deep")

    # Response style
    if mean_len > 500:
        parts.append("verbose")
    elif mean_len < 150:
        parts.append("concise")
    else:
        parts.append("moderate")

    # Trajectory
    if verb_traj > 0.2:
        parts.append("(escalating)")
    elif verb_traj < -0.2:
        parts.append("(tapering)")
    else:
        parts.append("(steady)")

    return " ".join(parts)

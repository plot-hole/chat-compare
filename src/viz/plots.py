"""All visualization functions.

Each function accepts the relevant results dict (loaded from JSON),
saves the figure to ``data/outputs/plots/`` as PNG, and returns the
figure object.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Shared configuration                                                #
# ------------------------------------------------------------------ #

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"
_PLOTS_DIR: Path | None = None
_PALETTE: dict[str, str] = {}
_STYLE: str = "seaborn-v0_8-whitegrid"
_FIGSIZE: tuple[int, int] = (12, 7)
_DPI: int = 150


def _init_config() -> None:
    """Load viz settings from config.yaml (called once)."""
    global _PLOTS_DIR, _PALETTE, _STYLE, _FIGSIZE, _DPI

    with open(_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    viz = config.get("viz", {})
    _STYLE = viz.get("style", _STYLE)
    _FIGSIZE = tuple(viz.get("figsize", list(_FIGSIZE)))
    _DPI = viz.get("dpi", _DPI)
    palette_list = viz.get("palette", ["#6366f1", "#f59e0b", "#10b981"])
    _PALETTE = {
        "claude": palette_list[0],
        "gemini": palette_list[1] if len(palette_list) > 1 else "#f59e0b",
        "chatgpt": palette_list[2] if len(palette_list) > 2 else "#10b981",
    }

    outputs = Path(config.get("paths", {}).get("outputs", "data/outputs"))
    _PLOTS_DIR = outputs / "plots"
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_config():
    if _PLOTS_DIR is None:
        _init_config()


def _color(source: str) -> str:
    _ensure_config()
    return _PALETTE.get(source, "#888888")


def _save(fig: plt.Figure, name: str) -> Path:
    _ensure_config()
    path = _PLOTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    logger.info("Saved plot: %s", path)
    return path


def _apply_style():
    _ensure_config()
    try:
        plt.style.use(_STYLE)
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")


# ------------------------------------------------------------------ #
#  1. Vocabulary comparison                                            #
# ------------------------------------------------------------------ #


def plot_vocabulary_comparison(lexical: dict) -> plt.Figure:
    """Grouped bar chart: MATTR, mean sentence length, FK readability."""
    _apply_style()
    sources = lexical.get("_meta", {}).get("sources", [])

    metrics = {
        "MATTR-500": [
            lexical.get("vocabulary", {}).get(s, {}).get("mattr_500", 0)
            for s in sources
        ],
        "Mean Sentence\nLength (words)": [
            lexical.get("sentence_stats", {}).get(s, {}).get("mean", 0)
            for s in sources
        ],
        "Flesch-Kincaid\nGrade Level": [
            lexical.get("readability", {}).get(s, {})
            .get("flesch_kincaid", {}).get("mean", 0)
            for s in sources
        ],
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        x = np.arange(len(sources))
        bars = ax.bar(
            x, values, width=0.5,
            color=[_color(s) for s in sources],
            edgecolor="white", linewidth=0.8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in sources], fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight="bold")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom", fontsize=10,
            )
        ax.set_ylim(0, max(values) * 1.2)

    fig.suptitle("Vocabulary & Readability Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "vocabulary_comparison")
    return fig


# ------------------------------------------------------------------ #
#  2. Distinctive words                                                #
# ------------------------------------------------------------------ #


def plot_distinctive_words(lexical: dict) -> plt.Figure:
    """Two-panel horizontal bar chart of top 15 TF-IDF words per source."""
    _apply_style()
    sources = lexical.get("_meta", {}).get("sources", [])
    dw = lexical.get("distinctive_words", {})

    n_panels = len(sources)
    fig, axes = plt.subplots(1, n_panels, figsize=(14, 7))
    if n_panels == 1:
        axes = [axes]

    for ax, s in zip(axes, sources):
        words_data = dw.get(s, [])[:15]
        words_data = list(reversed(words_data))  # Bottom-to-top
        words = [d["word"] for d in words_data]
        scores = [d["tfidf"] for d in words_data]

        ax.barh(
            range(len(words)), scores,
            color=_color(s), edgecolor="white", linewidth=0.5,
        )
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=10)
        ax.set_xlabel("TF-IDF Score", fontsize=10)
        ax.set_title(f"{s.capitalize()} Distinctive Words", fontsize=12, fontweight="bold")
        ax.invert_xaxis() if s == sources[0] and n_panels > 1 else None
        # Add value labels
        for i, (score, word) in enumerate(zip(scores, words)):
            ax.text(score + 0.001, i, f"{score:.3f}", va="center", fontsize=8)

    fig.suptitle("TF-IDF Distinctive Words by Source", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "distinctive_words")
    return fig


# ------------------------------------------------------------------ #
#  3. Response length distributions                                    #
# ------------------------------------------------------------------ #


def plot_response_length_distributions(
    lexical: dict,
    conversations: list | None = None,
) -> plt.Figure:
    """Overlapping KDE density plots of response length per source.

    If *conversations* are provided, computes lengths directly.
    Otherwise falls back to summary stats for a simulated distribution.
    """
    _apply_style()
    sources = lexical.get("_meta", {}).get("sources", [])

    fig, ax = plt.subplots(figsize=_FIGSIZE)

    lengths_by_source: dict[str, list[int]] = {}

    if conversations:
        for conv in conversations:
            for turn in conv.turns:
                if turn.role == "assistant" and len(turn.content) >= 10:
                    wc = len(turn.content.split())
                    lengths_by_source.setdefault(conv.source, []).append(wc)

    # Compute KDE or histogram
    for s in sources:
        if s in lengths_by_source and lengths_by_source[s]:
            data = np.array(lengths_by_source[s])
        else:
            # Fallback: create synthetic from mean/std
            rl = lexical.get("response_length", {}).get(s, {})
            mean = rl.get("mean", 300)
            std = rl.get("std", 200)
            count = rl.get("count", 500)
            rng = np.random.RandomState(42)
            data = np.abs(rng.normal(mean, std, count)).astype(int)

        # Clip at 95th percentile for readability
        p95 = np.percentile(data, 95)
        data_clipped = data[data <= p95]
        median_val = np.median(data)

        # KDE via histogram with smooth edges
        ax.hist(
            data_clipped, bins=60, density=True, alpha=0.35,
            color=_color(s), label=f"{s.capitalize()} (n={len(data):,})",
            edgecolor="white", linewidth=0.3,
        )
        # KDE line
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(data_clipped)
            x_range = np.linspace(0, p95, 300)
            ax.plot(x_range, kde(x_range), color=_color(s), linewidth=2)
        except Exception:
            pass

        ax.axvline(
            median_val, color=_color(s), linestyle="--", linewidth=1.5,
            label=f"{s.capitalize()} median: {median_val:.0f}",
        )

    ax.set_xlabel("Response Length (words)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Response Length Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(left=0)

    fig.tight_layout()
    _save(fig, "response_length_distributions")
    return fig


# ------------------------------------------------------------------ #
#  4. Formatting habits                                                #
# ------------------------------------------------------------------ #


def plot_formatting_habits(lexical: dict) -> plt.Figure:
    """Grouped bar chart of formatting elements per 1k words."""
    _apply_style()
    sources = lexical.get("_meta", {}).get("sources", [])
    fmt = lexical.get("formatting", {})

    elements = ["bold", "bullet_points", "headers", "numbered_lists", "code_blocks", "italic"]
    labels = ["Bold", "Bullets", "Headers", "Numbered\nLists", "Code\nBlocks", "Italic"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(elements))
    width = 0.35
    offsets = np.linspace(-width / 2, width / 2, len(sources))

    for i, s in enumerate(sources):
        vals = [
            fmt.get(s, {}).get("per_1k_words", {}).get(el, 0)
            for el in elements
        ]
        offset = (i - (len(sources) - 1) / 2) * width
        bars = ax.bar(
            x + offset, vals, width * 0.9,
            color=_color(s), label=s.capitalize(),
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            if val > 0.5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Occurrences per 1,000 Words", fontsize=11)
    ax.set_title("Formatting Habits by Source", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    # Annotation about Gemini HTML conversion
    ax.annotate(
        "Note: Gemini formatting was converted from\n"
        "native HTML (Google Takeout) to Markdown.",
        xy=(0.98, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=8, fontstyle="italic",
        color="#666666",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#cccccc"),
    )

    fig.tight_layout()
    _save(fig, "formatting_habits")
    return fig


# ------------------------------------------------------------------ #
#  5. Topic distribution                                               #
# ------------------------------------------------------------------ #


def plot_topic_distribution(semantic: dict) -> plt.Figure:
    """Grouped horizontal bar chart of top 10 topics by source %."""
    _apply_style()
    sources = semantic.get("_meta", {}).get("sources", [])
    topics = semantic.get("assistant_topics", {}).get("topics", [])

    # Filter to non-outlier topics, take top 10 by count
    display_topics = [t for t in topics if t.get("topic_id", -1) >= 0][:10]
    display_topics = list(reversed(display_topics))  # Bottom-to-top

    fig, ax = plt.subplots(figsize=(13, 7))

    y = np.arange(len(display_topics))
    height = 0.35

    for i, s in enumerate(sources):
        vals = []
        for t in display_topics:
            bd = t.get("source_breakdown", {})
            vals.append(bd.get(s, 0) * 100)
        offset = (i - (len(sources) - 1) / 2) * height
        ax.barh(
            y + offset, vals, height * 0.9,
            color=_color(s), label=s.capitalize(),
            edgecolor="white", linewidth=0.5,
        )

    # Labels: topic words + count
    ylabels = []
    for t in display_topics:
        words = [w["word"] for w in t.get("words", [])[:4]]
        count = t.get("count", 0)
        ylabels.append(f"{', '.join(words)} (n={count})")

    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("% of Topic from Source", fontsize=11)
    ax.set_title("Topic Distribution by Source (Top 10)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, 105)

    fig.tight_layout()
    _save(fig, "topic_distribution")
    return fig


# ------------------------------------------------------------------ #
#  6. Hedging comparison                                               #
# ------------------------------------------------------------------ #


def plot_hedging_comparison(pragmatic: dict) -> plt.Figure:
    """Two-part figure: category bars + top phrase bars."""
    _apply_style()
    sources = pragmatic.get("_meta", {}).get("sources", [])
    hedging = pragmatic.get("hedging", {})

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 10),
                                          gridspec_kw={"height_ratios": [1, 1.5]})

    # ---- Top: Category grouped bars ---- #
    categories = ["uncertainty", "caveat_softener", "epistemic"]
    cat_labels = ["Uncertainty\n(I think, I believe...)", "Caveat/Softener\n(however, that said...)",
                  "Epistemic\n(might, could, may...)"]

    x = np.arange(len(categories))
    width = 0.35

    for i, s in enumerate(sources):
        vals = [
            hedging.get(s, {}).get("category_per_1k", {}).get(cat, 0)
            for cat in categories
        ]
        offset = (i - (len(sources) - 1) / 2) * width
        bars = ax_top.bar(
            x + offset, vals, width * 0.9,
            color=_color(s), label=s.capitalize(),
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            ax_top.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom", fontsize=9,
            )

    ax_top.set_xticks(x)
    ax_top.set_xticklabels(cat_labels, fontsize=10)
    ax_top.set_ylabel("Per 1,000 Words", fontsize=11)
    ax_top.set_title("Hedge Density by Category", fontsize=13, fontweight="bold")
    ax_top.legend(fontsize=10)

    # ---- Bottom: Top 10 phrases ---- #
    # Collect all phrases and get top 10 by max rate across sources
    all_phrases: dict[str, float] = {}
    for s in sources:
        for phrase, rate in hedging.get(s, {}).get("phrase_per_1k", {}).items():
            all_phrases[phrase] = max(all_phrases.get(phrase, 0), rate)

    top_phrases = sorted(all_phrases.items(), key=lambda x: -x[1])[:10]
    phrase_names = [p[0] for p in top_phrases]
    phrase_names.reverse()  # Bottom-to-top

    y = np.arange(len(phrase_names))
    height = 0.35

    for i, s in enumerate(sources):
        vals = [
            hedging.get(s, {}).get("phrase_per_1k", {}).get(p, 0)
            for p in phrase_names
        ]
        offset = (i - (len(sources) - 1) / 2) * height
        ax_bot.barh(
            y + offset, vals, height * 0.9,
            color=_color(s), label=s.capitalize(),
            edgecolor="white", linewidth=0.5,
        )

    ax_bot.set_yticks(y)
    ax_bot.set_yticklabels(phrase_names, fontsize=10)
    ax_bot.set_xlabel("Per 1,000 Words", fontsize=11)
    ax_bot.set_title("Top Hedge Phrases by Source", fontsize=13, fontweight="bold")
    ax_bot.legend(fontsize=10)

    fig.tight_layout()
    _save(fig, "hedging_comparison")
    return fig


# ------------------------------------------------------------------ #
#  7. Style fingerprint (radar chart)                                  #
# ------------------------------------------------------------------ #


def plot_style_fingerprint(comparative: dict) -> plt.Figure:
    """Radar/spider chart of normalised style metrics."""
    _apply_style()
    fp = comparative.get("style_fingerprint", {})
    sources = comparative.get("_meta", {}).get("sources", [])
    metrics = fp.get("metrics", [])
    normalised = fp.get("normalized", {})
    raw = fp.get("raw", {})

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    # Pretty labels
    label_map = {
        "vocabulary_mattr": "Vocabulary\nRichness",
        "mean_response_length": "Response\nLength",
        "readability_fk": "Readability\n(FK Grade)",
        "hedge_density": "Hedge\nDensity",
        "question_rate": "Question\nRate",
        "verbosity_median": "Verbosity\nRatio",
        "first_person_I": "First-Person\n'I' Usage",
        "formatting_density": "Formatting\nDensity",
        "self_similarity": "Self-\nSimilarity",
    }

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for s in sources:
        values = normalised.get(s, [0] * n_metrics)
        values_closed = values + values[:1]
        ax.plot(
            angles, values_closed,
            color=_color(s), linewidth=2.5, label=s.capitalize(),
        )
        ax.fill(angles, values_closed, color=_color(s), alpha=0.15)

    # Labels
    display_labels = [label_map.get(m, m) for m in metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_labels, fontsize=10, fontweight="bold")

    # Grid
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=8, color="#888")
    ax.set_ylim(0, 1.05)

    ax.legend(
        loc="upper right", bbox_to_anchor=(1.25, 1.1),
        fontsize=12, frameon=True, facecolor="white", edgecolor="#ddd",
    )
    ax.set_title(
        "Style Fingerprint",
        fontsize=16, fontweight="bold", pad=25,
    )

    # Add raw values as annotation near each axis
    for i, m in enumerate(metrics):
        angle = angles[i]
        for j, s in enumerate(sources):
            raw_val = raw.get(s, [0] * n_metrics)[i]
            norm_val = normalised.get(s, [0] * n_metrics)[i]
            # Place annotation just outside the plot
            r = 1.12 + j * 0.08
            ax.annotate(
                f"{raw_val:.1f}" if raw_val >= 1 else f"{raw_val:.3f}",
                xy=(angle, r),
                ha="center", va="center",
                fontsize=7, color=_color(s),
                fontweight="bold",
            )

    fig.tight_layout()
    _save(fig, "style_fingerprint")
    return fig


# ------------------------------------------------------------------ #
#  8. Verbosity ratio                                                  #
# ------------------------------------------------------------------ #


def plot_verbosity_ratio(
    pragmatic: dict,
    conversations: list | None = None,
) -> plt.Figure:
    """Violin/box plots of verbosity ratio per source."""
    _apply_style()
    sources = pragmatic.get("_meta", {}).get("sources", [])

    fig, ax = plt.subplots(figsize=(8, 7))

    ratios_by_source: dict[str, list[float]] = {}

    if conversations:
        for conv in conversations:
            turns = conv.turns
            for i in range(len(turns) - 1):
                if turns[i].role == "user" and turns[i + 1].role == "assistant":
                    u_words = len(turns[i].content.split())
                    a_words = len(turns[i + 1].content.split())
                    if u_words >= 5 and len(turns[i + 1].content) >= 10:
                        ratios_by_source.setdefault(conv.source, []).append(
                            a_words / u_words
                        )

    # Fallback: synthetic from summary stats
    if not ratios_by_source:
        for s in sources:
            vr = pragmatic.get("verbosity_ratio", {}).get(s, {})
            mean = vr.get("mean", 20)
            std = vr.get("std", 30)
            count = vr.get("count", 500)
            rng = np.random.RandomState(42)
            data = np.abs(rng.normal(mean, std, count))
            ratios_by_source[s] = data.tolist()

    # Cap at 95th percentile for readability
    all_ratios = []
    for s in sources:
        all_ratios.extend(ratios_by_source.get(s, []))
    p95 = np.percentile(all_ratios, 95) if all_ratios else 100

    plot_data = []
    plot_positions = []
    plot_colors = []
    medians = []

    for i, s in enumerate(sources):
        data = np.array(ratios_by_source.get(s, []))
        data_clipped = data[data <= p95]
        plot_data.append(data_clipped)
        plot_positions.append(i + 1)
        plot_colors.append(_color(s))
        medians.append(np.median(data))

    # Violin plots
    parts = ax.violinplot(
        plot_data, positions=plot_positions,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for pc, color in zip(parts["bodies"], plot_colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.3)

    # Box plots on top
    bp = ax.boxplot(
        plot_data, positions=plot_positions,
        widths=0.15, patch_artist=True,
        showfliers=False, zorder=3,
    )
    for patch, color in zip(bp["boxes"], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for element in ["whiskers", "caps"]:
        for line in bp[element]:
            line.set_color("#555555")
    for line in bp["medians"]:
        line.set_color("white")
        line.set_linewidth(2)

    # Reference line at ratio=1
    ax.axhline(y=1.0, color="#aaaaaa", linestyle="--", linewidth=1, zorder=1)
    ax.text(0.55, 1.2, "1:1 ratio", fontsize=9, color="#888888")

    # Annotate medians
    for i, (pos, med) in enumerate(zip(plot_positions, medians)):
        ax.text(
            pos + 0.25, med, f"median: {med:.1f}x",
            fontsize=10, fontweight="bold", color=plot_colors[i],
            va="center",
        )

    ax.set_xticks(plot_positions)
    ax.set_xticklabels([s.capitalize() for s in sources], fontsize=12)
    ax.set_ylabel("Assistant Words / User Words", fontsize=12)
    ax.set_title("Verbosity Ratio Distribution", fontsize=14, fontweight="bold")
    ax.set_ylim(0, p95 * 1.1)

    fig.tight_layout()
    _save(fig, "verbosity_ratio")
    return fig


# ------------------------------------------------------------------ #
#  9. User language comparison                                         #
# ------------------------------------------------------------------ #


def plot_user_language_comparison(comparative: dict) -> plt.Figure:
    """Small multiples: 4 subplots comparing user language by platform."""
    _apply_style()
    sources = comparative.get("_meta", {}).get("sources", [])
    ul = comparative.get("user_language", {})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Mean message length
    ax = axes[0, 0]
    vals = [ul.get("message_length", {}).get(s, {}).get("mean", 0) for s in sources]
    bars = ax.bar(
        range(len(sources)), vals, color=[_color(s) for s in sources],
        edgecolor="white", linewidth=0.5,
    )
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([s.capitalize() for s in sources])
    ax.set_title("Mean Message Length (words)", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    # 2. Vocabulary richness (MATTR-200)
    ax = axes[0, 1]
    vals = [ul.get("vocabulary", {}).get(s, {}).get("mattr_200", 0) for s in sources]
    bars = ax.bar(
        range(len(sources)), vals, color=[_color(s) for s in sources],
        edgecolor="white", linewidth=0.5,
    )
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([s.capitalize() for s in sources])
    ax.set_title("Vocabulary Richness (MATTR-200)", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(min(vals) * 0.95, max(vals) * 1.05)

    # 3. Question rate
    ax = axes[1, 0]
    vals = [
        ul.get("question_rate", {}).get(s, {})
        .get("fraction_turns_with_question", 0) * 100
        for s in sources
    ]
    bars = ax.bar(
        range(len(sources)), vals, color=[_color(s) for s in sources],
        edgecolor="white", linewidth=0.5,
    )
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([s.capitalize() for s in sources])
    ax.set_title("% Turns with Question", fontsize=11, fontweight="bold")
    ax.set_ylabel("%")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    # 4. Politeness markers
    ax = axes[1, 1]
    vals = [
        ul.get("politeness", {}).get(s, {}).get("politeness_per_1k", 0)
        for s in sources
    ]
    bars = ax.bar(
        range(len(sources)), vals, color=[_color(s) for s in sources],
        edgecolor="white", linewidth=0.5,
    )
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([s.capitalize() for s in sources])
    ax.set_title("Politeness Markers (per 1k words)", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(min(vals) * 0.9, max(vals) * 1.15)

    fig.suptitle(
        "User Language Comparison by Platform",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "user_language_comparison")
    return fig


# ------------------------------------------------------------------ #
#  10. Opening patterns                                                #
# ------------------------------------------------------------------ #


def plot_opening_patterns(pragmatic: dict) -> plt.Figure:
    """Two-panel horizontal bar chart of top 10 opening words per source."""
    _apply_style()
    sources = pragmatic.get("_meta", {}).get("sources", [])
    op = pragmatic.get("opening_patterns", {})

    n_panels = len(sources)
    fig, axes = plt.subplots(1, n_panels, figsize=(14, 7))
    if n_panels == 1:
        axes = [axes]

    max_pct = 0  # For symmetric x-axis

    for ax, s in zip(axes, sources):
        data = op.get(s, {})
        n_turns = data.get("n_turns", 1)
        top_words = data.get("top_opening_words", [])[:10]
        top_words = list(reversed(top_words))  # Bottom-to-top

        words = [w for w, c in top_words]
        pcts = [c / max(n_turns, 1) * 100 for w, c in top_words]
        counts = [c for w, c in top_words]
        max_pct = max(max_pct, max(pcts) if pcts else 0)

        ax.barh(
            range(len(words)), pcts,
            color=_color(s), edgecolor="white", linewidth=0.5,
        )
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=11, fontweight="bold")
        ax.set_xlabel("% of Responses", fontsize=10)
        ax.set_title(
            f"{s.capitalize()} Opening Words\n(n={n_turns:,} turns)",
            fontsize=12, fontweight="bold",
        )

        # Value labels
        for i, (pct, count) in enumerate(zip(pcts, counts)):
            ax.text(
                pct + 0.3, i,
                f"{pct:.1f}% ({count})",
                va="center", fontsize=9,
            )

    # Symmetric x-axis
    for ax in axes:
        ax.set_xlim(0, max_pct * 1.35)

    fig.suptitle(
        "Opening Word Patterns",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "opening_patterns")
    return fig


# ------------------------------------------------------------------ #
#  11. Activity over time                                              #
# ------------------------------------------------------------------ #


def plot_activity_over_time(temporal: dict) -> plt.Figure:
    """Grouped bar chart showing conversations per month per source."""
    _apply_style()
    activity = temporal.get("activity", {})
    months = activity.get("months", [])
    per_source = activity.get("per_source", {})
    sources = sorted(per_source.keys())

    if not months:
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        ax.text(0.5, 0.5, "No temporal data", ha="center", va="center")
        _save(fig, "activity_over_time")
        return fig

    fig, ax = plt.subplots(figsize=(max(12, len(months) * 0.8), 6))

    x = np.arange(len(months))
    width = 0.35
    n_sources = len(sources)

    for i, s in enumerate(sources):
        vals = per_source.get(s, {}).get("conversations", [])
        # Pad if shorter than months
        vals = vals + [0] * (len(months) - len(vals))
        offset = (i - (n_sources - 1) / 2) * width
        ax.bar(
            x + offset, vals, width * 0.9,
            color=_color(s), label=s.capitalize(),
            edgecolor="white", linewidth=0.5, alpha=0.85,
        )

    # Mark non-overlapping period
    overlap = activity.get("overlap_period", {})
    overlap_start = overlap.get("start", "")
    if overlap_start and overlap_start in months:
        oi = months.index(overlap_start)
        if oi > 0:
            ax.axvspan(-0.5, oi - 0.5, alpha=0.06, color="#888888")
            ax.text(
                (oi - 1) / 2, ax.get_ylim()[1] * 0.92,
                "Non-overlapping",
                ha="center", fontsize=9, fontstyle="italic", color="#888888",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Conversations", fontsize=11)
    ax.set_title("Platform Activity Over Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(-0.5, len(months) - 0.5)

    fig.tight_layout()
    _save(fig, "activity_over_time")
    return fig


# ------------------------------------------------------------------ #
#  12. Bot metrics trends                                              #
# ------------------------------------------------------------------ #


def plot_bot_metrics_trends(temporal: dict) -> plt.Figure:
    """Small multiples: 6 subplots (one per bot metric), line per source."""
    _apply_style()
    bot = temporal.get("bot_metrics_over_time", {})
    months = bot.get("months", [])
    sources = temporal.get("_meta", {}).get("sources", [])
    overlap = temporal.get("activity", {}).get("overlap_period", {})

    metric_info = [
        ("response_length", "Mean Response Length (words)"),
        ("mattr", "Vocabulary Richness (MATTR-500)"),
        ("readability", "Readability (Flesch-Kincaid)"),
        ("hedge_density", "Hedge Density (per 1k words)"),
        ("formatting_density", "Formatting Density (per 1k words)"),
        ("question_rate", "Question Rate (per turn)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes_flat = axes.flatten()

    x = np.arange(len(months))

    for ax, (metric_key, title) in zip(axes_flat, metric_info):
        # Shade non-overlapping period
        overlap_start = overlap.get("start", "")
        if overlap_start and overlap_start in months:
            oi = months.index(overlap_start)
            if oi > 0:
                ax.axvspan(-0.5, oi - 0.5, alpha=0.08, color="#888888")

        for s in sources:
            src_data = bot.get(s, {})
            values = src_data.get(metric_key, [])
            # Build arrays handling None values
            valid_x = []
            valid_y = []
            for i, v in enumerate(values):
                if v is not None:
                    valid_x.append(i)
                    valid_y.append(v)

            if valid_x:
                ax.plot(
                    valid_x, valid_y,
                    color=_color(s), linewidth=2, marker="o",
                    markersize=4, label=s.capitalize(), alpha=0.85,
                )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(months, rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Bot Metrics Over Time",
        fontsize=16, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, "bot_metrics_trends")
    return fig


# ------------------------------------------------------------------ #
#  13. User metrics trends                                             #
# ------------------------------------------------------------------ #


def plot_user_metrics_trends(temporal: dict) -> plt.Figure:
    """Small multiples: 3 subplots for user behaviour metrics."""
    _apply_style()
    user = temporal.get("user_metrics_over_time", {})
    months = user.get("months", [])
    sources = temporal.get("_meta", {}).get("sources", [])
    overlap = temporal.get("activity", {}).get("overlap_period", {})

    metric_info = [
        ("message_length", "Mean Message Length (words)"),
        ("messages_per_convo", "Messages per Conversation"),
        ("question_rate", "Question Rate (fraction)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = np.arange(len(months))

    for ax, (metric_key, title) in zip(axes, metric_info):
        # Shade non-overlapping period
        overlap_start = overlap.get("start", "")
        if overlap_start and overlap_start in months:
            oi = months.index(overlap_start)
            if oi > 0:
                ax.axvspan(-0.5, oi - 0.5, alpha=0.08, color="#888888")

        for s in sources:
            src_data = user.get(s, {})
            values = src_data.get(metric_key, [])
            valid_x = []
            valid_y = []
            for i, v in enumerate(values):
                if v is not None:
                    valid_x.append(i)
                    valid_y.append(v)

            if valid_x:
                ax.plot(
                    valid_x, valid_y,
                    color=_color(s), linewidth=2, marker="o",
                    markersize=4, label=s.capitalize(), alpha=0.85,
                )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "User Metrics Over Time",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "user_metrics_trends")
    return fig


# ------------------------------------------------------------------ #
#  14. Inflection points overlay                                       #
# ------------------------------------------------------------------ #


def plot_inflection_points(temporal: dict) -> plt.Figure:
    """Bot metrics trends with inflection point annotations."""
    _apply_style()
    bot = temporal.get("bot_metrics_over_time", {})
    months = bot.get("months", [])
    sources = temporal.get("_meta", {}).get("sources", [])
    inflections = temporal.get("inflection_points", [])
    overlap = temporal.get("activity", {}).get("overlap_period", {})

    if not inflections:
        # Fall back to plain bot metrics trends
        return plot_bot_metrics_trends(temporal)

    metric_info = [
        ("response_length", "Mean Response Length (words)"),
        ("mattr", "Vocabulary Richness (MATTR-500)"),
        ("readability", "Readability (Flesch-Kincaid)"),
        ("hedge_density", "Hedge Density (per 1k words)"),
        ("formatting_density", "Formatting Density (per 1k words)"),
        ("question_rate", "Question Rate (per turn)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    axes_flat = axes.flatten()

    # Index inflection points by (metric, source)
    inflection_lookup: dict[tuple[str, str], list[dict]] = {}
    for ip in inflections:
        key = (ip["metric"], ip["source"])
        inflection_lookup.setdefault(key, []).append(ip)

    x = np.arange(len(months))

    for ax, (metric_key, title) in zip(axes_flat, metric_info):
        # Shade non-overlapping period
        overlap_start = overlap.get("start", "")
        if overlap_start and overlap_start in months:
            oi = months.index(overlap_start)
            if oi > 0:
                ax.axvspan(-0.5, oi - 0.5, alpha=0.08, color="#888888")

        for s in sources:
            src_data = bot.get(s, {})
            values = src_data.get(metric_key, [])
            valid_x = []
            valid_y = []
            for i, v in enumerate(values):
                if v is not None:
                    valid_x.append(i)
                    valid_y.append(v)

            if valid_x:
                ax.plot(
                    valid_x, valid_y,
                    color=_color(s), linewidth=2, marker="o",
                    markersize=4, label=s.capitalize(), alpha=0.85,
                )

            # Annotate inflection points for this metric+source
            ips = inflection_lookup.get((metric_key, s), [])
            for ip in ips:
                month = ip["month"]
                if month in months:
                    mi = months.index(month)
                    val = ip["value"]
                    direction = ip["direction"]
                    marker = "^" if direction == "increase" else "v"
                    ax.plot(
                        mi, val, marker=marker, color=_color(s),
                        markersize=12, markeredgecolor="black",
                        markeredgewidth=1.5, zorder=5,
                    )
                    ax.annotate(
                        f"z={ip['zscore']:+.1f}",
                        xy=(mi, val),
                        xytext=(5, 10 if direction == "increase" else -15),
                        textcoords="offset points",
                        fontsize=7, fontweight="bold",
                        color=_color(s),
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor="white",
                            edgecolor=_color(s),
                            alpha=0.8,
                        ),
                    )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(months, rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Bot Metrics Over Time — Inflection Points Annotated",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, "inflection_points")
    return fig


# ------------------------------------------------------------------ #
#  15. Depth distribution                                              #
# ------------------------------------------------------------------ #


def plot_depth_distribution(conv_struct: dict) -> plt.Figure:
    """Grouped bar chart of conversation depth categories per source."""
    _apply_style()
    sources = conv_struct.get("_meta", {}).get("sources", [])
    depth = conv_struct.get("depth_classification", {})
    per_source = depth.get("per_source", {})
    categories = depth.get("categories", [])
    resp_by_depth = depth.get("response_length_by_depth", {})

    cat_labels = {
        "quick_exchange": "Quick\n(1-2)",
        "short_session": "Short\n(3-6)",
        "working_session": "Working\n(7-20)",
        "deep_dive": "Deep Dive\n(21-50)",
        "marathon": "Marathon\n(51+)",
    }

    fig, (ax_main, ax_len) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.3, 1]}
    )

    # ---- Left panel: depth counts ---- #
    x = np.arange(len(categories))
    width = 0.35

    for i, s in enumerate(sources):
        vals = [per_source.get(s, {}).get(cat, 0) for cat in categories]
        offset = (i - (len(sources) - 1) / 2) * width
        bars = ax_main.bar(
            x + offset, vals, width * 0.9,
            color=_color(s), label=s.capitalize(),
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax_main.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    str(val), ha="center", va="bottom", fontsize=9,
                )

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(
        [cat_labels.get(c, c) for c in categories], fontsize=10,
    )
    ax_main.set_ylabel("Number of Conversations", fontsize=11)
    ax_main.set_title("Conversation Depth Distribution", fontsize=13, fontweight="bold")
    ax_main.legend(fontsize=10)

    # ---- Right panel: mean response length by depth ---- #
    for i, s in enumerate(sources):
        vals = []
        valid_cats = []
        for ci, cat in enumerate(categories):
            v = resp_by_depth.get(s, {}).get(cat)
            if v is not None:
                vals.append(v)
                valid_cats.append(ci)
        if vals:
            ax_len.plot(
                valid_cats, vals,
                color=_color(s), linewidth=2, marker="o",
                markersize=6, label=s.capitalize(),
            )

    ax_len.set_xticks(range(len(categories)))
    ax_len.set_xticklabels(
        [cat_labels.get(c, c) for c in categories], fontsize=9,
    )
    ax_len.set_ylabel("Mean Response Length (words)", fontsize=11)
    ax_len.set_title("Response Length by Depth", fontsize=13, fontweight="bold")
    ax_len.legend(fontsize=10)
    ax_len.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, "depth_distribution")
    return fig


# ------------------------------------------------------------------ #
#  16. Rephrasing rate                                                 #
# ------------------------------------------------------------------ #


def plot_rephrasing_rate(conv_struct: dict) -> plt.Figure:
    """Bar chart comparing rephrase rates per source."""
    _apply_style()
    sources = conv_struct.get("_meta", {}).get("sources", [])
    reph = conv_struct.get("rephrasing", {})
    per_source = reph.get("per_source", {})
    threshold = reph.get("threshold", 0.6)

    fig, (ax_rate, ax_conv) = plt.subplots(1, 2, figsize=(13, 6))

    # ---- Left: rate per 100 user turns ---- #
    rates = [
        per_source.get(s, {}).get("rate_per_100_user_turns", 0)
        for s in sources
    ]
    totals = [per_source.get(s, {}).get("total", 0) for s in sources]
    bars = ax_rate.bar(
        range(len(sources)), rates,
        color=[_color(s) for s in sources],
        edgecolor="white", linewidth=0.8,
    )
    for bar, rate, total in zip(bars, rates, totals):
        ax_rate.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{rate:.1f}%\n({total} events)",
            ha="center", va="bottom", fontsize=10,
        )
    ax_rate.set_xticks(range(len(sources)))
    ax_rate.set_xticklabels([s.capitalize() for s in sources], fontsize=12)
    ax_rate.set_ylabel("Rephrases per 100 User Turns", fontsize=11)
    ax_rate.set_title("Rephrase Rate", fontsize=13, fontweight="bold")
    ax_rate.set_ylim(0, max(rates) * 1.4 if rates else 1)

    # ---- Right: % conversations with at least one rephrase ---- #
    conv_pcts = []
    for s in sources:
        data = per_source.get(s, {})
        n_with = data.get("conversations_with_rephrase", 0)
        n_total = data.get("n_conversations", 1)
        conv_pcts.append(n_with / max(n_total, 1) * 100)

    bars2 = ax_conv.bar(
        range(len(sources)), conv_pcts,
        color=[_color(s) for s in sources],
        edgecolor="white", linewidth=0.8,
    )
    for bar, pct in zip(bars2, conv_pcts):
        ax_conv.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=11,
        )
    ax_conv.set_xticks(range(len(sources)))
    ax_conv.set_xticklabels([s.capitalize() for s in sources], fontsize=12)
    ax_conv.set_ylabel("% of Conversations", fontsize=11)
    ax_conv.set_title("Conversations with Rephrase", fontsize=13, fontweight="bold")
    ax_conv.set_ylim(0, max(conv_pcts) * 1.4 if conv_pcts else 1)

    fig.suptitle(
        f"Rephrase Detection (cosine similarity ≥ {threshold})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "rephrasing_rate")
    return fig


# ------------------------------------------------------------------ #
#  17. Resolution patterns                                             #
# ------------------------------------------------------------------ #


def plot_resolution_patterns(conv_struct: dict) -> plt.Figure:
    """Grouped bar chart of resolution pattern rates per source."""
    _apply_style()
    sources = conv_struct.get("_meta", {}).get("sources", [])
    resol = conv_struct.get("resolution_patterns", {})
    per_source = resol.get("per_source", {})

    pattern_keys = ["thank_you", "question", "correction", "short_close", "other"]
    pattern_labels = ["Thank You /\nAcknowledge", "Ends with\nQuestion",
                      "Correction /\nPushback", "Short\nClose", "Other"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(pattern_keys))
    width = 0.35

    for i, s in enumerate(sources):
        vals = [
            per_source.get(s, {}).get("rates", {}).get(k, 0) * 100
            for k in pattern_keys
        ]
        offset = (i - (len(sources) - 1) / 2) * width
        bars = ax.bar(
            x + offset, vals, width * 0.9,
            color=_color(s), label=s.capitalize(),
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            if val > 1:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(pattern_labels, fontsize=10)
    ax.set_ylabel("% of Conversations", fontsize=11)
    ax.set_title(
        "How Do Conversations End?",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11)

    fig.tight_layout()
    _save(fig, "resolution_patterns")
    return fig


# ------------------------------------------------------------------ #
#  18. Verbosity trajectory                                            #
# ------------------------------------------------------------------ #


def plot_verbosity_trajectory(
    conv_struct: dict,
    conversations: list | None = None,
) -> plt.Figure:
    """Histogram of verbosity trajectory correlations per source."""
    _apply_style()
    sources = conv_struct.get("_meta", {}).get("sources", [])
    shape = conv_struct.get("shape_metrics", {}).get("per_source", {})

    fig, (ax_bot, ax_user) = plt.subplots(1, 2, figsize=(14, 6))

    # We need the raw trajectory values — recompute from conversations
    bot_trajs: dict[str, list[float]] = {s: [] for s in sources}
    user_trajs: dict[str, list[float]] = {s: [] for s in sources}

    if conversations:
        for conv in conversations:
            turns = conv.turns
            asst_lens = [
                len(t.content.split()) for t in turns
                if t.role == "assistant" and len(t.content) >= 10
            ]
            user_lens = [
                len(t.content.split()) for t in turns
                if t.role == "user" and len(t.content) >= 10
            ]

            # Spearman-like: quick sign of correlation
            if len(asst_lens) >= 3:
                indices = list(range(len(asst_lens)))
                n = len(indices)
                mean_x = sum(indices) / n
                mean_y = sum(asst_lens) / n
                cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(indices, asst_lens))
                std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in indices) / n)
                std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in asst_lens) / n)
                if std_x > 0 and std_y > 0:
                    rho = cov / (n * std_x * std_y)
                    bot_trajs.setdefault(conv.source, []).append(rho)

            if len(user_lens) >= 3:
                indices = list(range(len(user_lens)))
                n = len(indices)
                mean_x = sum(indices) / n
                mean_y = sum(user_lens) / n
                cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(indices, user_lens))
                std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in indices) / n)
                std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in user_lens) / n)
                if std_x > 0 and std_y > 0:
                    rho = cov / (n * std_x * std_y)
                    user_trajs.setdefault(conv.source, []).append(rho)

    # Plot bot verbosity trajectories
    for s in sources:
        data = bot_trajs.get(s, [])
        if data:
            ax_bot.hist(
                data, bins=30, alpha=0.45,
                color=_color(s), label=s.capitalize(),
                edgecolor="white", linewidth=0.3,
            )
            median_val = np.median(data)
            ax_bot.axvline(
                median_val, color=_color(s), linestyle="--", linewidth=1.5,
            )

    ax_bot.axvline(0, color="#888888", linestyle="-", linewidth=0.8, alpha=0.5)
    ax_bot.set_xlabel("Correlation (negative=tapering, positive=escalating)", fontsize=10)
    ax_bot.set_ylabel("Conversations", fontsize=10)
    ax_bot.set_title("Bot Verbosity Trajectory", fontsize=12, fontweight="bold")
    ax_bot.legend(fontsize=10)

    # Plot user effort trajectories
    for s in sources:
        data = user_trajs.get(s, [])
        if data:
            ax_user.hist(
                data, bins=30, alpha=0.45,
                color=_color(s), label=s.capitalize(),
                edgecolor="white", linewidth=0.3,
            )
            median_val = np.median(data)
            ax_user.axvline(
                median_val, color=_color(s), linestyle="--", linewidth=1.5,
            )

    ax_user.axvline(0, color="#888888", linestyle="-", linewidth=0.8, alpha=0.5)
    ax_user.set_xlabel("Correlation (negative=less effort, positive=more effort)", fontsize=10)
    ax_user.set_ylabel("Conversations", fontsize=10)
    ax_user.set_title("User Effort Trajectory", fontsize=12, fontweight="bold")
    ax_user.legend(fontsize=10)

    fig.suptitle(
        "Verbosity & Effort Trajectories Within Conversations",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "verbosity_trajectory")
    return fig


# ------------------------------------------------------------------ #
#  User Behavior: Prompt Engineering Patterns                          #
# ------------------------------------------------------------------ #


def plot_prompt_engineering_patterns(ub: dict) -> plt.Figure:
    """Grouped bar chart of prompt engineering density by category per source."""
    _apply_style()
    sources = ub.get("_meta", {}).get("sources", [])
    pe = ub.get("prompt_engineering", {})
    by_source = pe.get("by_source", {})

    # Collect categories (exclude very noisy 'constraints')
    all_cats = set()
    for src_data in by_source.values():
        all_cats.update(src_data.get("category_per_1k", {}).keys())
    # Sort for consistent order, put constraints last
    cats = sorted(all_cats - {"constraints"}) + (["constraints"] if "constraints" in all_cats else [])

    cat_labels = {
        "step_by_step": "Step-by-step",
        "conciseness": "Conciseness",
        "simplification": "Simplification",
        "detail": "Detail/Elaborate",
        "formatting": "Formatting",
        "examples": "Examples",
        "role_persona": "Role/Persona",
        "constraints": "Constraints",
        "context_providing": "Context Providing",
    }

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(cats))
    width = 0.8 / max(len(sources), 1)

    for i, src in enumerate(sources):
        vals = [
            by_source.get(src, {}).get("category_per_1k", {}).get(c, 0)
            for c in cats
        ]
        offset = (i - len(sources) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=src.capitalize(),
            color=_color(src),
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            if val > 0.1:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [cat_labels.get(c, c) for c in cats],
        rotation=35, ha="right", fontsize=10,
    )
    ax.set_ylabel("Occurrences per 1,000 words", fontsize=11)
    ax.set_title(
        "Prompt Engineering Patterns by Platform",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=11, loc="upper right")

    # Add total density annotation
    totals = pe.get("total_density", {})
    if totals:
        total_str = "  |  ".join(
            f"{s.capitalize()}: {totals.get(s, 0):.1f}/1k" for s in sources
        )
        ax.annotate(
            f"Total density: {total_str}",
            xy=(0.5, -0.18), xycoords="axes fraction",
            ha="center", fontsize=10, style="italic",
            color="#555555",
        )

    fig.tight_layout()
    _save(fig, "prompt_engineering_patterns")
    return fig


# ------------------------------------------------------------------ #
#  User Behavior: Formality Comparison                                 #
# ------------------------------------------------------------------ #


def plot_user_formality(ub: dict) -> plt.Figure:
    """Side-by-side comparison of politeness, casual markers, and framing."""
    _apply_style()
    sources = ub.get("_meta", {}).get("sources", [])
    form = ub.get("formality", {})

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Panel 1: Politeness rate (per 1k words)
    ax = axes[0]
    pol = form.get("politeness", {})
    vals = [pol.get(s, {}).get("total_per_1k", 0) for s in sources]
    bars = ax.bar(
        range(len(sources)), vals, width=0.5,
        color=[_color(s) for s in sources],
        edgecolor="white", linewidth=0.8,
    )
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([s.capitalize() for s in sources], fontsize=11)
    ax.set_ylabel("Per 1,000 words", fontsize=10)
    ax.set_title("Politeness Markers", fontsize=12, fontweight="bold")

    # Panel 2: Casual marker rates
    ax = axes[1]
    cas = form.get("casual_markers", {})
    # Check if lowercase metric is unreliable for any source
    has_unreliable_lc = any(
        not cas.get(s, {}).get("lowercase_only_reliable_for_comparison", True)
        for s in sources
    )
    metrics = {
        "Contractions": [cas.get(s, {}).get("contractions_per_1k", 0) for s in sources],
        "Lowercase msgs*" if has_unreliable_lc else "Lowercase msgs": [
            cas.get(s, {}).get("lowercase_only_fraction", 0) * 100 for s in sources
        ],
        "No punctuation": [cas.get(s, {}).get("no_punctuation_fraction", 0) * 100 for s in sources],
    }
    x = np.arange(len(metrics))
    width = 0.8 / max(len(sources), 1)
    for i, src in enumerate(sources):
        vals = [metrics[m][i] for m in metrics]
        offset = (i - len(sources) / 2 + 0.5) * width
        ax.bar(
            x + offset, vals, width,
            label=src.capitalize(),
            color=_color(src),
            edgecolor="white", linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()), fontsize=10)
    ax.set_ylabel("Rate (per 1k / %)", fontsize=10)
    ax.set_title("Casual Markers", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    if has_unreliable_lc:
        ax.annotate(
            "*Gemini Takeout auto-capitalizes; lowercase gap is artificial",
            xy=(0.5, -0.15), xycoords="axes fraction",
            ha="center", fontsize=7.5, style="italic", color="#b91c1c",
        )

    # Panel 3: Imperative vs Request framing
    ax = axes[2]
    frm = form.get("imperative_vs_request", {})
    categories = ["imperative_fraction", "request_fraction", "other_fraction"]
    cat_labels = ["Imperative", "Request", "Other"]
    x = np.arange(len(categories))
    width = 0.8 / max(len(sources), 1)
    for i, src in enumerate(sources):
        vals = [frm.get(src, {}).get(c, 0) * 100 for c in categories]
        offset = (i - len(sources) / 2 + 0.5) * width
        ax.bar(
            x + offset, vals, width,
            label=src.capitalize(),
            color=_color(src),
            edgecolor="white", linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylabel("% of messages", fontsize=10)
    ax.set_title("Message Framing", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle(
        "User Formality & Tone by Platform",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "user_formality")
    return fig


# ------------------------------------------------------------------ #
#  User Behavior: Rephrasing Rate                                      #
# ------------------------------------------------------------------ #


def plot_user_rephrasing_rate(ub: dict) -> plt.Figure:
    """Bar chart of rephrase rate per source with annotation."""
    _apply_style()
    sources = ub.get("_meta", {}).get("sources", [])
    reph = ub.get("rephrasing", {})

    if reph.get("error"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, f"Rephrasing skipped: {reph['error']}",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_axis_off()
        _save(fig, "user_rephrasing_rate")
        return fig

    reph_src = reph.get("by_source", {})
    thresholds = reph.get("thresholds", {})
    strict_t = thresholds.get("strict", 0.85)
    loose_t = thresholds.get("loose", 0.70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Strict per 100 user turns (primary metric)
    ax = axes[0]
    rates_strict = [reph_src.get(s, {}).get("strict_per_100_turns", 0) for s in sources]
    totals_strict = [reph_src.get(s, {}).get("strict_total", 0) for s in sources]
    user_turns = [reph_src.get(s, {}).get("total_user_turns", 0) for s in sources]

    bars = ax.bar(
        range(len(sources)), rates_strict, width=0.5,
        color=[_color(s) for s in sources],
        edgecolor="white", linewidth=0.8,
    )
    for i, (bar, total, ut) in enumerate(zip(bars, totals_strict, user_turns)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{total} hits\n({ut} turns)",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([s.capitalize() for s in sources], fontsize=12)
    ax.set_ylabel("Rephrases per 100 User Turns", fontsize=11)
    ax.set_title(
        f"Strict Rephrasing (sim > {strict_t}, \u226510 words)",
        fontsize=12, fontweight="bold", pad=10,
    )

    # Panel 2: Loose per 100 user turns (secondary)
    ax = axes[1]
    rates_loose = [reph_src.get(s, {}).get("loose_per_100_turns", 0) for s in sources]
    totals_loose = [reph_src.get(s, {}).get("loose_total", 0) for s in sources]

    bars = ax.bar(
        range(len(sources)), rates_loose, width=0.5,
        color=[_color(s) for s in sources],
        edgecolor="white", linewidth=0.8, alpha=0.7,
    )
    for i, (bar, total, ut) in enumerate(zip(bars, totals_loose, user_turns)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{total} hits",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([s.capitalize() for s in sources], fontsize=12)
    ax.set_ylabel("Rephrases per 100 User Turns", fontsize=11)
    ax.set_title(
        f"Loose Rephrasing (sim > {loose_t}, no word filter)",
        fontsize=12, fontweight="bold", pad=10,
    )

    fig.suptitle(
        "User Rephrasing Rate by Platform",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "user_rephrasing_rate")
    return fig


# ------------------------------------------------------------------ #
#  LLM Judge plots                                                     #
# ------------------------------------------------------------------ #


def _judge_stacked_bar(
    results: dict, dimension: str, title: str, filename: str,
) -> plt.Figure:
    """Shared helper for depth / creativity stacked bar charts."""
    _apply_style()

    dim_data = results.get(dimension, {})
    by_source = dim_data.get("by_source", {})
    sources = sorted(by_source.keys())
    if not sources:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        _save(fig, filename)
        return fig

    # Build stacked bars.
    scores_1 = []
    scores_2 = []
    scores_3 = []
    means = []
    for s in sources:
        dist = by_source[s].get("distribution", {})
        total = sum(int(dist.get(str(k), 0)) for k in (1, 2, 3))
        if total == 0:
            scores_1.append(0)
            scores_2.append(0)
            scores_3.append(0)
        else:
            scores_1.append(int(dist.get("1", 0)) / total * 100)
            scores_2.append(int(dist.get("2", 0)) / total * 100)
            scores_3.append(int(dist.get("3", 0)) / total * 100)
        means.append(by_source[s].get("mean", 0))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sources))
    width = 0.5

    # Use a gradient from light (score 1) to dark (score 3).
    colors_map = {
        "depth": ["#e0e7ff", "#818cf8", "#4338ca"],
        "creativity": ["#fef3c7", "#fbbf24", "#d97706"],
    }
    c = colors_map.get(dimension, ["#d1d5db", "#6b7280", "#1f2937"])

    bars1 = ax.bar(x, scores_1, width, label="1 — Low", color=c[0], edgecolor="white")
    bars2 = ax.bar(x, scores_2, width, bottom=scores_1, label="2 — Moderate", color=c[1], edgecolor="white")
    bottoms3 = [a + b for a, b in zip(scores_1, scores_2)]
    bars3 = ax.bar(x, scores_3, width, bottom=bottoms3, label="3 — High", color=c[2], edgecolor="white")

    # Annotate mean scores.
    for i, mean in enumerate(means):
        if mean:
            ax.text(
                x[i], 102, f"mean: {mean:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    ax.set_ylabel("% of responses")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in sources], fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Caveat footnote.
    fig.text(
        0.5, -0.02,
        "Note: Claude is judging its own responses — potential self-bias.",
        ha="center", fontsize=8, style="italic", color="#666",
    )

    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_judge_depth(results: dict) -> plt.Figure:
    """Stacked bar chart of intellectual depth scores per source."""
    return _judge_stacked_bar(
        results, "depth",
        "Intellectual Depth — LLM Judge Scores",
        "judge_depth",
    )


def plot_judge_creativity(results: dict) -> plt.Figure:
    """Stacked bar chart of creativity scores per source."""
    return _judge_stacked_bar(
        results, "creativity",
        "Creativity — LLM Judge Scores",
        "judge_creativity",
    )


def plot_judge_combined(results: dict) -> plt.Figure:
    """Grouped bar chart: mean depth + creativity per source."""
    _apply_style()

    depth_by = results.get("depth", {}).get("by_source", {})
    creativity_by = results.get("creativity", {}).get("by_source", {})
    sources = sorted(set(depth_by.keys()) | set(creativity_by.keys()))

    if not sources:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        _save(fig, "judge_combined")
        return fig

    depth_means = [depth_by.get(s, {}).get("mean", 0) or 0 for s in sources]
    creativity_means = [creativity_by.get(s, {}).get("mean", 0) or 0 for s in sources]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sources))
    width = 0.3

    bars1 = ax.bar(
        x - width / 2, depth_means, width,
        label="Intellectual Depth", color="#6366f1", alpha=0.85,
    )
    bars2 = ax.bar(
        x + width / 2, creativity_means, width,
        label="Creativity", color="#f59e0b", alpha=0.85,
    )

    # Annotate values.
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            if h:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.03,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9,
                )

    ax.set_ylabel("Mean Score (1-3)")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in sources], fontsize=11)
    ax.set_ylim(0, 3.5)
    ax.legend(fontsize=10)
    ax.set_title(
        "LLM Judge — Depth & Creativity by Platform",
        fontsize=13, fontweight="bold", pad=12,
    )

    fig.text(
        0.5, -0.02,
        "Note: Claude is judging its own responses — potential self-bias.",
        ha="center", fontsize=8, style="italic", color="#666",
    )

    fig.tight_layout()
    _save(fig, "judge_combined")
    return fig


# ------------------------------------------------------------------ #
#  Public registry for CLI                                             #
# ------------------------------------------------------------------ #


PLOT_FUNCTIONS: dict[str, dict[str, Any]] = {
    "vocabulary_comparison": {
        "fn": plot_vocabulary_comparison,
        "data": "lexical",
        "needs_conversations": False,
    },
    "distinctive_words": {
        "fn": plot_distinctive_words,
        "data": "lexical",
        "needs_conversations": False,
    },
    "response_length_distributions": {
        "fn": plot_response_length_distributions,
        "data": "lexical",
        "needs_conversations": True,
    },
    "formatting_habits": {
        "fn": plot_formatting_habits,
        "data": "lexical",
        "needs_conversations": False,
    },
    "topic_distribution": {
        "fn": plot_topic_distribution,
        "data": "semantic",
        "needs_conversations": False,
    },
    "hedging_comparison": {
        "fn": plot_hedging_comparison,
        "data": "pragmatic",
        "needs_conversations": False,
    },
    "style_fingerprint": {
        "fn": plot_style_fingerprint,
        "data": "comparative",
        "needs_conversations": False,
    },
    "verbosity_ratio": {
        "fn": plot_verbosity_ratio,
        "data": "pragmatic",
        "needs_conversations": True,
    },
    "user_language_comparison": {
        "fn": plot_user_language_comparison,
        "data": "comparative",
        "needs_conversations": False,
    },
    "opening_patterns": {
        "fn": plot_opening_patterns,
        "data": "pragmatic",
        "needs_conversations": False,
    },
    "activity_over_time": {
        "fn": plot_activity_over_time,
        "data": "temporal",
        "needs_conversations": False,
    },
    "bot_metrics_trends": {
        "fn": plot_bot_metrics_trends,
        "data": "temporal",
        "needs_conversations": False,
    },
    "user_metrics_trends": {
        "fn": plot_user_metrics_trends,
        "data": "temporal",
        "needs_conversations": False,
    },
    "inflection_points": {
        "fn": plot_inflection_points,
        "data": "temporal",
        "needs_conversations": False,
    },
    "depth_distribution": {
        "fn": plot_depth_distribution,
        "data": "conversation_structure",
        "needs_conversations": False,
    },
    "rephrasing_rate": {
        "fn": plot_rephrasing_rate,
        "data": "conversation_structure",
        "needs_conversations": False,
    },
    "resolution_patterns": {
        "fn": plot_resolution_patterns,
        "data": "conversation_structure",
        "needs_conversations": False,
    },
    "verbosity_trajectory": {
        "fn": plot_verbosity_trajectory,
        "data": "conversation_structure",
        "needs_conversations": True,
    },
    "prompt_engineering_patterns": {
        "fn": plot_prompt_engineering_patterns,
        "data": "user_behavior",
        "needs_conversations": False,
    },
    "user_formality": {
        "fn": plot_user_formality,
        "data": "user_behavior",
        "needs_conversations": False,
    },
    "user_rephrasing_rate": {
        "fn": plot_user_rephrasing_rate,
        "data": "user_behavior",
        "needs_conversations": False,
    },
    "judge_depth": {
        "fn": plot_judge_depth,
        "data": "llm_judge",
        "needs_conversations": False,
    },
    "judge_creativity": {
        "fn": plot_judge_creativity,
        "data": "llm_judge",
        "needs_conversations": False,
    },
    "judge_combined": {
        "fn": plot_judge_combined,
        "data": "llm_judge",
        "needs_conversations": False,
    },
}

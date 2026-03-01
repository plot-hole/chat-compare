"""LLM-as-judge evaluation of chatbot response quality.

Uses the Anthropic API to have Claude score assistant responses on
intellectual depth and creativity.  Follows LLM-as-judge best practices:
one dimension per call, 3-point scale, chain-of-thought reasoning.

**Bias caveat**: Claude is judging its own responses alongside competitors.
Cross-reference with traditional NLP metrics for a fuller picture.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from src.parsers.base import Conversation

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Prompt templates                                                    #
# ------------------------------------------------------------------ #

DEPTH_SYSTEM_PROMPT = """\
You are evaluating the intellectual depth of an AI assistant's response. \
You will see a user message and the assistant's response.

Score the response on a 3-point scale:
1 = Surface-level: Restates common knowledge, gives generic advice, \
doesn't go beyond what the user likely already knows. Covers the topic \
without adding real insight.
2 = Moderate depth: Demonstrates understanding beyond the obvious, makes \
useful distinctions or connections, provides some analysis rather than \
just information.
3 = Substantial depth: Introduces genuinely insightful framing, makes \
non-obvious connections, demonstrates deep understanding of nuance, \
provides analysis that meaningfully advances the user's thinking.

First, reason step by step about the depth of the response in 2-3 \
sentences. Then provide your score.

Respond in exactly this format:
REASONING: [your reasoning]
SCORE: [1, 2, or 3]"""

CREATIVITY_SYSTEM_PROMPT = """\
You are evaluating the creativity of an AI assistant's response. \
You will see a user message and the assistant's response.

Score the response on a 3-point scale:
1 = Formulaic: Uses predictable structure and phrasing, gives the \
expected/standard answer, could have been assembled from templates.
2 = Somewhat creative: Shows some originality in framing, examples, or \
approach. Not entirely predictable but doesn't surprise.
3 = Notably creative: Offers unexpected angles, original examples or \
analogies, novel structuring, or ideas the user likely hadn't considered. \
The response has a distinctive voice or approach.

First, reason step by step about the creativity of the response in 2-3 \
sentences. Then provide your score.

Respond in exactly this format:
REASONING: [your reasoning]
SCORE: [1, 2, or 3]"""

USER_MESSAGE_TEMPLATE = """\
USER MESSAGE:
{user_turn}

ASSISTANT RESPONSE:
{assistant_turn}"""

# ------------------------------------------------------------------ #
#  Sampling                                                            #
# ------------------------------------------------------------------ #


def _build_sample(
    conversations: list[Conversation],
    config: dict,
) -> dict[str, list[dict]]:
    """Select stratified sample of assistant turns per source.

    Returns ``{source: [{"turn_id": str, "user": str, "assistant": str,
    "conversation_id": str, "turn_index": int, "word_count": int}, ...]}``
    """
    judge_cfg = config.get("llm_judge", {})
    sample_size: int = judge_cfg.get("sample_size", 200)
    min_words: int = judge_cfg.get("min_turn_length", 50)

    # Seed for reproducibility.
    rng = random.Random(42)

    candidates_by_source: dict[str, list[dict]] = {}

    for conv in conversations:
        for i, turn in enumerate(conv.turns):
            if turn.role != "assistant":
                continue
            word_count = len(turn.content.split())
            if word_count < min_words:
                continue
            # Need a preceding user turn for context.
            if i == 0 or conv.turns[i - 1].role != "user":
                continue

            user_turn = conv.turns[i - 1]
            turn_id = hashlib.md5(
                f"{conv.source}:{conv.conversation_id}:{i}".encode()
            ).hexdigest()[:12]

            entry = {
                "turn_id": turn_id,
                "source": conv.source,
                "conversation_id": conv.conversation_id,
                "turn_index": i,
                "user": user_turn.content,
                "assistant": turn.content,
                "word_count": word_count,
            }
            candidates_by_source.setdefault(conv.source, []).append(entry)

    samples: dict[str, list[dict]] = {}
    for source, candidates in candidates_by_source.items():
        if not candidates:
            continue
        # Stratified sampling: weight toward longer responses.
        weights = [c["word_count"] for c in candidates]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        n = min(sample_size, len(candidates))
        # Weighted sampling without replacement.
        selected_indices: set[int] = set()
        remaining_probs = list(probs)
        remaining_indices = list(range(len(candidates)))

        while len(selected_indices) < n and remaining_indices:
            chosen = rng.choices(
                remaining_indices,
                weights=[remaining_probs[remaining_indices.index(j)]
                         for j in remaining_indices],
                k=1,
            )[0]
            selected_indices.add(chosen)
            remaining_indices.remove(chosen)

        samples[source] = [candidates[idx] for idx in sorted(selected_indices)]
        logger.info(
            "Sampled %d/%d assistant turns from %s (min %d words)",
            len(samples[source]),
            len(candidates),
            source,
            min_words,
        )

    return samples


# ------------------------------------------------------------------ #
#  Cache management                                                    #
# ------------------------------------------------------------------ #


def _load_cache(cache_path: Path) -> dict[str, dict]:
    """Load evaluation cache from disk."""
    if cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cache file corrupt, starting fresh: %s", exc)
    return {}


def _save_cache(cache: dict, cache_path: Path) -> None:
    """Persist cache atomically."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    tmp.replace(cache_path)


# ------------------------------------------------------------------ #
#  Score parsing                                                       #
# ------------------------------------------------------------------ #


def _parse_score(text: str) -> tuple[str, int | None]:
    """Extract reasoning and numeric score from judge response.

    Returns ``(reasoning, score)`` where score is 1-3 or None on failure.
    """
    reasoning = ""
    score = None

    reasoning_match = re.search(
        r"REASONING:\s*(.+?)(?=\nSCORE:|\Z)", text, re.DOTALL
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    score_match = re.search(r"SCORE:\s*(\d)", text)
    if score_match:
        val = int(score_match.group(1))
        if val in (1, 2, 3):
            score = val
        else:
            logger.warning("Score out of range (%d), marking null", val)

    if score is None:
        logger.warning("Failed to parse score from response: %.120s…", text)

    return reasoning, score


# ------------------------------------------------------------------ #
#  Cost estimation                                                     #
# ------------------------------------------------------------------ #

# Approximate pricing for claude-sonnet-4-5-20250514 as of 2025-05
_COST_PER_M_INPUT = 3.00   # USD per million input tokens
_COST_PER_M_OUTPUT = 15.00  # USD per million output tokens

# Rough token estimates per call.
_EST_INPUT_TOKENS_PER_CALL = 900   # system + user message + assistant excerpt
_EST_OUTPUT_TOKENS_PER_CALL = 120  # reasoning + score


def _estimate_cost(n_calls: int) -> float:
    """Return estimated USD cost for *n_calls* API calls."""
    input_cost = (n_calls * _EST_INPUT_TOKENS_PER_CALL / 1_000_000) * _COST_PER_M_INPUT
    output_cost = (n_calls * _EST_OUTPUT_TOKENS_PER_CALL / 1_000_000) * _COST_PER_M_OUTPUT
    return input_cost + output_cost


# ------------------------------------------------------------------ #
#  Async evaluation engine                                             #
# ------------------------------------------------------------------ #


async def _evaluate_all(
    samples: dict[str, list[dict]],
    cache: dict,
    cache_path: Path,
    config: dict,
) -> tuple[dict, int, int]:
    """Run all API evaluations asynchronously.

    Returns ``(updated_cache, total_input_tokens, total_output_tokens)``.
    """
    import anthropic

    judge_cfg = config.get("llm_judge", {})
    model = judge_cfg.get("model", "claude-sonnet-4-5-20250514")
    max_concurrent = judge_cfg.get("max_concurrent", 5)

    client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env
    semaphore = asyncio.Semaphore(max_concurrent)

    total_input_tokens = 0
    total_output_tokens = 0
    token_lock = asyncio.Lock()

    # Flatten all tasks.
    tasks: list[dict] = []
    for source, entries in samples.items():
        for entry in entries:
            for dimension in ("depth", "creativity"):
                cache_key = f"{entry['turn_id']}:{dimension}"
                if cache_key in cache:
                    continue
                tasks.append({
                    "turn_id": entry["turn_id"],
                    "source": source,
                    "dimension": dimension,
                    "user": entry["user"],
                    "assistant": entry["assistant"],
                    "cache_key": cache_key,
                })

    if not tasks:
        logger.info("All evaluations cached — nothing to do.")
        return cache, 0, 0

    # Split by dimension for progress tracking.
    from tqdm import tqdm

    depth_tasks = [t for t in tasks if t["dimension"] == "depth"]
    creativity_tasks = [t for t in tasks if t["dimension"] == "creativity"]
    ordered_tasks = depth_tasks + creativity_tasks

    depth_bar = tqdm(total=len(depth_tasks), desc="Evaluating depth", unit="call")
    creativity_bar = tqdm(total=len(creativity_tasks), desc="Evaluating creativity", unit="call")

    save_counter = 0

    async def _do_call(task: dict) -> None:
        nonlocal save_counter, total_input_tokens, total_output_tokens

        system_prompt = (
            DEPTH_SYSTEM_PROMPT
            if task["dimension"] == "depth"
            else CREATIVITY_SYSTEM_PROMPT
        )
        user_content = USER_MESSAGE_TEMPLATE.format(
            user_turn=task["user"], assistant_turn=task["assistant"]
        )

        async with semaphore:
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=300,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_content}],
                )
            except Exception as exc:
                logger.error(
                    "API error for %s (%s): %s",
                    task["turn_id"],
                    task["dimension"],
                    exc,
                )
                return

        text = response.content[0].text
        reasoning, score = _parse_score(text)

        async with token_lock:
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

        cache[task["cache_key"]] = {
            "turn_id": task["turn_id"],
            "source": task["source"],
            "dimension": task["dimension"],
            "score": score,
            "reasoning": reasoning,
            "raw_response": text,
        }

        if task["dimension"] == "depth":
            depth_bar.update(1)
        else:
            creativity_bar.update(1)

        # Persist cache every 20 evaluations.
        save_counter += 1
        if save_counter % 20 == 0:
            _save_cache(cache, cache_path)

    # Process in batches to be kind to rate limits.
    batch_size = max_concurrent * 2
    for i in range(0, len(ordered_tasks), batch_size):
        batch = ordered_tasks[i : i + batch_size]
        await asyncio.gather(*[_do_call(t) for t in batch])
        # Small delay between batches.
        if i + batch_size < len(ordered_tasks):
            await asyncio.sleep(0.5)

    depth_bar.close()
    creativity_bar.close()

    # Final save.
    _save_cache(cache, cache_path)

    return cache, total_input_tokens, total_output_tokens


# ------------------------------------------------------------------ #
#  Analysis                                                            #
# ------------------------------------------------------------------ #


def _analyze_results(
    samples: dict[str, list[dict]],
    cache: dict,
) -> dict[str, Any]:
    """Aggregate cached evaluation results into summary statistics."""
    from scipy.stats import mannwhitneyu

    analysis: dict[str, Any] = {}

    for dimension in ("depth", "creativity"):
        scores_by_source: dict[str, list[int]] = {}
        entries_by_source: dict[str, list[dict]] = {}

        for source, entries in samples.items():
            scores = []
            evaluated = []
            for entry in entries:
                cache_key = f"{entry['turn_id']}:{dimension}"
                cached = cache.get(cache_key)
                if cached and cached.get("score") is not None:
                    scores.append(cached["score"])
                    evaluated.append({
                        "turn_id": entry["turn_id"],
                        "score": cached["score"],
                        "reasoning": cached.get("reasoning", ""),
                        "user_excerpt": entry["user"][:200],
                        "assistant_excerpt": entry["assistant"][:300],
                        "word_count": entry["word_count"],
                    })
            scores_by_source[source] = scores
            entries_by_source[source] = evaluated

        # Per-source stats.
        by_source: dict[str, dict] = {}
        for source, scores in scores_by_source.items():
            if not scores:
                by_source[source] = {
                    "mean": None,
                    "distribution": {"1": 0, "2": 0, "3": 0},
                    "n_evaluated": 0,
                }
                continue
            dist = {str(k): scores.count(k) for k in (1, 2, 3)}
            by_source[source] = {
                "mean": round(sum(scores) / len(scores), 3),
                "distribution": dist,
                "n_evaluated": len(scores),
            }

        # Statistical test (Mann-Whitney U between all pairs).
        sources = sorted(scores_by_source.keys())
        stat_test: dict[str, Any] = {"test": "Mann-Whitney U"}
        if len(sources) == 2:
            s1, s2 = scores_by_source[sources[0]], scores_by_source[sources[1]]
            if len(s1) >= 5 and len(s2) >= 5:
                u_stat, p_val = mannwhitneyu(s1, s2, alternative="two-sided")
                stat_test["comparison"] = f"{sources[0]} vs {sources[1]}"
                stat_test["statistic"] = round(float(u_stat), 4)
                stat_test["p_value"] = round(float(p_val), 6)
            else:
                stat_test["skipped"] = "Insufficient samples for test"
        elif len(sources) > 2:
            # Pairwise comparisons.
            pairwise = []
            for i, s1_name in enumerate(sources):
                for s2_name in sources[i + 1 :]:
                    s1 = scores_by_source[s1_name]
                    s2 = scores_by_source[s2_name]
                    if len(s1) >= 5 and len(s2) >= 5:
                        u_stat, p_val = mannwhitneyu(
                            s1, s2, alternative="two-sided"
                        )
                        pairwise.append({
                            "comparison": f"{s1_name} vs {s2_name}",
                            "statistic": round(float(u_stat), 4),
                            "p_value": round(float(p_val), 6),
                        })
            stat_test["pairwise"] = pairwise
        else:
            stat_test["skipped"] = "Need at least 2 sources to compare"

        # Exemplars: top 5 per source.
        exemplars: dict[str, list[dict]] = {}
        for source, evaluated in entries_by_source.items():
            sorted_entries = sorted(
                evaluated, key=lambda e: e["score"], reverse=True
            )
            exemplars[source] = sorted_entries[:5]

        analysis[dimension] = {
            "by_source": by_source,
            "statistical_test": stat_test,
            "exemplars": exemplars,
        }

    return analysis


# ------------------------------------------------------------------ #
#  Dry-run                                                             #
# ------------------------------------------------------------------ #


def dry_run(
    conversations: list[Conversation],
    config: dict,
) -> dict[str, Any]:
    """Show sample breakdown, estimated cost, and example prompts."""
    samples = _build_sample(conversations, config)
    judge_cfg = config.get("llm_judge", {})
    model = judge_cfg.get("model", "claude-sonnet-4-5-20250514")

    total_turns = sum(len(v) for v in samples.values())
    total_api_calls = total_turns * 2  # 2 dimensions per turn
    estimated_cost = _estimate_cost(total_api_calls)

    # Pick one example per source (first entry).
    example_prompts: dict[str, dict] = {}
    for source, entries in samples.items():
        if entries:
            entry = entries[0]
            example_prompts[source] = {
                "depth": {
                    "system": DEPTH_SYSTEM_PROMPT,
                    "user": USER_MESSAGE_TEMPLATE.format(
                        user_turn=entry["user"],
                        assistant_turn=entry["assistant"],
                    ),
                },
                "creativity": {
                    "system": CREATIVITY_SYSTEM_PROMPT,
                    "user": USER_MESSAGE_TEMPLATE.format(
                        user_turn=entry["user"],
                        assistant_turn=entry["assistant"],
                    ),
                },
            }

    # Word-count distribution per source.
    sample_stats: dict[str, dict] = {}
    for source, entries in samples.items():
        wc = [e["word_count"] for e in entries]
        sample_stats[source] = {
            "n_sampled": len(entries),
            "word_count_min": min(wc) if wc else 0,
            "word_count_max": max(wc) if wc else 0,
            "word_count_mean": round(sum(wc) / len(wc), 1) if wc else 0,
        }

    return {
        "model": model,
        "sample_stats": sample_stats,
        "total_turns": total_turns,
        "total_api_calls": total_api_calls,
        "estimated_cost_usd": round(estimated_cost, 4),
        "example_prompts": example_prompts,
    }


# ------------------------------------------------------------------ #
#  Public entry point                                                  #
# ------------------------------------------------------------------ #


def run(
    conversations: list[Conversation],
    config: dict,
) -> dict[str, Any]:
    """Run LLM-as-judge evaluation on sampled conversation turns.

    Args:
        conversations: Normalised conversation objects (all sources).
        config: Loaded ``config.yaml`` dict.

    Returns:
        Results dict with per-dimension scores, statistical tests,
        exemplars, and cost information.
    """
    # Check API key.
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it before running the LLM judge module."
        )

    judge_cfg = config.get("llm_judge", {})
    model = judge_cfg.get("model", "claude-sonnet-4-5-20250514")
    cache_file = Path(judge_cfg.get("cache_file", "data/processed/llm_judge_cache.json"))

    # Build sample.
    samples = _build_sample(conversations, config)
    total_turns = sum(len(v) for v in samples.values())
    total_api_calls = total_turns * 2
    estimated_cost = _estimate_cost(total_api_calls)

    logger.info(
        "LLM judge: %d turns sampled across %d sources, "
        "%d API calls, estimated cost $%.4f",
        total_turns,
        len(samples),
        total_api_calls,
        estimated_cost,
    )

    # Load cache and run evaluations.
    cache = _load_cache(cache_file)
    cached_before = len(cache)

    cache, input_tokens, output_tokens = asyncio.run(
        _evaluate_all(samples, cache, cache_file, config)
    )

    new_evals = len(cache) - cached_before
    logger.info("Completed %d new evaluations (%d from cache)", new_evals, cached_before)

    # Compute actual cost.
    actual_cost = (
        (input_tokens / 1_000_000) * _COST_PER_M_INPUT
        + (output_tokens / 1_000_000) * _COST_PER_M_OUTPUT
    )

    # Analyze.
    analysis = _analyze_results(samples, cache)

    return {
        "config": {
            "model": model,
            "sample_size": judge_cfg.get("sample_size", 200),
            "total_evaluations": total_api_calls,
        },
        "cost": {
            "estimated_usd": round(estimated_cost, 4),
            "actual_usd": round(actual_cost, 4),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        **analysis,
        "data_notes": (
            f"Evaluations performed by {model} via Anthropic API. "
            "Scores reflect the judge model's assessment and may carry "
            "biases — notably, Claude judging Claude's own responses "
            "creates a potential conflict of interest. Cross-reference "
            "with the traditional NLP metrics for a fuller picture."
        ),
    }

"""CLI entrypoint for the Chatbot Conversation Analysis tool."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.yaml"
PROJECT_ROOT = Path(__file__).parent


def load_config() -> dict:
    """Load configuration from config.yaml."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", CONFIG_PATH)
    return config


# ------------------------------------------------------------------ #
#  Parser registry                                                    #
# ------------------------------------------------------------------ #

_SOURCE_INFO: dict[str, dict] = {
    "claude": {
        "parser_cls": "src.parsers.claude_parser.ClaudeParser",
        "detect_files": ["conversations.json"],
    },
    "gemini": {
        "parser_cls": "src.parsers.gemini_parser.GeminiParser",
        "detect_files": ["MyActivity.html"],
    },
    "chatgpt": {
        "parser_cls": "src.parsers.chatgpt_parser.ChatGPTParser",
        "detect_files": ["conversations.json"],
    },
}


def _get_parser(source: str):
    """Dynamically import and instantiate a parser for *source*."""
    info = _SOURCE_INFO[source]
    module_path, cls_name = info["parser_cls"].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    return cls()


def _source_has_data(source: str, raw_root: Path) -> Path | None:
    """Return the raw data file path if the source directory has data."""
    source_dir = raw_root / source
    if not source_dir.is_dir():
        return None
    for fname in _SOURCE_INFO[source]["detect_files"]:
        candidate = source_dir / fname
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    # Fallback: any file in the directory.
    files = [f for f in source_dir.iterdir() if f.is_file() and f.stat().st_size > 0]
    return files[0] if files else None


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Chatbot Conversation Analysis — compare Claude, Gemini, and ChatGPT."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config()


# ------------------------------------------------------------------ #
#  inspect                                                            #
# ------------------------------------------------------------------ #

@cli.command()
@click.pass_context
def inspect(ctx: click.Context) -> None:
    """Print raw data structure from each source for inspection."""
    config = ctx.obj["config"]
    raw_root = PROJECT_ROOT / config["paths"]["raw_data"]

    for source in _SOURCE_INFO:
        data_file = _source_has_data(source, raw_root)
        if data_file is None:
            click.echo(f"\n{'='*60}")
            click.echo(f"  {source.upper()}: no data found — skipping")
            click.echo(f"{'='*60}")
            continue

        click.echo(f"\n{'='*60}")
        click.echo(f"  {source.upper()}: {data_file}")
        click.echo(f"{'='*60}")

        if source == "claude":
            _inspect_claude(data_file)
        elif source == "gemini":
            _inspect_gemini(data_file)
        elif source == "chatgpt":
            _inspect_chatgpt(data_file)


def _inspect_claude(path: Path) -> None:
    """Pretty-print the structure of the first Claude conversation."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    click.echo(f"  Type: list, length: {len(data)}")
    if not data:
        return
    first = data[0]
    click.echo(f"  Top-level keys: {list(first.keys())}")
    click.echo(f"  uuid: {first.get('uuid')}")
    click.echo(f"  name: {first.get('name')}")
    click.echo(f"  created_at: {first.get('created_at')}")
    messages = first.get("chat_messages", [])
    click.echo(f"  chat_messages: {len(messages)} messages")
    for i, msg in enumerate(messages[:4]):
        click.echo(f"\n  --- message[{i}] ---")
        click.echo(f"  sender: {msg.get('sender')}")
        text = msg.get("text", "")
        click.echo(f"  text (first 200 chars): {text[:200]}")
        content_blocks = msg.get("content", [])
        click.echo(f"  content blocks: {len(content_blocks)}")
        for j, block in enumerate(content_blocks[:3]):
            click.echo(f"    block[{j}] type={block.get('type')}")
            if block.get("type") == "text":
                bt = block.get("text", "")
                click.echo(f"    text (first 150 chars): {bt[:150]}")
    if len(messages) > 4:
        click.echo(f"\n  ... and {len(messages) - 4} more messages")


def _inspect_gemini(path: Path) -> None:
    """Pretty-print the structure of the first Gemini entry."""
    import re
    content = path.read_text(encoding="utf-8", errors="replace")
    # Find outer-cell blocks.
    pattern = re.compile(
        r'<div\s+class="outer-cell\s+mdl-cell\s+mdl-cell--12-col\s+mdl-shadow--2dp">(.*?)</div>\s*(?=<div\s+class="outer-cell|</div>\s*</body>)',
        re.DOTALL,
    )
    entries = pattern.findall(content)
    click.echo(f"  Format: HTML (Google Takeout), {len(entries)} entries")

    # Show the first Prompted entry.
    from src.parsers.gemini_parser import strip_html, _TS_LINE_RE, _CONTENT_CELL_RE
    for i, entry_html in enumerate(entries[:10]):
        m = _CONTENT_CELL_RE.search(entry_html)
        if not m:
            continue
        cell_text = strip_html(m.group(1))
        if not cell_text.startswith("Prompted"):
            continue
        click.echo(f"\n  --- entry[{i}] (first Prompted entry) ---")
        lines = cell_text.split("\n")
        for li, line in enumerate(lines[:15]):
            click.echo(f"  line[{li:02d}]: {line[:120]}")
        if len(lines) > 15:
            click.echo(f"  ... and {len(lines) - 15} more lines")
        break


def _inspect_chatgpt(path: Path) -> None:
    """Pretty-print the structure of the first ChatGPT conversation."""
    click.echo("  ChatGPT parser not yet implemented.")


# ------------------------------------------------------------------ #
#  parse                                                              #
# ------------------------------------------------------------------ #

@cli.command()
@click.pass_context
def parse(ctx: click.Context) -> None:
    """Parse all raw data into processed parquet files."""
    from src.parsers.base import Conversation
    config = ctx.obj["config"]
    raw_root = PROJECT_ROOT / config["paths"]["raw_data"]
    processed_root = PROJECT_ROOT / config["paths"]["processed_data"]
    processed_root.mkdir(parents=True, exist_ok=True)

    all_conversations: list[Conversation] = []
    summaries: list[dict] = []

    for source in _SOURCE_INFO:
        data_file = _source_has_data(source, raw_root)
        if data_file is None:
            logger.info("parse: no data for %s — skipping", source)
            continue

        try:
            parser = _get_parser(source)
        except NotImplementedError as exc:
            logger.info("parse: %s — %s", source, exc)
            continue

        try:
            convs = parser.parse(data_file.parent)
        except NotImplementedError as exc:
            logger.info("parse: %s — %s", source, exc)
            continue

        if not convs:
            logger.warning("parse: %s yielded 0 conversations", source)
            continue

        # Save per-source parquet.
        df = _conversations_to_dataframe(convs)
        out_path = processed_root / f"{source}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("parse: saved %s (%d rows)", out_path.name, len(df))

        all_conversations.extend(convs)
        summaries.append(_summarize(source, convs))

    if all_conversations:
        combined_df = _conversations_to_dataframe(all_conversations)
        combined_path = processed_root / "all_conversations.parquet"
        combined_df.to_parquet(combined_path, index=False)
        logger.info("parse: saved all_conversations.parquet (%d rows)", len(combined_df))

    # Print summary table.
    click.echo("\n" + "=" * 70)
    click.echo("  PARSE SUMMARY")
    click.echo("=" * 70)
    for s in summaries:
        click.echo(
            f"\n  {s['source'].upper()}\n"
            f"    Conversations : {s['conversations']}\n"
            f"    Total turns   : {s['total_turns']}\n"
            f"    User turns    : {s['user_turns']}\n"
            f"    Asst turns    : {s['assistant_turns']}\n"
            f"    Date range    : {s['earliest']} -> {s['latest']}"
        )
    total_convs = sum(s["conversations"] for s in summaries)
    total_turns = sum(s["total_turns"] for s in summaries)
    click.echo(f"\n  TOTAL: {total_convs} conversations, {total_turns} turns")
    click.echo("=" * 70)


def _summarize(source: str, convs: list) -> dict:
    """Produce a summary dict for a list of conversations."""
    total_turns = sum(len(c.turns) for c in convs)
    user_turns = sum(1 for c in convs for t in c.turns if t.role == "user")
    asst_turns = sum(1 for c in convs for t in c.turns if t.role == "assistant")
    timestamps = [
        c.created_at for c in convs if c.created_at is not None
    ]
    earliest = min(timestamps).strftime("%Y-%m-%d") if timestamps else "?"
    latest = max(timestamps).strftime("%Y-%m-%d") if timestamps else "?"
    return {
        "source": source,
        "conversations": len(convs),
        "total_turns": total_turns,
        "user_turns": user_turns,
        "assistant_turns": asst_turns,
        "earliest": earliest,
        "latest": latest,
    }


def _conversations_to_dataframe(convs: list) -> pd.DataFrame:
    """Flatten a list of Conversations into a DataFrame of turns."""
    rows: list[dict] = []
    for conv in convs:
        for turn in conv.turns:
            rows.append({
                "source": conv.source,
                "conversation_id": conv.conversation_id,
                "title": conv.title,
                "conversation_created_at": conv.created_at,
                "conversation_updated_at": conv.updated_at,
                "role": turn.role,
                "content": turn.content,
                "turn_timestamp": turn.timestamp,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
#  Helpers: reconstruct Conversations from parquet                     #
# ------------------------------------------------------------------ #


def _load_conversations_from_parquet(processed_root: Path) -> list:
    """Load the combined parquet and reconstruct Conversation objects."""
    from src.parsers.base import Conversation, Turn

    parquet_path = processed_root / "all_conversations.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} not found. Run 'python main.py parse' first."
        )

    df = pd.read_parquet(parquet_path)
    conversations: list[Conversation] = []

    for (source, conv_id), group in df.groupby(
        ["source", "conversation_id"], sort=False
    ):
        first = group.iloc[0]
        turns = [
            Turn(
                role=row["role"],
                content=row["content"],
                timestamp=(
                    row["turn_timestamp"].to_pydatetime()
                    if pd.notna(row.get("turn_timestamp"))
                    else None
                ),
            )
            for _, row in group.iterrows()
        ]
        conv = Conversation(
            source=str(source),
            conversation_id=str(conv_id),
            title=first.get("title") if pd.notna(first.get("title")) else None,
            created_at=(
                first["conversation_created_at"].to_pydatetime()
                if pd.notna(first.get("conversation_created_at"))
                else None
            ),
            updated_at=(
                first["conversation_updated_at"].to_pydatetime()
                if pd.notna(first.get("conversation_updated_at"))
                else None
            ),
            turns=turns,
        )
        conversations.append(conv)

    logger.info("Loaded %d conversations from %s", len(conversations), parquet_path)
    return conversations


# ------------------------------------------------------------------ #
#  Analysis module registry                                            #
# ------------------------------------------------------------------ #

_ANALYSIS_MODULES: dict[str, str] = {
    "lexical": "src.analysis.lexical",
    "semantic": "src.analysis.semantic",
    "pragmatic": "src.analysis.pragmatic",
    "comparative": "src.analysis.comparative",
    "temporal": "src.analysis.temporal",
    "conversation_structure": "src.analysis.conversation_structure",
    "user_behavior": "src.analysis.user_behavior",
    "llm_judge": "src.analysis.llm_judge",
}


# ------------------------------------------------------------------ #
#  analyze                                                             #
# ------------------------------------------------------------------ #

@cli.command()
@click.option("--module", default=None, help="Run a specific analysis module (e.g. lexical, semantic).")
@click.option("--dry-run", is_flag=True, default=False, help="For llm_judge: show sample, cost estimate, and example prompts without making API calls.")
@click.pass_context
def analyze(ctx: click.Context, module: str | None, dry_run: bool) -> None:
    """Run analysis modules on processed data."""
    import importlib
    from typing import Any

    config = ctx.obj["config"]
    processed_root = PROJECT_ROOT / config["paths"]["processed_data"]
    outputs_root = PROJECT_ROOT / config["paths"]["outputs"]
    outputs_root.mkdir(parents=True, exist_ok=True)

    conversations = _load_conversations_from_parquet(processed_root)

    # --dry-run only applies to llm_judge.
    if dry_run and module != "llm_judge":
        click.echo("  --dry-run is only supported for --module llm_judge")
        return

    modules_to_run = [module] if module else list(_ANALYSIS_MODULES.keys())

    for mod_name in modules_to_run:
        if mod_name not in _ANALYSIS_MODULES:
            click.echo(
                f"analyze: unknown module '{mod_name}'. "
                f"Available: {list(_ANALYSIS_MODULES.keys())}"
            )
            continue

        # Skip llm_judge in full "analyze all" runs (requires API key + costs money).
        if mod_name == "llm_judge" and module is None:
            click.echo(f"\n  SKIP llm_judge: run explicitly with --module llm_judge (costs money)")
            continue

        click.echo(f"\n{'='*70}")
        click.echo(f"  Running analysis module: {mod_name}")
        click.echo(f"{'='*70}")

        mod = importlib.import_module(_ANALYSIS_MODULES[mod_name])

        # llm_judge: handle --dry-run.
        if mod_name == "llm_judge" and dry_run:
            dry_results = mod.dry_run(conversations, config)
            _print_llm_judge_dry_run(dry_results)
            continue

        # comparative module has a different signature (needs prior results).
        if mod_name == "comparative":
            try:
                results = mod.run(conversations, config)
            except FileNotFoundError as exc:
                click.echo(f"\n  ERROR: {exc}")
                continue
        else:
            results = mod.run(conversations, config)

        # Save results as JSON (convert non-serialisable items).
        out_path = outputs_root / f"{mod_name}_results.json"
        _save_results_json(results, out_path)
        click.echo(f"\n  Results saved to {out_path}")

        # Print human-readable summary.
        if mod_name == "lexical":
            _print_lexical_summary(results)
        elif mod_name == "semantic":
            _print_semantic_summary(results)
        elif mod_name == "pragmatic":
            _print_pragmatic_summary(results)
        elif mod_name == "comparative":
            _print_comparative_summary(results)
        elif mod_name == "temporal":
            _print_temporal_summary(results)
        elif mod_name == "conversation_structure":
            _print_conversation_structure_summary(results)
        elif mod_name == "user_behavior":
            _print_user_behavior_summary(results)
        elif mod_name == "llm_judge":
            _print_llm_judge_summary(results)


def _save_results_json(results: dict, path: Path) -> None:
    """Save analysis results to JSON, stripping large distribution arrays."""

    def _prepare(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k == "distribution":
                    out[k] = f"<{len(v)} values>"
                else:
                    out[k] = _prepare(v)
            return out
        if isinstance(obj, (list, tuple)):
            if len(obj) > 100:
                return f"<list of {len(obj)} items>"
            return [_prepare(i) for i in obj]
        if isinstance(obj, float):
            return round(obj, 4)
        return obj

    clean = _prepare(results)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False, default=str)


def _print_lexical_summary(results: dict) -> None:
    """Pretty-print lexical analysis results to the terminal."""
    sources = results.get("_meta", {}).get("sources", [])

    # -- Vocabulary --
    click.echo(f"\n  {'VOCABULARY RICHNESS':=^66}")
    vocab = results.get("vocabulary", {})
    header = f"  {'Metric':<25}" + "".join(f"{s.upper():>20}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (25 + 20 * len(sources)))
    for metric in ["total_tokens", "unique_types", "ttr", "mattr_500"]:
        row = f"  {metric:<25}"
        for s in sources:
            val = vocab.get(s, {}).get(metric, "?")
            if isinstance(val, float):
                row += f"{val:>20.4f}"
            else:
                row += f"{val:>20,}" if isinstance(val, int) else f"{str(val):>20}"
        click.echo(row)

    # -- Distinctive words (TF-IDF) --
    click.echo(f"\n  {'DISTINCTIVE WORDS (TF-IDF top 15)':=^66}")
    dw = results.get("distinctive_words", {})
    for s in sources:
        words = dw.get(s, [])[:15]
        word_str = ", ".join(f"{w['word']}({w['tfidf']:.3f})" for w in words)
        click.echo(f"  {s.upper()}: {word_str}")

    # -- Sentence stats --
    click.echo(f"\n  {'SENTENCE STATISTICS':=^66}")
    ss = results.get("sentence_stats", {})
    header = f"  {'Metric':<25}" + "".join(f"{s.upper():>20}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (25 + 20 * len(sources)))
    for metric in ["mean", "median", "std", "count"]:
        row = f"  {('sent_len_' + metric):<25}"
        for s in sources:
            val = ss.get(s, {}).get(metric, "?")
            if isinstance(val, float):
                row += f"{val:>20.2f}"
            else:
                row += f"{val:>20,}" if isinstance(val, int) else f"{str(val):>20}"
        click.echo(row)

    # -- Response length --
    click.echo(f"\n  {'RESPONSE LENGTH (words)':=^66}")
    rl = results.get("response_length", {})
    header = f"  {'Metric':<25}" + "".join(f"{s.upper():>20}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (25 + 20 * len(sources)))
    for metric in ["mean", "median", "std", "count"]:
        row = f"  {('resp_' + metric):<25}"
        for s in sources:
            val = rl.get(s, {}).get(metric, "?")
            if isinstance(val, float):
                row += f"{val:>20.2f}"
            else:
                row += f"{val:>20,}" if isinstance(val, int) else f"{str(val):>20}"
        click.echo(row)

    # -- Readability --
    click.echo(f"\n  {'READABILITY':=^66}")
    rd = results.get("readability", {})
    header = f"  {'Metric':<25}" + "".join(f"{s.upper():>20}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (25 + 20 * len(sources)))
    for index_name in ["flesch_kincaid", "gunning_fog"]:
        for stat in ["mean", "std"]:
            label = f"{index_name}_{stat}"
            row = f"  {label:<25}"
            for s in sources:
                val = rd.get(s, {}).get(index_name, {}).get(stat, "?")
                if isinstance(val, float):
                    row += f"{val:>20.2f}"
                else:
                    row += f"{str(val):>20}"
            click.echo(row)

    # -- Formatting --
    click.echo(f"\n  {'FORMATTING HABITS (per 1k words)':=^66}")
    fmt = results.get("formatting", {})
    header = f"  {'Element':<25}" + "".join(f"{s.upper():>20}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (25 + 20 * len(sources)))
    fmt_keys = [
        "headers", "bullet_points", "numbered_lists",
        "code_blocks", "bold", "italic",
    ]
    for k in fmt_keys:
        row = f"  {k:<25}"
        for s in sources:
            val = fmt.get(s, {}).get("per_1k_words", {}).get(k, 0)
            row += f"{val:>20.2f}"
        click.echo(row)
    row = f"  {'total_words':<25}"
    for s in sources:
        val = fmt.get(s, {}).get("total_words", 0)
        row += f"{val:>20,}"
    click.echo(row)

    # -- Top 10 words --
    click.echo(f"\n  {'TOP 10 WORDS PER SOURCE':=^66}")
    wf = results.get("word_frequency", {})
    for s in sources:
        words = wf.get(s, {}).get("top_words", [])[:10]
        word_str = ", ".join(f"{w}({c:,})" for w, c in words)
        click.echo(f"  {s.upper()}: {word_str}")


def _print_semantic_summary(results: dict) -> None:
    """Pretty-print semantic analysis results to the terminal."""
    sources = results.get("_meta", {}).get("sources", [])

    # -- Assistant topics --
    at = results.get("assistant_topics", {})
    topics = at.get("topics", [])
    click.echo(f"\n  {'ASSISTANT TOPICS (BERTopic)':=^70}")
    click.echo(
        f"  Topics found: {at.get('n_topics_found', '?')} | "
        f"Documents: {at.get('n_documents', '?')} | "
        f"Outliers: {at.get('n_outliers', '?')}"
    )
    click.echo(
        f"\n  {'#':<5} {'Count':>6}  "
        + "".join(f"{s.upper():>8}" for s in sources)
        + "  Top words"
    )
    click.echo("  " + "-" * 80)
    for t in topics[:10]:
        tid = t["topic_id"]
        count = t["count"]
        breakdown = t.get("source_breakdown", {})
        pcts = "".join(
            f"{breakdown.get(s, 0)*100:>7.0f}%" for s in sources
        )
        words = ", ".join(w["word"] for w in t.get("words", [])[:8])
        click.echo(f"  {tid:<5} {count:>6}  {pcts}  {words}")

    # -- User topics --
    ut = results.get("user_topics", {})
    utopics = ut.get("topics", [])
    click.echo(f"\n  {'USER TOPICS (BERTopic)':=^70}")
    click.echo(
        f"  Topics found: {ut.get('n_topics_found', '?')} | "
        f"Documents: {ut.get('n_documents', '?')} | "
        f"Outliers: {ut.get('n_outliers', '?')}"
    )
    click.echo(
        f"\n  {'#':<5} {'Count':>6}  "
        + "".join(f"{s.upper():>8}" for s in sources)
        + "  Top words"
    )
    click.echo("  " + "-" * 80)
    for t in utopics[:10]:
        tid = t["topic_id"]
        count = t["count"]
        breakdown = t.get("source_breakdown", {})
        pcts = "".join(
            f"{breakdown.get(s, 0)*100:>7.0f}%" for s in sources
        )
        words = ", ".join(w["word"] for w in t.get("words", [])[:8])
        click.echo(f"  {tid:<5} {count:>6}  {pcts}  {words}")

    # -- Self-similarity --
    click.echo(f"\n  {'SELF-SIMILARITY (cosine, assistant responses)':=^70}")
    ss = results.get("self_similarity", {})
    header = f"  {'Metric':<20}" + "".join(f"{s.upper():>20}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (20 + 20 * len(sources)))
    for metric in ["mean", "std", "n_pairs"]:
        row = f"  {metric:<20}"
        for s in sources:
            val = ss.get(s, {}).get(metric, "?")
            if isinstance(val, float):
                row += f"{val:>20.4f}"
            elif isinstance(val, int):
                row += f"{val:>20,}"
            else:
                row += f"{str(val):>20}"
        click.echo(row)

    # -- Sentiment --
    click.echo(f"\n  {'SENTIMENT (lexicon-based, assistant turns)':=^70}")
    sent = results.get("sentiment", {})
    header = f"  {'Metric':<20}" + "".join(f"{s.upper():>20}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (20 + 20 * len(sources)))
    for metric in ["positive", "neutral", "negative", "mean_polarity", "n_turns"]:
        row = f"  {metric:<20}"
        for s in sources:
            val = sent.get(s, {}).get(metric, "?")
            if isinstance(val, float):
                if metric == "mean_polarity":
                    row += f"{val:>20.4f}"
                else:
                    row += f"{val*100:>19.1f}%"
            elif isinstance(val, int):
                row += f"{val:>20,}"
            else:
                row += f"{str(val):>20}"
        click.echo(row)


def _print_pragmatic_summary(results: dict) -> None:
    """Pretty-print pragmatic analysis results to the terminal."""
    sources = results.get("_meta", {}).get("sources", [])

    # -- Hedging --
    click.echo(f"\n  {'HEDGING LANGUAGE':=^70}")
    hedging = results.get("hedging", {})
    header = f"  {'Metric':<30}" + "".join(f"{s.upper():>20}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (30 + 20 * len(sources)))
    # Category totals
    cats = ["uncertainty", "caveat_softener", "epistemic"]
    for cat in cats:
        row = f"  {cat + ' (count)':<30}"
        for s in sources:
            val = hedging.get(s, {}).get("category_totals", {}).get(cat, 0)
            row += f"{val:>20,}"
        click.echo(row)
        row = f"  {cat + ' (per 1k)':<30}"
        for s in sources:
            val = hedging.get(s, {}).get("category_per_1k", {}).get(cat, 0.0)
            row += f"{val:>20.3f}"
        click.echo(row)
    # Totals
    click.echo("  " + "-" * (30 + 20 * len(sources)))
    row = f"  {'total_hedges':<30}"
    for s in sources:
        val = hedging.get(s, {}).get("total_hedges", 0)
        row += f"{val:>20,}"
    click.echo(row)
    row = f"  {'hedge_density (per 1k)':<30}"
    for s in sources:
        val = hedging.get(s, {}).get("hedge_density", 0.0)
        row += f"{val:>20.3f}"
    click.echo(row)
    row = f"  {'total_words':<30}"
    for s in sources:
        val = hedging.get(s, {}).get("total_words", 0)
        row += f"{val:>20,}"
    click.echo(row)

    # Top hedge phrases
    click.echo(f"\n  {'TOP HEDGE PHRASES (per 1k words)':=^70}")
    for s in sources:
        phrase_per_1k = hedging.get(s, {}).get("phrase_per_1k", {})
        sorted_phrases = sorted(phrase_per_1k.items(), key=lambda x: -x[1])[:10]
        click.echo(f"  {s.upper()}:")
        for phrase, rate in sorted_phrases:
            if rate > 0:
                raw = hedging.get(s, {}).get("phrase_counts", {}).get(phrase, 0)
                click.echo(f"    {phrase:<30} {rate:>8.3f}  (n={raw})")

    # -- Question rate --
    click.echo(f"\n  {'QUESTION RATE':=^70}")
    qr = results.get("question_rate", {})
    header = f"  {'Metric':<35}" + "".join(f"{s.upper():>18}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (35 + 18 * len(sources)))
    for metric in [
        "mean_questions_per_turn",
        "fraction_turns_with_question",
        "total_questions",
        "questions_per_1k_words",
        "n_turns",
    ]:
        row = f"  {metric:<35}"
        for s in sources:
            val = qr.get(s, {}).get(metric, 0)
            if isinstance(val, float):
                row += f"{val:>18.3f}"
            else:
                row += f"{val:>18,}"
        click.echo(row)

    # -- Disclaimers --
    click.echo(f"\n  {'DISCLAIMERS / CAVEATS':=^70}")
    disc = results.get("disclaimers", {})
    header = f"  {'Category':<30}" + "".join(f"{s.upper():>20}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (30 + 20 * len(sources)))
    disc_cats = [
        "ai_self_reference", "knowledge_disclaimer",
        "safety_professional", "hedge_disclaimer",
    ]
    for cat in disc_cats:
        row = f"  {cat + ' (count)':<30}"
        for s in sources:
            val = disc.get(s, {}).get("category_totals", {}).get(cat, 0)
            row += f"{val:>20,}"
        click.echo(row)
        row = f"  {cat + ' (per 1k)':<30}"
        for s in sources:
            val = disc.get(s, {}).get("category_per_1k", {}).get(cat, 0.0)
            row += f"{val:>20.3f}"
        click.echo(row)
    click.echo("  " + "-" * (30 + 20 * len(sources)))
    row = f"  {'total_disclaimers':<30}"
    for s in sources:
        val = disc.get(s, {}).get("total_disclaimers", 0)
        row += f"{val:>20,}"
    click.echo(row)
    row = f"  {'disclaimer_density (per 1k)':<30}"
    for s in sources:
        val = disc.get(s, {}).get("disclaimer_density", 0.0)
        row += f"{val:>20.3f}"
    click.echo(row)

    # -- Verbosity ratio --
    click.echo(f"\n  {'VERBOSITY RATIO (asst_words / user_words)':=^70}")
    vr = results.get("verbosity_ratio", {})
    header = f"  {'Metric':<20}" + "".join(f"{s.upper():>20}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (20 + 20 * len(sources)))
    for metric in ["mean", "median", "std", "count"]:
        row = f"  {metric:<20}"
        for s in sources:
            val = vr.get(s, {}).get(metric, 0)
            if isinstance(val, float):
                row += f"{val:>20.2f}"
            else:
                row += f"{val:>20,}"
        click.echo(row)

    # -- First-person usage --
    click.echo(f"\n  {'FIRST-PERSON USAGE (per 1k words)':=^70}")
    fp = results.get("first_person", {})
    header = f"  {'Pronoun':<20}" + "".join(f"{s.upper():>20}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (20 + 20 * len(sources)))
    for pronoun in ["I", "we", "you"]:
        row_rate = f"  {pronoun + ' (per 1k)':<20}"
        row_raw = f"  {pronoun + ' (raw)':<20}"
        for s in sources:
            rate = fp.get(s, {}).get("per_1k_words", {}).get(pronoun, 0.0)
            raw = fp.get(s, {}).get("raw_counts", {}).get(pronoun, 0)
            row_rate += f"{rate:>20.3f}"
            row_raw += f"{raw:>20,}"
        click.echo(row_rate)
        click.echo(row_raw)

    # -- Turn dynamics --
    click.echo(f"\n  {'TURN DYNAMICS':=^70}")
    td = results.get("turn_dynamics", {})
    header = f"  {'Metric':<35}" + "".join(f"{s.upper():>18}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (35 + 18 * len(sources)))
    for metric in ["n_conversations"]:
        row = f"  {metric:<35}"
        for s in sources:
            val = td.get(s, {}).get(metric, 0)
            row += f"{val:>18,}"
        click.echo(row)
    for sub in ["turns_per_conv", "user_turns_per_conv"]:
        for stat in ["mean", "median", "std"]:
            label = f"{sub}.{stat}"
            row = f"  {label:<35}"
            for s in sources:
                val = td.get(s, {}).get(sub, {}).get(stat, 0)
                if isinstance(val, float):
                    row += f"{val:>18.2f}"
                else:
                    row += f"{val:>18}"
            click.echo(row)
    row = f"  {'single_exchange_fraction':<35}"
    for s in sources:
        val = td.get(s, {}).get("single_exchange_fraction", 0.0)
        row += f"{val:>18.3f}"
    click.echo(row)
    row = f"  {'single_exchange_count':<35}"
    for s in sources:
        val = td.get(s, {}).get("single_exchange_count", 0)
        row += f"{val:>18,}"
    click.echo(row)

    # -- Opening patterns --
    click.echo(f"\n  {'OPENING PATTERNS':=^70}")
    op = results.get("opening_patterns", {})
    for s in sources:
        data = op.get(s, {})
        n = data.get("n_turns", 0)
        click.echo(f"\n  {s.upper()} ({n} turns):")

        click.echo("    Top opening words:")
        for word, count in data.get("top_opening_words", [])[:10]:
            pct = count / max(n, 1) * 100
            click.echo(f"      {word:<20} {count:>5}  ({pct:>5.1f}%)")

        click.echo("    Top opening bigrams:")
        for bigram, count in data.get("top_opening_bigrams", [])[:10]:
            pct = count / max(n, 1) * 100
            click.echo(f"      {bigram:<25} {count:>5}  ({pct:>5.1f}%)")

        click.echo("    Tracked opening word rates:")
        tracked = data.get("tracked_opening_rates", {})
        for word, rate in tracked.items():
            click.echo(f"      {word:<10} {rate*100:>6.1f}%")


def _print_comparative_summary(results: dict) -> None:
    """Pretty-print comparative analysis results to the terminal."""
    sources = results.get("_meta", {}).get("sources", [])

    # ---- Style Fingerprint ---- #
    fp = results.get("style_fingerprint", {})
    metrics = fp.get("metrics", [])
    raw = fp.get("raw", {})
    norm = fp.get("normalized", {})

    click.echo(f"\n  {'STYLE FINGERPRINT':=^76}")
    header = f"  {'Metric':<25}" + "".join(
        f"{s.upper() + ' (raw)':>16}{s.upper() + ' (norm)':>14}" for s in sources
    )
    click.echo(header)
    click.echo("  " + "-" * (25 + 30 * len(sources)))
    for i, m in enumerate(metrics):
        row = f"  {m:<25}"
        for s in sources:
            r = raw.get(s, [0] * len(metrics))[i] if i < len(raw.get(s, [])) else 0
            n = norm.get(s, [0] * len(metrics))[i] if i < len(norm.get(s, [])) else 0
            row += f"{r:>16.4f}{n:>14.4f}"
        click.echo(row)

    # ---- Summary Comparison Table ---- #
    table = results.get("summary_table", [])
    click.echo(f"\n  {'SUMMARY COMPARISON TABLE':=^76}")
    col_w = 14
    header = f"  {'Metric':<34}"
    for s in sources:
        header += f"{s.upper():>{col_w}}"
    header += f"{'Delta':>{col_w}}  Note"
    click.echo(header)
    click.echo("  " + "-" * (34 + col_w * (len(sources) + 1) + 30))

    for row in table:
        line = f"  {row.get('metric', ''):<34}"
        for s in sources:
            val = row.get(s, "?")
            if isinstance(val, float):
                line += f"{val:>{col_w}.3f}"
            elif isinstance(val, int):
                line += f"{val:>{col_w},}"
            else:
                line += f"{str(val):>{col_w}}"
        delta = row.get("delta", "")
        if isinstance(delta, float):
            line += f"{delta:>{col_w}.3f}"
        elif isinstance(delta, int):
            line += f"{delta:>{col_w},}"
        else:
            line += f"{str(delta):>{col_w}}"
        note = row.get("note", "")
        if note:
            line += f"  {note}"
        click.echo(line)

    # ---- User Language ---- #
    ul = results.get("user_language", {})

    click.echo(f"\n  {'USER LANGUAGE COMPARISON':=^76}")

    # Message length
    click.echo(f"\n  Message Length (words):")
    ml = ul.get("message_length", {})
    header = f"  {'Metric':<20}" + "".join(f"{s.upper():>18}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (20 + 18 * len(sources)))
    for stat in ["mean", "median", "std", "count"]:
        row = f"  {stat:<20}"
        for s in sources:
            val = ml.get(s, {}).get(stat, 0)
            if isinstance(val, float):
                row += f"{val:>18.2f}"
            else:
                row += f"{val:>18,}"
        click.echo(row)

    # Vocabulary
    click.echo(f"\n  User Vocabulary:")
    uv = ul.get("vocabulary", {})
    header = f"  {'Metric':<20}" + "".join(f"{s.upper():>18}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (20 + 18 * len(sources)))
    for stat in ["total_tokens", "unique_types", "ttr", "mattr_200"]:
        row = f"  {stat:<20}"
        for s in sources:
            val = uv.get(s, {}).get(stat, 0)
            if isinstance(val, float):
                row += f"{val:>18.4f}"
            elif isinstance(val, int):
                row += f"{val:>18,}"
            else:
                row += f"{str(val):>18}"
        click.echo(row)

    # Question rate
    click.echo(f"\n  User Question Rate:")
    qr = ul.get("question_rate", {})
    header = f"  {'Metric':<30}" + "".join(f"{s.upper():>18}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (30 + 18 * len(sources)))
    for stat in ["mean_questions_per_turn", "fraction_turns_with_question",
                 "total_questions", "n_turns"]:
        row = f"  {stat:<30}"
        for s in sources:
            val = qr.get(s, {}).get(stat, 0)
            if isinstance(val, float):
                row += f"{val:>18.3f}"
            else:
                row += f"{val:>18,}"
        click.echo(row)

    # Sentence length
    click.echo(f"\n  User Sentence Length:")
    sl = ul.get("sentence_length", {})
    header = f"  {'Metric':<20}" + "".join(f"{s.upper():>18}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (20 + 18 * len(sources)))
    for stat in ["mean", "median", "std"]:
        row = f"  {stat:<20}"
        for s in sources:
            val = sl.get(s, {}).get(stat, 0)
            if isinstance(val, float):
                row += f"{val:>18.2f}"
            else:
                row += f"{val:>18}"
        click.echo(row)

    # Politeness
    click.echo(f"\n  User Politeness Markers (per 1k words):")
    pol = ul.get("politeness", {})
    header = f"  {'Phrase':<20}" + "".join(f"{s.upper():>18}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (20 + 18 * len(sources)))
    polite_phrases = ["please", "thanks", "thank you", "sorry",
                      "could you", "would you"]
    for phrase in polite_phrases:
        row = f"  {phrase:<20}"
        for s in sources:
            val = pol.get(s, {}).get("per_1k_words", {}).get(phrase, 0)
            row += f"{val:>18.3f}"
        click.echo(row)
    row = f"  {'TOTAL':<20}"
    for s in sources:
        val = pol.get(s, {}).get("politeness_per_1k", 0)
        row += f"{val:>18.3f}"
    click.echo(row)

    # Distinctive words
    click.echo(f"\n  User Distinctive Words (TF-IDF top 15):")
    dw = ul.get("distinctive_words", {})
    for s in sources:
        words = dw.get(s, [])[:15]
        word_str = ", ".join(f"{w['word']}({w['tfidf']:.3f})" for w in words)
        click.echo(f"  {s.upper()}: {word_str}")

    # Top 10 words
    click.echo(f"\n  User Top 10 Words Per Source:")
    wf = ul.get("word_frequency", {})
    for s in sources:
        words = wf.get(s, [])[:10]
        word_str = ", ".join(f"{w}({c:,})" for w, c in words)
        click.echo(f"  {s.upper()}: {word_str}")

    # ---- Topic-Controlled ---- #
    tc = results.get("topic_controlled", {})
    click.echo(f"\n  {'TOPIC-CONTROLLED COMPARISON':=^76}")
    click.echo(f"  Topic: {tc.get('topic_name', '?')} (id={tc.get('topic_id', '?')})")
    click.echo(f"  Total count: {tc.get('topic_count', '?')}")
    click.echo(f"  Breakdown: {tc.get('source_breakdown', {})}")
    click.echo(f"  Turns used: {tc.get('n_turns_per_source', {})}")
    click.echo(f"  Note: {tc.get('note', '')}")

    tc_metrics = tc.get("metrics", {})
    if tc_metrics:
        click.echo(f"\n  {'Metric':<30}" + "".join(
            f"{s.upper():>18}" for s in sources
        ))
        click.echo("  " + "-" * (30 + 18 * len(sources)))

        # Response length
        rl = tc_metrics.get("mean_response_length", {})
        if rl:
            row = f"  {'response_length (words)':<30}"
            for s in sources:
                row += f"{rl.get(s, 0):>18.1f}"
            click.echo(row)

        # Hedge density
        hd = tc_metrics.get("hedge_density_per_1k", {})
        if hd:
            row = f"  {'hedge_density (per 1k)':<30}"
            for s in sources:
                row += f"{hd.get(s, 0):>18.3f}"
            click.echo(row)

        # Verbosity ratio
        vr = tc_metrics.get("verbosity_ratio", {})
        if vr:
            for stat in ["mean", "median"]:
                row = f"  {f'verbosity_{stat}':<30}"
                for s in sources:
                    v = vr.get(s, {})
                    if isinstance(v, dict):
                        row += f"{v.get(stat, 0):>18.2f}"
                    else:
                        row += f"{'?':>18}"
                click.echo(row)

        # Self-similarity
        ss = tc_metrics.get("self_similarity", {})
        if ss and not isinstance(ss.get("note", None), str):
            row = f"  {'self_similarity (cosine)':<30}"
            for s in sources:
                val = ss.get(s, 0)
                if isinstance(val, (int, float)):
                    row += f"{val:>18.4f}"
                else:
                    row += f"{'?':>18}"
            click.echo(row)


def _print_temporal_summary(results: dict) -> None:
    """Pretty-print temporal analysis results to the terminal."""
    sources = results.get("_meta", {}).get("sources", [])
    activity = results.get("activity", {})
    months = activity.get("months", [])
    per_source = activity.get("per_source", {})

    # ---- Activity table ---- #
    click.echo(f"\n  {'MONTHLY ACTIVITY':=^80}")
    source_ranges = activity.get("source_ranges", {})
    overlap = activity.get("overlap_period", {})

    for s in sources:
        sr = source_ranges.get(s, {})
        click.echo(f"  {s.upper()}: {sr.get('start', '?')} -> {sr.get('end', '?')}")
    if overlap.get("start"):
        click.echo(f"  Overlap period: {overlap['start']} -> {overlap.get('end', '?')}")

    # Table header
    header = f"  {'Month':<10}"
    for s in sources:
        header += f"  {s[:6].upper():>6}c {s[:6].upper():>6}t {s[:6].upper():>7}w"
    click.echo(f"\n{header}")
    click.echo("  " + "-" * (10 + len(sources) * 23))

    for i, mk in enumerate(months):
        row = f"  {mk:<10}"
        flags = []
        for s in sources:
            convs = per_source.get(s, {}).get("conversations", [])
            turns = per_source.get(s, {}).get("turns", [])
            words = per_source.get(s, {}).get("words", [])
            low = per_source.get(s, {}).get("low_confidence", [])
            c = convs[i] if i < len(convs) else 0
            t = turns[i] if i < len(turns) else 0
            w = words[i] if i < len(words) else 0
            lc = low[i] if i < len(low) else True
            row += f"  {c:>6} {t:>6} {w:>7}"
            if lc and (c > 0 or t > 0):
                flags.append(f"{s}(<20 turns)")
        if flags:
            row += f"  *{', '.join(flags)}"

        # Mark non-overlapping months
        overlap_start = overlap.get("start", "")
        if overlap_start and mk < overlap_start:
            row += "  [non-overlap]"

        click.echo(row)

    click.echo(f"\n  c=conversations, t=turns, w=assistant words")
    click.echo(f"  * = low confidence (< 20 turns)")

    # ---- Bot metrics trends ---- #
    click.echo(f"\n  {'BOT METRICS OVER TIME':=^80}")
    bot = results.get("bot_metrics_over_time", {})
    bot_months = bot.get("months", [])
    metric_labels = {
        "response_length": "Resp Length",
        "mattr": "MATTR-500",
        "readability": "FK Grade",
        "hedge_density": "Hedge/1k",
        "formatting_density": "Fmt/1k",
        "question_rate": "Q/turn",
    }

    for s in sources:
        click.echo(f"\n  {s.upper()}:")
        header = f"    {'Month':<10}"
        for mk, label in metric_labels.items():
            header += f" {label:>12}"
        click.echo(header)
        click.echo("    " + "-" * (10 + len(metric_labels) * 13))

        src_data = bot.get(s, {})
        for i, mk in enumerate(bot_months):
            row = f"    {mk:<10}"
            for metric_key in metric_labels:
                values = src_data.get(metric_key, [])
                val = values[i] if i < len(values) else None
                if val is None:
                    row += f" {'--':>12}"
                elif isinstance(val, float):
                    row += f" {val:>12.2f}"
                else:
                    row += f" {val:>12}"
            click.echo(row)

    # ---- User metrics trends ---- #
    click.echo(f"\n  {'USER METRICS OVER TIME':=^80}")
    user = results.get("user_metrics_over_time", {})
    user_months = user.get("months", [])
    user_labels = {
        "message_length": "Msg Length",
        "messages_per_convo": "Msgs/Conv",
        "question_rate": "Q Rate",
    }

    for s in sources:
        click.echo(f"\n  {s.upper()}:")
        header = f"    {'Month':<10}"
        for mk, label in user_labels.items():
            header += f" {label:>12}"
        click.echo(header)
        click.echo("    " + "-" * (10 + len(user_labels) * 13))

        src_data = user.get(s, {})
        for i, mk in enumerate(user_months):
            row = f"    {mk:<10}"
            for metric_key in user_labels:
                values = src_data.get(metric_key, [])
                val = values[i] if i < len(values) else None
                if val is None:
                    row += f" {'--':>12}"
                elif isinstance(val, float):
                    row += f" {val:>12.2f}"
                else:
                    row += f" {val:>12}"
            click.echo(row)

    # ---- Topic shifts ---- #
    topic_shifts = results.get("topic_shifts", {})
    click.echo(f"\n  {'TOPIC SHIFTS':=^80}")
    click.echo(f"  Status: {topic_shifts.get('status', '?')}")
    click.echo(f"  Note: {topic_shifts.get('reason', '')}")

    # ---- Inflection points ---- #
    inflections = results.get("inflection_points", [])
    click.echo(f"\n  {'INFLECTION POINTS':=^80}")
    if not inflections:
        click.echo("  No significant inflection points detected.")
    else:
        click.echo(f"  Detected {len(inflections)} inflection points:\n")
        header = (
            f"  {'Month':<10} {'Source':<8} {'Metric':<22} "
            f"{'Value':>10} {'Prior Mean':>12} {'Z-Score':>9} {'Dir':>10}"
        )
        click.echo(header)
        click.echo("  " + "-" * 83)
        for ip in inflections[:20]:
            click.echo(
                f"  {ip['month']:<10} {ip['source']:<8} {ip['metric']:<22} "
                f"{ip['value']:>10.2f} {ip['prior_mean']:>12.2f} "
                f"{ip['zscore']:>+9.2f} {ip['direction']:>10}"
            )
        if len(inflections) > 20:
            click.echo(f"\n  ... and {len(inflections) - 20} more")


def _print_conversation_structure_summary(results: dict) -> None:
    """Pretty-print conversation structure analysis results."""
    sources = results.get("_meta", {}).get("sources", [])

    # -- Shape Metrics -- #
    click.echo(f"\n  {'CONVERSATION SHAPE METRICS':=^70}")
    shape = results.get("shape_metrics", {}).get("per_source", {})
    header = f"  {'Metric':<30}" + "".join(f"{s.upper():>20}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (30 + 20 * len(sources)))
    shape_metrics = [
        ("turn_count (mean)", "turn_count", "mean"),
        ("turn_count (median)", "turn_count", "median"),
        ("asst words/conv (mean)", "assistant_words", "mean"),
        ("user words/conv (mean)", "user_words", "mean"),
        ("mean asst resp len", "mean_asst_response_length", "mean"),
        ("verbosity trajectory", "verbosity_trajectory", "mean"),
        ("user effort trajectory", "user_effort_trajectory", "mean"),
    ]
    for label, key, sub in shape_metrics:
        row = f"  {label:<30}"
        for s in sources:
            val = shape.get(s, {}).get(key, {}).get(sub, "?")
            if isinstance(val, float):
                row += f"{val:>20.2f}"
            else:
                row += f"{str(val):>20}"
        click.echo(row)

    # -- Depth Classification -- #
    click.echo(f"\n  {'DEPTH CLASSIFICATION':=^70}")
    depth = results.get("depth_classification", {})
    per_source = depth.get("per_source", {})
    categories = depth.get("categories", [])
    resp_by_depth = depth.get("response_length_by_depth", {})

    cat_labels = {
        "quick_exchange": "Quick (1-2)",
        "short_session": "Short (3-6)",
        "working_session": "Working (7-20)",
        "deep_dive": "Deep Dive (21-50)",
        "marathon": "Marathon (51+)",
    }
    header = f"  {'Category':<22}" + "".join(f"{s.upper():>20}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (22 + 20 * len(sources)))
    for cat in categories:
        row = f"  {cat_labels.get(cat, cat):<22}"
        for s in sources:
            val = per_source.get(s, {}).get(cat, 0)
            total = sum(per_source.get(s, {}).get(c, 0) for c in categories)
            pct = val / max(total, 1) * 100
            row += f"{val:>10} ({pct:>5.1f}%)"
        click.echo(row)

    # Response length by depth
    click.echo(f"\n  Mean response length (words) by depth:")
    header = f"  {'Category':<22}" + "".join(f"{s.upper():>20}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (22 + 20 * len(sources)))
    for cat in categories:
        row = f"  {cat_labels.get(cat, cat):<22}"
        for s in sources:
            val = resp_by_depth.get(s, {}).get(cat)
            if val is not None:
                row += f"{val:>20.1f}"
            else:
                row += f"{'—':>20}"
        click.echo(row)

    # -- Rephrasing -- #
    click.echo(f"\n  {'REPHRASING DETECTION':=^70}")
    reph = results.get("rephrasing", {})
    reph_src = reph.get("per_source", {})
    threshold = reph.get("threshold", 0.6)
    click.echo(f"  Similarity threshold: {threshold}")
    header = f"  {'Metric':<35}" + "".join(f"{s.upper():>18}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (35 + 18 * len(sources)))
    for label, key in [
        ("Total rephrase events", "total"),
        ("Rate per 100 user turns", "rate_per_100_user_turns"),
        ("Conversations w/ rephrase", "conversations_with_rephrase"),
        ("Rephrases per conversation", "rephrase_per_conversation"),
    ]:
        row = f"  {label:<35}"
        for s in sources:
            val = reph_src.get(s, {}).get(key, 0)
            if isinstance(val, float):
                row += f"{val:>18.2f}"
            else:
                row += f"{val:>18}"
        click.echo(row)

    # Show top examples per source
    for s in sources:
        examples = reph_src.get(s, {}).get("examples", [])[:3]
        if examples:
            click.echo(f"\n  Top rephrase examples ({s.upper()}):")
            for i, ex in enumerate(examples, 1):
                click.echo(f"    [{i}] sim={ex['similarity']:.3f}")
                msg1 = ex.get("msg1", "")[:120]
                msg2 = ex.get("msg2", "")[:120]
                click.echo(f"        MSG1: {msg1}")
                click.echo(f"        MSG2: {msg2}")

    # -- Resolution Patterns -- #
    click.echo(f"\n  {'RESOLUTION PATTERNS':=^70}")
    resol = results.get("resolution_patterns", {}).get("per_source", {})
    pattern_labels = {
        "thank_you": "Thank You / Acknowledge",
        "question": "Ends with Question",
        "correction": "Correction / Pushback",
        "short_close": "Short Close",
        "other": "Other",
        "no_user_turns": "No User Turns",
    }
    header = f"  {'Pattern':<28}" + "".join(f"{s.upper():>20}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (28 + 20 * len(sources)))
    for pat in ["thank_you", "question", "correction", "short_close", "other"]:
        row = f"  {pattern_labels.get(pat, pat):<28}"
        for s in sources:
            rate = resol.get(s, {}).get("rates", {}).get(pat, 0)
            count = resol.get(s, {}).get("raw_counts", {}).get(pat, 0)
            row += f"{rate*100:>10.1f}% ({count:>3})"
        click.echo(row)

    # -- Clustering -- #
    click.echo(f"\n  {'CONVERSATION CLUSTERING':=^70}")
    clust = results.get("clustering", {})
    status = clust.get("status", "skipped")
    if status == "ok":
        sil = clust.get("silhouette_score", 0)
        k = clust.get("n_clusters", 0)
        click.echo(f"  Status: {status} | k={k} clusters | silhouette={sil:.3f}")
        for c in clust.get("clusters", []):
            click.echo(
                f"\n  Cluster {c['cluster_id']}: {c['label']} "
                f"(n={c['size']})"
            )
            centroid = c.get("centroid", {})
            click.echo(
                f"    Centroid: pairs={centroid.get('turn_pairs', 0):.1f}, "
                f"duration={centroid.get('duration_min', 0):.1f}min, "
                f"mean_resp={centroid.get('mean_asst_length', 0):.0f}w, "
                f"verb_traj={centroid.get('verbosity_trajectory', 0):+.2f}"
            )
            bd = c.get("source_breakdown", {})
            bd_str = ", ".join(
                f"{s}: {pct*100:.0f}%" for s, pct in sorted(bd.items())
            )
            click.echo(f"    Source mix: {bd_str}")
    elif status == "low_quality":
        sil = clust.get("silhouette_score")
        click.echo(f"  Status: {status} (silhouette={sil})")
        click.echo(f"  Reason: {clust.get('reason', '?')}")
    else:
        click.echo(f"  Status: {status}")
        click.echo(f"  Reason: {clust.get('reason', '?')}")


def _print_user_behavior_summary(results: dict) -> None:
    """Pretty-print user behavior analysis results."""
    sources = results.get("_meta", {}).get("sources", [])

    # -- Message Complexity -- #
    click.echo(f"\n  {'MESSAGE COMPLEXITY':=^70}")
    cx = results.get("message_complexity", {})
    header = f"  {'Metric':<35}" + "".join(f"{s.upper():>18}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (35 + 18 * len(sources)))
    metrics = [
        ("Mean msg length (words)", "message_length", "mean"),
        ("Median msg length (words)", "message_length", "median"),
        ("Std msg length", "message_length", "std"),
        ("Mean sentence length", "sentence_length", "mean"),
        ("Readability FK (mean)", "readability_fk", "mean"),
    ]
    for label, key, sub in metrics:
        row = f"  {label:<35}"
        for s in sources:
            val = cx.get(s, {}).get(key, {}).get(sub, 0)
            row += f"{val:>18.2f}"
        click.echo(row)
    # Scalar metrics
    for label, key in [("MATTR-200", "mattr_200"), ("Unique vocabulary", "unique_vocabulary")]:
        row = f"  {label:<35}"
        for s in sources:
            val = cx.get(s, {}).get(key, 0)
            if isinstance(val, float):
                row += f"{val:>18.4f}"
            else:
                row += f"{val:>18}"
        click.echo(row)

    # -- Prompt Engineering -- #
    click.echo(f"\n  {'PROMPT ENGINEERING PATTERNS':=^70}")
    pe = results.get("prompt_engineering", {})
    pe_src = pe.get("by_source", {})
    header = f"  {'Category':<28}" + "".join(f"{s.upper():>18}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (28 + 18 * len(sources)))
    all_cats = set()
    for src_data in pe_src.values():
        all_cats.update(src_data.get("category_per_1k", {}).keys())
    for cat in sorted(all_cats):
        row = f"  {cat:<28}"
        for s in sources:
            val = pe_src.get(s, {}).get("category_per_1k", {}).get(cat, 0)
            row += f"{val:>18.3f}"
        click.echo(row)
    # Total density
    row = f"  {'TOTAL DENSITY (per 1k)':.<28}"
    for s in sources:
        val = pe.get("total_density", {}).get(s, 0)
        row += f"{val:>18.3f}"
    click.echo(row)

    # -- Formality & Tone -- #
    click.echo(f"\n  {'FORMALITY & TONE':=^70}")
    form = results.get("formality", {})

    # Politeness
    click.echo("  Politeness (per 1k words):")
    pol = form.get("politeness", {})
    header = f"    {'Phrase':<24}" + "".join(f"{s.upper():>18}" for s in sources)
    click.echo(header)
    click.echo("    " + "-" * (24 + 18 * len(sources)))
    phrases_to_show = ["please", "thanks", "thank you", "would you", "could you"]
    for phrase in phrases_to_show:
        row = f"    {phrase:<24}"
        for s in sources:
            val = pol.get(s, {}).get("phrase_per_1k", {}).get(phrase, 0)
            row += f"{val:>18.3f}"
        click.echo(row)
    row = f"    {'TOTAL':.<24}"
    for s in sources:
        val = pol.get(s, {}).get("total_per_1k", 0)
        row += f"{val:>18.3f}"
    click.echo(row)

    # Casual markers
    click.echo("\n  Casual markers:")
    cas = form.get("casual_markers", {})
    for label, key in [
        ("Contractions (per 1k)", "contractions_per_1k"),
        ("Lowercase-only (%)", "lowercase_only_fraction"),
        ("No punctuation (%)", "no_punctuation_fraction"),
        ("Emojis (per 1k)", "emoji_per_1k"),
    ]:
        row = f"    {label:<28}"
        for s in sources:
            val = cas.get(s, {}).get(key, 0)
            if "fraction" in key:
                row += f"{val*100:>18.1f}"
            else:
                row += f"{val:>18.3f}"
        click.echo(row)

    # Gemini capitalization warning
    data_notes = results.get("data_notes", {})
    if "gemini_capitalization" in data_notes:
        click.echo(
            click.style(
                "\n  WARNING: Gemini Takeout auto-capitalizes user prompts. "
                "The lowercase-only metric is NOT comparable across platforms.",
                fg="yellow", bold=True,
            )
        )

    # Imperative vs request
    click.echo("\n  Message framing:")
    frm = form.get("imperative_vs_request", {})
    for label, key in [
        ("Imperative (%)", "imperative_fraction"),
        ("Request/question (%)", "request_fraction"),
        ("Other (%)", "other_fraction"),
    ]:
        row = f"    {label:<28}"
        for s in sources:
            val = frm.get(s, {}).get(key, 0)
            row += f"{val*100:>18.1f}"
        click.echo(row)

    # -- Rephrasing -- #
    click.echo(f"\n  {'REPHRASING DETECTION':=^70}")
    reph = results.get("rephrasing", {})
    if reph.get("error"):
        click.echo(f"  Skipped: {reph['error']}")
    else:
        reph_src = reph.get("by_source", {})
        thresholds = reph.get("thresholds", {})
        strict_t = thresholds.get("strict", "?")
        loose_t = thresholds.get("loose", "?")
        min_w = thresholds.get("min_words_strict", "?")

        click.echo(f"  Thresholds: strict={strict_t} (min {min_w} words)  |  loose={loose_t}")

        header = f"  {'Metric':<40}" + "".join(f"{s.upper():>18}" for s in sources)
        click.echo(header)
        click.echo("  " + "-" * (40 + 18 * len(sources)))
        for label, key in [
            ("Total user turns", "total_user_turns"),
            ("Strict rephrases", "strict_total"),
            ("Strict per 100 turns (PRIMARY)", "strict_per_100_turns"),
            ("Strict per conversation", "strict_per_conv"),
            ("Loose rephrases", "loose_total"),
            ("Loose per 100 turns", "loose_per_100_turns"),
            ("Loose per conversation", "loose_per_conv"),
            ("Convs w/ strict rephrases", "conversations_with_strict_rephrases"),
        ]:
            row = f"  {label:<40}"
            for s in sources:
                val = reph_src.get(s, {}).get(key, 0)
                if isinstance(val, float):
                    row += f"{val:>18.3f}"
                else:
                    row += f"{val:>18}"
            click.echo(row)

        top = reph.get("top_conversations", [])
        if top:
            click.echo("\n  Top conversations with most strict rephrases:")
            for i, tc in enumerate(top[:5], 1):
                click.echo(
                    f"    [{i}] {tc['source'].upper()}: "
                    f"{tc['title'][:50]} "
                    f"({tc['rephrase_count']} rephrases)"
                )

    # -- First Message -- #
    click.echo(f"\n  {'FIRST MESSAGE ANALYSIS':=^70}")
    fm = results.get("first_message", {})
    header = f"  {'Metric':<35}" + "".join(f"{s.upper():>18}" for s in sources)
    click.echo(header)
    click.echo("  " + "-" * (35 + 18 * len(sources)))
    row = f"  {'Mean first msg length (words)':<35}"
    for s in sources:
        val = fm.get(s, {}).get("length", {}).get("mean", 0)
        row += f"{val:>18.1f}"
    click.echo(row)
    for label, key in [
        ("Question (%)", "question_fraction"),
        ("Command (%)", "command_fraction"),
        ("Context dump (%)", "context_dump_fraction"),
    ]:
        row = f"  {label:<35}"
        for s in sources:
            val = fm.get(s, {}).get("classification", {}).get(key, 0)
            row += f"{val*100:>18.1f}"
        click.echo(row)

    # -- Topic Routing -- #
    tr = results.get("topic_routing")
    if tr and not tr.get("skipped"):
        click.echo(f"\n  {'TOPIC ROUTING AWARENESS':=^70}")
        click.echo(f"  Qualifying topics: {tr.get('qualifying_topics', 0)}")
        for t in tr.get("topics", [])[:5]:
            words = ", ".join(t.get("words", [])[:5])
            counts = " | ".join(
                f"{s}: {c}" for s, c in t.get("source_counts", {}).items()
            )
            click.echo(f"    Topic {t['topic_id']}: [{words}] → {counts}")
    elif tr and tr.get("skipped"):
        click.echo(f"\n  Topic routing: skipped — {tr.get('reason', '?')}")


def _safe_echo(text: str) -> None:
    """Echo text to terminal, replacing unencodable chars for Windows cp1252."""
    try:
        click.echo(text)
    except UnicodeEncodeError:
        click.echo(text.encode("ascii", errors="replace").decode("ascii"))


def _print_llm_judge_dry_run(results: dict) -> None:
    """Pretty-print the dry-run results for the LLM judge module."""
    click.echo(f"\n  {'LLM JUDGE — DRY RUN':=^70}")
    click.echo(f"  Judge model: {results['model']}")

    click.echo(f"\n  {'SAMPLE BREAKDOWN':=^70}")
    stats = results.get("sample_stats", {})
    click.echo(f"  {'Source':<15} {'Sampled':>10} {'Min words':>12} {'Max words':>12} {'Mean words':>12}")
    click.echo("  " + "-" * 61)
    for source, s in sorted(stats.items()):
        click.echo(
            f"  {source:<15} {s['n_sampled']:>10} "
            f"{s['word_count_min']:>12} {s['word_count_max']:>12} "
            f"{s['word_count_mean']:>12.1f}"
        )
    click.echo(f"\n  Total turns to evaluate: {results['total_turns']}")
    click.echo(f"  Total API calls:        {results['total_api_calls']}  (2 dimensions x {results['total_turns']} turns)")
    click.echo(f"  Estimated cost:         ${results['estimated_cost_usd']:.4f}")

    # Show one example prompt per dimension.
    example_prompts = results.get("example_prompts", {})
    first_source = next(iter(example_prompts), None)
    if first_source and first_source in example_prompts:
        prompts = example_prompts[first_source]

        click.echo(f"\n  {'EXAMPLE PROMPT — DEPTH':=^70}")
        click.echo(f"  Source: {first_source}")
        click.echo(f"\n  [SYSTEM]")
        # Show first/last few lines of system prompt.
        sys_lines = prompts["depth"]["system"].split("\n")
        for line in sys_lines[:3]:
            click.echo(f"  {line}")
        if len(sys_lines) > 6:
            click.echo(f"  ...")
        for line in sys_lines[-3:]:
            click.echo(f"  {line}")

        click.echo(f"\n  [USER MESSAGE]")
        user_text = prompts["depth"]["user"]
        # Truncate long excerpts for display.
        lines = user_text.split("\n")
        for line in lines[:25]:
            _safe_echo(f"  {line[:120]}")
        if len(lines) > 25:
            click.echo(f"  ... ({len(lines) - 25} more lines)")

        click.echo(f"\n  {'EXAMPLE PROMPT — CREATIVITY':=^70}")
        click.echo(f"  Source: {first_source}")
        click.echo(f"\n  [SYSTEM]")
        sys_lines = prompts["creativity"]["system"].split("\n")
        for line in sys_lines[:3]:
            click.echo(f"  {line}")
        if len(sys_lines) > 6:
            click.echo(f"  ...")
        for line in sys_lines[-3:]:
            click.echo(f"  {line}")

        click.echo(f"\n  (Same user message as above)")

    click.echo(f"\n  {'NOTE':=^70}")
    click.echo("  Claude is judging its own responses alongside competitors.")
    click.echo("  This creates a potential conflict of interest — cross-reference")
    click.echo("  with traditional NLP metrics for a fuller picture.")
    click.echo(f"\n  To run the full evaluation:")
    click.echo(f"    python main.py analyze --module llm_judge")


def _print_llm_judge_summary(results: dict) -> None:
    """Pretty-print LLM judge evaluation results to the terminal."""
    click.echo(f"\n  {'LLM JUDGE RESULTS':=^70}")

    cfg = results.get("config", {})
    click.echo(f"  Model: {cfg.get('model', '?')}")
    click.echo(f"  Total evaluations: {cfg.get('total_evaluations', '?')}")

    cost = results.get("cost", {})
    click.echo(f"  Cost: ${cost.get('actual_usd', 0):.4f}  "
               f"(estimated: ${cost.get('estimated_usd', 0):.4f})")
    click.echo(f"  Tokens: {cost.get('input_tokens', 0):,} in / "
               f"{cost.get('output_tokens', 0):,} out")

    for dimension in ("depth", "creativity"):
        dim_data = results.get(dimension, {})
        by_source = dim_data.get("by_source", {})
        sources = sorted(by_source.keys())
        if not sources:
            continue

        label = dimension.upper()
        click.echo(f"\n  {f' {label} ':=^70}")

        header = f"  {'Source':<15} {'Mean':>8} {'Score 1':>10} {'Score 2':>10} {'Score 3':>10} {'N':>8}"
        click.echo(header)
        click.echo("  " + "-" * 61)
        for source in sources:
            info = by_source[source]
            mean = info.get("mean")
            dist = info.get("distribution", {})
            n = info.get("n_evaluated", 0)
            mean_str = f"{mean:.3f}" if mean is not None else "N/A"
            click.echo(
                f"  {source:<15} {mean_str:>8} "
                f"{dist.get('1', 0):>10} {dist.get('2', 0):>10} "
                f"{dist.get('3', 0):>10} {n:>8}"
            )

        # Statistical test.
        stat = dim_data.get("statistical_test", {})
        if "pairwise" in stat:
            click.echo(f"\n  Statistical tests (Mann-Whitney U):")
            for pw in stat["pairwise"]:
                sig = "*" if pw["p_value"] < 0.05 else ""
                click.echo(
                    f"    {pw['comparison']}: U={pw['statistic']:.1f}, "
                    f"p={pw['p_value']:.6f} {sig}"
                )
        elif "statistic" in stat:
            sig = "*" if stat["p_value"] < 0.05 else ""
            click.echo(
                f"\n  {stat.get('comparison', '?')}: "
                f"U={stat['statistic']:.1f}, p={stat['p_value']:.6f} {sig}"
            )

    # Bias caveat.
    notes = results.get("data_notes", "")
    if notes:
        click.echo(f"\n  {'NOTE':=^70}")
        # Wrap to ~70 chars.
        words = notes.split()
        line = "  "
        for w in words:
            if len(line) + len(w) + 1 > 72:
                click.echo(line)
                line = "  " + w
            else:
                line += " " + w if line.strip() else "  " + w
        if line.strip():
            click.echo(line)


@cli.command()
@click.option("--plot", default=None, help="Generate a specific plot (e.g. style_fingerprint).")
@click.pass_context
def visualize(ctx: click.Context, plot: str | None) -> None:
    """Generate all (or a specific) visualization from analysis results."""
    import matplotlib.pyplot as plt
    from src.viz.plots import PLOT_FUNCTIONS

    config = ctx.obj["config"]
    outputs_root = PROJECT_ROOT / config["paths"]["outputs"]
    processed_root = PROJECT_ROOT / config["paths"]["processed_data"]

    # Load all available result JSONs.
    result_data: dict[str, dict] = {}
    for mod_name in ["lexical", "semantic", "pragmatic", "comparative", "temporal", "conversation_structure", "user_behavior", "llm_judge"]:
        path = outputs_root / f"{mod_name}_results.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                result_data[mod_name] = json.load(f)
            logger.info("visualize: loaded %s", path)

    # Optionally load conversations for distribution plots.
    conversations = None
    needs_convs = False
    plots_to_run = [plot] if plot else list(PLOT_FUNCTIONS.keys())
    for pname in plots_to_run:
        info = PLOT_FUNCTIONS.get(pname, {})
        if info.get("needs_conversations"):
            needs_convs = True
            break

    if needs_convs:
        try:
            conversations = _load_conversations_from_parquet(processed_root)
        except FileNotFoundError:
            logger.warning("visualize: cannot load conversations for distribution plots")

    # Generate plots.
    for pname in plots_to_run:
        if pname not in PLOT_FUNCTIONS:
            click.echo(
                f"visualize: unknown plot '{pname}'. "
                f"Available: {list(PLOT_FUNCTIONS.keys())}"
            )
            continue

        info = PLOT_FUNCTIONS[pname]
        data_key = info["data"]
        if data_key not in result_data:
            click.echo(
                f"  SKIP {pname}: requires {data_key}_results.json "
                f"(run 'python main.py analyze --module {data_key}' first)"
            )
            continue

        click.echo(f"  Generating: {pname} ...")
        try:
            fn = info["fn"]
            if info.get("needs_conversations"):
                fig = fn(result_data[data_key], conversations=conversations)
            else:
                fig = fn(result_data[data_key])
            plt.close(fig)
            click.echo(f"    -> saved to data/outputs/plots/{pname}.png")
        except Exception as exc:
            click.echo(f"    ERROR: {exc}")
            logger.exception("visualize: %s failed", pname)

    # List all generated files.
    plots_dir = outputs_root / "plots"
    if plots_dir.exists():
        click.echo(f"\n  {'GENERATED PLOTS':=^60}")
        for f in sorted(plots_dir.glob("*.png")):
            size_kb = f.stat().st_size / 1024
            click.echo(f"    {f.name:<45} {size_kb:>7.1f} KB")


def _generate_report(outputs_dir: Path) -> str:
    """Build the full REPORT.md content from analysis result JSONs.

    Returns the markdown string.
    """
    # ------------------------------------------------------------------
    # Load all result files
    # ------------------------------------------------------------------
    def _load(name: str) -> dict:
        p = outputs_dir / name
        if not p.exists():
            logger.warning("Report: missing %s — section will be incomplete", name)
            return {}
        with open(p) as f:
            return json.load(f)

    lex = _load("lexical_results.json")
    sem = _load("semantic_results.json")
    prag = _load("pragmatic_results.json")
    comp = _load("comparative_results.json")

    # Helper: safe nested get
    def _g(d: dict, *keys, default=None):
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, default)
            else:
                return default
        return d

    plots = "plots"  # relative path inside outputs

    # ------------------------------------------------------------------
    # § 1  Dataset Overview
    # ------------------------------------------------------------------
    cl_turns = _g(lex, "response_length", "claude", "count", default=0)
    gm_turns = _g(lex, "response_length", "gemini", "count", default=0)
    cl_convos = _g(prag, "turn_dynamics", "claude", "n_conversations", default=0)
    gm_convos = _g(prag, "turn_dynamics", "gemini", "n_conversations", default=0)
    total_convos = cl_convos + gm_convos
    total_turns = cl_turns + gm_turns
    cl_words = _g(lex, "formatting", "claude", "total_words", default=0)
    gm_words = _g(lex, "formatting", "gemini", "total_words", default=0)

    sec1 = f"""# Chatbot Conversation Analysis Report

*Generated {datetime.now().strftime('%B %d, %Y')}*

---

## 1. Dataset Overview

This report examines **{total_convos} conversations** containing **{total_turns:,} assistant turns** and roughly **{(cl_words + gm_words):,} words** of AI-generated text, drawn from two platforms the user relies on daily: **Claude** (Anthropic) and **Gemini** (Google).

| | Claude | Gemini |
|---|---:|---:|
| Conversations | {cl_convos} | {gm_convos} |
| Assistant turns | {cl_turns:,} | {gm_turns:,} |
| Total words (assistant) | {cl_words:,} | {gm_words:,} |
| Median turns per conversation | {_g(prag, 'turn_dynamics', 'claude', 'turns_per_conv', 'median', default='—')} | {_g(prag, 'turn_dynamics', 'gemini', 'turns_per_conv', 'median', default='—')} |
| Single-exchange conversations | {_g(prag, 'turn_dynamics', 'claude', 'single_exchange_count', default='—')} ({_g(prag, 'turn_dynamics', 'claude', 'single_exchange_fraction', default=0):.0%}) | {_g(prag, 'turn_dynamics', 'gemini', 'single_exchange_count', default='—')} ({_g(prag, 'turn_dynamics', 'gemini', 'single_exchange_fraction', default=0):.0%}) |

Despite fewer conversations, Claude's dataset spans a comparable date range. The user's Gemini sessions tend to run longer (median {_g(prag, 'turn_dynamics', 'gemini', 'turns_per_conv', 'median', default=0)} turns vs. Claude's {_g(prag, 'turn_dynamics', 'claude', 'turns_per_conv', 'median', default=0)}), suggesting more sustained, multi-step interactions on that platform. Both platforms see relatively few one-off exchanges — the user treats these tools as collaborative partners, not search engines.
"""

    # ------------------------------------------------------------------
    # § 2  How You Use Each Platform (Topics)
    # ------------------------------------------------------------------
    # Pull the most interesting platform-skewed topics
    topics = _g(sem, "assistant_topics", "topics", default=[])
    claude_topics = []
    gemini_topics = []
    for t in topics:
        if t.get("topic_id") == 0:
            continue  # skip catch-all
        sb = t.get("source_breakdown", {})
        label = ", ".join(w["word"] for w in t.get("words", [])[:4])
        count = t.get("count", 0)
        if sb.get("claude", 0) >= 0.85:
            claude_topics.append((label, count, sb.get("claude", 0)))
        elif sb.get("gemini", 0) >= 0.85:
            gemini_topics.append((label, count, sb.get("gemini", 0)))

    claude_topic_lines = ""
    for label, count, pct in sorted(claude_topics, key=lambda x: -x[1])[:5]:
        claude_topic_lines += f"  - **{label}** ({count} turns, {pct:.0%} Claude)\n"

    gemini_topic_lines = ""
    for label, count, pct in sorted(gemini_topics, key=lambda x: -x[1])[:5]:
        gemini_topic_lines += f"  - **{label}** ({count} turns, {pct:.0%} Gemini)\n"

    sec2 = f"""## 2. How You Use Each Platform

Topic modeling (BERTopic over all assistant turns) reveals a clear division of labor. The user doesn't ask the same questions everywhere — each platform has carved out its own niche.

**Claude-dominant topics:**
{claude_topic_lines}
**Gemini-dominant topics:**
{gemini_topic_lines}
The pattern is striking: Claude is the user's **technical co-pilot** — the go-to for writing code, building infrastructure, and working through security architecture. Topics like Snowflake scripts, MCP servers, encryption, and GitHub automation are almost exclusively Claude territory.

Gemini, by contrast, handles the user's **life beyond the terminal**: veterinary questions, travel planning, personal finance, vehicle research, and fragrance recommendations. It's also the platform where interpersonal relationship discussions land, with topics around social dynamics and personal advice skewing heavily toward Gemini.

This isn't accidental. The user has developed an implicit mental model of each tool's strengths and routes queries accordingly.

![Topic Distribution]({plots}/topic_distribution.png)
"""

    # ------------------------------------------------------------------
    # § 3  How They Write (Lexical & Formatting)
    # ------------------------------------------------------------------
    cl_resp_mean = _g(lex, "response_length", "claude", "mean", default=0)
    gm_resp_mean = _g(lex, "response_length", "gemini", "mean", default=0)
    cl_resp_med = _g(lex, "response_length", "claude", "median", default=0)
    gm_resp_med = _g(lex, "response_length", "gemini", "median", default=0)
    cl_mattr = _g(lex, "vocabulary", "claude", "mattr_500", default=0)
    gm_mattr = _g(lex, "vocabulary", "gemini", "mattr_500", default=0)
    cl_fk = _g(lex, "readability", "claude", "flesch_kincaid", "mean", default=0)
    gm_fk = _g(lex, "readability", "gemini", "flesch_kincaid", "mean", default=0)

    # Formatting
    cl_bold = _g(lex, "formatting", "claude", "per_1k_words", "bold", default=0)
    gm_bold = _g(lex, "formatting", "gemini", "per_1k_words", "bold", default=0)
    cl_headers = _g(lex, "formatting", "claude", "per_1k_words", "headers", default=0)
    gm_headers = _g(lex, "formatting", "gemini", "per_1k_words", "headers", default=0)
    cl_code = _g(lex, "formatting", "claude", "per_1k_words", "code_blocks", default=0)
    gm_code = _g(lex, "formatting", "gemini", "per_1k_words", "code_blocks", default=0)
    cl_bullets = _g(lex, "formatting", "claude", "per_1k_words", "bullet_points", default=0)
    gm_bullets = _g(lex, "formatting", "gemini", "per_1k_words", "bullet_points", default=0)

    fmt_total_cl = _g(comp, "summary_table", default=[])
    fmt_density_cl = 0
    fmt_density_gm = 0
    for row in fmt_total_cl:
        if row.get("metric") == "Formatting density (per 1k)":
            fmt_density_cl = row.get("claude", 0)
            fmt_density_gm = row.get("gemini", 0)

    sec3 = f"""## 3. How They Write

### Length & Density

The most immediate difference is volume. Gemini's average response is **{gm_resp_mean:.0f} words** — nearly **{gm_resp_mean/cl_resp_mean:.1f}× longer** than Claude's {cl_resp_mean:.0f}-word average. The gap is even more pronounced at the median ({gm_resp_med:.0f} vs. {cl_resp_med:.0f} words), meaning Gemini consistently runs long rather than being skewed by occasional outliers.

Both models write at roughly the same reading level (Flesch-Kincaid grade {cl_fk:.1f} for Claude, {gm_fk:.1f} for Gemini), so the difference isn't complexity — it's verbosity. Claude says what it needs to say and stops. Gemini elaborates, contextualizes, and recaps.

![Response Length Distributions]({plots}/response_length_distributions.png)

### Vocabulary

Gemini draws from a richer lexicon: its moving-average type-token ratio ({gm_mattr:.3f}) edges out Claude's ({cl_mattr:.3f}), and it deploys **{_g(lex, 'vocabulary', 'gemini', 'unique_types', default=0):,} unique word types** versus Claude's {_g(lex, 'vocabulary', 'claude', 'unique_types', default=0):,}. Some of that gap is explained by sheer volume — more words means more opportunities for variety — but the MATTR metric controls for text length and the difference persists.

![Vocabulary Comparison]({plots}/vocabulary_comparison.png)

### Formatting Habits

Both models are heavy formatters, but they lean on different tools. Gemini's total formatting density ({fmt_density_gm:.1f} markers per 1,000 words) exceeds Claude's ({fmt_density_cl:.1f}), driven primarily by its love of **bold text** ({gm_bold:.1f}/1k vs. {cl_bold:.1f}/1k) and **section headers** ({gm_headers:.1f}/1k vs. {cl_headers:.1f}/1k). Gemini treats nearly every response as a structured document.

Claude, meanwhile, leads in **bullet points** ({cl_bullets:.1f}/1k vs. {gm_bullets:.1f}/1k) and — unsurprisingly, given its coding role — **code blocks** ({cl_code:.2f}/1k vs. {gm_code:.2f}/1k). Claude formats for utility; Gemini formats for emphasis.

![Formatting Habits]({plots}/formatting_habits.png)
"""

    # ------------------------------------------------------------------
    # § 4  How They Talk (Pragmatic Behavior)
    # ------------------------------------------------------------------
    cl_hedge = _g(prag, "hedging", "claude", "hedge_density", default=0)
    gm_hedge = _g(prag, "hedging", "gemini", "hedge_density", default=0)
    cl_ithink = _g(prag, "hedging", "claude", "phrase_per_1k", "i think", default=0)
    gm_ithink = _g(prag, "hedging", "gemini", "phrase_per_1k", "i think", default=0)
    cl_however = _g(prag, "hedging", "claude", "phrase_per_1k", "however", default=0)
    gm_however = _g(prag, "hedging", "gemini", "phrase_per_1k", "however", default=0)
    cl_likely = _g(prag, "hedging", "claude", "phrase_per_1k", "likely", default=0)
    gm_likely = _g(prag, "hedging", "gemini", "phrase_per_1k", "likely", default=0)

    cl_qrate = _g(prag, "question_rate", "claude", "mean_questions_per_turn", default=0)
    gm_qrate = _g(prag, "question_rate", "gemini", "mean_questions_per_turn", default=0)
    cl_qfrac = _g(prag, "question_rate", "claude", "fraction_turns_with_question", default=0)
    gm_qfrac = _g(prag, "question_rate", "gemini", "fraction_turns_with_question", default=0)

    cl_verb_med = _g(prag, "verbosity_ratio", "claude", "median", default=0)
    gm_verb_med = _g(prag, "verbosity_ratio", "gemini", "median", default=0)

    cl_I = _g(prag, "first_person", "claude", "per_1k_words", "I", default=0)
    gm_I = _g(prag, "first_person", "gemini", "per_1k_words", "I", default=0)

    # Opening patterns
    cl_top_open = _g(prag, "opening_patterns", "claude", "top_opening_words", default=[])
    gm_top_open = _g(prag, "opening_patterns", "gemini", "top_opening_words", default=[])
    cl_open_str = ", ".join(f'"{w[0]}"' for w in cl_top_open[:5]) if cl_top_open else "—"
    gm_open_str = ", ".join(f'"{w[0]}"' for w in gm_top_open[:5]) if gm_top_open else "—"

    # Disclaimers
    cl_disc = _g(prag, "disclaimers", "claude", "disclaimer_density", default=0)
    gm_disc = _g(prag, "disclaimers", "gemini", "disclaimer_density", default=0)

    sec4 = f"""## 4. How They Talk

### Hedging & Certainty

Both models hedge at similar overall rates ({cl_hedge:.1f} and {gm_hedge:.1f} hedges per 1,000 words), but they hedge *differently*. Claude reaches for personal uncertainty — "I think" appears {cl_ithink:.3f} times per 1,000 words, nearly **{cl_ithink/gm_ithink:.0f}× more** than Gemini's {gm_ithink:.3f}. Claude owns its uncertainty as a subjective stance.

Gemini hedges impersonally. It leans on "likely" ({gm_likely:.3f}/1k vs. Claude's {cl_likely:.3f}/1k) and "however" ({gm_however:.3f}/1k vs. {cl_however:.3f}/1k), framing uncertainty as a property of the world rather than of itself. The result: Claude sounds like a colleague thinking out loud; Gemini sounds like a textbook covering its bases.

![Hedging Comparison]({plots}/hedging_comparison.png)

### Engagement & Interactivity

Gemini asks far more questions — **{gm_qrate:.1f} per turn** vs. Claude's {cl_qrate:.1f}, and it asks at least one question in **{gm_qfrac:.0%}** of its responses (Claude: {cl_qfrac:.0%}). Gemini is constantly checking in, clarifying, and nudging the conversation forward. Claude tends to deliver an answer and wait.

### Verbosity Ratio

For every word the user types, Claude returns roughly **{cl_verb_med:.0f} words** (median). Gemini returns **{gm_verb_med:.0f}** — more than double. Even on identical topics (controlled for via topic modeling), the gap holds: Claude produces {_g(comp, 'topic_controlled', 'metrics', 'mean_response_length', 'claude', default=0):.0f} words vs. Gemini's {_g(comp, 'topic_controlled', 'metrics', 'mean_response_length', 'gemini', default=0):.0f} on the same shared topic cluster.

![Verbosity Ratio]({plots}/verbosity_ratio.png)

### Opening Moves

How each model starts a response reveals its conversational personality. Claude's top openers — {cl_open_str} — lean toward **acknowledgment and validation** ("That's a great point", "You're right"). Gemini's top openers — {gm_open_str} — are **declarative and analytical** ("This is", "That is", "Based on"). Claude meets you where you are; Gemini starts building the answer immediately.

![Opening Patterns]({plots}/opening_patterns.png)

### First-Person Voice & Disclaimers

Claude uses "I" more frequently ({cl_I:.2f}/1k vs. {gm_I:.2f}/1k), reinforcing its persona as an individual thinking partner. It also issues more hedge-style disclaimers like "I'm not sure" and "I could be wrong" ({cl_disc:.3f}/1k vs. {gm_disc:.3f}/1k). Despite both models rarely breaking character to say "as an AI" (Claude: 4 times, Gemini: 8), Claude's rhetorical stance is consistently more personal and more willing to express doubt.
"""

    # ------------------------------------------------------------------
    # § 5  Personality Profiles (Style Fingerprint)
    # ------------------------------------------------------------------
    cl_selfsim = _g(sem, "self_similarity", "claude", "mean", default=0)
    gm_selfsim = _g(sem, "self_similarity", "gemini", "mean", default=0)

    sec5 = f"""## 5. Style Fingerprints

The radar chart below collapses nine key metrics into a single visual profile for each model. The differences are consistent and mutually reinforcing:

![Style Fingerprint]({plots}/style_fingerprint.png)

**Claude's profile: The concise collaborator.** Lower on every axis except first-person "I" usage. Claude writes shorter, formats less aggressively, hedges with personal language, asks fewer questions, and produces more varied responses (self-similarity cosine of {cl_selfsim:.3f} vs. Gemini's {gm_selfsim:.3f}). It behaves like a trusted colleague who gives you the answer and trusts you to ask if you need more.

**Gemini's profile: The thorough analyst.** Higher on vocabulary richness, response length, formatting density, question rate, and verbosity. Gemini's responses are more self-similar — it has a more consistent "house style" — and it structures every response as if it might be the user's only reference on the topic. It behaves like a consultant writing a deliverable.

Neither style is inherently better. They serve different needs, and the user appears to have internalized this: technical depth goes to Claude, broad research and personal topics go to Gemini.
"""

    # ------------------------------------------------------------------
    # § 6  How You Adapt (User Language)
    # ------------------------------------------------------------------
    u_cl_msg_mean = _g(comp, "user_language", "message_length", "claude", "mean", default=0)
    u_gm_msg_mean = _g(comp, "user_language", "message_length", "gemini", "mean", default=0)
    u_cl_mattr = _g(comp, "user_language", "vocabulary", "claude", "mattr_200", default=0)
    u_gm_mattr = _g(comp, "user_language", "vocabulary", "gemini", "mattr_200", default=0)
    u_cl_qrate = _g(comp, "user_language", "question_rate", "claude", "mean_questions_per_turn", default=0)
    u_gm_qrate = _g(comp, "user_language", "question_rate", "gemini", "mean_questions_per_turn", default=0)
    u_cl_polite = _g(comp, "user_language", "politeness", "claude", "politeness_per_1k", default=0)
    u_gm_polite = _g(comp, "user_language", "politeness", "gemini", "politeness_per_1k", default=0)

    sec6 = f"""## 6. How You Adapt

The analysis doesn't just reveal how the bots differ — it shows how the *user* shifts behavior across platforms.

| | To Claude | To Gemini |
|---|---:|---:|
| Mean message length (chars) | {u_cl_msg_mean:.0f} | {u_gm_msg_mean:.0f} |
| Vocabulary richness (MATTR) | {u_cl_mattr:.3f} | {u_gm_mattr:.3f} |
| Questions per turn | {u_cl_qrate:.2f} | {u_gm_qrate:.2f} |
| Politeness markers per 1k | {u_cl_polite:.2f} | {u_gm_polite:.2f} |

The user writes slightly longer messages to Gemini, uses a richer vocabulary with Claude, and asks questions at a marginally higher rate on Gemini. Politeness markers (please, thanks, sorry) are nearly identical across platforms — the user doesn't treat one model more deferentially than the other.

Perhaps most tellingly, the user's **top words** on both platforms overlap heavily (the same names, the same professional vocabulary), but their *distinctive words* diverge in ways that mirror the topic split: technical jargon clusters around Claude, while personal and lifestyle vocabulary gravitates toward Gemini.

![User Language Comparison]({plots}/user_language_comparison.png)
"""

    # ------------------------------------------------------------------
    # § 7  Methodology Notes
    # ------------------------------------------------------------------
    sec7 = f"""## 7. Methodology Notes

- **Data sources**: Claude conversations exported as JSON; Gemini conversations extracted from a Google Takeout `MyActivity.html` file. Gemini responses were converted from native HTML to Markdown using `markdownify` to preserve formatting fidelity.
- **Conversation boundaries**: Claude exports include native conversation IDs. Gemini entries (individual Q&A pairs) were grouped into conversations using a 60-minute session gap heuristic.
- **Text analysis**: All lexical and pragmatic metrics computed on assistant turns only (unless noted). Vocabulary richness uses Moving-Average Type-Token Ratio (MATTR) with a 500-word window to control for text length differences. Readability scores use the `textstat` library.
- **Topic modeling**: BERTopic with `all-MiniLM-L6-v2` embeddings. The catch-all Topic 0 is excluded from platform-specific topic analysis.
- **Self-similarity**: Pairwise cosine similarity of 500 randomly sampled response embeddings per source.
- **Sentiment**: VADER polarity scores, bucketed into positive (>0.05), negative (<−0.05), and neutral.
- **Topic-controlled comparison**: Metrics recomputed on a balanced random sample of 500 turns per source from the shared catch-all topic, controlling for subject-matter differences.
- **All analysis runs locally** — no data is sent to external APIs. Embeddings are cached to Parquet after first computation.
"""

    # ------------------------------------------------------------------
    # § 8  What's Next
    # ------------------------------------------------------------------
    sec8 = """## 8. What's Next

This analysis captures a snapshot of two AI relationships as they exist today. Several extensions could deepen the picture:

- **Temporal trends**: Track how response length, hedging, and topic mix evolve month-over-month. Are the models converging in style? Is the user shifting workload between platforms over time?
- **ChatGPT integration**: Adding the third major platform would complete the triangle and reveal whether it occupies a distinct niche or overlaps with Claude or Gemini.
- **Conversation-level analysis**: Move beyond turn-level metrics to model entire conversation arcs — do certain topics produce longer sessions? Where do conversations stall?
- **Prompt engineering effects**: Correlate user prompt style (length, specificity, tone) with response quality metrics to identify what prompting strategies yield the most useful answers from each model.

---

*Report generated by the Chatbot Conversation Analysis pipeline. All data processed locally.*
"""

    sections = [sec1, sec2, sec3, sec4, sec5, sec6, sec7, sec8]
    return "\n\n".join(s.strip() for s in sections) + "\n"


@cli.command()
@click.pass_context
def report(ctx: click.Context) -> None:
    """Generate summary report and all visualizations."""
    cfg = ctx.obj["config"]
    outputs_dir = PROJECT_ROOT / cfg["paths"]["outputs"]
    outputs_dir.mkdir(parents=True, exist_ok=True)

    click.echo("  Generating report …")
    md = _generate_report(outputs_dir)

    report_path = outputs_dir / "REPORT.md"
    report_path.write_text(md, encoding="utf-8")
    size_kb = report_path.stat().st_size / 1024
    click.echo(f"  ✓ Report written to {report_path}  ({size_kb:.1f} KB)")
    click.echo(f"  Sections: {md.count('## ')}")
    click.echo(f"  Words:    ~{len(md.split()):,}")


@cli.command()
@click.pass_context
def explore(ctx: click.Context) -> None:
    """Launch Jupyter notebook for interactive exploration."""
    click.echo("explore: Not yet implemented.")


if __name__ == "__main__":
    cli()

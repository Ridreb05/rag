"""Dataset analysis for ai4bharat/MSMARCO-XI.

Produces, per language: corpus size (from cheap footer reads, all 14
languages), and content-level statistics (from a downloaded validation
file): query/passage length distributions, is_selected label distribution,
empty-translation rate, cross-query passage duplication rate, and a rough
script-consistency check (does the "translated" text actually contain
characters from the target language's Unicode block).

Usage:
    uv run python -m voice_rag.pipeline.ingestion.analyze --languages hi ta ur sa
    uv run python -m voice_rag.pipeline.ingestion.analyze --languages all --footer-only
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from voice_rag.pipeline.ingestion.hf_source import available_splits, download_split, read_footer_metadata
from voice_rag.pipeline.ingestion.schema import LANGUAGES, LANGUAGES_WITHOUT_TRAIN

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports/dataset_analysis")
SUMMARY_REPORT = Path("reports/dataset_analysis/summary.md")

# Rough per-language Unicode script block, used only as a coarse
# "did translation actually happen into this script" sanity check —
# not a substitute for a real language-ID model.
SCRIPT_PATTERN: dict[str, str] = {
    "hi": r"[ऀ-ॿ]",
    "mr": r"[ऀ-ॿ]",
    "ne": r"[ऀ-ॿ]",
    "sa": r"[ऀ-ॿ]",
    "bn": r"[ঀ-৿]",
    "as": r"[ঀ-৿]",
    "gu": r"[઀-૿]",
    "pa": r"[਀-੿]",
    "or": r"[଀-୿]",
    "ta": r"[஀-௿]",
    "te": r"[ఀ-౿]",
    "kn": r"[ಀ-೿]",
    "ml": r"[ഀ-ൿ]",
    "ur": r"[؀-ۿ]",
}


def _word_count_stats(series: pd.Series) -> dict[str, float]:
    lengths = series.fillna("").astype(str).str.split().map(len)
    if len(lengths) == 0:
        return {}
    return {
        "min": int(lengths.min()),
        "p50": float(lengths.quantile(0.50)),
        "p90": float(lengths.quantile(0.90)),
        "p99": float(lengths.quantile(0.99)),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
    }


def analyze_language(lang: str, sample_rows: int | None, cache_dir: str | None) -> dict[str, Any]:
    report: dict[str, Any] = {"language": lang, "language_name": LANGUAGES[lang]}

    splits = available_splits(lang)
    report["has_train_split"] = "train" in splits
    report["footer"] = {}
    for split in splits:
        meta = read_footer_metadata(lang, split)
        report["footer"][split] = {"num_rows": meta.num_rows, "num_row_groups": meta.num_row_groups}

    val_path = download_split(lang, "validation", cache_dir=cache_dir)
    table = pq.read_table(val_path)
    total_rows = table.num_rows
    n = total_rows if sample_rows is None else min(sample_rows, total_rows)
    if n == total_rows:
        df = table.to_pandas()
    else:
        # Rows are NOT randomly ordered in the source file — a head-of-file
        # slice measurably skews the is_selected distribution (verified:
        # first-20k-rows zero-relevant rate was 38.3% vs. 45.0% on the full
        # file for Hindi). Always sample uniformly at random instead.
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(total_rows, size=n, replace=False)
        idx.sort()
        df = table.take(idx.tolist()).to_pandas()
    report["rows_analyzed"] = int(n)
    report["rows_total_in_file"] = int(total_rows)
    report["sampling"] = "full" if n == total_rows else "uniform_random_seed42"

    # --- query-level stats ---
    report["query_type_distribution"] = df["query_type"].value_counts(dropna=False).to_dict()
    report["query_length_words"] = {
        "translated": _word_count_stats(df["query"]),
        "english": _word_count_stats(df["Eng_Query"]),
    }
    report["empty_translated_query_rate"] = float((df["query"].fillna("").str.strip() == "").mean())
    report["empty_translated_answer_rate"] = float((df["Answer"].fillna("").str.strip() == "").mean())

    # --- passage-level stats: explode the three parallel arrays ---
    passages = df["passages"]
    exploded_rows = []
    for qid, p in zip(df["query_id"], passages, strict=True):
        is_sel = list(p["is_selected"]) if p["is_selected"] is not None else []
        eng = list(p["English_passages"]) if p["English_passages"] is not None else []
        tr = list(p["Translated_passages"]) if p["Translated_passages"] is not None else []
        if not (len(is_sel) == len(eng) == len(tr)):
            report.setdefault("schema_warnings", []).append(
                f"query_id={qid}: non-parallel passage arrays "
                f"(is_selected={len(is_sel)}, English_passages={len(eng)}, Translated_passages={len(tr)})"
            )
            continue
        for i in range(len(is_sel)):
            exploded_rows.append(
                {
                    "query_id": qid,
                    "passage_index": i,
                    "is_selected": is_sel[i],
                    "english_passage": eng[i],
                    "translated_passage": tr[i],
                }
            )
    pdf = pd.DataFrame(exploded_rows)
    report["passage_slots_total"] = int(len(pdf))
    report["passages_per_query"] = {
        "mean": float(pdf.groupby("query_id").size().mean()) if len(pdf) else 0.0,
    }

    selected_per_query = pdf.groupby("query_id")["is_selected"].sum()
    report["is_selected_distribution"] = {
        "queries_with_zero_selected": int((selected_per_query == 0).sum()),
        "queries_with_exactly_one_selected": int((selected_per_query == 1).sum()),
        "queries_with_multiple_selected": int((selected_per_query > 1).sum()),
        "total_queries": int(len(selected_per_query)),
    }

    report["passage_length_words"] = {
        "translated": _word_count_stats(pdf["translated_passage"]),
        "english": _word_count_stats(pdf["english_passage"]),
    }
    report["empty_translated_passage_rate"] = float(
        (pdf["translated_passage"].fillna("").str.strip() == "").mean()
    )

    # --- duplicate passage detection (content hash on normalized translated text) ---
    norm = pdf["translated_passage"].fillna("").str.strip().str.lower()
    non_empty = norm[norm != ""]
    dup_counts = non_empty.value_counts()
    report["passage_duplication"] = {
        "unique_translated_passages": int(dup_counts.shape[0]),
        "total_non_empty_passage_slots": int(non_empty.shape[0]),
        "uniqueness_ratio": float(dup_counts.shape[0] / non_empty.shape[0]) if len(non_empty) else None,
        "passages_shared_across_multiple_queries": int((dup_counts > 1).sum()),
        "max_reuse_count_of_a_single_passage": int(dup_counts.max()) if len(dup_counts) else 0,
    }

    # --- rough script-consistency sanity check ---
    pattern = SCRIPT_PATTERN.get(lang)
    if pattern:
        has_script = pdf["translated_passage"].fillna("").str.contains(pattern, regex=True, na=False)
        non_empty_mask = pdf["translated_passage"].fillna("").str.strip() != ""
        denom = int(non_empty_mask.sum())
        report["script_consistency"] = {
            "expected_unicode_block": pattern,
            "fraction_of_non_empty_passages_containing_expected_script": (
                float((has_script & non_empty_mask).sum() / denom) if denom else None
            ),
        }

    # --- translation decoding metadata sanity ---
    meta = df["meta"]
    temps = sorted({m.get("temperature") for m in meta if m is not None})
    top_ps = sorted({m.get("top_p") for m in meta if m is not None})
    model_names = sorted({m.get("model_name") for m in meta if m is not None})
    report["translation_meta"] = {
        "distinct_temperature_values": temps[:20],
        "distinct_top_p_values": top_ps[:20],
        "distinct_model_names": model_names[:10],
    }

    return report


def write_markdown_summary(reports: list[dict[str, Any]], footer_only_langs: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# MSMARCO-XI Dataset Analysis Report")
    lines.append("")
    lines.append(
        "Generated by `voice_rag.pipeline.ingestion.analyze`. Corpus-size figures come from "
        "Parquet footer metadata (exact, cheap to read for all 14 languages). "
        "Content-level statistics come from a downloaded sample of each analyzed "
        "language's validation split."
    )
    lines.append("")
    lines.append("## Corpus size by language (footer metadata, exact)")
    lines.append("")
    lines.append("| Language | Code | Train rows | Validation rows |")
    lines.append("|---|---|---:|---:|")
    for row in footer_only_langs:
        train = row["footer"].get("train", {}).get("num_rows")
        val = row["footer"].get("validation", {}).get("num_rows")
        train_str = f"{train:,}" if train is not None else "**absent**"
        lines.append(f"| {row['language_name']} | `{row['language']}` | {train_str} | {val:,} |")
    lines.append("")
    lines.append(
        "> Telugu (`te`) has **no training split** in this dataset — confirmed from the "
        "repository's file listing, not inferred. Any language-conditioned fine-tuning or "
        "training-split-derived statistics must exclude Telugu or fall back to its "
        "validation split only."
    )
    lines.append("")

    for r in reports:
        lines.append(f"## {r['language_name']} (`{r['language']}`)")
        lines.append("")
        lines.append(
            f"Analyzed {r['rows_analyzed']:,} of {r['rows_total_in_file']:,} validation rows."
        )
        lines.append("")
        lines.append("**Query length (words):** " + json.dumps(r["query_length_words"]["translated"]))
        lines.append("")
        lines.append("**Passage length (words):** " + json.dumps(r["passage_length_words"]["translated"]))
        lines.append("")
        isd = r["is_selected_distribution"]
        lines.append(
            f"**Relevance labels:** {isd['queries_with_zero_selected']:,} / {isd['total_queries']:,} "
            f"queries have zero `is_selected=1` passages; "
            f"{isd['queries_with_exactly_one_selected']:,} have exactly one; "
            f"{isd['queries_with_multiple_selected']:,} have more than one."
        )
        lines.append("")
        dup = r["passage_duplication"]
        lines.append(
            f"**Passage duplication:** {dup['unique_translated_passages']:,} unique passages out of "
            f"{dup['total_non_empty_passage_slots']:,} non-empty slots "
            f"(uniqueness ratio {dup['uniqueness_ratio']:.3f}); "
            f"{dup['passages_shared_across_multiple_queries']:,} passages are reused across "
            f"more than one query (max reuse count: {dup['max_reuse_count_of_a_single_passage']})."
        )
        lines.append("")
        lines.append(
            f"**Empty translated passage rate:** {r['empty_translated_passage_rate']:.4%} · "
            f"**Empty translated query rate:** {r['empty_translated_query_rate']:.4%}"
        )
        lines.append("")
        sc = r.get("script_consistency")
        if sc and sc["fraction_of_non_empty_passages_containing_expected_script"] is not None:
            lines.append(
                f"**Script consistency:** {sc['fraction_of_non_empty_passages_containing_expected_script']:.2%} "
                f"of non-empty translated passages contain at least one character from the expected "
                f"Unicode block ({sc['expected_unicode_block']})."
            )
            lines.append("")
        warnings = r.get("schema_warnings")
        if warnings:
            lines.append(f"**Schema warnings:** {len(warnings)} row(s) had non-parallel passage arrays.")
            lines.append("")

    SUMMARY_REPORT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", SUMMARY_REPORT)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["hi"],
        help="Language codes to run content-level analysis on, or 'all'.",
    )
    parser.add_argument("--sample-rows", type=int, default=None, help="Cap rows analyzed per language.")
    parser.add_argument("--footer-only", action="store_true", help="Skip content download; sizes only.")
    parser.add_argument("--cache-dir", default=None, help="HF hub cache dir override.")
    args = parser.parse_args()

    langs = list(LANGUAGES) if args.languages == ["all"] else args.languages
    for lang in langs:
        if lang not in LANGUAGES:
            raise SystemExit(f"unknown language code {lang!r}; choices: {sorted(LANGUAGES)}")

    # Footer metadata for ALL 14 languages is cheap — always compute it for the size table.
    footer_rows: list[dict[str, Any]] = []
    for lang in LANGUAGES:
        row: dict[str, Any] = {"language": lang, "language_name": LANGUAGES[lang], "footer": {}}
        for split in available_splits(lang):
            meta = read_footer_metadata(lang, split)
            row["footer"][split] = {"num_rows": meta.num_rows, "num_row_groups": meta.num_row_groups}
        footer_rows.append(row)
        logger.info(
            "%s (%s): %s",
            LANGUAGES[lang],
            lang,
            {k: v["num_rows"] for k, v in row["footer"].items()},
        )

    if args.footer_only:
        write_markdown_summary([], footer_rows)
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    content_reports = []
    for lang in langs:
        logger.info("Analyzing content for %s (%s)...", LANGUAGES[lang], lang)
        report = analyze_language(lang, args.sample_rows, args.cache_dir)
        content_reports.append(report)
        out_path = REPORTS_DIR / f"{lang}.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote %s", out_path)

    write_markdown_summary(content_reports, footer_rows)


if __name__ == "__main__":
    main()

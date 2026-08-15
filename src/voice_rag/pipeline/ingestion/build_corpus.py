"""Ingestion: raw MSMARCO-XI parquet -> normalized, deduplicated corpus.

Two tables are produced per (language, split):

- ``passages`` — one row per *unique* passage (deduplicated by a content
  hash of the normalized translated text, per dataset analysis's finding
  that 1.4-3.6% of passage slots are exact duplicates and some passages are
  shared by 80+ queries). This is what actually gets embedded/indexed —
  indexing the raw exploded table would waste storage and skew retrieval
  towards over-represented passages.
- ``qrels`` — one row per (query_id, passage_id, is_selected), pointing at
  the deduplicated passage_id. This preserves the exact relevance signal
  needed for retrieval evaluation even though the underlying passage text is
  now shared across queries.

A ``queries`` table (one row per query) is also produced for convenience.

Usage:
    uv run python -m voice_rag.pipeline.ingestion.build_corpus --languages hi ta --split validation
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from voice_rag.pipeline.ingestion.hf_source import download_split
from voice_rag.pipeline.ingestion.schema import LANGUAGES

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


def _content_hash(text: str) -> str:
    normalized = " ".join(text.strip().split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_corpus(
    lang: str, split: str, cache_dir: str | None = None, limit: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (queries_df, passages_df, qrels_df) for one (language, split)."""
    local_path = download_split(lang, split, cache_dir=cache_dir)
    table = pq.read_table(local_path)
    n = table.num_rows if limit is None else min(limit, table.num_rows)
    df = table.slice(0, n).to_pandas()

    queries_df = pd.DataFrame(
        {
            "query_id": df["query_id"],
            "language": lang,
            "split": split,
            "query_type": df["query_type"],
            "query_text": df["query"],
            "eng_query_text": df["Eng_Query"],
            "translated_answer": df["Answer"],
            "eng_answer": df["Eng_Answer"],
        }
    )

    passage_hash_to_id: dict[str, str] = {}
    passage_records: dict[str, dict] = {}
    qrel_rows: list[dict] = []
    skipped_nonparallel = 0

    for qid, p in zip(df["query_id"], df["passages"], strict=True):
        is_sel = list(p["is_selected"]) if p["is_selected"] is not None else []
        eng = list(p["English_passages"]) if p["English_passages"] is not None else []
        tr = list(p["Translated_passages"]) if p["Translated_passages"] is not None else []
        if not (len(is_sel) == len(eng) == len(tr)):
            skipped_nonparallel += 1
            continue
        for idx in range(len(is_sel)):
            text = (tr[idx] or "").strip()
            if not text:
                continue
            h = _content_hash(text)
            passage_id = f"{lang}:{h[:16]}"
            if h not in passage_hash_to_id:
                passage_hash_to_id[h] = passage_id
                passage_records[passage_id] = {
                    "passage_id": passage_id,
                    "language": lang,
                    "content_hash": h,
                    "text_translated": text,
                    "text_english": (eng[idx] or "").strip(),
                    "first_seen_query_id": int(qid),
                }
            qrel_rows.append(
                {
                    "query_id": int(qid),
                    "passage_id": passage_hash_to_id[h],
                    "is_selected": int(is_sel[idx]),
                    "source_passage_index": idx,
                }
            )

    if skipped_nonparallel:
        logger.warning(
            "%s/%s: skipped %d rows with non-parallel passage arrays", lang, split, skipped_nonparallel
        )

    passages_df = pd.DataFrame(passage_records.values())
    qrels_df = pd.DataFrame(qrel_rows)

    dedup_ratio = len(passages_df) / len(qrels_df) if len(qrels_df) else 0.0
    logger.info(
        "%s/%s: %d queries, %d unique passages from %d passage occurrences (dedup ratio %.3f)",
        lang,
        split,
        len(queries_df),
        len(passages_df),
        len(qrels_df),
        dedup_ratio,
    )

    return queries_df, passages_df, qrels_df


def write_corpus(lang: str, split: str, queries_df: pd.DataFrame, passages_df: pd.DataFrame, qrels_df: pd.DataFrame) -> None:
    out_dir = PROCESSED_DIR / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    queries_df.to_parquet(out_dir / f"{split}_queries.parquet", index=False)
    passages_df.to_parquet(out_dir / f"{split}_passages.parquet", index=False)
    qrels_df.to_parquet(out_dir / f"{split}_qrels.parquet", index=False)
    logger.info("Wrote %s", out_dir)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    for lang in args.languages:
        if lang not in LANGUAGES:
            raise SystemExit(f"unknown language code {lang!r}")

    for lang in args.languages:
        queries_df, passages_df, qrels_df = build_corpus(lang, args.split, args.cache_dir, args.limit)
        write_corpus(lang, args.split, queries_df, passages_df, qrels_df)


if __name__ == "__main__":
    main()

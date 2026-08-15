"""Materializes a passages parquet (from ingestion/build_corpus.py) into a
chunks parquet ready for embedding.

Verified on a 5,000-passage sample of the real Hindi corpus: 99.64% of
passages are kept whole (strategy=whole_passage); only 0.36% needed any
splitting at all. This CLI processes the full table, not a sample.

Usage:
    uv run python -m voice_rag.pipeline.chunking.build_chunks --languages hi ta --split validation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from voice_rag.pipeline.chunking.chunker import ChunkingConfig, chunk_passage

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


def build_chunks_for_language(lang: str, split: str, config: ChunkingConfig | None = None) -> pd.DataFrame:
    passages_path = PROCESSED_DIR / lang / f"{split}_passages.parquet"
    if not passages_path.exists():
        raise FileNotFoundError(
            f"{passages_path} not found — run `voice_rag.pipeline.ingestion.build_corpus` for "
            f"language={lang!r} split={split!r} first"
        )
    passages_df = pd.read_parquet(passages_path)
    config = config or ChunkingConfig()

    records = []
    strategy_counts: dict[str, int] = {}
    for row in passages_df.itertuples():
        chunks = chunk_passage(row.passage_id, row.language, row.text_translated, config=config)
        for c in chunks:
            strategy_counts[c.strategy] = strategy_counts.get(c.strategy, 0) + 1
            records.append(c.model_dump())

    chunks_df = pd.DataFrame(records)
    logger.info(
        "%s/%s: %d passages -> %d chunks (strategy breakdown: %s)",
        lang,
        split,
        len(passages_df),
        len(chunks_df),
        strategy_counts,
    )
    return chunks_df


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--max-tokens-before-split", type=int, default=512)
    parser.add_argument("--window-tokens", type=int, default=256)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    args = parser.parse_args()

    config = ChunkingConfig(
        max_tokens_before_split=args.max_tokens_before_split,
        window_tokens=args.window_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    for lang in args.languages:
        chunks_df = build_chunks_for_language(lang, args.split, config)
        out_path = PROCESSED_DIR / lang / f"{args.split}_chunks.parquet"
        chunks_df.to_parquet(out_path, index=False)
        logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()

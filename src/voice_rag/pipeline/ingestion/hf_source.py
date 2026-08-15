"""Access to the raw ai4bharat/MSMARCO-XI parquet files on the Hugging Face Hub.

Two distinct access patterns are exposed on purpose:

- ``read_footer_metadata`` costs almost nothing (a handful of KB over HTTP
  range requests) and gives exact row counts without downloading data. Used
  for corpus-size accounting across all languages.
- ``download_split`` pulls a full parquet file into the local HF cache
  (resumable, content-addressed, safe to call repeatedly). Every validation
  file observed so far is a single Parquet row group, which means partial
  *content* reads (as opposed to footer reads) cannot save bandwidth over a
  full download anyway — so content-level analysis downloads the file once
  and reads it locally rather than fighting the row-group boundary.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import fsspec
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from voice_rag.pipeline.ingestion.schema import LANGUAGES, LANGUAGES_WITHOUT_TRAIN, REPO_ID, parquet_path
from voice_rag.pipeline.ingestion.schema import MSMarcoXIRow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FooterMetadata:
    language: str
    split: str
    num_rows: int
    num_row_groups: int


def available_splits(lang: str) -> list[str]:
    """Which splits actually exist for this language (accounts for the Telugu train gap)."""
    if lang not in LANGUAGES:
        raise ValueError(f"unknown language {lang!r}")
    splits = ["validation"]
    if lang not in LANGUAGES_WITHOUT_TRAIN:
        splits.insert(0, "train")
    return splits


def read_footer_metadata(lang: str, split: str, revision: str = "main") -> FooterMetadata:
    """Read only the Parquet footer (row count, row-group count) — no row data is fetched."""
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{revision}/{parquet_path(lang, split)}"
    fs = fsspec.filesystem("http")
    with fs.open(url) as f:
        pf = pq.ParquetFile(f)
        return FooterMetadata(
            language=lang,
            split=split,
            num_rows=pf.metadata.num_rows,
            num_row_groups=pf.num_row_groups,
        )


def download_split(lang: str, split: str, revision: str = "main", cache_dir: str | Path | None = None) -> Path:
    """Download a split's parquet file into the local HF cache; returns the local path.

    Safe to call repeatedly — huggingface_hub dedupes by content hash and
    resumes interrupted downloads rather than re-fetching from scratch.
    """
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=parquet_path(lang, split),
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(local_path)


def iter_rows(local_path: Path, limit: int | None = None) -> Iterator[MSMarcoXIRow]:
    """Validate and yield rows from a downloaded parquet file as MSMarcoXIRow.

    Note the parquet field names (``Answer``, ``English_passages``, ...) are
    matched exactly as-is by MSMarcoXIRow/Passages — see schema.py docstring
    for why this differs from a naive reading of the dataset card.
    """
    table = pq.read_table(local_path)
    n = table.num_rows if limit is None else min(limit, table.num_rows)
    # to_pylist() on a slice keeps peak memory bounded when limit is small,
    # at the cost of paying full-column decode for the slice's row group.
    records = table.slice(0, n).to_pylist()
    for rec in records:
        yield MSMarcoXIRow.model_validate(rec)

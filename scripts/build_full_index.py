"""Build a persistent, restart-safe full-corpus hybrid index.

The expensive unit of work is one embedding/upsert batch.  This script records
an atomic checkpoint *before* beginning a batch and advances it only after both
Qdrant and BM25 have durably accepted that batch.  A stopped Pod therefore
replays at most its in-flight batch instead of re-embedding the corpus.

Usage:
    uv run python scripts/build_full_index.py --language hi --split validation

Use ``--reset`` only when deliberately rebuilding this language/split/index
version.  It recreates that Qdrant collection and removes only its matching,
versioned BM25 directory and state manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_rag.pipeline.embeddings.service import EmbeddingService, MODEL_NAME
from voice_rag.pipeline.retrieval.dense.index import (
    IndexedChunk,
    collection_name,
    enable_dense_hnsw,
    ensure_collection,
    get_client,
    upsert_chunks,
    verify_bulk_upload_config,
    verify_search_index_config,
)
from voice_rag.pipeline.retrieval.sparse.bm25_index import Bm25Index

logger = logging.getLogger(__name__)
PROCESSED_DIR = Path("data/processed")
INDEX_DIR = Path("data/full_index")

STATE_SCHEMA_VERSION = 1
STATE_STAGES = {"indexing", "optimizing", "completed"}
FINGERPRINT_COLUMNS = ("chunk_id", "passage_id", "language", "text")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def state_path_for(language: str, split: str, index_version: str) -> Path:
    """Return the immutable identity path for one full-index build."""
    return INDEX_DIR / f"{language}_{split}_{index_version}.state.json"


def bm25_path_for(language: str, split: str, index_version: str) -> Path:
    """Keep lexical artifacts isolated per vector-index version as well."""
    return INDEX_DIR / "bm25" / f"{language}_{split}_{index_version}"


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Persist a checkpoint without ever leaving a truncated manifest behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        # os.replace has already removed it on success.  This also cleans up a
        # temporary file if JSON encoding or fsync raises.
        if tmp_path.exists():
            tmp_path.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Index state at {path} is not valid JSON; use --reset to rebuild safely.") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Index state at {path} must be a JSON object; use --reset to rebuild safely.")
    return value


def _chunks_fingerprint(chunks_df: pd.DataFrame) -> str:
    """Fingerprint both ordered IDs and source text without a second parquet read.

    Stable chunk IDs alone are insufficient: a chunking/data change can retain
    an ID while changing its content.  pandas produces one deterministic
    64-bit hash per row; SHA-256 then makes a compact manifest-safe fingerprint.
    """
    missing = [column for column in FINGERPRINT_COLUMNS if column not in chunks_df.columns]
    if missing:
        raise ValueError(f"chunks parquet is missing required columns: {', '.join(missing)}")
    row_hashes = pd.util.hash_pandas_object(
        chunks_df.loc[:, list(FINGERPRINT_COLUMNS)], index=True, categorize=False
    )
    digest = hashlib.sha256()
    digest.update(str(len(chunks_df)).encode("ascii"))
    digest.update(b"\0")
    digest.update(row_hashes.to_numpy(dtype="uint64", copy=False).tobytes())
    return digest.hexdigest()


def _state_identity(args: argparse.Namespace, chunks_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "language": args.language,
        "split": args.split,
        "index_version": args.index_version,
        "total": len(chunks_df),
        "chunks_fingerprint": _chunks_fingerprint(chunks_df),
        "limit": args.limit,
        "embedding_model": MODEL_NAME,
    }


def _new_state(identity: dict[str, Any]) -> dict[str, Any]:
    now = _utc_now()
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        **identity,
        "next_offset": 0,
        "stage": "indexing",
        "inflight_batch": None,
        # Every new build creates a collection in Qdrant's bulk mode.  This
        # durable bit also tells a restarted builder that finalization is
        # still required after the last upload checkpoint.
        "dense_hnsw_deferred": True,
        "created_at": now,
        "updated_at": now,
        "completed_at": None,
        "last_error": None,
    }


def _validate_state(state: dict[str, Any], identity: dict[str, Any], path: Path) -> None:
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise RuntimeError(
            f"Index state at {path} uses schema {state.get('schema_version')!r}, "
            f"expected {STATE_SCHEMA_VERSION}; use --reset to rebuild safely."
        )
    mismatches = {
        key: (state.get(key), expected)
        for key, expected in identity.items()
        if state.get(key) != expected
    }
    if mismatches:
        details = ", ".join(f"{key}={actual!r} (expected {expected!r})" for key, (actual, expected) in mismatches.items())
        raise RuntimeError(
            f"Index state at {path} does not match this corpus/configuration: {details}. "
            "Use a new --index-version or --reset to rebuild safely."
        )
    next_offset = state.get("next_offset")
    if not isinstance(next_offset, int) or not 0 <= next_offset <= identity["total"]:
        raise RuntimeError(f"Index state at {path} has invalid next_offset={next_offset!r}; use --reset to rebuild safely.")
    stage = state.get("stage")
    if stage not in STATE_STAGES:
        raise RuntimeError(f"Index state at {path} has invalid stage={stage!r}; use --reset to rebuild safely.")
    if stage in {"optimizing", "completed"} and next_offset != identity["total"]:
        raise RuntimeError(f"Index state at {path} is marked {stage} before all rows; use --reset to rebuild safely.")
    if state.get("dense_hnsw_deferred") not in (True, False):
        raise RuntimeError(f"Index state at {path} has invalid dense_hnsw_deferred; use --reset to rebuild safely.")


def _load_or_create_state(path: Path, identity: dict[str, Any], *, resume: bool) -> tuple[dict[str, Any], bool]:
    """Return ``(state, existed)`` and fail rather than silently mixing indexes."""
    if path.exists():
        if not resume:
            raise RuntimeError(f"A checkpoint already exists at {path}; use --reset instead of discarding it.")
        state = _load_json(path)
        _validate_state(state, identity, path)
        return state, True
    state = _new_state(identity)
    _atomic_write_json(path, state)
    return state, False


def _reset_bm25_path(path: Path) -> None:
    """Remove only the known versioned BM25 target requested via --reset."""
    allowed_root = (INDEX_DIR / "bm25").resolve()
    target = path.resolve()
    try:
        target.relative_to(allowed_root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to reset BM25 path outside {allowed_root}: {target}") from exc
    if not target.exists():
        return
    if not target.is_dir():
        raise RuntimeError(f"Refusing to reset non-directory BM25 path: {target}")
    shutil.rmtree(target)


def _collection_point_count(client: Any, collection: str) -> int:
    return int(client.count(collection_name=collection, exact=True).count)


def _status_value(value: Any) -> str:
    """Normalize Qdrant enums and status objects for logs/ready checks."""
    return str(getattr(value, "value", value)).lower()


def _wait_for_search_indexes(
    client: Any,
    collection: str,
    expected_points: int,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> None:
    """Wait for the post-upload Qdrant optimizer, with visible bounded logs.

    A deferred bulk upload intentionally leaves the collection without its
    dense HNSW and immutable sparse index until this point.  ``status=green``
    plus ``optimizer_status=ok`` and the expected exact point count is the
    service-ready condition.  For production-sized indexes, require Qdrant
    to report at least one indexed vector per point as well; very small smoke
    indexes legitimately remain below its indexing threshold.
    """
    if timeout_seconds <= 0:
        raise ValueError("optimizer wait timeout must be positive")
    deadline = time.monotonic() + timeout_seconds
    minimum_indexed_vectors = expected_points if expected_points >= 10_000 else 0

    while True:
        info = client.get_collection(collection)
        status = _status_value(getattr(info, "status", "unknown"))
        optimizer_status = _status_value(getattr(info, "optimizer_status", "unknown"))
        indexed_vectors = getattr(info, "indexed_vectors_count", None)
        point_count = _collection_point_count(client, collection)
        indexed_ready = minimum_indexed_vectors == 0 or (
            indexed_vectors is not None and int(indexed_vectors) >= minimum_indexed_vectors
        )
        ready = (
            status == "green"
            and optimizer_status == "ok"
            and point_count == expected_points
            and indexed_ready
        )
        logger.info(
            "Qdrant finalization: status=%s optimizer=%s points=%d/%d indexed_vectors=%s "
            "segments=%s queue=%s ready=%s",
            status,
            optimizer_status,
            point_count,
            expected_points,
            indexed_vectors,
            getattr(info, "segments_count", "unknown"),
            getattr(info, "update_queue", None),
            ready,
        )
        if ready:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Qdrant finalization for {collection!r} exceeded {timeout_seconds:.0f}s. "
                "The checkpoint remains at stage=optimizing; rerunning will resume finalization "
                "without re-embedding uploaded chunks."
            )
        time.sleep(poll_seconds)


def _validate_qdrant_for_state(client: Any, collection: str, state: dict[str, Any], *, existed: bool) -> None:
    """Ensure a resumed checkpoint cannot skip lost or unrelated vector rows."""
    point_count = _collection_point_count(client, collection)
    total = state["total"]
    next_offset = state["next_offset"]
    if point_count > total:
        raise RuntimeError(
            f"Qdrant collection {collection!r} has {point_count} points but this build has only {total}. "
            "Use a new --index-version or --reset to avoid mixing corpora."
        )
    if point_count < next_offset:
        raise RuntimeError(
            f"Qdrant collection {collection!r} has only {point_count} points, below checkpoint {next_offset}. "
            "The vector store was lost or changed; use --reset to rebuild safely."
        )
    if not existed and point_count:
        # A new manifest cannot prove that pre-existing points came from this
        # exact corpus; replaying only a suffix could leave stale documents.
        raise RuntimeError(
            f"Qdrant collection {collection!r} already contains {point_count} points but has no matching state manifest. "
            "Use --reset or a new --index-version to avoid mixing corpora."
        )
    if state["stage"] == "completed" and point_count != total:
        raise RuntimeError(
            f"Completed state expects {total} Qdrant points but found {point_count}; use --reset to rebuild safely."
        )


def _save_checkpoint(state_path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _utc_now()
    _atomic_write_json(state_path, state)


def _log_progress(state: dict[str, Any], session_started_at: float) -> None:
    next_offset = state["next_offset"]
    total = state["total"]
    elapsed = time.monotonic() - session_started_at
    rate = next_offset / elapsed if elapsed > 0 else 0.0
    eta_seconds = (total - next_offset) / rate if rate > 0 else float("inf")
    logger.info(
        "Progress: %d/%d (%.1f%%) | rate=%.1f chunks/sec | session_elapsed=%.0fs | eta=%.0fs (%.1f min)",
        next_offset,
        total,
        100 * next_offset / total if total else 100.0,
        rate,
        elapsed,
        eta_seconds,
        eta_seconds / 60,
    )


def _configure_logging() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(INDEX_DIR.parent / "full_index_build.log", encoding="utf-8"),
        ],
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embed-chunk-size", type=int, default=5000, help="rows embedded+upserted per checkpoint")
    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=1024,
        help="Qdrant points per durable localhost upsert (default: 1024)",
    )
    parser.add_argument(
        "--optimizer-wait-seconds",
        type=float,
        default=3600.0,
        help="maximum time to wait for post-upload Qdrant indexing before preserving stage=optimizing",
    )
    parser.add_argument(
        "--optimizer-poll-seconds",
        type=float,
        default=15.0,
        help="seconds between visible Qdrant finalization status logs",
    )
    parser.add_argument("--index-version", default="full1")
    parser.add_argument("--limit", type=int, default=None, help="cap total rows processed (smoke-testing only)")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume a matching state manifest (default: true)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="explicitly recreate this collection and its matching versioned BM25/state artifacts",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Point at a real Qdrant server instead of embedded local mode â€” required for deployment "
        "(docs/runpod-deployment.md): local mode's SQLite-backed storage is not the same on-disk "
        "format as the server and cannot be copied into one.",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.embed_chunk_size < 1:
        parser.error("--embed-chunk-size must be positive")
    if args.upsert_batch_size < 1:
        parser.error("--upsert-batch-size must be positive")
    if args.optimizer_wait_seconds <= 0:
        parser.error("--optimizer-wait-seconds must be positive")
    if args.optimizer_poll_seconds <= 0:
        parser.error("--optimizer-poll-seconds must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.reset and not args.resume:
        parser.error("--reset and --no-resume cannot be used together")
    return args


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _configure_logging()
    args = _parse_args(argv)

    chunks_path = PROCESSED_DIR / args.language / f"{args.split}_chunks.parquet"
    chunks_df = pd.read_parquet(chunks_path)
    if args.limit is not None:
        chunks_df = chunks_df.iloc[: args.limit].reset_index(drop=True)
    logger.info("Loaded %d chunks from %s", len(chunks_df), chunks_path)

    identity = _state_identity(args, chunks_df)
    state_path = state_path_for(args.language, args.split, args.index_version)
    bm25_path = bm25_path_for(args.language, args.split, args.index_version)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if args.reset:
        logger.warning("Reset requested for language=%s split=%s index_version=%s", args.language, args.split, args.index_version)
        _reset_bm25_path(bm25_path)
        state_path.unlink(missing_ok=True)

    state, state_existed = _load_or_create_state(state_path, identity, resume=args.resume)

    qdrant_path = str(INDEX_DIR / "qdrant")
    # The default client timeout (~5s) is shorter than a large wait=true bulk
    # upsert can legitimately take once segments are sizable; a client-side
    # timeout here does not stop the write, it just abandons the process
    # mid-flight and can leave Qdrant's on-disk segments inconsistent if the
    # process is then killed. Give bulk indexing a much longer budget.
    client = get_client(path=None if args.qdrant_url else qdrant_path, url=args.qdrant_url, timeout=120.0)
    collection = collection_name(args.language, args.index_version)
    collection_existed = client.collection_exists(collection)
    if not state_existed and collection_existed and not args.reset:
        raise RuntimeError(
            f"Qdrant collection {collection!r} exists without a matching state manifest. "
            "Use --reset or a new --index-version rather than mixing unknown partial data."
        )
    created_for_bulk_upload = args.reset or not collection_existed
    ensure_collection(
        client,
        collection,
        recreate=args.reset,
        defer_dense_hnsw=created_for_bulk_upload,
    )
    # Embedded local mode does not faithfully expose all optimizer settings;
    # the paid deployment uses a real HTTP Qdrant server and must prove the
    # bulk configuration before doing hours of embedding work.
    if created_for_bulk_upload and args.qdrant_url:
        verify_bulk_upload_config(client, collection)
    if created_for_bulk_upload:
        state["dense_hnsw_deferred"] = True
        _save_checkpoint(state_path, state)
    elif not state["dense_hnsw_deferred"] and state["stage"] != "completed":
        raise RuntimeError(
            f"Checkpoint {state_path} does not prove that {collection!r} was created in bulk mode. "
            "Use --reset to rebuild safely."
        )
    _validate_qdrant_for_state(client, collection, state, existed=state_existed)

    if state["stage"] == "completed":
        logger.info("Index already complete: %s (state=%s)", collection, state_path)
        return

    session_started_at = time.monotonic()
    start_offset = state["next_offset"]
    if start_offset:
        logger.info("Resuming from checkpoint at %d/%d", start_offset, state["total"])

    if start_offset < state["total"]:
        logger.info("Loading embedding service...")
        svc = EmbeddingService()
        bm25 = Bm25Index(bm25_path)
        for start in range(start_offset, state["total"], args.embed_chunk_size):
            end = min(start + args.embed_chunk_size, state["total"])
            batch_df = chunks_df.iloc[start:end]
            texts = batch_df["text"].tolist()

            # Persist intent first. A termination at any later point replays
            # this deterministic range, which is safe for Qdrant IDs and
            # idempotent BM25 upserts.
            state["stage"] = "indexing"
            state["inflight_batch"] = {"start": start, "end": end}
            state["last_error"] = None
            _save_checkpoint(state_path, state)
            try:
                result = svc.embed(texts, batch_size=args.batch_size)
                indexed_chunks = [
                    IndexedChunk(
                        chunk_id=row.chunk_id,
                        passage_id=row.passage_id,
                        language=row.language,
                        text=row.text,
                        dense_vector=result.dense[i].tolist(),
                        sparse_vector=result.sparse[i],
                    )
                    for i, row in enumerate(batch_df.itertuples())
                ]
                upsert_chunks(client, collection, indexed_chunks, batch_size=args.upsert_batch_size)
                bm25.upsert_batch(batch_df["chunk_id"].tolist(), texts)
            except BaseException as exc:
                state["last_error"] = f"{type(exc).__name__}: {exc}"
                _save_checkpoint(state_path, state)
                raise

            state["next_offset"] = end
            state["inflight_batch"] = None
            _save_checkpoint(state_path, state)
            _log_progress(state, session_started_at)

    # Empty corpora have no HNSW graph to construct, but their manifest still
    # needs a terminal state for readiness checks.
    if state["total"] == 0:
        state["stage"] = "completed"
        state["dense_hnsw_deferred"] = False
        state["completed_at"] = _utc_now()
        _save_checkpoint(state_path, state)
    else:
        # Do not declare ready merely because the rows were accepted. Qdrant
        # must finish the intentionally deferred dense+sparse index build.
        state["stage"] = "optimizing"
        state["inflight_batch"] = None
        state["last_error"] = None
        _save_checkpoint(state_path, state)
        try:
            enable_dense_hnsw(client, collection)
            if args.qdrant_url:
                verify_search_index_config(client, collection)
            _wait_for_search_indexes(
                client,
                collection,
                state["total"],
                timeout_seconds=args.optimizer_wait_seconds,
                poll_seconds=args.optimizer_poll_seconds,
            )
        except BaseException as exc:
            state["last_error"] = f"{type(exc).__name__}: {exc}"
            _save_checkpoint(state_path, state)
            raise
        state["stage"] = "completed"
        state["dense_hnsw_deferred"] = False
        state["completed_at"] = _utc_now()
        _save_checkpoint(state_path, state)

    logger.info("Done. Qdrant collection=%s, BM25 index=%s, state=%s", collection, bm25_path, state_path)


if __name__ == "__main__":
    main()

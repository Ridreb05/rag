"""Qdrant dense+sparse hybrid index.

Uses qdrant-client's embedded local mode (``QdrantClient(path=...)``) by
default — the same Rust HNSW engine as the server, running in-process with
on-disk persistence, no Docker/server required. This is legitimate for
development and correctness testing but is NOT the production topology.
Pass a ``url`` instead of ``path`` to point at a real server — the rest of
this module's API is identical either way.

Collection layout: one collection per (language, index_version), named
vectors "dense" (1024-dim BGE-M3) and "sparse" (BGE-M3 lexical weights).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

DENSE_DIM = 1024
DEFAULT_HNSW_M = 16
# Qdrant v1.13.4's production default, expressed in kilobytes.  Restoring a
# positive value after bulk upload lets Qdrant build its dense and immutable
# sparse indexes again.
DEFAULT_INDEXING_THRESHOLD_KB = 20_000


def collection_name(language: str, index_version: str) -> str:
    return f"chunks_{language}_v{index_version}"


def get_client(path: str | None = "data/qdrant_local", url: str | None = None, *, timeout: float | None = None) -> QdrantClient:
    if url:
        return QdrantClient(url=url, timeout=timeout)
    return QdrantClient(path=path, timeout=timeout)


def ensure_collection(
    client: QdrantClient,
    name: str,
    recreate: bool = False,
    *,
    defer_dense_hnsw: bool = False,
) -> None:
    """Create the hybrid collection when it does not already exist.

    ``defer_dense_hnsw`` is an explicit bulk-load mode.  Qdrant otherwise
    starts building dense HNSW segments and compact sparse indexes while
    points are still arriving, which competes with ingestion for CPU, memory,
    and disk I/O.  It sets ``m=0`` and ``indexing_threshold=0`` so those
    optimizations are deferred until :func:`enable_dense_hnsw` is called after
    the upload.  The dense vectors and named sparse vector themselves remain
    identical, so retrieval semantics are preserved once indexing is restored.

    We deliberately do not create a ``language`` payload index.  Collections
    are already partitioned by language and no query path filters on that
    field, so indexing it adds optimizer work without serving a request.
    """
    exists = client.collection_exists(name)
    if exists and not recreate:
        return
    if exists and recreate:
        client.delete_collection(name)

    hnsw_config = models.HnswConfigDiff(m=0) if defer_dense_hnsw else None
    optimizers_config = (
        models.OptimizersConfigDiff(indexing_threshold=0) if defer_dense_hnsw else None
    )
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
        hnsw_config=hnsw_config,
        optimizers_config=optimizers_config,
    )


def enable_dense_hnsw(client: QdrantClient, name: str, m: int = DEFAULT_HNSW_M) -> None:
    """Restore normal HNSW and optimizer settings after a bulk upload.

    The collection retains its existing dense/sparse points and payloads.
    Qdrant schedules the graph build asynchronously, so callers should wait
    for the collection optimizer to report healthy before accepting traffic.
    """
    if m < 1:
        raise ValueError("HNSW m must be at least 1 when enabling the index")
    client.update_collection(
        collection_name=name,
        hnsw_config=models.HnswConfigDiff(m=m),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=DEFAULT_INDEXING_THRESHOLD_KB
        ),
    )


def verify_bulk_upload_config(client: QdrantClient, name: str) -> None:
    """Fail fast if a Qdrant server did not apply deferred bulk settings.

    A silent fallback to Qdrant defaults would recreate the expensive
    optimizer work during upload.  Verify it immediately, before any model
    download or embedding work is started.
    """
    info = client.get_collection(name)
    hnsw_m = info.config.hnsw_config.m
    indexing_threshold = info.config.optimizer_config.indexing_threshold
    logger.info(
        "Qdrant bulk config collection=%s hnsw_m=%s indexing_threshold=%s",
        name,
        hnsw_m,
        indexing_threshold,
    )
    if hnsw_m != 0 or indexing_threshold != 0:
        raise RuntimeError(
            f"Qdrant did not apply deferred bulk settings for {name!r}: "
            f"hnsw_m={hnsw_m!r}, indexing_threshold={indexing_threshold!r}. "
            "Refusing to upload into a configuration that can optimize mid-ingestion."
        )


def verify_search_index_config(client: QdrantClient, name: str) -> None:
    """Confirm the normal retrieval indexes were restored before readiness."""
    info = client.get_collection(name)
    hnsw_m = info.config.hnsw_config.m
    indexing_threshold = info.config.optimizer_config.indexing_threshold
    logger.info(
        "Qdrant search config collection=%s hnsw_m=%s indexing_threshold=%s",
        name,
        hnsw_m,
        indexing_threshold,
    )
    if hnsw_m < 1 or indexing_threshold <= 0:
        raise RuntimeError(
            f"Qdrant did not restore search indexing for {name!r}: "
            f"hnsw_m={hnsw_m!r}, indexing_threshold={indexing_threshold!r}."
        )


@dataclass
class IndexedChunk:
    chunk_id: str
    passage_id: str
    language: str
    text: str
    dense_vector: list[float]
    sparse_vector: dict[int, float]
    payload_extra: dict = field(default_factory=dict)


def upsert_chunks(client: QdrantClient, collection: str, chunks: list[IndexedChunk], batch_size: int = 256) -> int:
    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        points = []
        for c in batch:
            sparse_indices = list(c.sparse_vector.keys())
            sparse_values = list(c.sparse_vector.values())
            points.append(
                models.PointStruct(
                    id=_stable_point_id(c.chunk_id),
                    vector={
                        "dense": c.dense_vector,
                        "sparse": models.SparseVector(indices=sparse_indices, values=sparse_values),
                    },
                    payload={
                        "chunk_id": c.chunk_id,
                        "passage_id": c.passage_id,
                        "language": c.language,
                        "text": c.text,
                        **c.payload_extra,
                    },
                )
            )
        client.upsert(collection_name=collection, points=points)
        total += len(points)
    return total


def _stable_point_id(chunk_id: str) -> int:
    """Qdrant point IDs must be int or UUID; derive a stable 63-bit int from
    the chunk_id so re-upserting the same chunk (e.g. re-running ingestion)
    overwrites rather than duplicates."""
    import hashlib

    h = hashlib.sha256(chunk_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFFFFFFFFFFFFFF

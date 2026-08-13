"""Phase 4/5 — Qdrant dense+sparse hybrid index.

Uses qdrant-client's embedded local mode (``QdrantClient(path=...)``) by
default — the same Rust HNSW engine as the server, running in-process with
on-disk persistence, no Docker/server required. This is legitimate for
development and correctness testing but is NOT the production topology
(docs/08-repo-and-stack.md calls for a 3-node HA cluster or Qdrant Cloud).
Pass a ``url`` instead of ``path`` to point at a real server — the rest of
this module's API is identical either way.

Collection layout matches docs/06-data-and-api.md's schema: one collection
per (language, index_version), named vectors "dense" (1024-dim BGE-M3) and
"sparse" (BGE-M3 lexical weights).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

DENSE_DIM = 1024


def collection_name(language: str, index_version: str) -> str:
    return f"chunks_{language}_v{index_version}"


def get_client(path: str | None = "data/qdrant_local", url: str | None = None) -> QdrantClient:
    if url:
        return QdrantClient(url=url)
    return QdrantClient(path=path)


def ensure_collection(client: QdrantClient, name: str, recreate: bool = False) -> None:
    exists = client.collection_exists(name)
    if exists and not recreate:
        return
    if exists and recreate:
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
    )
    # payload index on language for cheap filtered search, per docs/02
    client.create_payload_index(
        collection_name=name, field_name="language", field_schema=models.PayloadSchemaType.KEYWORD
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

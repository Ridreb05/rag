"""Phase 5 — BM25 sparse retrieval via Tantivy (embedded, no server).

A model-independent third signal alongside dense and BGE-M3 sparse search
(docs/02-architecture-and-retrieval.md's single-encoder hybrid decision):
this catches exact IDs, numbers, and proper nouns that an embedding-based
sparse representation can under-weight, and it keeps working even if the
embedding service is degraded (see the failure-injection design in
docs/07-observability-and-evaluation.md).
"""

from __future__ import annotations

from pathlib import Path

import tantivy


def build_schema() -> tantivy.Schema:
    builder = tantivy.SchemaBuilder()
    builder.add_text_field("chunk_id", stored=True, tokenizer_name="raw")
    builder.add_text_field("text", stored=False, tokenizer_name="default")
    return builder.build()


class Bm25Index:
    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.schema = build_schema()
        self._index = tantivy.Index(self.schema, path=str(self.index_dir))

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        """Index a complete corpus, replacing any matching chunk IDs.

        ``build`` used to be append-only.  That was harmless for a one-shot
        batch job, but made an interrupted bootstrap unsafe to resume because
        replayed batches produced duplicate BM25 documents.  Keep the public
        method, but give it the same idempotent semantics as the Qdrant
        upsert path.
        """
        self.upsert_batch(chunk_ids, texts)

    def upsert_batch(self, chunk_ids: list[str], texts: list[str]) -> None:
        """Atomically replace a small batch of documents and make it searchable.

        The ``raw`` tokenizer on ``chunk_id`` makes deletion an exact key
        lookup.  We commit once per batch so an index build can checkpoint
        after this call: if the process stops, replaying the last batch is
        correct and cannot create duplicate results.
        """
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must have the same length")
        if not chunk_ids:
            return
        writer = self._index.writer()
        for cid, text in zip(chunk_ids, texts, strict=True):
            writer.delete_documents_by_term("chunk_id", cid)
            writer.add_document(tantivy.Document(chunk_id=cid, text=text))
        writer.commit()
        self._index.reload()

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Returns [(chunk_id, bm25_score), ...] ordered best-first.

        Deliberately does NOT use ``Index.parse_query`` on raw text: that
        parser treats the input as a Lucene-style query-string language,
        where characters like ``-``, ``+``, ``"``, ``AND``/``OR`` are syntax,
        not content. Verified failure mode: an ordinary Hindi passage
        containing a hyphen ("...वाहन - यान ले जाना...") raised a
        ``ValueError: Syntax Error`` because ``-`` was parsed as a NOT
        operator. Real user/passage text must never be fed to a query-string
        parser — this builds a plain OR-of-terms boolean query instead,
        which has no syntax to misinterpret.
        """
        # lowercase to match the index's default tokenizer normalization
        # (a no-op for Devanagari/other Indic scripts, which have no case)
        terms = [t.lower() for t in query.strip().split() if t]
        if not terms:
            return []
        self._index.reload()
        searcher = self._index.searcher()
        subqueries = [
            (tantivy.Occur.Should, tantivy.Query.term_query(self.schema, "text", term)) for term in terms
        ]
        parsed = tantivy.Query.boolean_query(subqueries)
        hits = searcher.search(parsed, limit=top_k).hits
        results = []
        for score, addr in hits:
            doc = searcher.doc(addr)
            results.append((doc["chunk_id"][0], float(score)))
        return results

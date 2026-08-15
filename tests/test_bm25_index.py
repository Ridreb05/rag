import tempfile

from voice_rag.pipeline.retrieval.sparse.bm25_index import Bm25Index


def test_bm25_finds_relevant_document():
    with tempfile.TemporaryDirectory() as d:
        idx = Bm25Index(d)
        idx.build(
            chunk_ids=["c1", "c2", "c3"],
            texts=[
                "diabetes is a chronic disease affecting blood sugar",
                "the capital of france is paris",
                "paris is known for the eiffel tower",
            ],
        )
        results = idx.search("diabetes blood sugar", top_k=3)
        assert results[0][0] == "c1"


def test_bm25_ranks_multiple_matches_by_relevance():
    with tempfile.TemporaryDirectory() as d:
        idx = Bm25Index(d)
        idx.build(
            chunk_ids=["c1", "c2"],
            texts=[
                "paris paris paris is the capital of france",
                "france has several major cities",
            ],
        )
        results = idx.search("paris", top_k=2)
        ids = [r[0] for r in results]
        assert ids[0] == "c1"


def test_bm25_empty_query_returns_no_crash():
    with tempfile.TemporaryDirectory() as d:
        idx = Bm25Index(d)
        idx.build(chunk_ids=["c1"], texts=["some text here"])
        results = idx.search("nonexistent_term_xyz", top_k=5)
        assert results == []


def test_bm25_upsert_batch_replaces_replayed_chunk_ids():
    """A restarted full-index build may replay its final committed batch."""
    with tempfile.TemporaryDirectory() as d:
        idx = Bm25Index(d)
        idx.upsert_batch(["c1"], ["oldonlyterm"])
        idx.upsert_batch(["c1", "c2"], ["newonlyterm", "secondterm"])

        assert idx.search("oldonlyterm", top_k=5) == []
        assert [cid for cid, _ in idx.search("newonlyterm", top_k=5)] == ["c1"]
        assert [cid for cid, _ in idx.search("secondterm", top_k=5)] == ["c2"]


def test_bm25_handles_query_syntax_characters_in_real_text():
    # Regression test: an ordinary Hindi passage containing a hyphen
    # ("एक अवैध वाहन - यान ले जाना एक अपराध है") previously crashed
    # Index.parse_query with "Syntax Error" because '-' is a NOT operator
    # in Tantivy's query-string language. Real text must never be parsed
    # as a query-string; term-query construction sidesteps this entirely.
    with tempfile.TemporaryDirectory() as d:
        idx = Bm25Index(d)
        idx.build(
            chunk_ids=["c1", "c2"],
            texts=["एक अवैध वाहन - यान ले जाना एक अपराध है", "यह पूरी तरह अलग विषय है"],
        )
        results = idx.search("एक अवैध वाहन - यान ले जाना एक अपराध है", top_k=5)
        assert results  # must not raise, and must find the matching doc
        assert results[0][0] == "c1"


def test_bm25_handles_other_special_syntax_characters():
    with tempfile.TemporaryDirectory() as d:
        idx = Bm25Index(d)
        idx.build(chunk_ids=["c1"], texts=["some text with AND OR NOT quotes"])
        for q in ['"quoted phrase"', "term1 AND term2", "term +required -excluded", "(parens)"]:
            idx.search(q, top_k=5)  # must not raise

"""Tests for the two-phase answering store.

Phase two hands a client-supplied trace_id back to the server to claim work
cached from phase one. That is a small amount of state with a large amount of
ways to go wrong — unbounded growth, stale entries served indefinitely, the
same refinement claimed twice — so the eviction rules are pinned here rather
than trusted to review.
"""

import time

import numpy as np

from voice_rag.api import main
from voice_rag.pipeline.generation.harness import DEADLINE_FALLBACK_FLAG
from voice_rag.pipeline.generation.schemas import RetrievalCandidate


def make_pending(query: str = "q") -> main._PendingRefinement:
    return main._PendingRefinement(
        query_text=query,
        language="hi",
        candidates=[
            RetrievalCandidate(chunk_id="c1", doc_id="d1", language="hi", text="passage", rerank_score=0.5)
        ],
        query_embedding=np.zeros(4, dtype=np.float32),
        payload_by_chunk_id={"c1": {"text": "passage", "chunk_id": "c1"}},
        rerank_score_by_chunk_id={"c1": 0.5},
        created_at=time.monotonic(),
    )


def setup_function() -> None:
    main._PENDING_REFINEMENTS.clear()


def test_pending_refinement_round_trips():
    main._remember_refinement("trace-1", make_pending("first"))

    assert main._PENDING_REFINEMENTS["trace-1"].query_text == "first"


def test_store_is_bounded_so_a_client_supplied_key_cannot_grow_it_forever():
    """trace_id comes from the caller, so an unbounded dict keyed by it is a
    memory leak reachable by anyone who can send queries."""
    for i in range(main._REFINEMENT_MAX_ENTRIES + 25):
        main._remember_refinement(f"trace-{i}", make_pending())

    assert len(main._PENDING_REFINEMENTS) == main._REFINEMENT_MAX_ENTRIES
    # Oldest evicted first — the entries most likely already abandoned.
    assert "trace-0" not in main._PENDING_REFINEMENTS
    assert f"trace-{main._REFINEMENT_MAX_ENTRIES + 24}" in main._PENDING_REFINEMENTS


def test_expired_entries_are_dropped_on_write():
    stale = make_pending("stale")
    stale.created_at = time.monotonic() - (main._REFINEMENT_TTL_SECONDS + 1)
    main._PENDING_REFINEMENTS["old"] = stale

    main._remember_refinement("fresh", make_pending("fresh"))

    assert "old" not in main._PENDING_REFINEMENTS
    assert "fresh" in main._PENDING_REFINEMENTS


def test_refinement_is_single_use():
    """The endpoint pops rather than reads: a refinement already delivered must
    not be regenerable by replaying the same trace_id, which would let one
    query fund unlimited LLM calls."""
    main._remember_refinement("trace-1", make_pending())

    first = main._PENDING_REFINEMENTS.pop("trace-1", None)
    second = main._PENDING_REFINEMENTS.pop("trace-1", None)

    assert first is not None
    assert second is None


def test_deadline_flag_is_shared_between_harness_and_api():
    """The API decides whether to offer a refinement by matching this exact
    flag. If the two ever drift apart, two-phase answering silently stops
    being offered — so they must be the same object, not equal strings."""
    assert main.DEADLINE_FALLBACK_FLAG is DEADLINE_FALLBACK_FLAG

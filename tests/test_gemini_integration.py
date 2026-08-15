"""Real Gemini API integration tests — network + a valid GEMINI_API_KEY
required. Excluded from the default run; run explicitly with:

    uv run pytest -m slow tests/test_gemini_integration.py
"""

import pytest

from voice_rag.settings import settings

pytestmark = pytest.mark.slow

requires_gemini_key = pytest.mark.skipif(not settings.gemini_api_key, reason="GEMINI_API_KEY not set in .env")


@requires_gemini_key
def test_generate_grounds_answer_and_cites_correct_chunk():
    from voice_rag.pipeline.generation.gemini_service import GeminiGenerationService
    from voice_rag.pipeline.generation.schemas import GenerationRequest, RetrievalCandidate

    svc = GeminiGenerationService()
    req = GenerationRequest(
        trace_id="test-int-1",
        query_final="What is diabetes?",
        query_language="en",
        candidates=[
            RetrievalCandidate(
                chunk_id="c1",
                doc_id="d1",
                language="en",
                text="Diabetes is a chronic disease that affects how the body processes blood sugar (glucose).",
                rerank_score=0.6,
            ),
            RetrievalCandidate(
                chunk_id="c2",
                doc_id="d2",
                language="en",
                text="The Eiffel Tower is located in Paris, France.",
                rerank_score=0.1,
            ),
        ],
        retrieval_confidence=0.6,
        mode="generative",
    )

    result = svc.generate(req)

    assert result is not None
    assert "diabetes" in result.answer_text.lower()
    assert len(result.claims) >= 1
    # must cite the real chunk_id, not the [C#] marker, and not the irrelevant passage
    all_cited = {cid for c in result.claims for cid in c.cited_chunk_ids}
    assert "c1" in all_cited
    assert "c2" not in all_cited

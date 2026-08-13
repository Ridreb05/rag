"""Shared context-building and citation-marker resolution, used by both
generation backends (anthropic_service.py, gemini_service.py) so the
[C#]-marker <-> chunk_id mapping logic exists in exactly one place."""

from __future__ import annotations

from voice_rag.generation.schemas import GenerationRequest


def build_context_block(request: GenerationRequest) -> tuple[str, dict[str, str]]:
    """Returns (context_text_with_C#_markers, marker_to_chunk_id_map)."""
    marker_to_chunk_id: dict[str, str] = {}
    lines = []
    for i, c in enumerate(request.candidates, start=1):
        marker = f"C{i}"
        marker_to_chunk_id[marker] = c.chunk_id
        lines.append(f"[{marker}] {c.text}")
    return "\n\n".join(lines), marker_to_chunk_id


def resolve_markers_to_chunk_ids(cited: list[str], marker_to_chunk_id: dict[str, str]) -> list[str]:
    """The model may cite either the bare marker ('C1') or a chunk_id it
    copied verbatim; resolve both to real chunk_ids, dropping anything
    that doesn't match a real citation (never trust unresolvable cites)."""
    resolved = []
    valid_chunk_ids = set(marker_to_chunk_id.values())
    for tag in cited:
        tag = tag.strip("[]")
        if tag in marker_to_chunk_id:
            resolved.append(marker_to_chunk_id[tag])
        elif tag in valid_chunk_ids:
            resolved.append(tag)
    return resolved

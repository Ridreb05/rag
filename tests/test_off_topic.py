import numpy as np
import pytest

from voice_rag.pipeline.guardrails.off_topic import OffTopicGate, compute_corpus_centroid, should_refuse


def test_compute_corpus_centroid_is_unit_norm():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    centroid = compute_corpus_centroid(embeddings)
    assert np.isclose(np.linalg.norm(centroid), 1.0)


def test_compute_corpus_centroid_rejects_empty():
    with pytest.raises(ValueError):
        compute_corpus_centroid(np.zeros((0, 4)))


def test_off_topic_gate_flags_orthogonal_query():
    centroid = np.array([1.0, 0.0])
    gate = OffTopicGate(centroid, similarity_threshold=0.35)
    orthogonal_query = np.array([0.0, 1.0])
    assert gate.is_off_topic(orthogonal_query) is True


def test_off_topic_gate_accepts_aligned_query():
    centroid = np.array([1.0, 0.0])
    gate = OffTopicGate(centroid, similarity_threshold=0.35)
    aligned_query = np.array([1.0, 0.1])
    assert gate.is_off_topic(aligned_query) is False


def test_off_topic_gate_handles_zero_vector():
    centroid = np.array([1.0, 0.0])
    gate = OffTopicGate(centroid, similarity_threshold=0.35)
    assert gate.similarity(np.array([0.0, 0.0])) == 0.0
    assert gate.is_off_topic(np.array([0.0, 0.0])) is True


def test_should_refuse_on_off_topic_even_with_high_confidence():
    centroid = np.array([1.0, 0.0])
    gate = OffTopicGate(centroid, similarity_threshold=0.35)
    orthogonal_query = np.array([0.0, 1.0])
    assert should_refuse(gate, orthogonal_query, retrieval_confidence=0.99) is True


def test_should_refuse_on_low_confidence_even_when_on_topic():
    centroid = np.array([1.0, 0.0])
    gate = OffTopicGate(centroid, similarity_threshold=0.35)
    aligned_query = np.array([1.0, 0.0])
    assert should_refuse(gate, aligned_query, retrieval_confidence=0.05) is True


def test_should_not_refuse_when_on_topic_and_confident():
    centroid = np.array([1.0, 0.0])
    gate = OffTopicGate(centroid, similarity_threshold=0.35)
    aligned_query = np.array([1.0, 0.0])
    assert should_refuse(gate, aligned_query, retrieval_confidence=0.8) is False

import math

from evaluation.retrieval_metrics import aggregate, hit_rate_at_k, ndcg_at_k, reciprocal_rank, recall_at_k


def test_recall_at_k_perfect():
    assert recall_at_k(["a", "b", "c"], {"a"}, k=3) == 1.0


def test_recall_at_k_miss():
    assert recall_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0


def test_recall_at_k_partial():
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5


def test_recall_at_k_respects_k_cutoff():
    # relevant item is at rank 5, so recall@3 should be 0
    assert recall_at_k(["x", "y", "z", "w", "a"], {"a"}, k=3) == 0.0
    assert recall_at_k(["x", "y", "z", "w", "a"], {"a"}, k=5) == 1.0


def test_recall_at_k_zero_relevant_is_nan():
    assert math.isnan(recall_at_k(["a", "b"], set(), k=3))


def test_hit_rate_at_k():
    assert hit_rate_at_k(["a", "b"], {"b"}, k=2) == 1.0
    assert hit_rate_at_k(["a", "b"], {"c"}, k=2) == 0.0
    assert math.isnan(hit_rate_at_k(["a"], set(), k=2))


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == 1 / 3


def test_reciprocal_rank_not_found():
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_reciprocal_rank_zero_relevant_is_nan():
    assert math.isnan(reciprocal_rank(["a"], set()))


def test_ndcg_at_k_perfect_order():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_ndcg_at_k_worse_order_scores_less_than_one():
    # relevant item pushed to rank 2 instead of rank 1
    score = ndcg_at_k(["x", "a"], {"a"}, k=2)
    assert 0 < score < 1.0


def test_ndcg_at_k_zero_relevant_is_nan():
    assert math.isnan(ndcg_at_k(["a"], set(), k=2))


def test_aggregate_skips_nan():
    assert aggregate([1.0, float("nan"), 0.5]) == 0.75


def test_aggregate_all_nan_is_nan():
    assert math.isnan(aggregate([float("nan"), float("nan")]))


def test_aggregate_empty_is_nan():
    assert math.isnan(aggregate([]))

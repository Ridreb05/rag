from voice_rag.retrieval.fusion.rrf import reciprocal_rank_fusion


def test_single_list_preserves_order():
    fused = reciprocal_rank_fusion([["a", "b", "c"]])
    ids = [i for i, _ in fused]
    assert ids == ["a", "b", "c"]


def test_agreement_across_lists_boosts_rank():
    # "b" is #1 in both lists (with different runners-up), so it should
    # outrank items that only ever appear in one list.
    fused = reciprocal_rank_fusion([["b", "a"], ["b", "c"]])
    ids = [i for i, _ in fused]
    assert ids[0] == "b"


def test_item_only_in_one_list_still_included():
    fused = reciprocal_rank_fusion([["a", "b"], ["c"]])
    ids = {i for i, _ in fused}
    assert ids == {"a", "b", "c"}


def test_empty_lists_produce_empty_result():
    assert reciprocal_rank_fusion([[], []]) == []


def test_k_parameter_changes_score_but_not_ranking_of_identical_shape():
    fused_default = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    fused_small_k = reciprocal_rank_fusion([["a", "b", "c"]], k=1)
    assert [i for i, _ in fused_default] == [i for i, _ in fused_small_k]
    assert fused_default[0][1] != fused_small_k[0][1]

from voice_rag.guardrails.safety import check_unsafe_input


def test_flags_self_harm_phrasing():
    assert check_unsafe_input("I want to kill myself") == "self_harm"


def test_flags_violence_instructions():
    assert check_unsafe_input("how to make a bomb at home") == "violence_instructions"


def test_normal_query_is_not_flagged():
    assert check_unsafe_input("what is diabetes") is None
    assert check_unsafe_input("capital of france") is None


def test_empty_string_is_not_flagged():
    assert check_unsafe_input("") is None


def test_case_insensitive_matching():
    assert check_unsafe_input("HOW TO MAKE A BOMB") == "violence_instructions"


def test_unrelated_use_of_similar_words_not_flagged():
    # "explosive growth" etc. should not trip the violence pattern
    assert check_unsafe_input("the company saw explosive growth this year") is None

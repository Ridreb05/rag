from voice_rag.query_processing.granularity import Granularity, classify_granularity


def test_short_query_is_narrow():
    assert classify_granularity("capital of France") == Granularity.NARROW


def test_query_with_digit_is_narrow_even_if_long():
    q = "what was the population of the city in the year 2011 census"
    assert classify_granularity(q) == Granularity.NARROW


def test_query_with_devanagari_digit_is_narrow():
    q = "सन् २०११ की जनगणना में शहर की जनसंख्या कितनी थी और कैसे बढ़ी"
    assert classify_granularity(q) == Granularity.NARROW


def test_long_query_without_digits_is_broad():
    q = "what is diabetes and how does it affect the human body over time"
    assert classify_granularity(q) == Granularity.BROAD


def test_medium_query_without_digits_is_ambiguous():
    q = "why does diabetes affect the body"  # 6 tokens: > narrow_max(3), < broad_min(8)
    assert classify_granularity(q) == Granularity.AMBIGUOUS


def test_empty_query_is_ambiguous():
    assert classify_granularity("") == Granularity.AMBIGUOUS
    assert classify_granularity("   ") == Granularity.AMBIGUOUS

import pytest
from pydantic import ValidationError

from voice_rag.pipeline.ingestion.schema import LANGUAGES, LANGUAGES_WITHOUT_TRAIN, Passages, parquet_path, resolve_url


def test_parquet_path_uses_real_filename_stems_not_iso_codes():
    # Confirmed against the live HF repo listing: files use 3-letter stems
    # (asm, ben, guj, hin, ...), not the dataset script's 2-letter ISO codes.
    assert parquet_path("hi", "train") == "train/hintrain.parquet"
    assert parquet_path("as", "validation") == "validation/asmval.parquet"
    assert parquet_path("ur", "train") == "train/urdtrain.parquet"
    assert parquet_path("sa", "validation") == "validation/sanval.parquet"


def test_all_languages_have_validation_path():
    for lang in LANGUAGES:
        path = parquet_path(lang, "validation")
        assert path.startswith("validation/") and path.endswith(".parquet")


def test_telugu_has_no_train_split():
    assert "te" in LANGUAGES_WITHOUT_TRAIN
    with pytest.raises(ValueError, match="no train split"):
        parquet_path("te", "train")


def test_non_telugu_languages_have_train_path():
    for lang in LANGUAGES:
        if lang in LANGUAGES_WITHOUT_TRAIN:
            continue
        path = parquet_path(lang, "train")
        assert path.startswith("train/") and path.endswith(".parquet")


def test_parquet_path_rejects_unknown_language():
    with pytest.raises(ValueError, match="unknown language"):
        parquet_path("xx", "train")


def test_parquet_path_rejects_bad_split():
    with pytest.raises(ValueError, match="split must be"):
        parquet_path("hi", "test")


def test_resolve_url_shape():
    url = resolve_url("hi", "validation")
    assert url == "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet"


def test_passages_accepts_parallel_arrays():
    p = Passages(is_selected=[0, 1], English_passages=["a", "b"], Translated_passages=["x", "y"])
    assert len(p) == 2


def test_passages_rejects_non_parallel_arrays():
    with pytest.raises(ValidationError, match="not parallel"):
        Passages(is_selected=[0, 1], English_passages=["a"], Translated_passages=["x", "y"])


def test_passages_empty_is_valid():
    p = Passages()
    assert len(p) == 0

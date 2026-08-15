"""Schema for ai4bharat/MSMARCO-XI, verified against the dataset's own
loading script (ms_marco_translations.py) and the live parquet footer
schema on 2026-08-13 — not assumed from the dataset card prose.

Verified facts that differ from a naive reading of the dataset card:
  - ``passages`` is NOT a list of per-passage dicts. It is a struct of
    three *parallel* arrays (``is_selected``, ``English_passages``,
    ``Translated_passages``), all indexed by the same passage position.
  - The translated answer field is capitalized ``Answer``, not ``answer``.
  - ``meta.temperature`` / ``meta.top_p`` are stored as int64 in the
    parquet files (not float), despite the loader script casting to
    float — treat them as numeric-but-possibly-degenerate (see
    ``analyze.py`` for the sanity check this implies).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

# (ISO 639-1 code, English name, HF filename stem)
LANGUAGES: dict[str, str] = {
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}

# Empirically confirmed via the HF API file listing (api/datasets/.../tree/main):
# every language has a validation/{stem}val.parquet file, but "te" (Telugu)
# has NO train/teltrain.parquet — train coverage is 13 of 14 languages.
LANGUAGES_WITHOUT_TRAIN: frozenset[str] = frozenset({"te"})

# The dataset's own loading script (ms_marco_translations.py) builds filenames
# from the 2-letter ISO codes above (e.g. "train/astrain.jsonl") — but the
# actual files in the repo use different 3-letter stems, AND the repo ships
# .parquet, not the .jsonl the script requests. Loading via the script's
# BuilderConfig would 404. Confirmed directly against the HF tree listing
# on 2026-08-13; do not "correct" this back to the 2-letter codes.
_FILENAME_STEM: dict[str, str] = {
    "as": "asm",
    "bn": "ben",
    "gu": "guj",
    "hi": "hin",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    "ne": "nep",
    "or": "ori",
    "pa": "pan",
    "sa": "san",
    "ta": "tam",
    "te": "tel",
    "ur": "urd",
}

REPO_ID = "ai4bharat/MSMARCO-XI"


def parquet_path(lang: str, split: str) -> str:
    """Path of the parquet file within the HF dataset repo, e.g. 'train/hintrain.parquet'.

    Uses the actual on-disk 3-letter filename stem (_FILENAME_STEM), not the
    2-letter ISO code the dataset's own loading script incorrectly assumes.
    """
    if split not in ("train", "validation"):
        raise ValueError(f"split must be 'train' or 'validation', got {split!r}")
    if lang not in LANGUAGES:
        raise ValueError(f"unknown language code {lang!r}, expected one of {sorted(LANGUAGES)}")
    if split == "train" and lang in LANGUAGES_WITHOUT_TRAIN:
        raise ValueError(
            f"language {lang!r} ({LANGUAGES[lang]}) has no train split in this dataset "
            "(confirmed absent from the repo file listing)"
        )
    stem = _FILENAME_STEM[lang]
    suffix = "train" if split == "train" else "val"
    return f"{split}/{stem}{suffix}.parquet"


def resolve_url(lang: str, split: str, revision: str = "main") -> str:
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/{revision}/{parquet_path(lang, split)}"


class TranslationMeta(BaseModel):
    model_name: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None


class Passages(BaseModel):
    """Three parallel arrays — position i in each array describes the same passage."""

    is_selected: list[int] = Field(default_factory=list)
    English_passages: list[str] = Field(default_factory=list)
    Translated_passages: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_parallel_lengths(self) -> "Passages":
        lengths = {len(self.is_selected), len(self.English_passages), len(self.Translated_passages)}
        if len(lengths) > 1:
            raise ValueError(
                f"passages arrays are not parallel: is_selected={len(self.is_selected)}, "
                f"English_passages={len(self.English_passages)}, "
                f"Translated_passages={len(self.Translated_passages)}"
            )
        return self

    def __len__(self) -> int:
        return len(self.is_selected)


class MSMarcoXIRow(BaseModel):
    """One row of the raw dataset, as it actually appears in the parquet files."""

    source_lang: str = ""
    target_lang: str = ""
    meta: TranslationMeta = Field(default_factory=TranslationMeta)
    query: str = ""
    Answer: str = ""
    query_id: int
    query_type: str = ""
    passages: Passages = Field(default_factory=Passages)
    Eng_Query: str = ""
    Eng_Answer: str = ""


class NormalizedPassage(BaseModel):
    """One (query, passage) pair, flattened out of a MSMarcoXIRow for downstream use."""

    query_id: int
    language: str
    passage_index: int
    is_selected: bool
    text_translated: str
    text_english: str

    @property
    def passage_uid(self) -> str:
        return f"{self.language}:{self.query_id}:{self.passage_index}"

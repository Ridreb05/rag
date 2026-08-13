# Dataset Analysis

The architecture is adapted to what MSMARCO-XI actually contains, not to a
generic "documents with titles and sections" assumption. Verified against the
[dataset card](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) and its
accompanying paper ([IndicRAGSuite, arXiv:2506.01615](https://arxiv.org/abs/2506.01615)).

| Property | Finding |
|---|---|
| Source | Direct machine translation of the original MS MARCO passage-ranking dataset into 14 Indic languages. |
| Languages | Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Nepali, Odia, Punjabi, Sanskrit, Tamil, Telugu, Urdu. |
| Splits | Verified via live Parquet footer reads (exact, not estimated): **validation is exactly 97,941 rows for every one of the 14 languages** — the same underlying query set, translated 14 ways. **Train varies per language**, 754,154–782,282 rows, and **Telugu has no train file at all** (confirmed from the repo's file listing — `train/teltrain.parquet` does not exist). Format is parquet only; the dataset's own loading script requests `.jsonl`, which doesn't exist in the repo (see below). |
| Query fields | `query_id`, `query_type`, translated `query`, translated `Answer` (capitalized), plus preserved `Eng_Query` / `Eng_Answer`. |
| Passage fields | **Correction to the original card-based summary:** `passages` is not a list of per-passage dicts. It's a struct of **three parallel arrays** — `is_selected`, `English_passages`, `Translated_passages` — all indexed by the same passage position. ~10 passage slots per query. Verified against the live Parquet schema and the dataset's own `ms_marco_translations.py` loader. |
| Translation metadata | source/target language codes, decoding params (`temperature`, `top_p`, penalties) used to produce the translation — useful for filtering low-quality machine-translated rows. |
| Document hierarchy | **None.** No titles, sections, or multi-passage documents — this is a flat corpus of short, already-segmented passages, structurally identical to the original MS MARCO passage-ranking format. |
| Relevance labels | `is_selected` is a ready-made qrel: usable directly for Recall@K / MRR@K / NDCG@K without manual annotation. |
| License | Inherits original MS MARCO dataset terms (non-commercial research use) — flag for legal review before any production/commercial deployment. |

### The dataset's loading script is broken — load via parquet directly

`ms_marco_translations.py` (the repo's HF `datasets` builder script) does two
things that don't match the actual repo contents: it looks up files by
2-letter ISO code (e.g. `train/astrain.jsonl`) when the real files use
3-letter stems (`train/asmtrain.parquet`), and it requests `.jsonl` when the
repo only ships `.parquet`. Calling `datasets.load_dataset("ai4bharat/MSMARCO-XI", "as")`
through the script 404s. The ingestion layer (`voice_rag.ingestion.hf_source`)
bypasses the script entirely and reads the parquet files directly by their
real paths — see `voice_rag/ingestion/schema.py` for the verified
code → filename-stem mapping.

## Adapting the architecture to this finding

Because the corpus has **no natural document hierarchy**, hierarchical
chunking (Document → Section → Passage → Sentence) is *not load-bearing for
this dataset as shipped* — there's no section boundary to hierarchically
chunk. It's retained in the architecture as a forward-compatible layer for
when the system is pointed at real long-form documents (PDFs, wikis) in a
later deployment, but for MSMARCO-XI the operative unit is the passage, and
"chunking" here mostly means: (a) whether to further split unusually long
passages, and (b) how to aggregate multiple retrieved passages per query into
a coherent context. See [Architecture & Retrieval](02-architecture-and-retrieval.md#chunking-strategy).

## Data quality risk

Machine-translated text carries translation artifacts (calques, occasional
target-language disfluency, inconsistent transliteration of named entities).
This directly affects embedding quality and BM25 tokenization for
morphologically rich languages (Tamil, Telugu, Malayalam). The pipeline keeps
`Eng_Query`/`Eng_Answer` available as a fallback cross-lingual retrieval path
and logs per-language retrieval quality separately rather than assuming
uniform performance across all 14 languages (see
[Observability & Evaluation](07-observability-and-evaluation.md)).

For evaluation, `is_selected` is exploited directly as ground truth: each
validation-split query with its 10 passages and binary labels becomes one
qrel entry, giving a MS MARCO-style dev set per language without extra
annotation work.

## Verified against live data (2026-08-13)

Content-level analysis (`voice_rag.ingestion.analyze`) run against the
validation split for Hindi, Tamil, Sanskrit, Telugu, and Urdu (20,000
uniformly-randomly-sampled rows each, seed 42; full run script and
per-language JSON reports in `reports/dataset_analysis/`, combined table in
[`dataset-analysis-report.md`](dataset-analysis-report.md)). Findings that
materially affect the architecture:

- **~45% of queries have zero relevant passages.** Across every language
  sampled, 8,914 of 20,000 queries have no `is_selected=1` passage at all in
  their 10-passage pool (10,456 have exactly one; 630 have more than one —
  identical across languages, as expected, since it's the same underlying
  query set translated 14 ways). Cross-checked against the full 97,941-row
  Hindi/Tamil corpus built in ingestion: 44,045/97,941 = 45.0% zero-relevant,
  matching the sample. (An earlier head-of-file sample without randomization
  showed 38.3% — a ~7-point bias from non-random row order in the source
  file; the sampler now draws uniformly at random. Documented here as a
  concrete example of why the sampling method itself needs verification.)
  Any retrieval-quality evaluation (Recall@K, MRR@K) must explicitly exclude
  or separately track these zero-relevant queries rather than silently
  scoring them as failures — and the guardrail design's "not enough
  information" response ([Harness & Guardrails](03-harness-and-guardrails.md))
  is effectively the correct behavior for close to half of this dataset's
  queries by construction, not an edge case.
- **Passages are measurably reused across queries.** On the full Hindi/Tamil
  validation corpus (built via `voice_rag.ingestion.build_corpus`), content-hash
  deduplication collapsed 977,545 passage occurrences down to ~953–957K
  unique passages — a 2.1–2.5% duplicate rate, with some individual passages
  shared by dozens of different queries. This confirms the corpus-building
  step needs content-hash deduplication before indexing (implemented in
  `voice_rag/ingestion/build_corpus.py` — see
  [Architecture & Retrieval](02-architecture-and-retrieval.md#retrieval-architecture))
  — otherwise a real index would carry redundant near-identical vectors and
  a popular passage would be over-represented in retrieval results.
- **Translation quality looks solid by a coarse script-consistency check.**
  99.83–99.97% of non-empty translated passages contain at least one
  character from the target language's expected Unicode block, and the
  empty-translation rate was 0.00% across all five sampled languages in the
  20,000-row sample. This is a weak signal (it can't catch a fluent
  mistranslation), but it rules out the worst failure mode — passages
  silently left in English or empty.
- **Sanskrit's shorter word counts are a tokenization artifact, not a data
  quality issue.** Sanskrit passages average 42.9 words vs. Hindi's 60.0,
  but Sanskrit's extensive sandhi (compounding) means a whitespace-based
  word count under-counts its actual information content relative to more
  isolating languages. Any chunking or length-budgeting logic
  ([Architecture & Retrieval](02-architecture-and-retrieval.md#chunking-strategy))
  should token-count with the actual embedding model's tokenizer per
  language rather than relying on whitespace splitting for Sanskrit-like
  languages.
- **At least one query-length outlier exists.** The Hindi sample's max
  query length was 2,185 words — clearly not a real search query, likely a
  malformed row where a longer text block ended up in the `query` field.
  The p99 query length (19 words) is a far more representative ceiling;
  ingestion should treat extreme outliers as a data-quality signal to log
  and cap, not a real "extremely long query" case to design UX around
  (that case is still handled at the API layer regardless, per
  [Data & API](06-data-and-api.md)).

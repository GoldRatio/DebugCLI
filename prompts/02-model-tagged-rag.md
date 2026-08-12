# Prompt 02 — Platform-tagged RAG + model-filtered retrieval

Implements: the document library is model-aware — chunks carry the platforms
they apply to, and retrieval is filtered to the detected server model so a
wrong-model snippet can never become evidence.

## Mission

No chunk from a foreign server platform can be retrieved for a diagnosis; the
prompt states which platform the snippets were (or weren't) filtered by.

## Read first

- `src/harness/docs/ingest/chunk.py` (Chunk dataclass, Chunker)
- `src/harness/docs/ingest/library.py` (`DocLibrary.add`, `_all_chunks`,
  `build_retriever`, manifest, chunks.jsonl persistence)
- `src/harness/docs/retrieval/hybrid_search.py` (HybridRetriever)
- `src/harness/docs/retrieval/rag.py` (RagPipeline)
- `src/harness/inspect/decoder.py`, `src/harness/inspect/catalog/catalog_loader.py`
  (optional catalog `platforms` support)
- `src/harness/operator/cli.py` — `docs` subcommand (`add`/`ls`/`rm`/`reindex`)
- `tests/unit/test_docs_library.py` and `tests/unit/test_docs_operator.py` for style

## Changes

### 1. `docs/ingest/chunk.py`

- `Chunk` gains `platforms: list[str] = field(default_factory=list)` — the
  canonical `model_key` values this chunk applies to.
- Chunker is unchanged; platforms are attached at the library layer.

### 2. `docs/ingest/library.py`

- `DocLibrary.add(files, platform: str | None = None)`:
  - Stores `platform` in the manifest entry for each indexed doc.
  - `_parse_doc` / `_all_chunks` attach `platforms=[platform]` when set,
    otherwise `[]` (untagged = applies to all platforms so far).
- `_save_chunks` / `_load_chunks`: persist and restore `platforms` (JSON field
  present in new lines; old lines without it load via the dataclass default —
  add a migration test for this exact case).
- `reindex`/`remove` preserve the stored platform.
- `DocEntry` gains `platform: str | None`.

### 3. `docs/retrieval/hybrid_search.py`

- `HybridRetriever.__init__(chunks, semantic_weight=0.5, k=60,
  platform_filter: str | None = None)`.
- `query()` restricts the ranked candidate set to chunks whose `platforms` is
  empty or contains the filter; corpus BM25 stats computed over that subset.
  Return `[]` when the filtered corpus is empty.
- Untagged chunks are always eligible (they claim platform-neutral knowledge).

### 4. `docs/retrieval/rag.py`

- `RagPipeline.retrieve(query, top_k=5, platform: str | None = None)` and
  `lines(...)` pass through to the retriever filter.

### 5. `docs/worker` (CLI)

- `harness docs add <files...> --platform <key>` — optional `--platform`,
  default None (untagged).

### 6. `diagnosis/engine.py` + `diagnosis/prompt.py`

- `run()` already passes `model_key` to `docs_retriever` (Prompt 01); the CLI
  now honors it by building the retriever with `platform_filter=model_key`.
- `build_prompt` / `build_turn_evidence` add to `## System`:
  `rag_platform_filter={key|'none'}` and when a `key` was used, one line in
  `## Relevant Architecture Snippets`: `(retrieved for platform <key>)`.

### 7. Catalog platform scope (surface, don't enforce)

- `catalog_loader.py`: entries may carry an optional `platforms: list[str]`;
  `decoder.py` copies it onto `RegisterDecode.platforms` (new optional field,
  default `[]`).
- Prompt rendering: for a decode whose `platforms` is non-empty AND excludes
  the detected model_key, append `[PLATFORM MISMATCH — verify]` next to the
  register line. Do NOT change scoring in this prompt.

## Acceptance criteria

- `tests/unit/test_rag_platform_filter.py`:
  - tagged vs untagged chunks: filter returns tagged+untagged, never
    foreign-tagged; empty filtered corpus returns [].
  - library round-trip: add with platform → chunks.jsonl reload has the tag;
    legacy JSON line without `platforms` loads.
  - CLI: `docs add --platform` records the manifest entry.
- Prompt renders `rag_platform_filter` and the snippets filter note; a
  platform-mismatched decode renders `[PLATFORM MISMATCH — verify]`.
- `pytest tests/unit -q`, `ruff check src tests` green.
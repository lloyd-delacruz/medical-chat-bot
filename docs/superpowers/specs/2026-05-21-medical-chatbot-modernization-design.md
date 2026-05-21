# Medical Chatbot — Pipeline Modernization & Up-to-Date Data

**Date:** 2026-05-21
**Status:** Approved (design)
**Author:** Lloyd dela Cruz (with Claude Code)

## 1. Background

The project is a Retrieval-Augmented Generation (RAG) medical Q&A chatbot built from a
YouTube tutorial. Current pipeline:

- A single 2003 PDF (`data/Medical_book.pdf`, Gale Encyclopedia of Medicine) is loaded,
  reduced to minimal metadata, and split into ~500-char chunks.
- Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` (384-dim) and stored in
  a Pinecone serverless index named `medical-chatbot`.
- A LangChain retrieval chain answers questions using OpenAI `gpt-4o`.

The same logic is duplicated across `research/trials.ipynb`, `src/helper.py`,
`store_index.py`, and `app.py`. The code uses several **deprecated** LangChain imports, the
data source is **outdated** (2003), and there is **no medical safety guardrail**.

## 2. Goals

1. Modernize the entire pipeline (notebook + `src/` + `store_index.py` + `app.py`) using
   current, non-deprecated LangChain APIs.
2. Add **up-to-date** medical reference data via a unified ingestion layer with three
   sources: a curated built-in dataset, a drop-in PDF/text folder, and a live web fetcher
   for authoritative public-health sources (WHO / CDC / NIH / MedlinePlus).
3. Upgrade embeddings to `BAAI/bge-small-en-v1.5` (384-dim, free/local).
4. Upgrade the LLM to OpenAI `gpt-4.1` (with `gpt-4o-mini` as a cheaper toggle).
5. Add a safety-aware prompt: a "not a substitute for professional medical care" disclaimer,
   source citation, and scope guardrails.
6. Recreate `research/trials.ipynb` as a clean, runnable notebook mirroring the new pipeline.

### Non-goals (YAGNI)

- No switch of LLM provider to Anthropic (kept configurable-free; OpenAI only).
- No chat history / multi-turn memory, auth, or new UI work.
- No automated evaluation harness or test suite for retrieval quality (manual verification
  steps only).
- No unrelated refactoring of the Flask front-end (`templates/`, `static/`).

## 3. Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scope | Modernize whole pipeline | User selected. |
| Data sources | Curated + drop-in folder + live web (all three) | User selected. |
| LLM | OpenAI `gpt-4.1`, `gpt-4o-mini` toggle | User already has `OPENAI_API_KEY`. |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim) | Stronger than MiniLM, same dimension. |
| Index strategy | **Option A** — delete & recreate the `medical-chatbot` index | Mixing two embedding models in one index gives incoherent retrieval; re-index is required anyway because we add new data. |
| Safety | Disclaimer + citations + scope guardrails in system prompt | Medical domain; reduces harmful/over-confident answers. |

### Index re-indexing rationale

The existing `medical-chatbot` index holds MiniLM vectors. BGE vectors are also 384-dim, so
the index *dimension* is unchanged, but vectors from two different models are not comparable.
`store_index.py` will therefore **delete the index if it exists and recreate it** before
upserting freshly embedded documents from all enabled sources. This is a one-time wipe of the
old index contents.

## 4. Architecture

```
src/
  config.py        # NEW: all configuration from .env with safe defaults
  helper.py        # modernized loaders/splitter/embeddings; PDF + .txt/.md support
  data_sources.py  # NEW: unified ingestion -> List[Document] from enabled sources
  curated_data.py  # NEW: hand-authored, dated, cited current reference documents
  prompt.py        # safety-aware system prompt (disclaimer + citations + scope)
store_index.py     # modernized: (re)create index, ingest all sources, upsert
app.py             # modernized imports; configurable model; shows sources + disclaimer
research/trials.ipynb  # recreated clean notebook mirroring the new pipeline
requirements.txt   # + langchain-huggingface, langchain-text-splitters, beautifulsoup4, requests
.env(.example)     # documents new configuration variables
```

### 4.1 `src/config.py` (new)

Single source of truth, read from environment with defaults. Avoids the config drift of
hard-coded values duplicated across files.

| Variable | Default | Purpose |
|---|---|---|
| `PINECONE_API_KEY` | (required) | Pinecone auth |
| `OPENAI_API_KEY` | (required) | OpenAI auth |
| `INDEX_NAME` | `medical-chatbot` | Pinecone index |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HF embedding model |
| `EMBEDDING_DIM` | `384` | Index dimension |
| `LLM_MODEL` | `gpt-4.1` | OpenAI chat model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | Splitter params |
| `RETRIEVER_K` | `4` | Top-k retrieved chunks |
| `DATA_DIR` | `data/` | Drop-in folder |
| `ENABLE_CURATED` / `ENABLE_DROPIN` / `ENABLE_WEB` | `true` / `true` / `true` | Toggle sources |
| `PINECONE_CLOUD` / `PINECONE_REGION` | `aws` / `us-east-1` | Serverless spec |

### 4.2 `src/helper.py` (modernized)

- `from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader`
- `from langchain_huggingface import HuggingFaceEmbeddings`
- `from langchain_text_splitters import RecursiveCharacterTextSplitter`
- `from langchain_core.documents import Document`
- `load_files(data_dir)` — loads `*.pdf`, `*.txt`, `*.md` from the folder (drop-in support).
- `filter_to_minimal_docs(docs)` — preserves `source` and any existing rich metadata
  (`source_type`, `url`, `retrieved_at`) rather than dropping it.
- `text_split(docs)` — uses configured chunk size/overlap.
- `download_embeddings()` — returns BGE embeddings from config.

### 4.3 `src/data_sources.py` (new) — unified ingestion

`gather_documents() -> List[Document]` aggregates only the enabled sources:

1. **Curated** (`ENABLE_CURATED`): imports the list from `curated_data.py`.
2. **Drop-in** (`ENABLE_DROPIN`): `helper.load_files(DATA_DIR)`.
3. **Web** (`ENABLE_WEB`): `fetch_web_sources()` — iterates a hard-coded list of authoritative
   URLs, fetches with `requests`, extracts readable text with BeautifulSoup, wraps each as a
   `Document`. **Fail-soft:** a failed/empty fetch logs a warning and is skipped; it never
   aborts the build.

**Metadata schema** on every Document: `source` (human label), `source_type`
(`curated|file|web`), `url` (if applicable), `retrieved_at` (ISO date). This lets `app.py`
cite where an answer came from and how current it is.

### 4.4 `src/curated_data.py` (new)

A Python list of `Document` objects with current, **cited** reference content covering common
needs: common conditions, normal vital sign ranges, red-flag/emergency symptoms, basic
medication-interaction cautions, and general prevention guidance. Content will be grounded in
real WHO/CDC/NIH/MedlinePlus pages fetched at authoring time (via WebFetch) — not invented —
and each entry carries its `url` and `retrieved_at`. This guarantees a reproducible,
network-free baseline of up-to-date data even when `ENABLE_WEB=false`.

### 4.5 `src/prompt.py` (safety-aware)

System prompt additions:
- Role + concise-answer instruction (kept from original).
- **Disclaimer**: state that responses are educational and not a substitute for a licensed
  professional; advise seeking care for emergencies.
- **Citation**: reference the source(s) of retrieved context when answering.
- **Scope guardrail**: if context is insufficient, say so rather than guessing; do not provide
  individualized diagnosis or dosing.

### 4.6 `store_index.py` (modernized)

1. Load config + keys.
2. `docs = data_sources.gather_documents()`.
3. `minimal = filter_to_minimal_docs(docs)`; `chunks = text_split(minimal)`.
4. Pinecone: if index exists, **delete**, then create with `EMBEDDING_DIM` + serverless spec.
5. `PineconeVectorStore.from_documents(chunks, embedding, index_name=INDEX_NAME)`.
6. Print a summary (per-source doc counts, total chunks).

### 4.7 `app.py` (modernized)

- Modernized imports; model + retriever-k from config.
- Connect to existing index via `from_existing_index`.
- Optionally append a short disclaimer and the cited sources to the response.
- Flask routes unchanged.

### 4.8 `research/trials.ipynb` (recreated)

A clean, runnable notebook that walks the new pipeline end to end (load → gather sources →
chunk → embed → create index → upsert → retrieve → RAG answer) with markdown explanations,
mirroring the structure shown in the reference screenshot but using the modernized code.

## 5. Dependencies (`requirements.txt`)

Add: `langchain-huggingface`, `langchain-text-splitters`, `beautifulsoup4`, `requests`.
Keep existing pinned libs; bump only where required for the new imports. `langchain-community`
is already present.

## 6. Data Flow

```
[curated_data.py]  [data/ folder]  [WHO/CDC/NIH/MedlinePlus URLs]
        \               |                     /
         \              |                    /
          v             v                   v
         data_sources.gather_documents()  (rich metadata)
                        |
                        v
        filter_to_minimal_docs -> text_split (chunks)
                        |
                        v
            BGE embeddings (384-dim)
                        |
                        v
        Pinecone index "medical-chatbot" (recreated)
                        |
        app.py: retriever (top-k) -> gpt-4.1 + safety prompt -> answer + sources
```

## 7. Error Handling

- **Missing keys:** `config.py` raises a clear error naming the missing variable.
- **Web fetch failures:** logged and skipped (fail-soft); build proceeds with other sources.
- **Empty drop-in folder:** allowed (no error) as long as at least one source yields docs.
- **No documents at all:** `store_index.py` aborts with an explicit message.

## 8. Verification (manual — user runs these)

The author cannot run these (requires the user's Pinecone/OpenAI keys):

1. `pip install -r requirements.txt` succeeds.
2. `python -c "from src import data_sources, helper, config"` imports cleanly (no deprecation
   errors).
3. `python store_index.py` prints per-source counts and "index built" without errors.
4. `python app.py` starts; asking "What are the warning signs of a stroke?" returns an answer
   that includes a disclaimer and cites a source.
5. Notebook runs top-to-bottom without deprecation warnings.

## 9. Open Risks

- Live source URLs may change/return different markup over time; mitigated by fail-soft
  fetching + the curated fallback.
- BGE model download (~130MB) on first run; documented in README/notebook.
- Recreating the index destroys old vectors (intended; Option A).

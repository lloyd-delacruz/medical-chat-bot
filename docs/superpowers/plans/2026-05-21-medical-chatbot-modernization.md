# Medical Chatbot Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modernize the medical-chatbot RAG pipeline (current LangChain APIs), add a unified up-to-date ingestion layer (curated + drop-in files + live WHO/CDC/NIH/MedlinePlus), upgrade embeddings to BAAI/bge-small-en-v1.5 and the LLM to OpenAI gpt-4.1, and add a safety-aware prompt.

**Architecture:** A central `src/config.py` reads all settings from `.env`. `src/data_sources.py` aggregates documents from three toggleable sources into `List[Document]` with rich metadata. `store_index.py` recreates the Pinecone index and upserts BGE-embedded chunks. `app.py` serves a Flask RAG endpoint built lazily so modules import cleanly for testing. A recreated notebook reuses these functions end-to-end.

**Tech Stack:** Python 3.12, LangChain 0.3.x (`langchain-community`, `langchain-huggingface`, `langchain-text-splitters`, `langchain-pinecone`, `langchain-openai`), Pinecone serverless, BAAI/bge-small-en-v1.5 (sentence-transformers), OpenAI gpt-4.1, Flask, BeautifulSoup4 + requests, pytest.

**Spec:** `docs/superpowers/specs/2026-05-21-medical-chatbot-modernization-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `requirements.txt` | Modify | Add new runtime deps |
| `requirements-dev.txt` | Create | pytest |
| `pytest.ini` | Create | Make `src` importable from tests |
| `.env.example` | Create | Document all config vars |
| `tests/__init__.py` | Create | Test package marker |
| `src/config.py` | Create | All config from env + key validation |
| `src/helper.py` | Modify | Modern loaders/splitter/embeddings; PDF+txt+md |
| `src/curated_data.py` | Create | Hand-authored, cited current reference docs |
| `src/data_sources.py` | Create | Unified ingestion → `List[Document]` |
| `src/prompt.py` | Modify | Safety-aware system prompt |
| `store_index.py` | Modify | Recreate index + ingest all sources |
| `app.py` | Modify | Lazy `create_app()` / `build_rag_chain()` |
| `research/trials.ipynb` | Recreate | Clean notebook over the new pipeline |
| `README.md` | Modify | Updated setup/run instructions |
| `tests/test_config.py` | Create | Config tests |
| `tests/test_helper.py` | Create | helper tests |
| `tests/test_curated_data.py` | Create | curated_data tests |
| `tests/test_data_sources.py` | Create | data_sources tests |
| `tests/test_prompt.py` | Create | prompt tests |
| `tests/test_store_index.py` | Create | store_index unit tests |
| `tests/test_app_smoke.py` | Create | app import smoke test |

---

## Task 0: Project scaffolding (deps + test config)

**Files:**
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`, `pytest.ini`, `.env.example`, `tests/__init__.py`

- [ ] **Step 1: Update `requirements.txt`** to this exact content:

```
langchain==0.3.26
langchain-community==0.3.26
langchain-huggingface
langchain-text-splitters
langchain-pinecone==0.2.8
langchain-openai==0.3.24
sentence-transformers==4.1.0
pypdf==5.6.1
beautifulsoup4
requests
flask==3.1.1
python-dotenv==1.1.0
-e .
```

- [ ] **Step 2: Create `requirements-dev.txt`:**

```
-r requirements.txt
pytest
```

- [ ] **Step 3: Create `pytest.ini`** (lets tests do `from src... import ...` from repo root):

```ini
[pytest]
pythonpath = .
testpaths = tests
```

- [ ] **Step 4: Create `tests/__init__.py`** (empty file, just `touch`).

- [ ] **Step 5: Create `.env.example`:**

```
# Required secrets
PINECONE_API_KEY=your-pinecone-key
OPENAI_API_KEY=your-openai-key

# Index / embeddings
INDEX_NAME=medical-chatbot
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIM=384

# LLM (use gpt-4o-mini for cheaper runs)
LLM_MODEL=gpt-4.1

# Chunking / retrieval
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RETRIEVER_K=4

# Data sources
DATA_DIR=data/
ENABLE_CURATED=true
ENABLE_DROPIN=true
ENABLE_WEB=true

# Pinecone serverless location
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

- [ ] **Step 6: Install dev deps** (so later tasks can run tests):

Run: `pip install -r requirements-dev.txt`
Expected: completes without resolver errors (first run downloads the BGE model lazily later, not now).

- [ ] **Step 7: Commit**

```bash
git add requirements.txt requirements-dev.txt pytest.ini tests/__init__.py .env.example
git commit -m "chore: add deps, pytest config, and .env.example"
```

---

## Task 1: `src/config.py`

**Files:**
- Create: `src/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests** in `tests/test_config.py`:

```python
import pytest
from src.config import Config


def test_defaults_when_env_empty():
    cfg = Config.from_env({})
    assert cfg.index_name == "medical-chatbot"
    assert cfg.embedding_model == "BAAI/bge-small-en-v1.5"
    assert cfg.embedding_dim == 384
    assert cfg.llm_model == "gpt-4.1"
    assert cfg.chunk_size == 500
    assert cfg.chunk_overlap == 50
    assert cfg.retriever_k == 4
    assert cfg.enable_curated is True
    assert cfg.enable_web is True
    assert cfg.pinecone_cloud == "aws"
    assert cfg.pinecone_region == "us-east-1"


def test_env_overrides_and_types():
    cfg = Config.from_env(
        {"LLM_MODEL": "gpt-4o-mini", "EMBEDDING_DIM": "1536", "ENABLE_WEB": "false"}
    )
    assert cfg.llm_model == "gpt-4o-mini"
    assert cfg.embedding_dim == 1536
    assert cfg.enable_web is False


def test_require_keys_raises_naming_missing():
    cfg = Config.from_env({"PINECONE_API_KEY": "x"})
    with pytest.raises(ValueError) as exc:
        cfg.require_keys()
    assert "OPENAI_API_KEY" in str(exc.value)


def test_require_keys_passes_when_present():
    cfg = Config.from_env({"PINECONE_API_KEY": "x", "OPENAI_API_KEY": "y"})
    cfg.require_keys()  # should not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 3: Create `src/config.py`:**

```python
from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    pinecone_api_key: str | None
    openai_api_key: str | None
    index_name: str
    embedding_model: str
    embedding_dim: int
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    retriever_k: int
    data_dir: str
    enable_curated: bool
    enable_dropin: bool
    enable_web: bool
    pinecone_cloud: str
    pinecone_region: str

    @classmethod
    def from_env(cls, env: dict | None = None) -> "Config":
        env = os.environ if env is None else env
        return cls(
            pinecone_api_key=env.get("PINECONE_API_KEY"),
            openai_api_key=env.get("OPENAI_API_KEY"),
            index_name=env.get("INDEX_NAME", "medical-chatbot"),
            embedding_model=env.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            embedding_dim=int(env.get("EMBEDDING_DIM", "384")),
            llm_model=env.get("LLM_MODEL", "gpt-4.1"),
            chunk_size=int(env.get("CHUNK_SIZE", "500")),
            chunk_overlap=int(env.get("CHUNK_OVERLAP", "50")),
            retriever_k=int(env.get("RETRIEVER_K", "4")),
            data_dir=env.get("DATA_DIR", "data/"),
            enable_curated=_as_bool(env.get("ENABLE_CURATED", "true")),
            enable_dropin=_as_bool(env.get("ENABLE_DROPIN", "true")),
            enable_web=_as_bool(env.get("ENABLE_WEB", "true")),
            pinecone_cloud=env.get("PINECONE_CLOUD", "aws"),
            pinecone_region=env.get("PINECONE_REGION", "us-east-1"),
        )

    def require_keys(self) -> None:
        missing = [
            name
            for name, val in (
                ("PINECONE_API_KEY", self.pinecone_api_key),
                ("OPENAI_API_KEY", self.openai_api_key),
            )
            if not val
        ]
        if missing:
            raise ValueError(
                "Missing required environment variable(s): " + ", ".join(missing)
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add central Config loaded from environment"
```

---

## Task 2: Modernize `src/helper.py`

**Files:**
- Modify: `src/helper.py` (full rewrite)
- Test: `tests/test_helper.py`

- [ ] **Step 1: Write the failing tests** in `tests/test_helper.py`:

```python
from langchain_core.documents import Document
from src import helper
from src.config import Config


def test_filter_to_minimal_docs_keeps_rich_metadata():
    docs = [
        Document(
            page_content="hello",
            metadata={
                "source": "WHO",
                "source_type": "web",
                "url": "https://x",
                "retrieved_at": "2026-05-21",
                "page": 3,  # should be dropped
            },
        )
    ]
    out = helper.filter_to_minimal_docs(docs)
    assert out[0].page_content == "hello"
    assert out[0].metadata == {
        "source": "WHO",
        "source_type": "web",
        "url": "https://x",
        "retrieved_at": "2026-05-21",
    }


def test_filter_defaults_missing_source():
    out = helper.filter_to_minimal_docs([Document(page_content="x", metadata={})])
    assert out[0].metadata["source"] == "unknown"


def test_text_split_chunks_long_doc():
    long_text = "word " * 400  # ~2000 chars
    cfg = Config.from_env({"CHUNK_SIZE": "200", "CHUNK_OVERLAP": "20"})
    chunks = helper.text_split([Document(page_content=long_text, metadata={"source": "s"})], cfg)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)


def test_load_files_reads_txt_and_md(tmp_path):
    (tmp_path / "a.txt").write_text("alpha content")
    (tmp_path / "b.md").write_text("# beta content")
    docs = helper.load_files(str(tmp_path))
    contents = " ".join(d.page_content for d in docs)
    assert "alpha content" in contents
    assert "beta content" in contents


def test_download_embeddings_uses_config_model(monkeypatch):
    captured = {}

    class FakeEmb:
        def __init__(self, model_name=None):
            captured["model_name"] = model_name

    monkeypatch.setattr(helper, "HuggingFaceEmbeddings", FakeEmb)
    cfg = Config.from_env({"EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5"})
    helper.download_embeddings(cfg)
    assert captured["model_name"] == "BAAI/bge-small-en-v1.5"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_helper.py -v`
Expected: FAIL (current `helper.py` uses deprecated imports / lacks `load_files`, `download_embeddings`)

- [ ] **Step 3: Rewrite `src/helper.py`:**

```python
from __future__ import annotations

from typing import List

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Config

_METADATA_KEYS = ("source", "source_type", "url", "retrieved_at")


def load_files(data_dir: str) -> List[Document]:
    """Load every PDF, TXT and Markdown file from a directory."""
    documents: List[Document] = []
    loader_specs = [
        ("**/*.pdf", PyPDFLoader, {}),
        ("**/*.txt", TextLoader, {"autodetect_encoding": True}),
        ("**/*.md", TextLoader, {"autodetect_encoding": True}),
    ]
    for glob, loader_cls, loader_kwargs in loader_specs:
        loader = DirectoryLoader(
            data_dir, glob=glob, loader_cls=loader_cls, loader_kwargs=loader_kwargs
        )
        documents.extend(loader.load())
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Keep page_content plus only the metadata keys we rely on downstream."""
    minimal_docs: List[Document] = []
    for doc in docs:
        metadata = {k: doc.metadata[k] for k in _METADATA_KEYS if k in doc.metadata}
        metadata.setdefault("source", doc.metadata.get("source") or "unknown")
        minimal_docs.append(Document(page_content=doc.page_content, metadata=metadata))
    return minimal_docs


def text_split(docs: List[Document], config: Config | None = None) -> List[Document]:
    config = config or Config.from_env()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    return splitter.split_documents(docs)


def download_embeddings(config: Config | None = None) -> HuggingFaceEmbeddings:
    config = config or Config.from_env()
    return HuggingFaceEmbeddings(model_name=config.embedding_model)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_helper.py -v`
Expected: 5 passed (no `LangChainDeprecationWarning` for the rewritten imports)

- [ ] **Step 5: Commit**

```bash
git add src/helper.py tests/test_helper.py
git commit -m "feat: modernize helper with current imports and multi-format loader"
```

---

## Task 3: `src/curated_data.py` (current, cited reference docs)

**Files:**
- Create: `src/curated_data.py`
- Test: `tests/test_curated_data.py`

- [ ] **Step 1: Write the failing tests** in `tests/test_curated_data.py`:

```python
from langchain_core.documents import Document
from src.curated_data import CURATED_DOCS


def test_curated_docs_nonempty_list_of_documents():
    assert isinstance(CURATED_DOCS, list)
    assert len(CURATED_DOCS) >= 5
    assert all(isinstance(d, Document) for d in CURATED_DOCS)


def test_each_doc_has_required_metadata_and_content():
    for d in CURATED_DOCS:
        assert d.page_content.strip()
        assert d.metadata["source_type"] == "curated"
        assert d.metadata["source"]
        assert d.metadata["url"].startswith("http")
        assert d.metadata["retrieved_at"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_curated_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.curated_data'`

- [ ] **Step 3: Verify the citation URLs are live** (do this before writing the file). Use WebFetch on each URL below; if any returns 404/moved, replace it with the current authoritative page for the same topic and update the content if the facts changed. Set `VERIFIED_ON` to today's date.

URLs to verify:
- `https://www.cdc.gov/stroke/signs-symptoms/`
- `https://medlineplus.gov/ency/article/002341.htm` (Vital signs)
- `https://medlineplus.gov/ency/article/001927.htm` (When to use the emergency room — adult)
- `https://www.who.int/news-room/fact-sheets/detail/hypertension`
- `https://www.who.int/news-room/fact-sheets/detail/diabetes`
- `https://medlineplus.gov/druginformation.html`
- `https://www.cdc.gov/prevention/index.html`

- [ ] **Step 4: Create `src/curated_data.py`** (the content below is stable clinical guidance; correct any URL that failed verification in Step 3):

```python
"""Curated, current medical reference documents.

Grounded in public guidance from WHO, CDC, and NIH/MedlinePlus. Each entry
carries its source URL and the date it was last verified. These are stable
clinical facts that provide a reproducible, offline baseline of up-to-date
information even when the live web fetcher is disabled.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document

VERIFIED_ON = "2026-05-21"

_ENTRIES = [
    {
        "title": "Stroke warning signs (FAST)",
        "url": "https://www.cdc.gov/stroke/signs-symptoms/",
        "content": (
            "Stroke is a medical emergency. Use FAST to recognize warning signs. "
            "F - Face drooping: ask the person to smile; one side may droop. "
            "A - Arm weakness: ask them to raise both arms; one may drift down. "
            "S - Speech difficulty: speech may be slurred or hard to understand. "
            "T - Time to call emergency services immediately. Other sudden signs "
            "include numbness on one side of the body, confusion, trouble seeing, "
            "trouble walking or dizziness, and a severe headache with no known cause. "
            "Note the time symptoms first appeared, because some treatments work best "
            "when given soon after onset."
        ),
    },
    {
        "title": "Normal adult vital sign ranges",
        "url": "https://medlineplus.gov/ency/article/002341.htm",
        "content": (
            "Typical resting vital signs for healthy adults. Body temperature averages "
            "about 98.6 F (37 C), with a normal range of roughly 97 to 99 F "
            "(36.1 to 37.2 C). Resting heart rate is about 60 to 100 beats per minute. "
            "Respiratory rate is about 12 to 20 breaths per minute at rest. A blood "
            "pressure below 120/80 mm Hg is normal, while 130/80 mm Hg or higher is "
            "considered high. Individual normal values vary with age, fitness, and "
            "health conditions."
        ),
    },
    {
        "title": "When to seek emergency care (adults)",
        "url": "https://medlineplus.gov/ency/article/001927.htm",
        "content": (
            "Seek emergency care for difficulty breathing or shortness of breath; chest "
            "or upper-abdominal pain or pressure; fainting or sudden dizziness or "
            "weakness; sudden changes in vision; confusion or change in mental status; "
            "any sudden or severe pain; uncontrolled bleeding; severe or persistent "
            "vomiting or diarrhea; coughing or vomiting blood; suicidal feelings; or a "
            "head injury with loss of consciousness. When in doubt about a serious "
            "symptom, seek emergency care."
        ),
    },
    {
        "title": "Hypertension (high blood pressure) basics",
        "url": "https://www.who.int/news-room/fact-sheets/detail/hypertension",
        "content": (
            "Hypertension is when blood pressure is too high. It is written as systolic "
            "over diastolic; a reading at or above 140/90 mm Hg on two different days is "
            "commonly used to diagnose it. It often has no symptoms, so measurement "
            "matters. Risk factors include older age, family history, excess weight, "
            "physical inactivity, high-salt diets, and excessive alcohol use. It raises "
            "the risk of heart attack, stroke, and kidney disease. Management includes a "
            "healthy diet, less salt, physical activity, avoiding tobacco, limiting "
            "alcohol, and prescribed medicines."
        ),
    },
    {
        "title": "Type 2 diabetes basics",
        "url": "https://www.who.int/news-room/fact-sheets/detail/diabetes",
        "content": (
            "Diabetes is a chronic condition with high blood glucose. In type 2 diabetes "
            "the body does not use insulin well. Symptoms can include increased thirst, "
            "frequent urination, increased hunger, fatigue, and blurred vision, though "
            "many people have no early symptoms. Risk factors include excess weight, "
            "physical inactivity, and family history. Over time it can damage the heart, "
            "blood vessels, eyes, kidneys, and nerves. Prevention and management focus on "
            "healthy eating, regular activity, a healthy weight, and prescribed medicines."
        ),
    },
    {
        "title": "Medication safety and interactions (general guidance)",
        "url": "https://medlineplus.gov/druginformation.html",
        "content": (
            "Use medicines safely. Keep an up-to-date list of all prescription drugs, "
            "over-the-counter medicines, vitamins, and supplements, and share it with "
            "your doctor and pharmacist. Interactions can change how a medicine works or "
            "increase side effects. Combining medicines that cause drowsiness (such as "
            "opioids, sedatives, and alcohol) can dangerously slow breathing. Some pain "
            "relievers (NSAIDs such as ibuprofen) can interact with blood-pressure "
            "medicines and blood thinners. Read labels, follow dosing instructions, and "
            "ask a pharmacist before combining medicines. This is general information, "
            "not personalized advice."
        ),
    },
    {
        "title": "Everyday preventive health measures",
        "url": "https://www.cdc.gov/prevention/index.html",
        "content": (
            "Core preventive steps that lower the risk of many diseases. Do not smoke and "
            "avoid tobacco. Be physically active; adults should aim for at least 150 "
            "minutes of moderate activity per week. Eat a balanced diet rich in "
            "vegetables, fruits, and whole grains while limiting salt, added sugar, and "
            "saturated fat. Maintain a healthy weight, limit alcohol, get recommended "
            "vaccinations and screenings, wash hands regularly, and get adequate sleep. "
            "Routine check-ups help detect problems early."
        ),
    },
]


def _build() -> List[Document]:
    docs: List[Document] = []
    for entry in _ENTRIES:
        docs.append(
            Document(
                page_content=f"{entry['title']}.\n{entry['content']}",
                metadata={
                    "source": entry["title"],
                    "source_type": "curated",
                    "url": entry["url"],
                    "retrieved_at": VERIFIED_ON,
                },
            )
        )
    return docs


CURATED_DOCS: List[Document] = _build()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_curated_data.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add src/curated_data.py tests/test_curated_data.py
git commit -m "feat: add curated, cited current medical reference documents"
```

---

## Task 4: `src/data_sources.py` (unified ingestion)

**Files:**
- Create: `src/data_sources.py`
- Test: `tests/test_data_sources.py`

- [ ] **Step 1: Write the failing tests** in `tests/test_data_sources.py`:

```python
from langchain_core.documents import Document
from src import data_sources
from src.config import Config


def test_html_to_text_strips_scripts_and_collapses_space():
    html = "<html><head><style>x{}</style></head><body><script>bad()</script>" \
           "<p>Hello   world</p><p>Stay   safe</p></body></html>"
    text = data_sources.html_to_text(html)
    assert "bad()" not in text
    assert "x{}" not in text
    assert "Hello world" in text
    assert "Stay safe" in text


class _FakeResp:
    def __init__(self, text, ok=True):
        self.text = text
        self._ok = ok

    def raise_for_status(self):
        if not self._ok:
            raise RuntimeError("HTTP 500")


def test_fetch_web_sources_parses_and_tags(monkeypatch):
    monkeypatch.setattr(
        data_sources.requests,
        "get",
        lambda url, **kw: _FakeResp("<p>fact about health</p>"),
    )
    docs = data_sources.fetch_web_sources([{"label": "L", "url": "https://x"}])
    assert len(docs) == 1
    assert docs[0].metadata["source_type"] == "web"
    assert docs[0].metadata["url"] == "https://x"
    assert docs[0].metadata["retrieved_at"]
    assert "fact about health" in docs[0].page_content


def test_fetch_web_sources_is_failsoft(monkeypatch):
    def boom(url, **kw):
        raise ConnectionError("no network")

    monkeypatch.setattr(data_sources.requests, "get", boom)
    docs = data_sources.fetch_web_sources([{"label": "L", "url": "https://x"}])
    assert docs == []


def test_gather_respects_toggles_curated_only(monkeypatch):
    monkeypatch.setattr(
        data_sources, "CURATED_DOCS", [Document(page_content="c", metadata={"source_type": "curated"})]
    )
    cfg = Config.from_env({"ENABLE_WEB": "false", "ENABLE_DROPIN": "false"})
    docs = data_sources.gather_documents(cfg)
    assert len(docs) == 1
    assert docs[0].metadata["source_type"] == "curated"


def test_gather_all_disabled_returns_empty():
    cfg = Config.from_env(
        {"ENABLE_WEB": "false", "ENABLE_DROPIN": "false", "ENABLE_CURATED": "false"}
    )
    assert data_sources.gather_documents(cfg) == []


def test_gather_dropin_tags_source_type(monkeypatch):
    monkeypatch.setattr(
        data_sources.helper,
        "load_files",
        lambda data_dir: [Document(page_content="f", metadata={"source": "a.pdf"})],
    )
    cfg = Config.from_env(
        {"ENABLE_WEB": "false", "ENABLE_CURATED": "false", "ENABLE_DROPIN": "true"}
    )
    docs = data_sources.gather_documents(cfg)
    assert docs[0].metadata["source_type"] == "file"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data_sources.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data_sources'`

- [ ] **Step 3: Create `src/data_sources.py`:**

```python
from __future__ import annotations

import logging
from datetime import date
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from src import helper
from src.config import Config
from src.curated_data import CURATED_DOCS

logger = logging.getLogger(__name__)

WEB_SOURCES = [
    {"label": "WHO: Hypertension", "url": "https://www.who.int/news-room/fact-sheets/detail/hypertension"},
    {"label": "WHO: Diabetes", "url": "https://www.who.int/news-room/fact-sheets/detail/diabetes"},
    {"label": "WHO: Cardiovascular diseases", "url": "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)"},
    {"label": "MedlinePlus: Stroke", "url": "https://medlineplus.gov/stroke.html"},
    {"label": "MedlinePlus: Asthma", "url": "https://medlineplus.gov/asthma.html"},
]

REQUEST_TIMEOUT = 20
USER_AGENT = "medical-chatbot-ingest/1.0 (educational use)"
_STRIP_TAGS = ["script", "style", "noscript", "header", "footer", "nav", "form"]


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def fetch_web_sources(sources: list | None = None) -> List[Document]:
    """Fetch and clean each web source. Fail-soft: skip on any error."""
    sources = WEB_SOURCES if sources is None else sources
    today = date.today().isoformat()
    docs: List[Document] = []
    for src in sources:
        try:
            resp = requests.get(
                src["url"], timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}
            )
            resp.raise_for_status()
            text = html_to_text(resp.text)
            if not text.strip():
                logger.warning("Empty content from %s; skipping", src["url"])
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": src["label"],
                        "source_type": "web",
                        "url": src["url"],
                        "retrieved_at": today,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001 - fail-soft by design
            logger.warning("Failed to fetch %s: %s", src["url"], exc)
    return docs


def gather_documents(config: Config | None = None) -> List[Document]:
    """Aggregate documents from every enabled source."""
    config = config or Config.from_env()
    docs: List[Document] = []
    if config.enable_curated:
        docs.extend(CURATED_DOCS)
    if config.enable_dropin:
        file_docs = helper.load_files(config.data_dir)
        for doc in file_docs:
            doc.metadata.setdefault("source_type", "file")
        docs.extend(file_docs)
    if config.enable_web:
        docs.extend(fetch_web_sources())
    return docs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data_sources.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_sources.py tests/test_data_sources.py
git commit -m "feat: add unified multi-source ingestion (curated + files + web)"
```

---

## Task 5: Safety-aware `src/prompt.py`

**Files:**
- Modify: `src/prompt.py`
- Test: `tests/test_prompt.py`

- [ ] **Step 1: Write the failing tests** in `tests/test_prompt.py`:

```python
from src.prompt import system_prompt


def test_prompt_has_context_placeholder():
    assert "{context}" in system_prompt


def test_prompt_has_safety_disclaimer():
    lower = system_prompt.lower()
    assert "not a substitute for professional medical care" in lower
    assert "emergenc" in lower  # emergency / emergencies


def test_prompt_instructs_uncertainty_and_no_diagnosis():
    lower = system_prompt.lower()
    assert "don't know" in lower or "do not know" in lower
    assert "diagnos" in lower
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prompt.py -v`
Expected: FAIL (current prompt lacks disclaimer/diagnosis/citation language)

- [ ] **Step 3: Rewrite `src/prompt.py`:**

```python
system_prompt = (
    "You are a careful medical information assistant for question-answering tasks. "
    "Use ONLY the following pieces of retrieved context to answer the question. "
    "If the answer is not in the context, say you don't know and suggest consulting "
    "a licensed healthcare professional. Keep the answer concise, at most four sentences. "
    "Do not give a personal diagnosis, specific drug dosages, or individualized treatment "
    "decisions. After answering, briefly cite the source(s) you used from each context "
    "item's 'source' metadata. "
    "Always finish with this exact sentence on its own line: "
    "'This is general health information, not a substitute for professional medical care; "
    "for emergencies, contact your local emergency services.'"
    "\n\n"
    "{context}"
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prompt.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/prompt.py tests/test_prompt.py
git commit -m "feat: safety-aware system prompt with disclaimer and citations"
```

---

## Task 6: Modernize `store_index.py`

**Files:**
- Modify: `store_index.py` (full rewrite)
- Test: `tests/test_store_index.py`

- [ ] **Step 1: Write the failing tests** in `tests/test_store_index.py`:

```python
from langchain_core.documents import Document
import store_index
from src.config import Config


def test_summarize_counts_by_source_type():
    docs = [
        Document(page_content="a", metadata={"source_type": "curated"}),
        Document(page_content="b", metadata={"source_type": "curated"}),
        Document(page_content="c", metadata={"source_type": "web"}),
        Document(page_content="d", metadata={}),
    ]
    counts = store_index.summarize(docs)
    assert counts == {"curated": 2, "web": 1, "unknown": 1}


class _FakePC:
    def __init__(self, exists):
        self._exists = exists
        self.deleted = False
        self.created_with = None

    def has_index(self, name):
        return self._exists

    def delete_index(self, name):
        self.deleted = True

    def create_index(self, name, dimension, metric, spec):
        self.created_with = {"name": name, "dimension": dimension, "metric": metric}


def test_recreate_index_deletes_existing_then_creates():
    pc = _FakePC(exists=True)
    cfg = Config.from_env({"INDEX_NAME": "medical-chatbot", "EMBEDDING_DIM": "384"})
    store_index.recreate_index(pc, cfg)
    assert pc.deleted is True
    assert pc.created_with["name"] == "medical-chatbot"
    assert pc.created_with["dimension"] == 384
    assert pc.created_with["metric"] == "cosine"


def test_recreate_index_skips_delete_when_absent():
    pc = _FakePC(exists=False)
    cfg = Config.from_env({})
    store_index.recreate_index(pc, cfg)
    assert pc.deleted is False
    assert pc.created_with is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_store_index.py -v`
Expected: FAIL (current `store_index.py` runs ingestion at import time and lacks `summarize`/`recreate_index`)

- [ ] **Step 3: Rewrite `store_index.py`:**

```python
import os

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src import data_sources, helper
from src.config import Config


def summarize(docs) -> dict:
    counts: dict = {}
    for doc in docs:
        source_type = doc.metadata.get("source_type", "unknown")
        counts[source_type] = counts.get(source_type, 0) + 1
    return counts


def recreate_index(pc: Pinecone, config: Config) -> None:
    if pc.has_index(config.index_name):
        pc.delete_index(config.index_name)
    pc.create_index(
        name=config.index_name,
        dimension=config.embedding_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud=config.pinecone_cloud, region=config.pinecone_region),
    )


def main() -> None:
    load_dotenv()
    config = Config.from_env()
    config.require_keys()
    os.environ["PINECONE_API_KEY"] = config.pinecone_api_key
    os.environ["OPENAI_API_KEY"] = config.openai_api_key

    print("Gathering documents from enabled sources...")
    docs = data_sources.gather_documents(config)
    if not docs:
        raise SystemExit("No documents gathered. Enable at least one source in .env.")
    print("Documents by source:", summarize(docs))

    minimal = helper.filter_to_minimal_docs(docs)
    chunks = helper.text_split(minimal, config)
    print(f"Total chunks to index: {len(chunks)}")

    embeddings = helper.download_embeddings(config)

    pc = Pinecone(api_key=config.pinecone_api_key)
    print(f"(Re)creating index '{config.index_name}' (dim={config.embedding_dim})...")
    recreate_index(pc, config)

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=config.index_name,
    )
    print(f"Index '{config.index_name}' built successfully.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_store_index.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add store_index.py tests/test_store_index.py
git commit -m "feat: recreate index and ingest all sources in store_index"
```

---

## Task 7: Modernize `app.py`

**Files:**
- Modify: `app.py` (full rewrite)
- Test: `tests/test_app_smoke.py`

- [ ] **Step 1: Write the failing test** in `tests/test_app_smoke.py`:

```python
import app


def test_app_exposes_factories_without_side_effects():
    # Importing app must NOT connect to Pinecone/OpenAI.
    assert callable(app.create_app)
    assert callable(app.build_rag_chain)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app_smoke.py -v`
Expected: FAIL — importing current `app.py` connects to Pinecone at import time (errors without keys) and has no `create_app`/`build_rag_chain`.

- [ ] **Step 3: Rewrite `app.py`:**

```python
import os

from dotenv import load_dotenv
from flask import Flask, render_template, request
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.config import Config
from src.helper import download_embeddings
from src.prompt import system_prompt


def build_rag_chain(config: Config):
    embeddings = download_embeddings(config)
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=config.index_name, embedding=embeddings
    )
    retriever = docsearch.as_retriever(
        search_type="similarity", search_kwargs={"k": config.retriever_k}
    )
    chat_model = ChatOpenAI(model=config.llm_model)
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    qa_chain = create_stuff_documents_chain(chat_model, prompt)
    return create_retrieval_chain(retriever, qa_chain)


def create_app(config: Config | None = None) -> Flask:
    load_dotenv()
    config = config or Config.from_env()
    config.require_keys()
    os.environ["PINECONE_API_KEY"] = config.pinecone_api_key
    os.environ["OPENAI_API_KEY"] = config.openai_api_key

    app = Flask(__name__)
    rag_chain = build_rag_chain(config)

    @app.route("/")
    def index():
        return render_template("chat.html")

    @app.route("/get", methods=["GET", "POST"])
    def chat():
        msg = request.form["msg"]
        print("User:", msg)
        response = rag_chain.invoke({"input": msg})
        print("Bot:", response["answer"])
        return str(response["answer"])

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=8080, debug=True)
```

Note: add `from __future__ import annotations` at the top is unnecessary on Python 3.12 for `Config | None`, but harmless. Leave it out to match the file above.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_app_smoke.py -v`
Expected: 1 passed

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: all tests pass (config 4, helper 5, curated 2, data_sources 7, prompt 3, store_index 3, app 1)

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app_smoke.py
git commit -m "feat: lazy Flask app factory with configurable model"
```

---

## Task 8: Recreate `research/trials.ipynb`

**Files:**
- Recreate: `research/trials.ipynb`

This notebook reuses the new functions (DRY) and walks the pipeline end-to-end. Build it with the NotebookEdit tool (or `jupyter nbconvert`/`nbformat`). Create cells in order with exactly this content.

- [ ] **Step 1: Create the notebook with these cells, in order:**

Cell 1 (markdown):
```
# Medical Chatbot — Research Notebook (modernized)
End-to-end walkthrough of the RAG pipeline: gather up-to-date data → chunk → embed (BGE) → build Pinecone index → retrieve → answer with a safety-aware prompt. Run this from the `research/` folder; the first cell wires up imports from the project root.
```

Cell 2 (code):
```python
import sys, os
sys.path.append(os.path.abspath(".."))   # make `src`, store_index, app importable
os.chdir(os.path.abspath(".."))          # run from project root so DATA_DIR='data/' works
from dotenv import load_dotenv
load_dotenv()
from src.config import Config
config = Config.from_env()
config.require_keys()
os.environ["PINECONE_API_KEY"] = config.pinecone_api_key
os.environ["OPENAI_API_KEY"] = config.openai_api_key
print("Index:", config.index_name, "| Model:", config.llm_model, "| Embeddings:", config.embedding_model)
```

Cell 3 (markdown):
```
## 1. Gather up-to-date data from all enabled sources
Curated (built-in, cited) + drop-in files in `data/` + live WHO/CDC/NIH/MedlinePlus.
```

Cell 4 (code):
```python
from src import data_sources, helper
from store_index import summarize
docs = data_sources.gather_documents(config)
print("Documents by source:", summarize(docs))
print("Total documents:", len(docs))
docs[0].metadata
```

Cell 5 (markdown):
```
## 2. Reduce metadata and split into chunks
```

Cell 6 (code):
```python
minimal_docs = helper.filter_to_minimal_docs(docs)
chunks = helper.text_split(minimal_docs, config)
print("Number of chunks:", len(chunks))
chunks[0]
```

Cell 7 (markdown):
```
## 3. Embeddings (BAAI/bge-small-en-v1.5, 384-dim)
First run downloads the model (~130 MB).
```

Cell 8 (code):
```python
embedding = helper.download_embeddings(config)
vector = embedding.embed_query("What are the warning signs of a stroke?")
print("Vector length:", len(vector))
```

Cell 9 (markdown):
```
## 4. (Re)create the Pinecone index and upsert
This deletes and rebuilds the index so it only contains current BGE vectors.
```

Cell 10 (code):
```python
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from store_index import recreate_index
pc = Pinecone(api_key=config.pinecone_api_key)
recreate_index(pc, config)
docsearch = PineconeVectorStore.from_documents(
    documents=chunks, embedding=embedding, index_name=config.index_name
)
print("Index built:", config.index_name)
```

Cell 11 (markdown):
```
## 5. Retrieve relevant chunks
```

Cell 12 (code):
```python
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": config.retriever_k})
retriever.invoke("What are the warning signs of a stroke?")
```

Cell 13 (markdown):
```
## 6. Answer questions with the safety-aware RAG chain
```

Cell 14 (code):
```python
from app import build_rag_chain
rag_chain = build_rag_chain(config)
for q in ["What are the warning signs of a stroke?",
          "What is a normal resting heart rate?",
          "How can I prevent type 2 diabetes?"]:
    print("Q:", q)
    print(rag_chain.invoke({"input": q})["answer"])
    print("-" * 80)
```

- [ ] **Step 2: Sanity-check the notebook JSON is valid**

Run: `python -c "import nbformat; nbformat.read('research/trials.ipynb', as_version=4); print('valid notebook')"`
Expected: `valid notebook`

- [ ] **Step 3: Commit**

```bash
git add research/trials.ipynb
git commit -m "feat: recreate research notebook over the modernized pipeline"
```

---

## Task 9: Update README and final verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read the current README** to match its style: `Read README.md`.

- [ ] **Step 2: Update `README.md`** — ensure it documents:
  - New env vars (point to `.env.example`).
  - Embedding model is now `BAAI/bge-small-en-v1.5` (384-dim); LLM default `gpt-4.1`.
  - Run order: `pip install -r requirements.txt` → `python store_index.py` → `python app.py` (http://localhost:8080).
  - Data sources: curated + drop files into `data/` + live web (toggle with `ENABLE_*`).
  - A short medical disclaimer line (educational use; not a substitute for professional care).
  - How to run tests: `pip install -r requirements-dev.txt && pytest`.

- [ ] **Step 3: Run the full test suite one final time**

Run: `pytest -v`
Expected: all tests pass (25 total).

- [ ] **Step 4: Import smoke check (no deprecation warnings)**

Run: `python -W error::DeprecationWarning -c "from src import config, helper, data_sources, curated_data, prompt; import store_index, app; print('imports clean')"`
Expected: `imports clean` (if a third-party lib emits an unrelated DeprecationWarning, narrow the check to our modules and note it; our own imports must be clean).

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: update README for modernized pipeline and data sources"
```

---

## Manual Verification (user runs — requires real Pinecone/OpenAI keys)

These need live credentials and network, so they are run by the user, not the implementer:

1. Copy `.env.example` to `.env` and fill in `PINECONE_API_KEY` and `OPENAI_API_KEY`.
2. `pip install -r requirements.txt`
3. `python store_index.py` → prints per-source counts and "Index ... built successfully."
4. `python app.py` → open http://localhost:8080, ask "What are the warning signs of a stroke?" → answer includes the FAST guidance, cites a source, and ends with the disclaimer line.
5. Optionally run the notebook top-to-bottom in `research/`.

---

## Self-Review Notes (completed during planning)

- **Spec coverage:** modular layout (Tasks 1–7), three-source ingestion + metadata schema (Tasks 3–4), BGE embeddings (Tasks 1–2), gpt-4.1 (Tasks 1, 7), safety prompt (Task 5), Option A index recreation (Task 6), config knobs (Task 1 + `.env.example` Task 0), notebook recreation (Task 8), README/verification (Task 9). All spec sections mapped.
- **Placeholders:** none — full code in every code step; the only "verify then adjust" step (Task 3 Step 3) is a real WebFetch verification, not a content placeholder.
- **Type consistency:** `Config.from_env`/`require_keys`, `helper.load_files`/`filter_to_minimal_docs`/`text_split`/`download_embeddings`, `data_sources.html_to_text`/`fetch_web_sources`/`gather_documents`, `store_index.summarize`/`recreate_index`/`main`, `app.build_rag_chain`/`create_app` are named identically everywhere they appear.

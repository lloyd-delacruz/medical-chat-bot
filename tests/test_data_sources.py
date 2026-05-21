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


def test_fetch_web_sources_is_failsoft_on_http_error(monkeypatch):
    monkeypatch.setattr(
        data_sources.requests,
        "get",
        lambda url, **kw: _FakeResp("error page", ok=False),
    )
    docs = data_sources.fetch_web_sources([{"label": "L", "url": "https://x"}])
    assert docs == []


def test_fetch_web_sources_skips_empty_content(monkeypatch):
    monkeypatch.setattr(
        data_sources.requests,
        "get",
        lambda url, **kw: _FakeResp("<p>   </p>"),  # collapses to ''
    )
    docs = data_sources.fetch_web_sources([{"label": "L", "url": "https://x"}])
    assert docs == []

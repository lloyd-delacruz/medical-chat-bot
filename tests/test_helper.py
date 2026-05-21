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
    assert all(len(c.page_content) <= cfg.chunk_size + cfg.chunk_overlap for c in chunks)


def test_filter_normalizes_empty_source():
    out = helper.filter_to_minimal_docs(
        [Document(page_content="x", metadata={"source": ""})]
    )
    assert out[0].metadata["source"] == "unknown"


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

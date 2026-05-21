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

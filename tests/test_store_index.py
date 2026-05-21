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
        self.created_with = {
            "name": name,
            "dimension": dimension,
            "metric": metric,
            "spec": spec,
        }


def test_recreate_index_deletes_existing_then_creates():
    pc = _FakePC(exists=True)
    cfg = Config.from_env({"INDEX_NAME": "medical-chatbot", "EMBEDDING_DIM": "384"})
    store_index.recreate_index(pc, cfg)
    assert pc.deleted is True
    assert pc.created_with["name"] == "medical-chatbot"
    assert pc.created_with["dimension"] == 384
    assert pc.created_with["metric"] == "cosine"
    assert pc.created_with["spec"].cloud == "aws"


def test_recreate_index_skips_delete_when_absent():
    pc = _FakePC(exists=False)
    cfg = Config.from_env({})
    store_index.recreate_index(pc, cfg)
    assert pc.deleted is False
    assert pc.created_with["name"] == "medical-chatbot"
    assert pc.created_with["dimension"] == 384

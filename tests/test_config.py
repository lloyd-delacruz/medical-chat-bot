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
    assert cfg.enable_dropin is True
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


def test_require_keys_raises_both_missing():
    cfg = Config.from_env({})
    with pytest.raises(ValueError) as exc:
        cfg.require_keys()
    assert "PINECONE_API_KEY" in str(exc.value)
    assert "OPENAI_API_KEY" in str(exc.value)


def test_invalid_int_env_names_variable():
    with pytest.raises(ValueError) as exc:
        Config.from_env({"EMBEDDING_DIM": "not-a-number"})
    assert "EMBEDDING_DIM" in str(exc.value)

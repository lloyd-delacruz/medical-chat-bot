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

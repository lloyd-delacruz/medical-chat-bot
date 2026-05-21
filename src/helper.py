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
            data_dir,
            glob=glob,
            loader_cls=loader_cls,
            loader_kwargs=loader_kwargs,
            recursive=True,
        )
        documents.extend(loader.load())
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Keep page_content plus only the metadata keys we rely on downstream."""
    minimal_docs: List[Document] = []
    for doc in docs:
        metadata = {k: doc.metadata[k] for k in _METADATA_KEYS if k in doc.metadata}
        metadata["source"] = doc.metadata.get("source") or "unknown"
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

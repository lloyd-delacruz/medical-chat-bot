import os
import time

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src import data_sources, helper
from src.config import Config


def summarize(docs: list) -> dict:
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


def _wait_until_ready(
    pc: Pinecone, index_name: str, attempts: int = 30, delay: float = 2.0
) -> None:
    """Poll until the index reports ready before writing to it.

    Modern Pinecone SDKs usually block on create_index until the index is
    ready, but a fresh serverless index can still need a moment after a
    delete+create. This bounded poll makes the upsert that follows reliable.
    """
    for _ in range(attempts):
        try:
            if pc.describe_index(index_name).status.get("ready"):
                return
        except Exception:  # index may briefly 404 right after creation
            pass
        time.sleep(delay)


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
    if not chunks:
        raise SystemExit(
            "No chunks produced from the gathered documents; aborting to avoid "
            "wiping the existing index."
        )
    print(f"Total chunks to index: {len(chunks)}")

    embeddings = helper.download_embeddings(config)

    pc = Pinecone(api_key=config.pinecone_api_key)
    print(
        f"WARNING: rebuilding index '{config.index_name}' deletes its current "
        "contents. If the upsert below fails, re-run this script to recover."
    )
    print(f"(Re)creating index '{config.index_name}' (dim={config.embedding_dim})...")
    recreate_index(pc, config)
    _wait_until_ready(pc, config.index_name)

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=config.index_name,
    )
    print(f"Index '{config.index_name}' built successfully.")


if __name__ == "__main__":
    main()

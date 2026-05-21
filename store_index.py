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

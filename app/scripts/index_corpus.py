"""Build/refresh the Qdrant collection from data/corpus_chunks.jsonl."""

import json
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

CHUNKS_PATH = Path("data/corpus_chunks.jsonl")
COLLECTION_NAME = "sklearn_docs"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"  # must match app/rag/chain.py
EMBEDDING_DIM = 384


def load_chunks(path: Path) -> list[Document]:
    docs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            docs.append(Document(page_content=row["content"], metadata=row["metadata"]))
    print(f"Loaded {len(docs)} chunks from {path}")
    return docs


def main() -> None:
    qdrant_url = os.environ["QDRANT_URL"]
    client = QdrantClient(url=qdrant_url)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"Collection {COLLECTION_NAME} (re)created at {qdrant_url}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    chunks = load_chunks(CHUNKS_PATH)
    vectorstore.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks")

    for query in [
        "How does Ridge regression work?",
        "Что такое переобучение?",
        "what do you know about?",
    ]:
        hits = vectorstore.similarity_search(query, k=3)
        print(f"\n--- Sanity: {query!r} ---")
        for i, hit in enumerate(hits, 1):
            print(f"  {i}. {hit.metadata.get('source', '?')[:80]}")
            print(f"     {hit.page_content[:100].strip()!r}")


if __name__ == "__main__":
    main()

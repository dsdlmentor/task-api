"""RAG chain assembly.

Embedding model: intfloat/multilingual-e5-small (118 MB, 384-dim).
Multilingual coverage means the system answers questions in English or
Russian over the English scikit-learn documentation corpus.

If you don't need Russian and want a slightly faster purely-English
embedder, swap EMBEDDING_MODEL to "sentence-transformers/all-MiniLM-L6-v2"
(same 384-dim, no re-indexing needed).
"""

import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.llm import get_llm

COLLECTION_NAME = "sklearn_docs"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"  # 384-dim, multilingual
TOP_K = 4

SYSTEM_PROMPT = """You are a study assistant for the Classic ML cycle of an ML/DS course.
The context below is taken from the official scikit-learn documentation and from
the service's internal "about" pages.

Rules:
- Use ONLY the provided context. If the answer is not in the context, say so honestly.
- Cite sources using [1], [2], ... — the numbers correspond to the source list in the context block.
- Reply in the SAME LANGUAGE as the user's question (English question -> English answer,
  Russian question -> Russian answer). Translate the relevant facts; keep code identifiers
  (function names, parameter names, classes) in English.
- If the user asks meta-questions ("what do you know about?", "what topics do you cover?",
  "что ты умеешь?") — answer based on the internal "About this RAG assistant" context.

Context:
{context}

Question: {question}

Answer (with citations):"""


def get_vectorstore() -> QdrantVectorStore:
    """Connect to the running Qdrant and the indexed collection."""
    client = QdrantClient(url=os.environ["QDRANT_URL"])
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


def format_docs_with_sources(docs: list[Document]) -> str:
    """Render top-k chunks as a numbered context block for the LLM prompt."""
    lines = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        lines.append(f"[{i}] Source: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(lines)


def build_rag_chain():
    """Assemble the LCEL pipeline: retriever -> prompt -> LLM -> parser."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs_with_sources),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

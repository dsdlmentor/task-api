from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI

from app.rag.chain import build_rag_chain
from app.schemas.chat import ChatRequest, ChatResponse, Source

_chain = None
_retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chain, _retriever
    _chain, _retriever = build_rag_chain()
    print("RAG chain ready")
    yield
    _chain = None
    _retriever = None


app = FastAPI(title="RAG service", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    docs = _retriever.invoke(payload.question)
    answer = _chain.invoke(payload.question)
    sources = [
        Source(
            url=doc.metadata.get("source", "unknown"),
            snippet=doc.page_content[:200].strip(),
        )
        for doc in docs
    ]
    return ChatResponse(answer=answer, sources=sources)


def respond(message: str, history: list[tuple[str, str]]) -> str:
    docs = _retriever.invoke(message)
    answer = _chain.invoke(message)
    links = "\n\n**Источники:**\n" + "\n".join(
        f"- {doc.metadata.get('source', 'unknown')}" for doc in docs
    )
    return answer + links


demo = gr.ChatInterface(
    fn=respond,
    title="scikit-learn docs assistant",
    description="Задайте вопрос по документации scikit-learn — RAG найдёт релевантные разделы и ответит с цитатами.",
    examples=[
        "How does Ridge regression work?",
        "What is the difference between Lasso and Ridge?",
        "How do I tune max_depth in a decision tree?",
    ],
)

app = gr.mount_gradio_app(app, demo, path="/")

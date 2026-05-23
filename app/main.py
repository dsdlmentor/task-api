from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI, HTTPException

from app.rag.chain import build_rag_chain
from app.schemas.chat import ChatRequest, ChatResponse, Source

_chain = None
_retriever = None

# LaTeX delimiters for Gradio's Chatbot — LLM answers about Ridge, Lasso,
# gradient boosting and metrics use $\alpha$ / \[..\] / $$..$$ regularly,
# without this they show as raw `$\ell_1$`-strings.
LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
]


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
    try:
        answer = _chain.invoke(payload.question)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"LLM provider temporarily unavailable. "
                f"Try again in 30-60 seconds. Raw: {type(exc).__name__}"
            ),
        ) from exc
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
    try:
        answer = _chain.invoke(message)
    except Exception as exc:
        return (
            f"⚠️ LLM-провайдер сейчас недоступен ({type(exc).__name__}). "
            f"На бесплатном тарифе OpenRouter это нормально — upstream "
            f"ушёл в rate-limit. Попробуй через 30-60 секунд."
        )
    links = "\n\n**Источники:**\n" + "\n".join(
        f"- {doc.metadata.get('source', 'unknown')}" for doc in docs
    )
    return answer + links


demo = gr.ChatInterface(
    fn=respond,
    chatbot=gr.Chatbot(
        latex_delimiters=LATEX_DELIMITERS,
        height=500,
    ),
    title="scikit-learn docs assistant",
    description="Задайте вопрос по документации scikit-learn — RAG найдёт релевантные разделы и ответит с цитатами.",
    examples=[
        "How does Ridge regression work?",
        "What is the difference between Lasso and Ridge?",
        "Что ты умеешь?",
    ],
)

app = gr.mount_gradio_app(app, demo, path="/")

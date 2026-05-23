"""FastAPI + Gradio entry point.

Layout: Gradio Blocks with two columns under a single Row.
- Left (scale=3): full-height chat with LaTeX rendering
- Right (scale=1): timings panel + sources list, updated on every turn

Timings split the request into two measurable phases:
- retrieval (query embed + Qdrant top-k)
- LLM (OpenRouter call: prompt assembly is sub-millisecond, lumped in here)

Students see WHERE the latency lives and can reason about caching,
provider swap, or model size without guesswork.
"""

import time
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI, HTTPException

from app.rag.chain import build_rag_chain
from app.schemas.chat import ChatRequest, ChatResponse, Source

_chain = None
_retriever = None

# LaTeX delimiters for Gradio's Chatbot. LLM answers about Ridge, Lasso,
# precision/recall use $$..$$ / \[..\] / $..$ regularly. Without this
# block they render as raw `$\ell_1$`-strings.
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


def _format_timings(retrieval_ms: float, llm_ms: float | None, llm_error: str | None) -> str:
    lines = [
        "### ⏱ Тайминги последнего запроса",
        "",
        f"- 🔍 **Retrieval (embed + Qdrant):** {retrieval_ms:.0f} ms",
    ]
    if llm_ms is not None:
        lines.append(f"- 🤖 **LLM call:** {llm_ms:.0f} ms")
        lines.append(f"- 📊 **Total:** {retrieval_ms + llm_ms:.0f} ms")
    else:
        lines.append(f"- 🤖 **LLM call:** ❌ {llm_error}")
    return "\n".join(lines)


def _format_sources(docs: list) -> str:
    if not docs:
        return "### 📚 Источники\n\n_Ничего не найдено_"
    lines = ["### 📚 Источники", ""]
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        snippet = doc.page_content[:140].strip().replace("\n", " ")
        lines.append(f"**[{i}]** `{source}`")
        lines.append(f"> {snippet}…")
        lines.append("")
    return "\n".join(lines)


def respond(message: str, history: list) -> tuple[list, str, str, str]:
    """Gradio handler. Returns updated history, cleared textbox, timings, sources."""
    if not message or not message.strip():
        return history, "", "### ⏱ Тайминги\n\n_Пустой запрос_", "### 📚 Источники\n\n_—_"

    history = history + [{"role": "user", "content": message}]

    t0 = time.perf_counter()
    docs = _retriever.invoke(message)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    try:
        answer = _chain.invoke(message)
        llm_ms = (time.perf_counter() - t1) * 1000
        history.append({"role": "assistant", "content": answer})
        return history, "", _format_timings(retrieval_ms, llm_ms, None), _format_sources(docs)
    except Exception as exc:
        msg = (
            f"⚠️ LLM-провайдер сейчас недоступен ({type(exc).__name__}). "
            f"На бесплатном тарифе OpenRouter это бывает — upstream-провайдер "
            f"ушёл в rate-limit. Попробуй через 30-60 секунд."
        )
        history.append({"role": "assistant", "content": msg})
        return history, "", _format_timings(retrieval_ms, None, type(exc).__name__), _format_sources(docs)


CSS = """
.gradio-container { max-width: 100% !important; padding: 1rem !important; }
#chatbot { height: calc(100vh - 220px) !important; min-height: 500px !important; }
#side-panel { height: calc(100vh - 220px) !important; overflow-y: auto !important;
              padding: 1rem !important; border-left: 1px solid #ddd !important; }
"""

with gr.Blocks(
    title="scikit-learn docs RAG",
    css=CSS,
    fill_height=True,
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# 📖 scikit-learn docs RAG assistant\n"
        "_Спрашивай про Linear models, Decision trees, Metrics — на русском или английском._"
    )
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                type="messages",
                latex_delimiters=LATEX_DELIMITERS,
                show_copy_button=True,
                avatar_images=(None, None),
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Например: «Покажи формулу Ridge» или «Чем precision отличается от recall»",
                    scale=8,
                    container=False,
                    autofocus=True,
                )
                send = gr.Button("Отправить", scale=1, variant="primary")
            gr.Examples(
                examples=[
                    "How does Ridge regression work?",
                    "Что ты умеешь?",
                    "Объясни разницу между precision и recall с формулами",
                    "When does a decision tree overfit?",
                ],
                inputs=msg,
            )
        with gr.Column(scale=1, elem_id="side-panel"):
            timings_md = gr.Markdown(
                "### ⏱ Тайминги последнего запроса\n\n_Задайте вопрос, чтобы увидеть тайминги._"
            )
            sources_md = gr.Markdown("### 📚 Источники\n\n_—_")

    msg.submit(respond, [msg, chatbot], [chatbot, msg, timings_md, sources_md])
    send.click(respond, [msg, chatbot], [chatbot, msg, timings_md, sources_md])


app = gr.mount_gradio_app(app, demo, path="/")

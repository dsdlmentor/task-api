"""LLM client factory.

Wraps any OpenAI-compatible provider (OpenRouter, Groq, Mistral,
Cerebras, DeepSeek, ...). All parameters come from app.config so
that swapping providers is a one-line edit in config.py — no env
juggling, no secrets reshuffle.
"""

from langchain_openai import ChatOpenAI

from app.config import settings


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )

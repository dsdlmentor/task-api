"""Central configuration for the RAG service.

All tunable parameters live here as a single source of truth. Values
can be overridden via environment variables (env vars take precedence
over the defaults declared below), but in normal operation the defaults
are what runs in production.

Only true secrets — the LLM API key — must come from the environment.
Everything else (model name, retriever top_k, embedding model, qdrant
URL) has a sensible default and would not be a "secret" anyway.

Pattern: pydantic-settings. Pydantic validates types at import time,
so a typo in the env file or a missing required value fails loudly at
container start rather than silently breaking a request later.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Secrets (env-only) ---
    # No default — container fails fast at startup if missing.
    llm_api_key: str

    # --- LLM provider ---
    # OpenAI-compatible endpoint. Switching providers is a one-line change.
    llm_base_url: str = "https://openrouter.ai/api/v1"

    # Default pinned model: DeepSeek v4 Flash via OpenRouter free tier.
    # Pinned (not auto-routed) because it reliably follows the "reply in
    # the same language as the user" system-prompt instruction. The
    # openrouter/free auto-router can dispatch to ANY underlying provider,
    # including small English-centric models that ignore the language rule.
    # See content.md "Стабильность языка ответов" for the production
    # pattern (multi-provider fallback chain + per-language eval).
    #
    # Verified-available alternatives on OpenRouter free tier:
    #   - meta-llama/llama-3.3-70b-instruct:free  (CURRENT — reliable, holds Russian)
    #   - qwen/qwen3-next-80b-a3b-instruct:free   (best multilingual, but quota-shared with llama)
    #   - deepseek/deepseek-v4-flash:free         (fast, but Crucible upstream often out of credits)
    #
    # Why not deepseek-v4-flash:free as default: its underlying provider
    # (Crucible) regularly returns 402 "out of credits" on the free tier
    # even when the OpenRouter daily quota is fresh. llama-3.3-70b is
    # hosted by multiple providers — more resilient.
    llm_model: str = "meta-llama/llama-3.3-70b-instruct:free"

    # 0 = deterministic. Bump to 0.2-0.3 only if you want variety in answers.
    llm_temperature: float = 0.0

    # --- Vector store ---
    # Inside Docker network the qdrant service is reachable by name.
    # Locally (running scripts from host) override to http://localhost:6333.
    qdrant_url: str = "http://qdrant:6333"
    collection_name: str = "sklearn_docs"

    # Number of chunks retrieved for each query. 4 is a good baseline:
    # enough context for the LLM to cite, not so much it overflows the
    # context window of small free-tier models.
    top_k: int = 4

    # --- Embeddings ---
    # multilingual-e5-small: 118 MB, 384-dim, covers 100 languages.
    # MUST match the model used at index time (app/scripts/index_corpus.py)
    # — a mismatch silently degrades retrieval quality to ~random.
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384

    # e5 models REQUIRE L2 normalization for cosine distance to behave correctly.
    normalize_embeddings: bool = True


settings = Settings()

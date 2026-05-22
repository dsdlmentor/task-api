import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return a configured ChatOpenAI client.

    Reads LLM_BASE_URL, LLM_API_KEY, LLM_MODEL from environment.
    Works with any OpenAI-compatible provider: OpenRouter, Groq,
    Mistral, Cerebras, DeepSeek.
    """
    return ChatOpenAI(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        model=os.environ["LLM_MODEL"],
        temperature=0,
    )

from __future__ import annotations

from typing import Any

from agent.config import AgentConfig


def create_chat_model(config: AgentConfig) -> Any:
    if config.langchain_provider == "openai":
        if not config.langchain_api_key:
            raise ValueError("LANGCHAIN_PROVIDER=openai requires OPENAI_API_KEY or LANGCHAIN_API_KEY.")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install langchain-openai to use LANGCHAIN_PROVIDER=openai.") from exc

        return ChatOpenAI(
            model=config.langchain_model,
            temperature=config.langchain_temperature,
            api_key=config.langchain_api_key,
        )

    raise ValueError(f"Unsupported LANGCHAIN_PROVIDER: {config.langchain_provider}")

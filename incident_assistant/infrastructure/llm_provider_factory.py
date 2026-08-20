"""Single composition point for selecting an LLM provider."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from incident_assistant.config import LLMProviderName, Settings
from incident_assistant.domain.ports import AgentLLMProvider
from incident_assistant.infrastructure.gemini_agent import GeminiAgentProvider
from incident_assistant.infrastructure.groq_agent import GroqAgentProvider
from incident_assistant.infrastructure.openai_agent import OpenAIAgentProvider


def create_llm_provider(
    settings: Settings,
    *,
    clients: Mapping[str, Any] | None = None,
) -> AgentLLMProvider | None:
    """Create only the selected provider, or return None for the guaranteed rules path."""

    if not settings.ai_provider_configured:
        return None
    client = (clients or {}).get(settings.llm_provider.value)
    common = {
        "api_key": settings.selected_api_key or "",
        "model": settings.selected_model or "",
        "timeout_seconds": settings.llm_timeout_seconds,
        "client": client,
    }
    if settings.llm_provider is LLMProviderName.GEMINI:
        return GeminiAgentProvider(**common)
    if settings.llm_provider is LLMProviderName.GROQ:
        return GroqAgentProvider(**common)
    if settings.llm_provider is LLMProviderName.OPENAI:
        return OpenAIAgentProvider(**common)
    if settings.llm_provider is LLMProviderName.RULES:
        return None
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

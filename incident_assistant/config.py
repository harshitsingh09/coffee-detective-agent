"""Environment-based configuration with project-local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LLMProviderName(StrEnum):
    """Supported diagnosis backends, including the no-network rules mode."""

    GEMINI = "gemini"
    GROQ = "groq"
    OPENAI = "openai"
    RULES = "rules"


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _environment_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false.")


def _bounded_environment_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw_value = os.getenv(name)
    value = default if raw_value is None else int(raw_value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _bounded_environment_float(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw_value = os.getenv(name)
    value = default if raw_value is None else float(raw_value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    database_path: Path
    log_path: Path
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.4-mini"
    llm_provider: LLMProviderName = LLMProviderName.GEMINI
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-20b"
    enable_ai_agent: bool = False
    max_agent_steps: int = 5
    max_log_results: int = 20
    similar_incident_top_k: int = 3
    similarity_threshold: float = 0.35
    llm_timeout_seconds: float = 30.0
    embedding_model: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    embedding_index_path: Path = PROJECT_ROOT / "data" / "incident_embeddings.json"
    embedding_cache_path: Path = PROJECT_ROOT / ".cache" / "sentence_transformers"

    @property
    def selected_api_key(self) -> str | None:
        return {
            LLMProviderName.GEMINI: self.gemini_api_key,
            LLMProviderName.GROQ: self.groq_api_key,
            LLMProviderName.OPENAI: self.openai_api_key,
            LLMProviderName.RULES: None,
        }[self.llm_provider]

    @property
    def selected_model(self) -> str | None:
        return {
            LLMProviderName.GEMINI: self.gemini_model,
            LLMProviderName.GROQ: self.groq_model,
            LLMProviderName.OPENAI: self.openai_model,
            LLMProviderName.RULES: None,
        }[self.llm_provider]

    @property
    def ai_provider_configured(self) -> bool:
        return bool(
            self.enable_ai_agent
            and self.llm_provider is not LLMProviderName.RULES
            and self.selected_api_key
        )

    @classmethod
    def from_environment(cls) -> Settings:
        try:
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env")
        except ImportError:
            pass

        raw_provider = os.getenv("LLM_PROVIDER", LLMProviderName.GEMINI.value).strip().casefold()
        try:
            llm_provider = LLMProviderName(raw_provider)
        except ValueError as exc:
            choices = ", ".join(provider.value for provider in LLMProviderName)
            raise ValueError(f"LLM_PROVIDER must be one of: {choices}.") from exc

        return cls(
            database_path=_project_path(os.getenv("INCIDENT_DATABASE_PATH", "data/support.db")),
            log_path=_project_path(os.getenv("INCIDENT_LOG_PATH", "data/app_logs.txt")),
            llm_provider=llm_provider,
            gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip(),
            groq_api_key=os.getenv("GROQ_API_KEY") or None,
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip(),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip(),
            enable_ai_agent=_environment_flag("ENABLE_AI_AGENT", False),
            max_agent_steps=_bounded_environment_int("MAX_AGENT_STEPS", 5, 1, 8),
            max_log_results=_bounded_environment_int("MAX_LOG_RESULTS", 20, 1, 100),
            similar_incident_top_k=_bounded_environment_int(
                "SIMILAR_INCIDENT_TOP_K",
                _bounded_environment_int("MAX_SIMILAR_INCIDENTS", 3, 1, 10),
                1,
                10,
            ),
            similarity_threshold=_bounded_environment_float("SIMILARITY_THRESHOLD", 0.35, 0.0, 1.0),
            llm_timeout_seconds=_bounded_environment_float(
                "LLM_TIMEOUT_SECONDS",
                _bounded_environment_float("OPENAI_TIMEOUT_SECONDS", 30.0, 1.0, 120.0),
                1.0,
                120.0,
            ),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            ).strip(),
            embedding_index_path=_project_path(
                os.getenv(
                    "INCIDENT_EMBEDDING_INDEX_PATH",
                    "data/incident_embeddings.json",
                )
            ),
            embedding_cache_path=_project_path(
                os.getenv("EMBEDDING_CACHE_PATH", ".cache/sentence_transformers")
            ),
        )

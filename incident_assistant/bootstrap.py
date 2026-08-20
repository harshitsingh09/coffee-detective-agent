"""Composition root: the only place concrete adapters are wired together."""

import sqlite3
from pathlib import Path

from incident_assistant.application.agent_service import AgentInvestigationService
from incident_assistant.application.service import InvestigationService
from incident_assistant.config import Settings
from incident_assistant.infrastructure.diagnosticians import RuleBasedDiagnostician
from incident_assistant.infrastructure.extraction import RegexMachineIdExtractor
from incident_assistant.infrastructure.file_log_repository import FileLogRepository
from incident_assistant.infrastructure.incident_retriever import (
    LexicalHistoricalIncidentRetriever,
    PersistentSemanticIncidentRetriever,
    ResilientHistoricalIncidentRetriever,
    SentenceTransformerEmbeddingProvider,
)
from incident_assistant.infrastructure.llm_provider_factory import create_llm_provider
from incident_assistant.infrastructure.sqlite_repository import SqliteRepository
from incident_assistant.tools.incident_tools import build_investigation_tool_registry


def ensure_demo_data(settings: Settings | None = None) -> bool:
    """Create deterministic demo data when a fresh deployment has no data files."""

    active_settings = settings or Settings.from_environment()
    required_paths = (active_settings.database_path, active_settings.log_path)
    if all(path.exists() for path in required_paths):
        try:
            with sqlite3.connect(active_settings.database_path) as connection:
                table = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='brew_cycles'"
                ).fetchone()
            if table is not None:
                return False
        except sqlite3.Error:
            pass

    parents = {path.parent.resolve() for path in required_paths}
    if len(parents) != 1:
        missing = ", ".join(str(path) for path in required_paths if not path.exists())
        raise FileNotFoundError(
            f"Demo data cannot be initialized across different directories. Missing: {missing}"
        )

    from seed_data import seed

    output_dir = Path(next(iter(parents)))
    seed(output_dir)
    return True


def build_investigation_service(settings: Settings | None = None) -> InvestigationService:
    """Construct the application from replaceable adapters."""

    active_settings = settings or Settings.from_environment()
    database = SqliteRepository(active_settings.database_path)
    return InvestigationService(
        extractor=RegexMachineIdExtractor(),
        operations=database,
        logs=FileLogRepository(active_settings.log_path),
        incidents=database,
        diagnostician=RuleBasedDiagnostician(),
    )


def build_agent_investigation_service(
    settings: Settings | None = None,
) -> AgentInvestigationService:
    """Construct the bounded agent with a deterministic no-key fallback."""

    active_settings = settings or Settings.from_environment()
    database = SqliteRepository(active_settings.database_path)
    logs = FileLogRepository(active_settings.log_path)
    semantic = PersistentSemanticIncidentRetriever(
        database,
        SentenceTransformerEmbeddingProvider(
            active_settings.embedding_model,
            active_settings.embedding_cache_path,
        ),
        active_settings.embedding_index_path,
        active_settings.similarity_threshold,
    )
    retriever = ResilientHistoricalIncidentRetriever(
        primary=semantic,
        fallback=LexicalHistoricalIncidentRetriever(database),
    )
    tools = build_investigation_tool_registry(
        database,
        logs,
        retriever,
        max_log_results=active_settings.max_log_results,
        max_similar_incidents=active_settings.similar_incident_top_k,
    )
    agent_model = create_llm_provider(active_settings)
    return AgentInvestigationService(
        extractor=RegexMachineIdExtractor(),
        tools=tools,
        fallback=build_investigation_service(active_settings),
        agent_model=agent_model,
        max_steps=active_settings.max_agent_steps,
    )

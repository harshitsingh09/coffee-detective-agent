"""Ports implemented by infrastructure adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from incident_assistant.domain.agent_models import (
    AgentStep,
    HistoricalIncidentDocument,
    SimilarIncident,
    ToolDefinition,
    ToolExecutionResult,
)
from incident_assistant.domain.models import Diagnosis, Evidence


class MachineIdExtractor(Protocol):
    def extract(self, incident_description: str) -> str | None:
        """Extract and normalize a machine identifier."""


class OperationsRepository(Protocol):
    def check_brews(self, machine_id: str) -> Sequence[Evidence]:
        """Return recent brew-quality evidence for a machine."""

    def check_machine_status(self, machine_id: str) -> Sequence[Evidence]:
        """Return supply, sensor, temperature, and cleaning evidence."""


class LogRepository(Protocol):
    def search(self, machine_id: str, limit: int = 50) -> Sequence[Evidence]:
        """Return relevant application-log evidence."""


class IncidentRepository(Protocol):
    def find_similar(
        self,
        incident_description: str,
        machine_id: str,
        limit: int = 3,
    ) -> Sequence[Evidence]:
        """Return evidence from similar historical incidents."""


class Diagnostician(Protocol):
    def diagnose(
        self,
        incident_description: str,
        machine_id: str,
        evidence: Sequence[Evidence],
    ) -> Diagnosis:
        """Interpret evidence and recommend next actions."""


class HistoricalIncidentRetriever(Protocol):
    def search(self, description: str, top_k: int = 3) -> Sequence[SimilarIncident]:
        """Return semantically similar historical incidents."""


class HistoricalIncidentDocumentRepository(Protocol):
    def list_incident_documents(self) -> Sequence[HistoricalIncidentDocument]:
        """Return the incident corpus used to build a retrieval index."""


class EmbeddingProvider(Protocol):
    @property
    def model_id(self) -> str:
        """Stable identifier stored with persisted vectors."""

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Embed historical-incident passages."""

    def embed_query(self, text: str) -> Sequence[float]:
        """Embed one search query."""


class AgentSession(Protocol):
    def next_step(
        self,
        tool_results: Sequence[ToolExecutionResult] = (),
    ) -> AgentStep:
        """Choose safe tool calls or return a structured final diagnosis."""


class AgentLLMProvider(Protocol):
    @property
    def provider_name(self) -> str:
        """Stable provider identifier used for observability."""

    def start_session(
        self,
        incident_id: str,
        incident_description: str,
        machine_id: str,
        tools: Sequence[ToolDefinition],
    ) -> AgentSession:
        """Start one provider-managed agent conversation."""


# Backwards-compatible name for integrations built before provider selection existed.
AgentModel = AgentLLMProvider


class InvestigationOperations(Protocol):
    """Structured, read-only operational queries available to safe tools."""

    def get_machine_status(self, machine_id: str) -> Mapping[str, Any]: ...

    def get_recent_brews(self, machine_id: str, limit: int = 20) -> Mapping[str, Any]: ...

    def get_supply_levels(self, machine_id: str) -> Mapping[str, Any]: ...

    def get_sensor_alerts(self, machine_id: str, limit: int = 10) -> Mapping[str, Any]: ...

    def get_temperature_history(self, machine_id: str, limit: int = 10) -> Mapping[str, Any]: ...

    def get_cleaning_status(self, machine_id: str) -> Mapping[str, Any]: ...

    def get_machine_health(self, machine_id: str) -> Mapping[str, Any]: ...


class StructuredLogSearcher(Protocol):
    def search_structured(
        self,
        machine_id: str,
        keywords: Sequence[str] = (),
        limit: int = 20,
    ) -> Mapping[str, Any]:
        """Return bounded JSON-compatible log-search results."""

"""Structured, provider-independent models for bounded agent investigations."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

MACHINE_ID_PATTERN = r"^CM-\d{4}$"
TOOL_NAME_PATTERN = r"^[a-z][a-z0-9_]{2,63}$"


class StrictAgentModel(BaseModel):
    """Base class that rejects unexpected model or tool output fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DiagnosisStatus(StrEnum):
    DIAGNOSED = "diagnosed"
    NORMAL = "normal"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FALLBACK_DIAGNOSIS = "fallback_diagnosis"


class TraceEventType(StrEnum):
    MACHINE_EXTRACTED = "machine_extracted"
    TOOL_SELECTED = "tool_selected"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    FALLBACK_STARTED = "fallback_started"
    INVESTIGATION_COMPLETED = "investigation_completed"


class ToolExecutionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"


class ExecutionMode(StrEnum):
    AI_AGENT = "ai_agent"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"
    SAFETY_STOP = "safety_stop"


class InvestigationToolName(StrEnum):
    GET_MACHINE_STATUS = "get_machine_status"
    GET_RECENT_BREWS = "get_recent_brews"
    GET_SUPPLY_LEVELS = "get_supply_levels"
    GET_SENSOR_ALERTS = "get_sensor_alerts"
    GET_TEMPERATURE_HISTORY = "get_temperature_history"
    GET_CLEANING_STATUS = "get_cleaning_status"
    SEARCH_APPLICATION_LOGS = "search_application_logs"
    SEARCH_SIMILAR_INCIDENTS = "search_similar_incidents"
    GET_MACHINE_HEALTH = "get_machine_health"


class SimilarIncident(StrictAgentModel):
    incident_id: str = Field(min_length=1, max_length=64)
    similarity_score: float = Field(ge=0.0, le=1.0)
    description: str = Field(min_length=1, max_length=4_000)
    root_cause: str = Field(min_length=1, max_length=1_000)
    resolution: str = Field(min_length=1, max_length=4_000)
    retrieval_method: str = Field(default="semantic", min_length=1, max_length=64)


class HistoricalIncidentDocument(StrictAgentModel):
    incident_id: str = Field(min_length=1, max_length=64)
    machine_id: str = Field(pattern=MACHINE_ID_PATTERN)
    title: str = Field(min_length=1, max_length=1_000)
    description: str = Field(min_length=1, max_length=4_000)
    root_cause: str = Field(min_length=1, max_length=1_000)
    resolution: str = Field(min_length=1, max_length=4_000)
    error_codes: tuple[str, ...] = Field(default_factory=tuple, max_length=20)
    service: str = Field(min_length=1, max_length=200)
    severity: str = Field(min_length=1, max_length=32)

    def embedding_text(self) -> str:
        error_codes = ", ".join(self.error_codes) if self.error_codes else "none"
        return (
            f"Title: {self.title}\n"
            f"Description: {self.description}\n"
            f"Root cause: {self.root_cause}\n"
            f"Resolution: {self.resolution}\n"
            f"Error codes: {error_codes}\n"
            f"Service: {self.service}\n"
            f"Severity: {self.severity}"
        )


class ToolDefinition(StrictAgentModel):
    """A safe tool description exposed to an agent model."""

    name: str = Field(pattern=TOOL_NAME_PATTERN)
    description: str = Field(min_length=1, max_length=500)
    input_schema: dict[str, Any]


class RequestedToolCall(StrictAgentModel):
    call_id: str = Field(min_length=1, max_length=128)
    name: str = Field(pattern=TOOL_NAME_PATTERN)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionResult(StrictAgentModel):
    call_id: str = Field(min_length=1, max_length=128)
    name: str = Field(pattern=TOOL_NAME_PATTERN)
    status: ToolExecutionStatus
    arguments: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] | None = None
    error: str | None = Field(default=None, max_length=1_000)
    duration_ms: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_outcome(self) -> ToolExecutionResult:
        if self.status is ToolExecutionStatus.SUCCEEDED and self.data is None:
            raise ValueError("A successful tool result must include data.")
        if self.status is not ToolExecutionStatus.SUCCEEDED and not self.error:
            raise ValueError("A rejected or failed tool result must include an error.")
        return self


class InvestigationTraceEvent(StrictAgentModel):
    """An observable action or outcome; never hidden model reasoning."""

    sequence: int = Field(ge=1)
    event_type: TraceEventType
    message: str = Field(min_length=1, max_length=1_000)
    tool_name: str | None = Field(default=None, pattern=TOOL_NAME_PATTERN)
    result_summary: str | None = Field(default=None, max_length=1_000)
    duration_ms: float | None = Field(default=None, ge=0.0)


class IncidentDiagnosis(StrictAgentModel):
    """Structured final diagnosis shared by AI and fallback paths."""

    incident_id: str = Field(min_length=1, max_length=64)
    machine_id: str = Field(pattern=MACHINE_ID_PATTERN)
    status: DiagnosisStatus
    root_cause: str | None = Field(default=None, max_length=1_000)
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1, max_length=4_000)
    supporting_evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=20)
    recommended_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=20)
    tools_used: tuple[str, ...] = Field(default_factory=tuple, max_length=20)
    similar_incidents: tuple[SimilarIncident, ...] = Field(default_factory=tuple, max_length=10)
    warnings: tuple[str, ...] = Field(default_factory=tuple, max_length=20)

    @model_validator(mode="after")
    def require_grounded_cause(self) -> IncidentDiagnosis:
        if (
            self.status
            in {
                DiagnosisStatus.DIAGNOSED,
                DiagnosisStatus.FALLBACK_DIAGNOSIS,
            }
            and not self.root_cause
        ):
            raise ValueError("A diagnosed result must include a root cause.")
        return self


class AgentStep(StrictAgentModel):
    """One observable agent decision: use tools or finish with a diagnosis."""

    tool_calls: tuple[RequestedToolCall, ...] = Field(default_factory=tuple, max_length=8)
    diagnosis: IncidentDiagnosis | None = None

    @model_validator(mode="after")
    def require_one_outcome(self) -> AgentStep:
        has_calls = bool(self.tool_calls)
        has_diagnosis = self.diagnosis is not None
        if has_calls == has_diagnosis:
            raise ValueError("An agent step must contain tool calls or one diagnosis, not both.")
        return self


class AgentInvestigationReport(StrictAgentModel):
    incident_description: str = Field(min_length=1, max_length=4_000)
    machine_id: str = Field(pattern=MACHINE_ID_PATTERN)
    diagnosis: IncidentDiagnosis
    execution_mode: ExecutionMode
    llm_provider: str | None = Field(default=None, min_length=1, max_length=64)
    trace: tuple[InvestigationTraceEvent, ...]
    tool_results: tuple[ToolExecutionResult, ...]

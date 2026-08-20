"""Structured inputs and outputs for reproducible agent evaluation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from incident_assistant.domain.agent_models import DiagnosisStatus


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvaluationCase(EvaluationModel):
    case_id: str = Field(min_length=1, max_length=64)
    scenario: str = Field(min_length=1, max_length=64)
    incident_description: str = Field(min_length=1, max_length=4_000)
    expected_root_cause: str | None
    expected_tools: tuple[str, ...]
    expected_machine: str | None
    expected_status: DiagnosisStatus
    unavailable_sources: tuple[str, ...] = ()


class EvaluationResult(EvaluationModel):
    case_id: str
    scenario: str
    incident_description: str
    expected_root_cause: str | None
    actual_root_cause: str | None
    expected_status: DiagnosisStatus
    actual_status: DiagnosisStatus
    expected_tools: tuple[str, ...]
    selected_tools: tuple[str, ...]
    expected_machine: str | None
    actual_machine: str | None
    fallback_used: bool
    latency_ms: float = Field(ge=0.0)
    root_cause_correct: bool
    status_correct: bool
    machine_correct: bool
    error: str | None = None


class EvaluationMetrics(EvaluationModel):
    total_cases: int = Field(ge=0)
    root_cause_accuracy: float = Field(ge=0.0, le=1.0)
    status_accuracy: float = Field(ge=0.0, le=1.0)
    machine_accuracy: float = Field(ge=0.0, le=1.0)
    tool_selection_precision: float = Field(ge=0.0, le=1.0)
    tool_selection_recall: float = Field(ge=0.0, le=1.0)
    average_tool_calls: float = Field(ge=0.0)
    unnecessary_tool_calls: int = Field(ge=0)
    fallback_success_rate: float = Field(ge=0.0, le=1.0)
    average_investigation_latency_ms: float = Field(ge=0.0)


class EvaluationReport(EvaluationModel):
    generated_at: str
    mode: str
    metrics: EvaluationMetrics
    results: tuple[EvaluationResult, ...]

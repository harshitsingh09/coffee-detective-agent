"""Validated arguments for every planned investigation tool."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from incident_assistant.domain.agent_models import (
    MACHINE_ID_PATTERN,
    InvestigationToolName,
)


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class MachineArguments(ToolArguments):
    machine_id: str = Field(pattern=MACHINE_ID_PATTERN)

    @field_validator("machine_id", mode="before")
    @classmethod
    def normalize_machine_id(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value


class ApplicationLogSearchArguments(MachineArguments):
    keywords: tuple[str, ...] = Field(max_length=8)

    @field_validator("keywords")
    @classmethod
    def normalize_keywords(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for value in values:
            keyword = value.strip().casefold()
            if not keyword:
                continue
            if len(keyword) > 64:
                raise ValueError("Log-search keywords cannot exceed 64 characters.")
            if keyword not in normalized:
                normalized.append(keyword)
        return tuple(normalized)


class SimilarIncidentSearchArguments(ToolArguments):
    description: str = Field(min_length=3, max_length=4_000)
    top_k: int = Field(ge=1, le=10)


STANDARD_TOOL_ARGUMENT_MODELS: dict[InvestigationToolName, type[ToolArguments]] = {
    InvestigationToolName.GET_MACHINE_STATUS: MachineArguments,
    InvestigationToolName.GET_RECENT_BREWS: MachineArguments,
    InvestigationToolName.GET_SUPPLY_LEVELS: MachineArguments,
    InvestigationToolName.GET_SENSOR_ALERTS: MachineArguments,
    InvestigationToolName.GET_TEMPERATURE_HISTORY: MachineArguments,
    InvestigationToolName.GET_CLEANING_STATUS: MachineArguments,
    InvestigationToolName.SEARCH_APPLICATION_LOGS: ApplicationLogSearchArguments,
    InvestigationToolName.SEARCH_SIMILAR_INCIDENTS: SimilarIncidentSearchArguments,
    InvestigationToolName.GET_MACHINE_HEALTH: MachineArguments,
}

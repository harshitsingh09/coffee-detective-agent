"""Provider-independent allowlist and execution boundary for safe agent tools."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ValidationError

from incident_assistant.domain.agent_models import (
    RequestedToolCall,
    ToolDefinition,
    ToolExecutionResult,
    ToolExecutionStatus,
)

ToolHandler = Callable[[BaseModel], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class SafeTool:
    """One allowlisted handler with validated, structured arguments."""

    name: str
    description: str
    arguments_model: type[BaseModel]
    handler: ToolHandler

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.arguments_model.model_json_schema(),
        )


class ToolRegistry:
    """Rejects unknown tools and converts every outcome to structured data."""

    def __init__(self, tools: Sequence[SafeTool] = ()) -> None:
        self._tools: dict[str, SafeTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: SafeTool) -> None:
        definition = tool.definition()  # Validate name, description, and schema.
        if definition.name in self._tools:
            raise ValueError(f"Tool {definition.name!r} is already registered.")
        self._tools[definition.name] = tool

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(tool.definition() for tool in self._tools.values())

    def execute(self, call: RequestedToolCall) -> ToolExecutionResult:
        started = perf_counter()
        tool = self._tools.get(call.name)
        if tool is None:
            return self._result(
                call,
                ToolExecutionStatus.REJECTED,
                started,
                error=f"Tool {call.name!r} is not allowlisted.",
            )

        try:
            arguments = tool.arguments_model.model_validate(call.arguments)
        except ValidationError as exc:
            return self._result(
                call,
                ToolExecutionStatus.REJECTED,
                started,
                error=f"Invalid tool arguments: {exc}",
            )

        normalized_arguments = arguments.model_dump(mode="json")
        try:
            raw_data = dict(tool.handler(arguments))
            json.dumps(raw_data)
        except Exception as exc:
            return self._result(
                call,
                ToolExecutionStatus.FAILED,
                started,
                arguments=normalized_arguments,
                error=f"{type(exc).__name__}: {exc}",
            )

        return self._result(
            call,
            ToolExecutionStatus.SUCCEEDED,
            started,
            arguments=normalized_arguments,
            data=raw_data,
        )

    @staticmethod
    def _result(
        call: RequestedToolCall,
        status: ToolExecutionStatus,
        started: float,
        *,
        arguments: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            call_id=call.call_id,
            name=call.name,
            status=status,
            arguments=arguments if arguments is not None else dict(call.arguments),
            data=data,
            error=error,
            duration_ms=(perf_counter() - started) * 1_000,
        )

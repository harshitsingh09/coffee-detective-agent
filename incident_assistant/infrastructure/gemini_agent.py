"""Gemini Interactions API adapter for bounded function-calling investigations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from incident_assistant.domain.agent_models import (
    AgentStep,
    IncidentDiagnosis,
    RequestedToolCall,
    ToolDefinition,
    ToolExecutionResult,
)
from incident_assistant.infrastructure.agent_provider_common import (
    AGENT_INSTRUCTIONS,
    flat_function_tool,
    incident_context_json,
    parse_diagnosis,
    parse_json_object,
)


class GeminiAgentProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
        *,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self._client = client
        self._model = model
        self._timeout_seconds = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "gemini"

    def start_session(
        self,
        incident_id: str,
        incident_description: str,
        machine_id: str,
        tools: Sequence[ToolDefinition],
    ) -> GeminiAgentSession:
        return GeminiAgentSession(
            client=self._client,
            model=self._model,
            timeout_seconds=self._timeout_seconds,
            incident_id=incident_id,
            incident_description=incident_description,
            machine_id=machine_id,
            tools=tools,
        )


class GeminiAgentSession:
    def __init__(
        self,
        *,
        client: Any,
        model: str,
        timeout_seconds: float,
        incident_id: str,
        incident_description: str,
        machine_id: str,
        tools: Sequence[ToolDefinition],
    ) -> None:
        self._client = client
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._incident_id = incident_id
        self._incident_description = incident_description
        self._machine_id = machine_id
        self._tools = tuple(flat_function_tool(tool) for tool in tools)
        self._previous_interaction_id: str | None = None
        self._started = False

    def next_step(self, tool_results: Sequence[ToolExecutionResult] = ()) -> AgentStep:
        if not self._started:
            if tool_results:
                raise ValueError("The initial agent step cannot contain tool results.")
            input_payload: Any = incident_context_json(
                self._incident_id,
                self._incident_description,
                self._machine_id,
            )
            self._started = True
        else:
            if not tool_results:
                raise ValueError("A continued agent step requires tool results.")
            input_payload = [
                {
                    "type": "function_result",
                    "call_id": result.call_id,
                    "name": result.name,
                    "result": parse_json_object(result.model_dump_json()),
                    "is_error": result.error is not None,
                }
                for result in tool_results
            ]

        request: dict[str, Any] = {
            "model": self._model,
            "system_instruction": AGENT_INSTRUCTIONS,
            "input": input_payload,
            "tools": self._tools,
            "generation_config": {"tool_choice": "auto", "thinking_summaries": "none"},
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": IncidentDiagnosis.model_json_schema(),
            },
            "timeout": self._timeout_seconds,
        }
        if self._previous_interaction_id is not None:
            request["previous_interaction_id"] = self._previous_interaction_id
        interaction = self._client.interactions.create(**request)
        self._previous_interaction_id = str(interaction.id)
        calls = tuple(
            RequestedToolCall(
                call_id=str(step.id),
                name=str(step.name),
                arguments=parse_json_object(step.arguments),
            )
            for step in (interaction.steps or ())
            if step.type == "function_call"
        )
        if len(calls) > 1:
            raise ValueError("Gemini requested more than one tool in a bounded step.")
        if calls:
            return AgentStep(tool_calls=calls)
        if not interaction.output_text:
            raise ValueError("Gemini returned neither a tool call nor a diagnosis.")
        return AgentStep(diagnosis=parse_diagnosis(interaction.output_text))

"""OpenAI Responses API adapter for bounded function-calling investigations."""

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
    parse_json_object,
)


class OpenAIAgentModel:
    """Create provider-managed sessions while keeping SDK details out of the use case."""

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
        *,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, timeout=timeout_seconds)
        self._client = client
        self._model = model
        self._timeout_seconds = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "openai"

    def start_session(
        self,
        incident_id: str,
        incident_description: str,
        machine_id: str,
        tools: Sequence[ToolDefinition],
    ) -> OpenAIAgentSession:
        return OpenAIAgentSession(
            client=self._client,
            model=self._model,
            timeout_seconds=self._timeout_seconds,
            incident_id=incident_id,
            incident_description=incident_description,
            machine_id=machine_id,
            tools=tools,
        )


class OpenAIAgentSession:
    """Translate one safe controller step at a time to Responses API calls."""

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
        self._tools = tuple(self._openai_tool(tool) for tool in tools)
        self._previous_response_id: str | None = None
        self._started = False

    def next_step(
        self,
        tool_results: Sequence[ToolExecutionResult] = (),
    ) -> AgentStep:
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
                    "type": "function_call_output",
                    "call_id": result.call_id,
                    "output": result.model_dump_json(),
                }
                for result in tool_results
            ]

        request: dict[str, Any] = {
            "model": self._model,
            "instructions": AGENT_INSTRUCTIONS,
            "input": input_payload,
            "tools": self._tools,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "max_tool_calls": 1,
            "text_format": IncidentDiagnosis,
            "reasoning": {"effort": "low"},
            "verbosity": "low",
            "store": True,
            "safety_identifier": f"synthetic-incident-{self._machine_id}",
            "timeout": self._timeout_seconds,
        }
        if self._previous_response_id is not None:
            request["previous_response_id"] = self._previous_response_id
        response = self._client.responses.parse(**request)
        self._previous_response_id = str(response.id)
        calls = tuple(
            RequestedToolCall(
                call_id=str(item.call_id),
                name=str(item.name),
                arguments=self._arguments(item.arguments),
            )
            for item in response.output
            if item.type == "function_call"
        )
        if calls:
            return AgentStep(tool_calls=calls)
        if response.output_parsed is None:
            raise ValueError("The model returned neither a tool call nor a diagnosis.")
        diagnosis = IncidentDiagnosis.model_validate(response.output_parsed)
        return AgentStep(diagnosis=diagnosis)

    @staticmethod
    def _openai_tool(definition: ToolDefinition) -> dict[str, Any]:
        return {**flat_function_tool(definition), "strict": True}

    @staticmethod
    def _arguments(raw_arguments: str) -> dict[str, Any]:
        return parse_json_object(raw_arguments)


OpenAIAgentProvider = OpenAIAgentModel

"""Groq Chat Completions adapter for bounded function-calling investigations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from incident_assistant.domain.agent_models import (
    AgentStep,
    RequestedToolCall,
    ToolDefinition,
    ToolExecutionResult,
)
from incident_assistant.infrastructure.agent_provider_common import (
    AGENT_INSTRUCTIONS,
    chat_function_tool,
    diagnosis_json_instruction,
    incident_context_json,
    parse_diagnosis,
    parse_json_object,
)


class GroqAgentProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
        *,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from groq import Groq

            client = Groq(api_key=api_key, timeout=timeout_seconds)
        self._client = client
        self._model = model
        self._timeout_seconds = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "groq"

    def start_session(
        self,
        incident_id: str,
        incident_description: str,
        machine_id: str,
        tools: Sequence[ToolDefinition],
    ) -> GroqAgentSession:
        return GroqAgentSession(
            client=self._client,
            model=self._model,
            timeout_seconds=self._timeout_seconds,
            incident_id=incident_id,
            incident_description=incident_description,
            machine_id=machine_id,
            tools=tools,
        )


class GroqAgentSession:
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
        self._tools = tuple(chat_function_tool(tool) for tool in tools)
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": f"{AGENT_INSTRUCTIONS}\n{diagnosis_json_instruction()}"},
            {
                "role": "user",
                "content": incident_context_json(
                    incident_id,
                    incident_description,
                    machine_id,
                ),
            },
        ]
        self._started = False

    def next_step(self, tool_results: Sequence[ToolExecutionResult] = ()) -> AgentStep:
        if not self._started:
            if tool_results:
                raise ValueError("The initial agent step cannot contain tool results.")
            self._started = True
        else:
            if not tool_results:
                raise ValueError("A continued agent step requires tool results.")
            self._messages.extend(
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "name": result.name,
                    "content": result.model_dump_json(),
                }
                for result in tool_results
            )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._messages,
            tools=self._tools,
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=0,
            timeout=self._timeout_seconds,
        )
        message = response.choices[0].message
        raw_calls = tuple(message.tool_calls or ())
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": message.content,
        }
        if raw_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in raw_calls
            ]
        self._messages.append(assistant_message)
        if len(raw_calls) > 1:
            raise ValueError("Groq requested more than one tool in a bounded step.")
        if raw_calls:
            calls = tuple(
                RequestedToolCall(
                    call_id=str(call.id),
                    name=str(call.function.name),
                    arguments=parse_json_object(call.function.arguments),
                )
                for call in raw_calls
            )
            return AgentStep(tool_calls=calls)
        if not message.content:
            raise ValueError("Groq returned neither a tool call nor a diagnosis.")
        return AgentStep(diagnosis=parse_diagnosis(message.content))

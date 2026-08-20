import unittest
from types import SimpleNamespace

from incident_assistant.domain.agent_models import (
    DiagnosisStatus,
    IncidentDiagnosis,
    ToolDefinition,
    ToolExecutionResult,
    ToolExecutionStatus,
)
from incident_assistant.infrastructure.gemini_agent import GeminiAgentProvider
from incident_assistant.tools.schemas import MachineArguments


class FakeInteractions:
    def __init__(self, interactions):
        self._interactions = list(interactions)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self._interactions.pop(0)


class FakeClient:
    def __init__(self, interactions):
        self.interactions = FakeInteractions(interactions)


def diagnosis_json() -> str:
    return IncidentDiagnosis(
        incident_id="INV-TEST",
        machine_id="CM-1001",
        status=DiagnosisStatus.DIAGNOSED,
        root_cause="Coffee bean hopper nearly empty",
        confidence=0.9,
        explanation="Watery brews and a low-bean alert align.",
        supporting_evidence=("34 watery brews",),
        recommended_actions=("Refill the bean hopper.",),
    ).model_dump_json()


class GeminiAgentAdapterTests(unittest.TestCase):
    def tool(self):
        return ToolDefinition(
            name="get_machine_status",
            description="Return recent brew cycles.",
            input_schema=MachineArguments.model_json_schema(),
        )

    def test_tool_call_continuation_and_structured_diagnosis(self) -> None:
        client = FakeClient(
            (
                SimpleNamespace(
                    id="interaction-1",
                    steps=(
                        SimpleNamespace(
                            type="function_call",
                            id="call-1",
                            name="get_machine_status",
                            arguments={"machine_id": "CM-1001"},
                        ),
                    ),
                    output_text=None,
                ),
                SimpleNamespace(
                    id="interaction-2",
                    steps=(),
                    output_text=diagnosis_json(),
                ),
            )
        )
        provider = GeminiAgentProvider("test", "gemini-test", client=client)
        self.assertEqual(provider.provider_name, "gemini")
        session = provider.start_session(
            "INV-TEST", "Investigate CM-1001", "CM-1001", (self.tool(),)
        )

        step = session.next_step()
        self.assertEqual(step.tool_calls[0].name, "get_machine_status")
        final = session.next_step(
            (
                ToolExecutionResult(
                    call_id="call-1",
                    name="get_machine_status",
                    status=ToolExecutionStatus.SUCCEEDED,
                    arguments={"machine_id": "CM-1001"},
                    data={"pending_count": 34},
                ),
            )
        )
        self.assertEqual(final.diagnosis.root_cause, "Coffee bean hopper nearly empty")
        first, second = client.interactions.requests
        self.assertEqual(first["tools"][0]["name"], "get_machine_status")
        self.assertEqual(first["response_format"]["mime_type"], "application/json")
        self.assertEqual(second["previous_interaction_id"], "interaction-1")
        self.assertEqual(second["input"][0]["type"], "function_result")

    def test_rejects_multiple_or_malformed_tool_calls(self) -> None:
        malformed = SimpleNamespace(
            id="interaction-1",
            steps=(
                SimpleNamespace(
                    type="function_call",
                    id="call-1",
                    name="get_machine_status",
                    arguments=[],
                ),
            ),
            output_text=None,
        )
        session = GeminiAgentProvider(
            "test", "gemini-test", client=FakeClient((malformed,))
        ).start_session("INV-TEST", "Investigate CM-1001", "CM-1001", (self.tool(),))
        with self.assertRaises(ValueError):
            session.next_step()

    def test_rejects_malformed_final_diagnosis(self) -> None:
        response = SimpleNamespace(id="interaction-1", steps=(), output_text='{"status":"bad"}')
        session = GeminiAgentProvider(
            "test", "gemini-test", client=FakeClient((response,))
        ).start_session("INV-TEST", "Investigate CM-1001", "CM-1001", (self.tool(),))
        with self.assertRaises(ValueError):
            session.next_step()


if __name__ == "__main__":
    unittest.main()

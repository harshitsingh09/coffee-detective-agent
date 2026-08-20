import unittest
from types import SimpleNamespace

from incident_assistant.domain.agent_models import (
    DiagnosisStatus,
    IncidentDiagnosis,
    ToolDefinition,
    ToolExecutionResult,
    ToolExecutionStatus,
)
from incident_assistant.infrastructure.groq_agent import GroqAgentProvider
from incident_assistant.tools.schemas import MachineArguments


class FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def response(message):
    return SimpleNamespace(choices=(SimpleNamespace(message=message),))


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


class GroqAgentAdapterTests(unittest.TestCase):
    def tool(self):
        return ToolDefinition(
            name="get_machine_status",
            description="Return recent brew cycles.",
            input_schema=MachineArguments.model_json_schema(),
        )

    def test_tool_call_continuation_and_structured_diagnosis(self) -> None:
        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="get_machine_status",
                arguments='{"machine_id":"CM-1001"}',
            ),
        )
        client = FakeClient(
            (
                response(SimpleNamespace(content=None, tool_calls=(tool_call,))),
                response(SimpleNamespace(content=diagnosis_json(), tool_calls=())),
            )
        )
        provider = GroqAgentProvider("test", "groq-test", client=client)
        self.assertEqual(provider.provider_name, "groq")
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
        first, second = client.chat.completions.requests
        self.assertFalse(first["parallel_tool_calls"])
        self.assertNotIn("response_format", first)
        self.assertEqual(first["tools"][0]["function"]["name"], "get_machine_status")
        self.assertEqual(second["messages"][-2]["role"], "tool")
        self.assertEqual(second["messages"][-2]["tool_call_id"], "call-1")

    def test_rejects_malformed_tool_arguments(self) -> None:
        call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="get_machine_status", arguments="[]"),
        )
        session = GroqAgentProvider(
            "test",
            "groq-test",
            client=FakeClient((response(SimpleNamespace(content=None, tool_calls=(call,))),)),
        ).start_session("INV-TEST", "Investigate CM-1001", "CM-1001", (self.tool(),))
        with self.assertRaises(ValueError):
            session.next_step()

    def test_rejects_malformed_final_diagnosis(self) -> None:
        session = GroqAgentProvider(
            "test",
            "groq-test",
            client=FakeClient(
                (response(SimpleNamespace(content='{"status":"bad"}', tool_calls=())),)
            ),
        ).start_session("INV-TEST", "Investigate CM-1001", "CM-1001", (self.tool(),))
        with self.assertRaises(ValueError):
            session.next_step()


if __name__ == "__main__":
    unittest.main()

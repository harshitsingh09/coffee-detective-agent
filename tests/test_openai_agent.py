import json
import unittest
from types import SimpleNamespace

from incident_assistant.domain.agent_models import (
    DiagnosisStatus,
    IncidentDiagnosis,
    ToolDefinition,
    ToolExecutionResult,
    ToolExecutionStatus,
)
from incident_assistant.infrastructure.openai_agent import OpenAIAgentModel
from incident_assistant.tools.schemas import MachineArguments


class FakeResponses:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    def parse(self, **kwargs):
        self.requests.append(kwargs)
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.responses = FakeResponses(responses)


class OpenAIAgentAdapterTests(unittest.TestCase):
    def tool(self):
        return ToolDefinition(
            name="get_machine_status",
            description="Return recent brew cycles.",
            input_schema=MachineArguments.model_json_schema(),
        )

    def diagnosis(self):
        return IncidentDiagnosis(
            incident_id="INV-TEST",
            machine_id="CM-1001",
            status=DiagnosisStatus.DIAGNOSED,
            root_cause="Coffee bean hopper nearly empty",
            confidence=0.9,
            explanation="Watery brews were observed.",
            supporting_evidence=("34 recent brews were watery.",),
            recommended_actions=("Refill the bean hopper.",),
            tools_used=(),
            similar_incidents=(),
            warnings=(),
        )

    def test_translates_function_call_and_continuation_to_responses_api(self) -> None:
        first_response = SimpleNamespace(
            id="response-1",
            output=(
                SimpleNamespace(
                    type="function_call",
                    call_id="call-1",
                    name="get_machine_status",
                    arguments=json.dumps({"machine_id": "CM-1001"}),
                ),
            ),
            output_parsed=None,
        )
        second_response = SimpleNamespace(
            id="response-2",
            output=(),
            output_parsed=self.diagnosis(),
        )
        client = FakeClient((first_response, second_response))
        session = OpenAIAgentModel(
            api_key="test",
            model="gpt-5.4-mini",
            client=client,
        ).start_session(
            "INV-TEST",
            "CM-1001 is making watery espresso",
            "CM-1001",
            (self.tool(),),
        )

        self.assertEqual(
            OpenAIAgentModel(api_key="test", model="gpt-5.4-mini", client=client).provider_name,
            "openai",
        )

        step = session.next_step()
        self.assertEqual(step.tool_calls[0].name, "get_machine_status")
        result = ToolExecutionResult(
            call_id="call-1",
            name="get_machine_status",
            status=ToolExecutionStatus.SUCCEEDED,
            arguments={"machine_id": "CM-1001"},
            data={"pending_central_count": 34},
        )
        final_step = session.next_step((result,))
        self.assertEqual(
            final_step.diagnosis.root_cause,
            "Coffee bean hopper nearly empty",
        )

        first_request, second_request = client.responses.requests
        self.assertEqual(first_request["model"], "gpt-5.4-mini")
        self.assertFalse(first_request["parallel_tool_calls"])
        self.assertEqual(first_request["max_tool_calls"], 1)
        self.assertTrue(first_request["tools"][0]["strict"])
        self.assertFalse(first_request["tools"][0]["parameters"]["additionalProperties"])
        self.assertEqual(second_request["previous_response_id"], "response-1")
        self.assertEqual(second_request["input"][0]["type"], "function_call_output")
        self.assertEqual(second_request["input"][0]["call_id"], "call-1")

    def test_rejects_malformed_function_arguments(self) -> None:
        response = SimpleNamespace(
            id="response-1",
            output=(
                SimpleNamespace(
                    type="function_call",
                    call_id="call-1",
                    name="get_machine_status",
                    arguments="[]",
                ),
            ),
            output_parsed=None,
        )
        session = OpenAIAgentModel(
            api_key="test",
            model="gpt-5.4-mini",
            client=FakeClient((response,)),
        ).start_session(
            "INV-TEST",
            "Investigate CM-1001",
            "CM-1001",
            (self.tool(),),
        )
        with self.assertRaises(ValueError):
            session.next_step()


if __name__ == "__main__":
    unittest.main()

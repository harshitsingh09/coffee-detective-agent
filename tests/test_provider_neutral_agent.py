import json
import unittest
from types import SimpleNamespace

from incident_assistant.application.agent_service import AgentInvestigationService
from incident_assistant.domain.agent_models import (
    DiagnosisStatus,
    ExecutionMode,
    IncidentDiagnosis,
)
from incident_assistant.infrastructure.extraction import RegexMachineIdExtractor
from incident_assistant.infrastructure.gemini_agent import GeminiAgentProvider
from incident_assistant.infrastructure.groq_agent import GroqAgentProvider
from incident_assistant.infrastructure.openai_agent import OpenAIAgentProvider
from incident_assistant.tools.registry import SafeTool, ToolRegistry
from incident_assistant.tools.schemas import MachineArguments


class SequentialMethod:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self, **_kwargs):
        return self.values.pop(0)


class UnusedFallback:
    def investigate(self, _description):
        raise AssertionError("Successful provider flow must not invoke fallback.")


def diagnosis() -> IncidentDiagnosis:
    return IncidentDiagnosis(
        incident_id="INV-COMMON",
        machine_id="CM-1001",
        status=DiagnosisStatus.DIAGNOSED,
        root_cause="Coffee bean hopper nearly empty",
        confidence=0.91,
        explanation="Watery brews coincide with low-bean alerts.",
        supporting_evidence=("Watery brews and low-bean alerts were observed.",),
        recommended_actions=("Refill the bean hopper.",),
    )


def openai_provider():
    def tool_response(response_id, call_id, name):
        return SimpleNamespace(
            id=response_id,
            output=(
                SimpleNamespace(
                    type="function_call",
                    call_id=call_id,
                    name=name,
                    arguments=json.dumps({"machine_id": "CM-1001"}),
                ),
            ),
            output_parsed=None,
        )

    responses = SimpleNamespace(
        parse=SequentialMethod(
            (
                tool_response("oa-1", "call-1", "get_recent_brews"),
                tool_response("oa-2", "call-2", "get_supply_levels"),
                SimpleNamespace(id="oa-3", output=(), output_parsed=diagnosis()),
            )
        )
    )
    return OpenAIAgentProvider("test", "openai-test", client=SimpleNamespace(responses=responses))


def gemini_provider():
    def tool_response(response_id, call_id, name):
        return SimpleNamespace(
            id=response_id,
            steps=(
                SimpleNamespace(
                    type="function_call",
                    id=call_id,
                    name=name,
                    arguments={"machine_id": "CM-1001"},
                ),
            ),
            output_text=None,
        )

    interactions = SimpleNamespace(
        create=SequentialMethod(
            (
                tool_response("gm-1", "call-1", "get_recent_brews"),
                tool_response("gm-2", "call-2", "get_supply_levels"),
                SimpleNamespace(id="gm-3", steps=(), output_text=diagnosis().model_dump_json()),
            )
        )
    )
    return GeminiAgentProvider(
        "test", "gemini-test", client=SimpleNamespace(interactions=interactions)
    )


def groq_provider():
    def tool_response(call_id, name):
        call = SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(
                name=name,
                arguments='{"machine_id":"CM-1001"}',
            ),
        )
        return SimpleNamespace(
            choices=(SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=(call,))),)
        )

    final = SimpleNamespace(
        choices=(
            SimpleNamespace(
                message=SimpleNamespace(
                    content=diagnosis().model_dump_json(),
                    tool_calls=(),
                )
            ),
        )
    )
    completions = SimpleNamespace(
        create=SequentialMethod(
            (
                tool_response("call-1", "get_recent_brews"),
                tool_response("call-2", "get_supply_levels"),
                final,
            )
        )
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return GroqAgentProvider("test", "groq-test", client=client)


class ProviderNeutralAgentTests(unittest.TestCase):
    def test_all_providers_drive_the_same_bounded_tool_workflow(self) -> None:
        providers = (openai_provider(), gemini_provider(), groq_provider())
        for provider in providers:
            with self.subTest(provider=provider.provider_name):
                registry = ToolRegistry(
                    (
                        SafeTool(
                            name="get_recent_brews",
                            description="Return recent brew cycles.",
                            arguments_model=MachineArguments,
                            handler=lambda arguments: {
                                "machine_id": arguments.machine_id,
                                "pending_count": 34,
                            },
                        ),
                        SafeTool(
                            name="get_supply_levels",
                            description="Return machine sensor alerts.",
                            arguments_model=MachineArguments,
                            handler=lambda arguments: {
                                "machine_id": arguments.machine_id,
                                "failure_count": 4,
                            },
                        ),
                    )
                )
                report = AgentInvestigationService(
                    extractor=RegexMachineIdExtractor(),
                    tools=registry,
                    fallback=UnusedFallback(),
                    agent_model=provider,
                    incident_id_factory=lambda: "INV-COMMON",
                ).investigate("CM-1001 is making watery espresso")

                self.assertEqual(report.execution_mode, ExecutionMode.AI_AGENT)
                self.assertEqual(report.llm_provider, provider.provider_name)
                self.assertEqual(
                    report.diagnosis.tools_used,
                    ("get_recent_brews", "get_supply_levels"),
                )
                self.assertEqual(
                    report.diagnosis.root_cause,
                    "Coffee bean hopper nearly empty",
                )


if __name__ == "__main__":
    unittest.main()

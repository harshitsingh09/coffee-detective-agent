import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from incident_assistant.config import LLMProviderName, Settings
from incident_assistant.domain.agent_models import (
    AgentStep,
    DiagnosisStatus,
    IncidentDiagnosis,
    InvestigationToolName,
    RequestedToolCall,
    ToolExecutionStatus,
)
from incident_assistant.tools.registry import SafeTool, ToolRegistry
from incident_assistant.tools.schemas import (
    STANDARD_TOOL_ARGUMENT_MODELS,
    MachineArguments,
)


class AgentModelContractTests(unittest.TestCase):
    def test_all_planned_tools_have_an_argument_contract(self) -> None:
        self.assertEqual(
            set(STANDARD_TOOL_ARGUMENT_MODELS),
            set(InvestigationToolName),
        )

    def test_diagnosis_rejects_unknown_fields(self) -> None:
        with self.assertRaises(ValidationError):
            IncidentDiagnosis(
                incident_id="INV-1",
                machine_id="CM-1001",
                status=DiagnosisStatus.DIAGNOSED,
                root_cause="Coffee bean hopper nearly empty",
                confidence=0.9,
                explanation="The bean sensor reports an empty hopper.",
                unsupported_field="must fail",
            )

    def test_agent_step_is_either_tool_calls_or_diagnosis(self) -> None:
        call = RequestedToolCall(
            call_id="call-1",
            name="get_machine_status",
            arguments={"machine_id": "CM-1001"},
        )
        self.assertEqual(AgentStep(tool_calls=(call,)).tool_calls, (call,))
        with self.assertRaises(ValidationError):
            AgentStep()


class ToolRegistryTests(unittest.TestCase):
    @staticmethod
    def _summary(arguments: MachineArguments):
        return {"machine_id": arguments.machine_id, "total_brews": 10}

    def setUp(self) -> None:
        self.registry = ToolRegistry(
            (
                SafeTool(
                    name="get_machine_status",
                    description="Return a bounded status summary for one coffee machine.",
                    arguments_model=MachineArguments,
                    handler=self._summary,
                ),
            )
        )

    def test_executes_allowlisted_tool_with_normalized_arguments(self) -> None:
        result = self.registry.execute(
            RequestedToolCall(
                call_id="call-1",
                name="get_machine_status",
                arguments={"machine_id": "cm-1001"},
            )
        )
        self.assertEqual(result.status, ToolExecutionStatus.SUCCEEDED)
        self.assertEqual(result.arguments["machine_id"], "CM-1001")
        self.assertEqual(result.data["total_brews"], 10)

    def test_rejects_unknown_tool_without_executing_any_code(self) -> None:
        result = self.registry.execute(
            RequestedToolCall(call_id="call-2", name="delete_database", arguments={})
        )
        self.assertEqual(result.status, ToolExecutionStatus.REJECTED)
        self.assertIn("not allowlisted", result.error)

    def test_rejects_invalid_machine_identifier(self) -> None:
        result = self.registry.execute(
            RequestedToolCall(
                call_id="call-3",
                name="get_machine_status",
                arguments={"machine_id": "../../database"},
            )
        )
        self.assertEqual(result.status, ToolExecutionStatus.REJECTED)
        self.assertIn("Invalid tool arguments", result.error)


class AgentSettingsTests(unittest.TestCase):
    def test_agent_settings_are_bounded_and_opt_in(self) -> None:
        environment = {
            "LLM_PROVIDER": "gemini",
            "ENABLE_AI_AGENT": "true",
            "MAX_AGENT_STEPS": "5",
            "MAX_LOG_RESULTS": "20",
            "SIMILAR_INCIDENT_TOP_K": "3",
            "SIMILARITY_THRESHOLD": "0.35",
            "OPENAI_TIMEOUT_SECONDS": "30",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = Settings.from_environment()
        self.assertTrue(settings.enable_ai_agent)
        self.assertEqual(settings.max_agent_steps, 5)
        self.assertEqual(settings.max_log_results, 20)
        self.assertEqual(settings.similar_incident_top_k, 3)
        self.assertEqual(settings.similarity_threshold, 0.35)
        self.assertEqual(settings.llm_provider, LLMProviderName.GEMINI)
        self.assertEqual(settings.llm_timeout_seconds, 30.0)
        self.assertEqual(
            settings.embedding_model,
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        )

    def test_rejects_unbounded_agent_steps(self) -> None:
        with patch.dict(os.environ, {"MAX_AGENT_STEPS": "100"}, clear=True):
            with self.assertRaises(ValueError):
                Settings.from_environment()

    def test_rejects_invalid_similarity_threshold(self) -> None:
        with patch.dict(os.environ, {"SIMILARITY_THRESHOLD": "1.1"}, clear=True):
            with self.assertRaises(ValueError):
                Settings.from_environment()

    def test_rejects_unknown_llm_provider(self) -> None:
        with patch.dict(os.environ, {"LLM_PROVIDER": "unknown"}, clear=True):
            with self.assertRaisesRegex(ValueError, "LLM_PROVIDER"):
                Settings.from_environment()


if __name__ == "__main__":
    unittest.main()

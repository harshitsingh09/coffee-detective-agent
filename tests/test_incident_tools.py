import tempfile
import unittest
from pathlib import Path

from incident_assistant.domain.agent_models import (
    InvestigationToolName,
    RequestedToolCall,
    SimilarIncident,
    ToolExecutionStatus,
)
from incident_assistant.infrastructure.file_log_repository import FileLogRepository
from incident_assistant.infrastructure.sqlite_repository import SqliteRepository
from incident_assistant.tools.incident_tools import build_investigation_tool_registry
from seed_data import seed


class StubHistoricalIncidentRetriever:
    def search(self, description: str, top_k: int = 3):
        return (
            SimilarIncident(
                incident_id="INC-TEST",
                similarity_score=0.91,
                description=description,
                root_cause="Known test cause",
                resolution="Known test resolution",
            ),
        )[:top_k]


class SafeIncidentToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        data_dir = Path(cls._temporary_directory.name)
        database_path, log_path, _ = seed(data_dir, brews_per_machine=50)
        cls.registry = build_investigation_tool_registry(
            SqliteRepository(database_path),
            FileLogRepository(log_path),
            StubHistoricalIncidentRetriever(),
            max_log_results=2,
            max_similar_incidents=1,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    @classmethod
    def execute(cls, name: InvestigationToolName, arguments: dict):
        return cls.registry.execute(
            RequestedToolCall(
                call_id=f"call-{name.value}",
                name=name.value,
                arguments=arguments,
            )
        )

    def test_registers_exactly_the_standard_allowlist(self) -> None:
        self.assertEqual(
            {definition.name for definition in self.registry.definitions()},
            {name.value for name in InvestigationToolName},
        )

    def test_every_function_schema_is_strict_compatible(self) -> None:
        for definition in self.registry.definitions():
            schema = definition.input_schema
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["properties"]), set(schema["required"]))

    def test_recent_brews_show_seeded_watery_espresso_cycles(self) -> None:
        result = self.execute(
            InvestigationToolName.GET_RECENT_BREWS,
            {"machine_id": "CM-1001"},
        )
        self.assertEqual(result.status, ToolExecutionStatus.SUCCEEDED)
        self.assertEqual(result.data["abnormal_brew_count"], 34)
        self.assertLessEqual(result.data["returned_count"], 20)

    def test_sensor_alerts_show_seeded_milk_fault(self) -> None:
        result = self.execute(
            InvestigationToolName.GET_SENSOR_ALERTS,
            {"machine_id": "CM-1002"},
        )
        self.assertEqual(result.data["alert_count"], 3)
        self.assertEqual(result.data["alerts"][0]["error_code"], "MILK_LINE_DISCONNECTED")

    def test_temperature_history_shows_seeded_overheating(self) -> None:
        result = self.execute(
            InvestigationToolName.GET_TEMPERATURE_HISTORY,
            {"machine_id": "CM-1003"},
        )
        self.assertEqual(result.data["reading_count"], 6)
        self.assertEqual(result.data["maximum_temperature_c"], 101.0)

    def test_cleaning_status_shows_seeded_overdue_machine(self) -> None:
        result = self.execute(
            InvestigationToolName.GET_CLEANING_STATUS,
            {"machine_id": "CM-1004"},
        )
        self.assertEqual(result.data["cycles_since_cleaning"], 221)
        self.assertTrue(result.data["cleaning_due"])

    def test_health_tool_reports_normal_machine_as_healthy(self) -> None:
        result = self.execute(
            InvestigationToolName.GET_MACHINE_HEALTH,
            {"machine_id": "CM-1005"},
        )
        self.assertEqual(result.data["health"], "healthy")

    def test_log_tool_applies_keyword_and_result_limits(self) -> None:
        result = self.execute(
            InvestigationToolName.SEARCH_APPLICATION_LOGS,
            {
                "machine_id": "CM-1001",
                "keywords": ["error", "beans", "watery"],
            },
        )
        self.assertLessEqual(result.data["match_count"], 2)
        self.assertTrue(all("CM-1001" in entry for entry in result.data["entries"]))

    def test_similar_incident_tool_returns_structured_matches(self) -> None:
        result = self.execute(
            InvestigationToolName.SEARCH_SIMILAR_INCIDENTS,
            {"description": "espresso is watery and beans are low", "top_k": 10},
        )
        self.assertEqual(result.data["match_count"], 1)
        self.assertEqual(result.data["matches"][0]["incident_id"], "INC-TEST")

    def test_tool_arguments_cannot_inject_sql_or_extra_parameters(self) -> None:
        result = self.execute(
            InvestigationToolName.GET_MACHINE_STATUS,
            {"machine_id": "CM-1001", "sql": "DROP TABLE brew_cycles"},
        )
        self.assertEqual(result.status, ToolExecutionStatus.REJECTED)
        self.assertIn("Invalid tool arguments", result.error)


if __name__ == "__main__":
    unittest.main()

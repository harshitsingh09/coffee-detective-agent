import unittest

from incident_assistant.application.errors import InvalidIncidentError
from incident_assistant.application.service import InvestigationService
from incident_assistant.domain.models import Diagnosis, Evidence


class StubExtractor:
    def extract(self, incident_description: str) -> str | None:
        return "CM-1001" if "CM-1001" in incident_description else None


class StubOperations:
    def check_brews(self, machine_id: str):
        return (Evidence("brews.test", "database", "Brew check completed."),)

    def check_machine_status(self, machine_id: str):
        raise RuntimeError("sensor source offline")


class StubLogs:
    def search(self, machine_id: str, limit: int = 50):
        return (Evidence("logs.test", "logs", "Log check completed."),)


class StubIncidents:
    def find_similar(self, incident_description: str, machine_id: str, limit: int = 3):
        return ()


class StubDiagnostician:
    def diagnose(self, incident_description: str, machine_id: str, evidence):
        return Diagnosis(
            root_cause="Test cause",
            explanation="Test explanation",
            supporting_evidence=tuple(item.summary for item in evidence),
            recommended_actions=("Test action",),
            confidence=0.8,
            generated_by="test",
        )


class InvestigationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = InvestigationService(
            extractor=StubExtractor(),
            operations=StubOperations(),
            logs=StubLogs(),
            incidents=StubIncidents(),
            diagnostician=StubDiagnostician(),
        )

    def test_rejects_description_without_machine_id(self) -> None:
        with self.assertRaises(InvalidIncidentError):
            self.service.investigate("Coffee tastes strange")

    def test_keeps_partial_evidence_when_one_adapter_fails(self) -> None:
        report = self.service.investigate("CM-1001 coffee is watery")
        codes = {item.code for item in report.evidence}
        self.assertIn("brews.test", codes)
        self.assertIn("source.machine_status.unavailable", codes)
        self.assertIn("logs.test", codes)
        self.assertEqual(report.diagnosis.root_cause, "Test cause")


if __name__ == "__main__":
    unittest.main()

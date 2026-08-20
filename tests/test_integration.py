import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from incident_assistant.bootstrap import build_investigation_service
from incident_assistant.config import Settings
from seed_data import seed


class SeededInvestigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.data_dir = Path(cls._temporary_directory.name)
        database_path, log_path, _ = seed(cls.data_dir, brews_per_machine=50)
        settings = Settings(
            database_path=database_path,
            log_path=log_path,
            openai_api_key=None,
            openai_model="unused",
        )
        cls.service = build_investigation_service(settings)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def test_seed_contains_twenty_machines_and_one_thousand_brews(self) -> None:
        with closing(sqlite3.connect(self.data_dir / "support.db")) as connection:
            machines = connection.execute("SELECT COUNT(*) FROM machines").fetchone()[0]
            brews = connection.execute("SELECT COUNT(*) FROM brew_cycles").fetchone()[0]
        self.assertEqual(machines, 20)
        self.assertEqual(brews, 1000)

    def test_diagnoses_low_bean_hopper(self) -> None:
        report = self.service.investigate("CM-1001 is making watery espresso")
        self.assertEqual(report.diagnosis.root_cause, "Coffee bean hopper nearly empty")

    def test_diagnoses_milk_system_fault(self) -> None:
        report = self.service.investigate("CM-1002 cappuccino has no milk foam")
        self.assertEqual(
            report.diagnosis.root_cause,
            "Milk line disconnected or milk supply empty",
        )

    def test_diagnoses_overheating(self) -> None:
        report = self.service.investigate("CM-1003 is extremely hot and aborts drinks")
        self.assertEqual(report.diagnosis.root_cause, "Brewing system overheating")

    def test_diagnoses_overdue_cleaning(self) -> None:
        report = self.service.investigate("CM-1004 coffee tastes bitter and dirty")
        self.assertEqual(report.diagnosis.root_cause, "Cleaning cycle overdue")

    def test_reports_normal_operation(self) -> None:
        report = self.service.investigate("Check whether CM-1005 is brewing normally")
        self.assertEqual(report.diagnosis.root_cause, "No machine fault detected")


if __name__ == "__main__":
    unittest.main()

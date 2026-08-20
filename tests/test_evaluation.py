import unittest

from incident_assistant.domain.agent_models import DiagnosisStatus
from incident_assistant.evaluation.models import EvaluationResult
from incident_assistant.evaluation.runner import calculate_metrics, normalize_root_cause
from scripts.generate_eval_cases import build_cases


class EvaluationDatasetTests(unittest.TestCase):
    def test_dataset_has_required_size_and_coverage(self) -> None:
        cases = build_cases()
        self.assertGreaterEqual(len(cases), 50)
        self.assertLessEqual(len(cases), 100)
        scenarios = {case.scenario for case in cases}
        self.assertTrue(
            {
                "low_beans",
                "milk_fault",
                "overheating",
                "cleaning_overdue",
                "healthy",
                "missing_id",
                "partial_failure",
                "malicious",
            }.issubset(scenarios)
        )
        self.assertTrue(any(case.unavailable_sources for case in cases))

    def test_case_ids_are_unique(self) -> None:
        cases = build_cases()
        self.assertEqual(len(cases), len({case.case_id for case in cases}))


class EvaluationMetricTests(unittest.TestCase):
    @staticmethod
    def result(
        case_id,
        *,
        root_correct,
        status_correct,
        selected,
        expected,
        fallback,
    ):
        return EvaluationResult(
            case_id=case_id,
            scenario="test",
            incident_description="Investigate CM-1001",
            expected_root_cause="low_bean_hopper",
            actual_root_cause=("low_bean_hopper" if root_correct else "other"),
            expected_status=DiagnosisStatus.DIAGNOSED,
            actual_status=(
                DiagnosisStatus.DIAGNOSED if status_correct else DiagnosisStatus.FALLBACK_DIAGNOSIS
            ),
            expected_tools=expected,
            selected_tools=selected,
            expected_machine="CM-1001",
            actual_machine="CM-1001",
            fallback_used=fallback,
            latency_ms=10.0,
            root_cause_correct=root_correct,
            status_correct=status_correct,
            machine_correct=True,
        )

    def test_calculates_requested_metrics_from_actual_results(self) -> None:
        results = (
            self.result(
                "one",
                root_correct=True,
                status_correct=True,
                selected=("get_recent_brews", "search_application_logs"),
                expected=("get_recent_brews", "get_supply_levels"),
                fallback=False,
            ),
            self.result(
                "two",
                root_correct=False,
                status_correct=False,
                selected=("get_machine_status",),
                expected=("get_recent_brews",),
                fallback=True,
            ),
        )
        metrics = calculate_metrics(results)
        self.assertEqual(metrics.total_cases, 2)
        self.assertEqual(metrics.root_cause_accuracy, 0.5)
        self.assertEqual(metrics.status_accuracy, 0.5)
        self.assertAlmostEqual(metrics.tool_selection_precision, 1 / 3)
        self.assertAlmostEqual(metrics.tool_selection_recall, 1 / 3)
        self.assertEqual(metrics.unnecessary_tool_calls, 2)
        self.assertEqual(metrics.fallback_success_rate, 0.0)

    def test_normalizes_diagnosis_labels(self) -> None:
        self.assertEqual(
            normalize_root_cause("Coffee bean hopper nearly empty"),
            "low_bean_hopper",
        )
        self.assertIsNone(normalize_root_cause(None))


if __name__ == "__main__":
    unittest.main()

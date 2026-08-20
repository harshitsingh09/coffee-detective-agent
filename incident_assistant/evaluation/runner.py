"""Run controlled incident cases and calculate honest agent-system metrics."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from incident_assistant.application.agent_service import AgentInvestigationService
from incident_assistant.application.errors import InvalidIncidentError
from incident_assistant.application.service import InvestigationService
from incident_assistant.config import Settings
from incident_assistant.domain.agent_models import DiagnosisStatus, TraceEventType
from incident_assistant.evaluation.models import (
    EvaluationCase,
    EvaluationMetrics,
    EvaluationReport,
    EvaluationResult,
)
from incident_assistant.infrastructure.diagnosticians import RuleBasedDiagnostician
from incident_assistant.infrastructure.extraction import RegexMachineIdExtractor
from incident_assistant.infrastructure.file_log_repository import FileLogRepository
from incident_assistant.infrastructure.incident_retriever import (
    LexicalHistoricalIncidentRetriever,
    PersistentSemanticIncidentRetriever,
    ResilientHistoricalIncidentRetriever,
    SentenceTransformerEmbeddingProvider,
)
from incident_assistant.infrastructure.llm_provider_factory import create_llm_provider
from incident_assistant.infrastructure.sqlite_repository import SqliteRepository
from incident_assistant.tools.incident_tools import build_investigation_tool_registry

_ROOT_CAUSE_LABELS = (
    ("coffee bean hopper nearly empty", "low_bean_hopper"),
    ("bean hopper nearly empty", "low_bean_hopper"),
    ("low bean", "low_bean_hopper"),
    ("milk line disconnected", "milk_system_fault"),
    ("milk supply empty", "milk_system_fault"),
    ("milk system", "milk_system_fault"),
    ("brewing system overheating", "brewing_system_overheating"),
    ("overheat", "brewing_system_overheating"),
    ("cleaning cycle overdue", "cleaning_cycle_overdue"),
    ("cleaning overdue", "cleaning_cycle_overdue"),
    ("no machine fault detected", "no_machine_fault"),
)


class _FailureProxy:
    _operational_methods = {
        "check_brews",
        "check_machine_status",
        "get_machine_status",
        "get_recent_brews",
        "get_supply_levels",
        "get_temperature_history",
        "get_sensor_alerts",
        "get_cleaning_status",
        "get_machine_health",
    }
    _history_methods = {"find_similar", "list_incident_documents"}
    _log_methods = {"search", "search_structured"}

    def __init__(self, target: Any, unavailable_sources: Sequence[str]) -> None:
        self._target = target
        self._unavailable = frozenset(unavailable_sources)

    def __getattr__(self, name: str) -> Any:
        operation = getattr(self._target, name)
        unavailable = (
            (name in self._operational_methods and "database" in self._unavailable)
            or (name in self._history_methods and "history" in self._unavailable)
            or (name in self._log_methods and "logs" in self._unavailable)
        )
        if not unavailable:
            return operation

        def fail(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError(f"Injected evaluation failure for {name}.")

        return fail


class _RetrieverFailureProxy:
    def __init__(self, target: Any, unavailable_sources: Sequence[str]) -> None:
        self._target = target
        self._unavailable = frozenset(unavailable_sources)

    def search(self, description: str, top_k: int = 3):
        if "history" in self._unavailable:
            raise RuntimeError("Injected evaluation failure for incident history.")
        return self._target.search(description, top_k)


class EvaluationRunner:
    """Evaluate either live agent mode or the actual no-key fallback path."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._database = SqliteRepository(settings.database_path)
        self._logs = FileLogRepository(settings.log_path)
        semantic = PersistentSemanticIncidentRetriever(
            self._database,
            SentenceTransformerEmbeddingProvider(
                settings.embedding_model,
                settings.embedding_cache_path,
            ),
            settings.embedding_index_path,
            settings.similarity_threshold,
        )
        self._retriever = ResilientHistoricalIncidentRetriever(
            semantic,
            LexicalHistoricalIncidentRetriever(self._database),
        )
        self._agent_model = create_llm_provider(settings)

    @property
    def mode(self) -> str:
        return "ai_agent" if self._agent_model is not None else "deterministic_fallback"

    @staticmethod
    def load_cases(path: Path) -> tuple[EvaluationCase, ...]:
        cases: list[EvaluationCase] = []
        with Path(path).open("r", encoding="utf-8") as case_file:
            for line_number, line in enumerate(case_file, start=1):
                if not line.strip():
                    continue
                try:
                    cases.append(EvaluationCase.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(f"Invalid evaluation case on line {line_number}.") from exc
        if not cases:
            raise ValueError("The evaluation dataset contains no cases.")
        return tuple(cases)

    def evaluate(self, cases: Sequence[EvaluationCase]) -> EvaluationReport:
        results = tuple(self._evaluate_case(case) for case in cases)
        return EvaluationReport(
            generated_at=datetime.now(UTC).isoformat(),
            mode=self.mode,
            metrics=calculate_metrics(results),
            results=results,
        )

    @staticmethod
    def save(report: EvaluationReport, output_directory: Path) -> tuple[Path, Path]:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        json_path = output_directory / "evaluation_report.json"
        csv_path = output_directory / "evaluation_results.csv"
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            fieldnames = (
                "case_id",
                "scenario",
                "incident_description",
                "expected_root_cause",
                "actual_root_cause",
                "expected_status",
                "actual_status",
                "expected_tools",
                "selected_tools",
                "expected_machine",
                "actual_machine",
                "fallback_used",
                "latency_ms",
                "root_cause_correct",
                "status_correct",
                "machine_correct",
                "error",
            )
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for result in report.results:
                row = result.model_dump(mode="json")
                row["expected_tools"] = "|".join(result.expected_tools)
                row["selected_tools"] = "|".join(result.selected_tools)
                writer.writerow({name: row[name] for name in fieldnames})
        return json_path, csv_path

    def _evaluate_case(self, case: EvaluationCase) -> EvaluationResult:
        service = self._build_service(case.unavailable_sources)
        started = perf_counter()
        error: str | None = None
        actual_root_cause: str | None = None
        actual_machine: str | None = None
        actual_status = DiagnosisStatus.INSUFFICIENT_EVIDENCE
        selected_tools: tuple[str, ...] = ()
        fallback_used = False
        try:
            report = service.investigate(case.incident_description)
            actual_root_cause = normalize_root_cause(report.diagnosis.root_cause)
            actual_machine = report.machine_id
            actual_status = report.diagnosis.status
            selected_tools = tuple(dict.fromkeys(result.name for result in report.tool_results))
            fallback_used = any(
                event.event_type is TraceEventType.FALLBACK_STARTED for event in report.trace
            )
        except InvalidIncidentError as exc:
            error = str(exc)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (perf_counter() - started) * 1_000
        return EvaluationResult(
            case_id=case.case_id,
            scenario=case.scenario,
            incident_description=case.incident_description,
            expected_root_cause=case.expected_root_cause,
            actual_root_cause=actual_root_cause,
            expected_status=case.expected_status,
            actual_status=actual_status,
            expected_tools=case.expected_tools,
            selected_tools=selected_tools,
            expected_machine=case.expected_machine,
            actual_machine=actual_machine,
            fallback_used=fallback_used,
            latency_ms=latency_ms,
            root_cause_correct=actual_root_cause == case.expected_root_cause,
            status_correct=actual_status is case.expected_status,
            machine_correct=actual_machine == case.expected_machine,
            error=error,
        )

    def _build_service(
        self,
        unavailable_sources: Sequence[str],
    ) -> AgentInvestigationService:
        operations = _FailureProxy(self._database, unavailable_sources)
        logs = _FailureProxy(self._logs, unavailable_sources)
        retriever = _RetrieverFailureProxy(self._retriever, unavailable_sources)
        tools = build_investigation_tool_registry(
            operations,
            logs,
            retriever,
            max_log_results=self._settings.max_log_results,
            max_similar_incidents=self._settings.similar_incident_top_k,
        )
        fallback = InvestigationService(
            extractor=RegexMachineIdExtractor(),
            operations=operations,
            logs=logs,
            incidents=operations,
            diagnostician=RuleBasedDiagnostician(),
        )
        return AgentInvestigationService(
            extractor=RegexMachineIdExtractor(),
            tools=tools,
            fallback=fallback,
            agent_model=self._agent_model,
            max_steps=self._settings.max_agent_steps,
        )


def normalize_root_cause(root_cause: str | None) -> str | None:
    if root_cause is None:
        return None
    normalized = root_cause.casefold()
    for phrase, label in _ROOT_CAUSE_LABELS:
        if phrase in normalized:
            return label
    return "other"


def calculate_metrics(results: Sequence[EvaluationResult]) -> EvaluationMetrics:
    total = len(results)
    if not total:
        return EvaluationMetrics(
            total_cases=0,
            root_cause_accuracy=0.0,
            status_accuracy=0.0,
            machine_accuracy=0.0,
            tool_selection_precision=0.0,
            tool_selection_recall=0.0,
            average_tool_calls=0.0,
            unnecessary_tool_calls=0,
            fallback_success_rate=0.0,
            average_investigation_latency_ms=0.0,
        )
    true_positive_tools = 0
    selected_tool_count = 0
    expected_tool_count = 0
    unnecessary_tool_calls = 0
    for result in results:
        selected = set(result.selected_tools)
        expected = set(result.expected_tools)
        overlap = selected & expected
        true_positive_tools += len(overlap)
        selected_tool_count += len(selected)
        expected_tool_count += len(expected)
        unnecessary_tool_calls += len(selected - expected)
    fallback_results = [result for result in results if result.fallback_used]
    successful_fallbacks = sum(result.root_cause_correct for result in fallback_results)
    return EvaluationMetrics(
        total_cases=total,
        root_cause_accuracy=sum(result.root_cause_correct for result in results) / total,
        status_accuracy=sum(result.status_correct for result in results) / total,
        machine_accuracy=sum(result.machine_correct for result in results) / total,
        tool_selection_precision=(
            true_positive_tools / selected_tool_count if selected_tool_count else 0.0
        ),
        tool_selection_recall=(
            true_positive_tools / expected_tool_count if expected_tool_count else 0.0
        ),
        average_tool_calls=sum(len(result.selected_tools) for result in results) / total,
        unnecessary_tool_calls=unnecessary_tool_calls,
        fallback_success_rate=(
            successful_fallbacks / len(fallback_results) if fallback_results else 0.0
        ),
        average_investigation_latency_ms=(sum(result.latency_ms for result in results) / total),
    )

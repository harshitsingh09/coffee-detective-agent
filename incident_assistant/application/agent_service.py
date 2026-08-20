"""Bounded orchestration for a safe tool-using incident investigation agent."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any
from uuid import uuid4

from incident_assistant.application.errors import InvalidIncidentError
from incident_assistant.application.service import InvestigationService
from incident_assistant.domain.agent_models import (
    AgentInvestigationReport,
    DiagnosisStatus,
    ExecutionMode,
    IncidentDiagnosis,
    InvestigationTraceEvent,
    SimilarIncident,
    ToolExecutionResult,
    ToolExecutionStatus,
    TraceEventType,
)
from incident_assistant.domain.ports import AgentLLMProvider, MachineIdExtractor
from incident_assistant.tools.registry import ToolRegistry


class AgentInvestigationService:
    """Run model-selected tools within hard safety, cost, and reliability bounds."""

    _unsafe_phrases = (
        "ignore previous instructions",
        "ignore all instructions",
        "delete the database",
        "drop table",
        "execute shell",
        "run a command",
    )

    def __init__(
        self,
        extractor: MachineIdExtractor,
        tools: ToolRegistry,
        fallback: InvestigationService,
        agent_model: AgentLLMProvider | None,
        *,
        max_steps: int = 5,
        incident_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._extractor = extractor
        self._tools = tools
        self._fallback_service = fallback
        self._agent_model = agent_model
        self._provider_name = (
            getattr(agent_model, "provider_name", "custom") if agent_model is not None else None
        )
        self._max_steps = max(1, min(max_steps, 8))
        self._incident_id_factory = incident_id_factory or (
            lambda: f"INV-{uuid4().hex[:12].upper()}"
        )

    def investigate(self, incident_description: str) -> AgentInvestigationReport:
        description = incident_description.strip()
        if not description:
            raise InvalidIncidentError("Enter an incident description.")
        machine_id = self._extractor.extract(description)
        if machine_id is None:
            raise InvalidIncidentError("No machine ID was found. Include an ID such as CM-1001.")

        incident_id = self._incident_id_factory()
        trace = [
            InvestigationTraceEvent(
                sequence=1,
                event_type=TraceEventType.MACHINE_EXTRACTED,
                message=f"Machine extracted: {machine_id}",
            )
        ]
        if self._contains_unsafe_instruction(description):
            diagnosis = IncidentDiagnosis(
                incident_id=incident_id,
                machine_id=machine_id,
                status=DiagnosisStatus.INSUFFICIENT_EVIDENCE,
                root_cause=None,
                confidence=0.0,
                explanation=(
                    "The incident text contains action instructions unrelated to safe "
                    "diagnosis, so no investigation tools were executed."
                ),
                supporting_evidence=(),
                recommended_actions=(
                    "Remove action instructions and submit only the operational symptom.",
                ),
                tools_used=(),
                similar_incidents=(),
                warnings=("Potential prompt-injection text was rejected.",),
            )
            trace.append(
                self._trace(
                    trace,
                    TraceEventType.INVESTIGATION_COMPLETED,
                    "Investigation stopped safely without executing tools.",
                )
            )
            return AgentInvestigationReport(
                incident_description=description,
                machine_id=machine_id,
                diagnosis=diagnosis,
                execution_mode=ExecutionMode.SAFETY_STOP,
                llm_provider=self._provider_name,
                trace=tuple(trace),
                tool_results=(),
            )

        if self._agent_model is None:
            return self._fallback(
                description,
                machine_id,
                incident_id,
                trace,
                (),
                "AI agent is disabled, rules mode is selected, or the selected provider key "
                "is not configured.",
            )

        results: list[ToolExecutionResult] = []
        seen_calls: set[str] = set()
        previous_results: Sequence[ToolExecutionResult] = ()
        try:
            session = self._agent_model.start_session(
                incident_id,
                description,
                machine_id,
                self._tools.definitions(),
            )
            for _ in range(self._max_steps):
                step = session.next_step(previous_results)
                if step.diagnosis is not None:
                    diagnosis = self._validate_and_normalize_diagnosis(
                        step.diagnosis,
                        incident_id,
                        machine_id,
                        results,
                    )
                    trace.append(
                        self._trace(
                            trace,
                            TraceEventType.INVESTIGATION_COMPLETED,
                            "Investigation completed with a structured diagnosis.",
                        )
                    )
                    return AgentInvestigationReport(
                        incident_description=description,
                        machine_id=machine_id,
                        diagnosis=diagnosis,
                        execution_mode=ExecutionMode.AI_AGENT,
                        llm_provider=self._provider_name,
                        trace=tuple(trace),
                        tool_results=tuple(results),
                    )

                if len(results) + len(step.tool_calls) > self._max_steps:
                    raise RuntimeError("Agent exceeded the allowed tool-call budget.")
                current_results: list[ToolExecutionResult] = []
                for call in step.tool_calls:
                    trace.append(
                        self._trace(
                            trace,
                            TraceEventType.TOOL_SELECTED,
                            f"Selected tool: {call.name}",
                            tool_name=call.name,
                        )
                    )
                    signature = self._call_signature(call.name, call.arguments)
                    if signature in seen_calls:
                        result = ToolExecutionResult(
                            call_id=call.call_id,
                            name=call.name,
                            status=ToolExecutionStatus.REJECTED,
                            arguments=call.arguments,
                            error="Identical tool call was already executed.",
                        )
                    elif self._wrong_machine(call.arguments, machine_id):
                        result = ToolExecutionResult(
                            call_id=call.call_id,
                            name=call.name,
                            status=ToolExecutionStatus.REJECTED,
                            arguments=call.arguments,
                            error="Tool machine_id does not match the scoped incident.",
                        )
                    else:
                        seen_calls.add(signature)
                        result = self._tools.execute(call)
                    results.append(result)
                    current_results.append(result)
                    trace.append(self._result_trace(trace, result))
                previous_results = tuple(current_results)
        except Exception as exc:
            return self._fallback(
                description,
                machine_id,
                incident_id,
                trace,
                tuple(results),
                f"AI agent unavailable: {type(exc).__name__}: {exc}",
            )

        return self._fallback(
            description,
            machine_id,
            incident_id,
            trace,
            tuple(results),
            f"AI agent reached the maximum of {self._max_steps} investigation steps.",
        )

    def _validate_and_normalize_diagnosis(
        self,
        diagnosis: IncidentDiagnosis,
        incident_id: str,
        machine_id: str,
        results: Sequence[ToolExecutionResult],
    ) -> IncidentDiagnosis:
        if diagnosis.incident_id != incident_id:
            raise ValueError("Diagnosis incident_id does not match the active investigation.")
        if diagnosis.machine_id != machine_id:
            raise ValueError("Diagnosis machine_id does not match the active investigation.")
        if diagnosis.status is DiagnosisStatus.FALLBACK_DIAGNOSIS:
            raise ValueError("The AI model cannot label its own output as a fallback diagnosis.")
        succeeded = [result for result in results if result.status is ToolExecutionStatus.SUCCEEDED]
        if diagnosis.status in {DiagnosisStatus.DIAGNOSED, DiagnosisStatus.NORMAL}:
            if not succeeded:
                raise ValueError("A conclusive diagnosis requires successful tool evidence.")
            if diagnosis.status is DiagnosisStatus.DIAGNOSED and not diagnosis.supporting_evidence:
                raise ValueError("A diagnosed result must cite supporting evidence.")

        tools_used = tuple(dict.fromkeys(result.name for result in succeeded))
        similar_incidents = self._observed_similar_incidents(succeeded)
        tool_warnings = tuple(
            f"{result.name}: {result.error}"
            for result in results
            if result.status is not ToolExecutionStatus.SUCCEEDED and result.error
        )
        warnings = tuple(dict.fromkeys((*diagnosis.warnings, *tool_warnings)))
        return diagnosis.model_copy(
            update={
                "tools_used": tools_used,
                "similar_incidents": similar_incidents,
                "warnings": warnings,
            }
        )

    def _fallback(
        self,
        description: str,
        machine_id: str,
        incident_id: str,
        trace: list[InvestigationTraceEvent],
        results: Sequence[ToolExecutionResult],
        reason: str,
    ) -> AgentInvestigationReport:
        trace.append(
            self._trace(
                trace,
                TraceEventType.FALLBACK_STARTED,
                "Deterministic investigation fallback started.",
                result_summary=reason,
            )
        )
        legacy_report = self._fallback_service.investigate(description)
        is_normal = legacy_report.diagnosis.root_cause == "No machine fault detected"
        diagnosis = IncidentDiagnosis(
            incident_id=incident_id,
            machine_id=machine_id,
            status=(DiagnosisStatus.NORMAL if is_normal else DiagnosisStatus.FALLBACK_DIAGNOSIS),
            root_cause=legacy_report.diagnosis.root_cause,
            confidence=legacy_report.diagnosis.confidence,
            explanation=legacy_report.diagnosis.explanation,
            supporting_evidence=legacy_report.diagnosis.supporting_evidence,
            recommended_actions=legacy_report.diagnosis.recommended_actions,
            tools_used=(),
            similar_incidents=(),
            warnings=(reason, "Deterministic fallback diagnosis was used."),
        )
        trace.append(
            self._trace(
                trace,
                TraceEventType.INVESTIGATION_COMPLETED,
                "Investigation completed using deterministic fallback.",
            )
        )
        return AgentInvestigationReport(
            incident_description=description,
            machine_id=machine_id,
            diagnosis=diagnosis,
            execution_mode=ExecutionMode.DETERMINISTIC_FALLBACK,
            llm_provider=self._provider_name,
            trace=tuple(trace),
            tool_results=tuple(results),
        )

    @classmethod
    def _contains_unsafe_instruction(cls, description: str) -> bool:
        normalized = description.casefold()
        return any(phrase in normalized for phrase in cls._unsafe_phrases)

    @staticmethod
    def _wrong_machine(arguments: dict[str, Any], machine_id: str) -> bool:
        requested = arguments.get("machine_id")
        return requested is not None and str(requested).upper() != machine_id

    @staticmethod
    def _call_signature(name: str, arguments: dict[str, Any]) -> str:
        return f"{name}:{json.dumps(arguments, sort_keys=True, separators=(',', ':'))}"

    @staticmethod
    def _observed_similar_incidents(
        results: Sequence[ToolExecutionResult],
    ) -> tuple[SimilarIncident, ...]:
        matches: list[SimilarIncident] = []
        for result in results:
            if result.name != "search_similar_incidents" or not result.data:
                continue
            for payload in result.data.get("matches", []):
                matches.append(SimilarIncident.model_validate(payload))
        return tuple(matches)

    @staticmethod
    def _trace(
        trace: Sequence[InvestigationTraceEvent],
        event_type: TraceEventType,
        message: str,
        *,
        tool_name: str | None = None,
        result_summary: str | None = None,
        duration_ms: float | None = None,
    ) -> InvestigationTraceEvent:
        return InvestigationTraceEvent(
            sequence=len(trace) + 1,
            event_type=event_type,
            message=message,
            tool_name=tool_name,
            result_summary=result_summary,
            duration_ms=duration_ms,
        )

    def _result_trace(
        self,
        trace: Sequence[InvestigationTraceEvent],
        result: ToolExecutionResult,
    ) -> InvestigationTraceEvent:
        succeeded = result.status is ToolExecutionStatus.SUCCEEDED
        return self._trace(
            trace,
            (TraceEventType.TOOL_COMPLETED if succeeded else TraceEventType.TOOL_FAILED),
            f"Tool {result.name} {result.status.value}.",
            tool_name=result.name,
            result_summary=(self._result_summary(result.data or {}) if succeeded else result.error),
            duration_ms=result.duration_ms,
        )

    @staticmethod
    def _result_summary(data: dict[str, Any]) -> str:
        summary_fields = (
            "abnormal_brew_count",
            "alert_count",
            "maximum_temperature_c",
            "cycles_since_cleaning",
            "match_count",
            "health",
            "bean_level_pct",
            "milk_level_pct",
            "water_level_pct",
        )
        values = [f"{field}={data[field]}" for field in summary_fields if field in data]
        return ", ".join(values) if values else "Structured result returned."

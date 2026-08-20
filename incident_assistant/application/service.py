"""Orchestration for the incident-investigation use case."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from incident_assistant.application.errors import InvalidIncidentError
from incident_assistant.domain.models import Evidence, EvidenceSeverity, InvestigationReport
from incident_assistant.domain.ports import (
    Diagnostician,
    IncidentRepository,
    LogRepository,
    MachineIdExtractor,
    OperationsRepository,
)


class InvestigationService:
    """Coordinates replaceable adapters without depending on their implementations."""

    def __init__(
        self,
        extractor: MachineIdExtractor,
        operations: OperationsRepository,
        logs: LogRepository,
        incidents: IncidentRepository,
        diagnostician: Diagnostician,
    ) -> None:
        self._extractor = extractor
        self._operations = operations
        self._logs = logs
        self._incidents = incidents
        self._diagnostician = diagnostician

    def investigate(self, incident_description: str) -> InvestigationReport:
        description = incident_description.strip()
        if not description:
            raise InvalidIncidentError("Enter an incident description.")

        machine_id = self._extractor.extract(description)
        if machine_id is None:
            raise InvalidIncidentError("No machine ID was found. Include an ID such as CM-1001.")

        evidence: list[Evidence] = []
        evidence.extend(
            self._collect("brew cycles", lambda: self._operations.check_brews(machine_id))
        )
        evidence.extend(
            self._collect(
                "machine status",
                lambda: self._operations.check_machine_status(machine_id),
            )
        )
        evidence.extend(self._collect("logs", lambda: self._logs.search(machine_id)))
        evidence.extend(
            self._collect(
                "incident history",
                lambda: self._incidents.find_similar(description, machine_id),
            )
        )

        diagnosis = self._diagnostician.diagnose(description, machine_id, evidence)
        return InvestigationReport(
            incident_description=description,
            machine_id=machine_id,
            evidence=tuple(evidence),
            diagnosis=diagnosis,
        )

    @staticmethod
    def _collect(
        source_name: str,
        operation: Callable[[], Sequence[Evidence]],
    ) -> Sequence[Evidence]:
        try:
            return operation()
        except Exception as exc:  # Adapters fail independently; diagnosis can use partial evidence.
            return (
                Evidence(
                    code=f"source.{source_name.replace(' ', '_')}.unavailable",
                    source=source_name,
                    summary=f"{source_name.title()} evidence is unavailable.",
                    severity=EvidenceSeverity.WARNING,
                    details=(str(exc),),
                ),
            )

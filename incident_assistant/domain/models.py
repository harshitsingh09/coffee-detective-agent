"""Framework-independent domain objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class EvidenceSeverity(StrEnum):
    """Operational significance of a piece of evidence."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Evidence:
    """A fact gathered from one investigation source."""

    code: str
    source: str
    summary: str
    severity: EvidenceSeverity = EvidenceSeverity.INFO
    details: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", tuple(self.details))
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """A diagnosis derived from collected evidence."""

    root_cause: str
    explanation: str
    supporting_evidence: tuple[str, ...]
    recommended_actions: tuple[str, ...]
    confidence: float
    generated_by: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "supporting_evidence", tuple(self.supporting_evidence))
        object.__setattr__(self, "recommended_actions", tuple(self.recommended_actions))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, self.confidence)))


@dataclass(frozen=True, slots=True)
class InvestigationReport:
    """Complete result returned by the investigation use case."""

    incident_description: str
    machine_id: str
    evidence: tuple[Evidence, ...]
    diagnosis: Diagnosis

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence))

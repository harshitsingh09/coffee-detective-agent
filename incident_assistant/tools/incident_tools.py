"""Nine allowlisted coffee-machine investigation tools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from incident_assistant.domain.agent_models import InvestigationToolName
from incident_assistant.domain.ports import (
    HistoricalIncidentRetriever,
    InvestigationOperations,
    StructuredLogSearcher,
)
from incident_assistant.tools.registry import SafeTool, ToolRegistry
from incident_assistant.tools.schemas import (
    ApplicationLogSearchArguments,
    MachineArguments,
    SimilarIncidentSearchArguments,
)


class IncidentToolHandlers:
    """Typed handlers exposing no arbitrary SQL, path, URL, or code parameters."""

    def __init__(
        self,
        operations: InvestigationOperations,
        logs: StructuredLogSearcher,
        incidents: HistoricalIncidentRetriever,
        *,
        max_log_results: int = 20,
        max_similar_incidents: int = 3,
    ) -> None:
        self._operations = operations
        self._logs = logs
        self._incidents = incidents
        self._max_log_results = max(1, min(max_log_results, 100))
        self._max_similar_incidents = max(1, min(max_similar_incidents, 10))

    def get_machine_status(self, arguments: MachineArguments) -> Mapping[str, Any]:
        return self._operations.get_machine_status(arguments.machine_id)

    def get_recent_brews(self, arguments: MachineArguments) -> Mapping[str, Any]:
        return self._operations.get_recent_brews(arguments.machine_id)

    def get_supply_levels(self, arguments: MachineArguments) -> Mapping[str, Any]:
        return self._operations.get_supply_levels(arguments.machine_id)

    def get_sensor_alerts(self, arguments: MachineArguments) -> Mapping[str, Any]:
        return self._operations.get_sensor_alerts(arguments.machine_id)

    def get_temperature_history(self, arguments: MachineArguments) -> Mapping[str, Any]:
        return self._operations.get_temperature_history(arguments.machine_id)

    def get_cleaning_status(self, arguments: MachineArguments) -> Mapping[str, Any]:
        return self._operations.get_cleaning_status(arguments.machine_id)

    def search_application_logs(
        self, arguments: ApplicationLogSearchArguments
    ) -> Mapping[str, Any]:
        return self._logs.search_structured(
            arguments.machine_id,
            arguments.keywords,
            self._max_log_results,
        )

    def search_similar_incidents(
        self, arguments: SimilarIncidentSearchArguments
    ) -> Mapping[str, Any]:
        top_k = min(arguments.top_k, self._max_similar_incidents)
        matches = self._incidents.search(arguments.description, top_k)
        return {
            "query": arguments.description,
            "match_count": len(matches),
            "matches": [match.model_dump(mode="json") for match in matches],
        }

    def get_machine_health(self, arguments: MachineArguments) -> Mapping[str, Any]:
        return self._operations.get_machine_health(arguments.machine_id)


_TOOL_DESCRIPTIONS: dict[InvestigationToolName, str] = {
    InvestigationToolName.GET_MACHINE_STATUS: (
        "Return the machine identity, location, model, and latest controller state."
    ),
    InvestigationToolName.GET_RECENT_BREWS: (
        "Return recent drink cycles and count brews with quality or safety warnings."
    ),
    InvestigationToolName.GET_SUPPLY_LEVELS: (
        "Return current water, coffee-bean, and milk levels from ingredient sensors."
    ),
    InvestigationToolName.GET_SENSOR_ALERTS: (
        "Return recent bounded sensor alerts and their coffee-machine error codes."
    ),
    InvestigationToolName.GET_TEMPERATURE_HISTORY: (
        "Return recent boiler temperature and pressure readings."
    ),
    InvestigationToolName.GET_CLEANING_STATUS: (
        "Return cycles since cleaning and the latest maintenance event."
    ),
    InvestigationToolName.SEARCH_APPLICATION_LOGS: (
        "Search bounded coffee-machine logs for one machine and optional keywords."
    ),
    InvestigationToolName.SEARCH_SIMILAR_INCIDENTS: (
        "Return semantically similar historical coffee-machine incidents."
    ),
    InvestigationToolName.GET_MACHINE_HEALTH: (
        "Return a compact deterministic overview of machine health indicators."
    ),
}


def build_investigation_tool_registry(
    operations: InvestigationOperations,
    logs: StructuredLogSearcher,
    incidents: HistoricalIncidentRetriever,
    *,
    max_log_results: int = 20,
    max_similar_incidents: int = 3,
) -> ToolRegistry:
    """Register exactly the nine tools the detective may request."""

    handlers = IncidentToolHandlers(
        operations,
        logs,
        incidents,
        max_log_results=max_log_results,
        max_similar_incidents=max_similar_incidents,
    )
    machine_tools = (
        InvestigationToolName.GET_MACHINE_STATUS,
        InvestigationToolName.GET_RECENT_BREWS,
        InvestigationToolName.GET_SUPPLY_LEVELS,
        InvestigationToolName.GET_SENSOR_ALERTS,
        InvestigationToolName.GET_TEMPERATURE_HISTORY,
        InvestigationToolName.GET_CLEANING_STATUS,
        InvestigationToolName.GET_MACHINE_HEALTH,
    )
    tools = [
        SafeTool(
            name=name.value,
            description=_TOOL_DESCRIPTIONS[name],
            arguments_model=MachineArguments,
            handler=cast(Any, getattr(handlers, name.value)),
        )
        for name in machine_tools
    ]
    tools.extend(
        (
            SafeTool(
                name=InvestigationToolName.SEARCH_APPLICATION_LOGS.value,
                description=_TOOL_DESCRIPTIONS[InvestigationToolName.SEARCH_APPLICATION_LOGS],
                arguments_model=ApplicationLogSearchArguments,
                handler=cast(Any, handlers.search_application_logs),
            ),
            SafeTool(
                name=InvestigationToolName.SEARCH_SIMILAR_INCIDENTS.value,
                description=_TOOL_DESCRIPTIONS[InvestigationToolName.SEARCH_SIMILAR_INCIDENTS],
                arguments_model=SimilarIncidentSearchArguments,
                handler=cast(Any, handlers.search_similar_incidents),
            ),
        )
    )
    return ToolRegistry(tools)

"""Replaceable evidence diagnosticians."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from incident_assistant.domain.models import Diagnosis, Evidence, EvidenceSeverity
from incident_assistant.domain.ports import Diagnostician


class RuleBasedDiagnostician:
    """Deterministic fallback that keeps the project useful without an API key."""

    def diagnose(
        self,
        incident_description: str,
        machine_id: str,
        evidence: Sequence[Evidence],
    ) -> Diagnosis:
        del incident_description, machine_id
        by_code = {item.code: item for item in evidence}
        supplies = by_code.get("machine.supplies")
        thermal = by_code.get("machine.temperature")
        cleaning = by_code.get("machine.cleaning")
        alert = by_code.get("machine.alert")
        beans = self._attribute_float(supplies, "beans")
        milk = self._attribute_float(supplies, "milk")
        temperature = self._attribute_float(thermal, "temperature_c")
        cycles_since_cleaning = self._attribute_int(cleaning, "cycles_since_cleaning")
        error_code = str(alert.attributes.get("error_code", "")) if alert else ""
        if error_code == "OVERHEAT" or temperature >= 96:
            diagnosis = (
                "Brewing system overheating",
                "The boiler is above its safe operating temperature and brews may abort.",
                (
                    "Stop new brews and allow the machine to cool before inspection.",
                    "Check the ventilation path, heater relay, and thermostat before a test brew.",
                ),
                0.97,
            )
        elif error_code == "MILK_LINE_DISCONNECTED" or milk <= 2:
            diagnosis = (
                "Milk line disconnected or milk supply empty",
                "The milk system cannot detect usable milk, so foam-based drinks fail.",
                (
                    "Refill or reconnect the milk container and inspect the intake tube.",
                    "Prime the milk line, then prepare one test cappuccino.",
                ),
                0.95,
            )
        elif error_code == "LOW_BEANS" or 0 < beans <= 5:
            diagnosis = (
                "Coffee bean hopper nearly empty",
                "Too few beans are reaching the grinder, producing fast and watery extractions.",
                (
                    "Refill the bean hopper with the approved coffee beans.",
                    "Run one calibration espresso and confirm extraction time returns to normal.",
                ),
                0.96,
            )
        elif error_code == "CLEANING_OVERDUE" or cycles_since_cleaning >= 200:
            diagnosis = (
                "Cleaning cycle overdue",
                "The brew group has exceeded its cleaning interval, which can make drinks bitter.",
                (
                    "Run the manufacturer-approved cleaning and descaling program.",
                    "Discard the first rinse drink and verify taste with a fresh test brew.",
                ),
                0.94,
            )
        elif error_code:
            diagnosis = (
                "Coffee machine alert requires inspection",
                f"The latest sensor reading reports {error_code} without a known rule match.",
                (
                    "Inspect the matching sensor and the machine service guide.",
                    "Run one controlled test brew after correcting the alert.",
                ),
                0.72,
            )
        else:
            diagnosis = (
                "No machine fault detected",
                "Supplies, temperature, cleaning state, alerts, and brew telemetry look normal.",
                (
                    "Confirm the selected drink recipe and cup size.",
                    "If the symptom returns, record the brew time and inspect that cycle.",
                ),
                0.82,
            )

        notable = [
            item.summary
            for item in evidence
            if item.severity in (EvidenceSeverity.ERROR, EvidenceSeverity.WARNING)
        ]
        if not notable:
            notable = [item.summary for item in evidence[:3]]
        return Diagnosis(
            root_cause=diagnosis[0],
            explanation=diagnosis[1],
            supporting_evidence=tuple(notable[:5]),
            recommended_actions=diagnosis[2],
            confidence=diagnosis[3],
            generated_by="deterministic rules",
        )

    @staticmethod
    def _attribute_int(item: Evidence | None, name: str) -> int:
        if item is None:
            return 0
        try:
            return int(item.attributes.get(name, 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _attribute_float(item: Evidence | None, name: str) -> float:
        if item is None:
            return 0.0
        try:
            return float(item.attributes.get(name, 0.0))
        except (TypeError, ValueError):
            return 0.0


class OpenAIDiagnostician:
    """OpenAI Responses API adapter; all SDK details stay outside the use case."""

    _instructions = """You are a coffee-machine incident diagnostician.
Use only the supplied evidence. Distinguish observed facts from inference.
Return one JSON object with keys: root_cause (string), explanation (string),
supporting_evidence (array of strings), recommended_actions (array of strings),
and confidence (number from 0 to 1). Do not include Markdown fences."""

    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def diagnose(
        self,
        incident_description: str,
        machine_id: str,
        evidence: Sequence[Evidence],
    ) -> Diagnosis:
        payload = {
            "incident": incident_description,
            "machine_id": machine_id,
            "evidence": [self._serialize_evidence(item) for item in evidence],
        }
        response = self._client.responses.create(
            model=self._model,
            instructions=self._instructions,
            input=json.dumps(payload, indent=2),
        )
        parsed = self._parse_json(response.output_text)
        return Diagnosis(
            root_cause=self._required_text(parsed, "root_cause"),
            explanation=self._required_text(parsed, "explanation"),
            supporting_evidence=self._string_list(parsed, "supporting_evidence"),
            recommended_actions=self._string_list(parsed, "recommended_actions"),
            confidence=float(parsed.get("confidence", 0.5)),
            generated_by=f"OpenAI {self._model}",
        )

    @staticmethod
    def _serialize_evidence(item: Evidence) -> dict[str, Any]:
        return {
            "code": item.code,
            "source": item.source,
            "summary": item.summary,
            "severity": item.severity.value,
            "details": list(item.details),
            "attributes": dict(item.attributes),
        }

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("The model did not return a JSON object.")
        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("The model response must be a JSON object.")
        return parsed

    @staticmethod
    def _required_text(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"The model response is missing {key}.")
        return value.strip()

    @staticmethod
    def _string_list(payload: dict[str, Any], key: str) -> tuple[str, ...]:
        value = payload.get(key)
        if not isinstance(value, list):
            raise ValueError(f"The model response is missing {key}.")
        strings = tuple(str(item).strip() for item in value if str(item).strip())
        if not strings:
            raise ValueError(f"The model response contains no values for {key}.")
        return strings


class ResilientDiagnostician:
    """Falls back to a local diagnostician when a remote provider fails."""

    def __init__(self, primary: Diagnostician, fallback: Diagnostician) -> None:
        self._primary = primary
        self._fallback = fallback

    def diagnose(
        self,
        incident_description: str,
        machine_id: str,
        evidence: Sequence[Evidence],
    ) -> Diagnosis:
        try:
            return self._primary.diagnose(incident_description, machine_id, evidence)
        except Exception as exc:
            fallback_result = self._fallback.diagnose(incident_description, machine_id, evidence)
            return replace(
                fallback_result,
                generated_by=(
                    f"{fallback_result.generated_by} "
                    f"(remote diagnosis unavailable: {type(exc).__name__})"
                ),
            )

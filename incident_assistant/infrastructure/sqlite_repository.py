"""Read-only SQLite adapter for coffee-machine telemetry and incident history."""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from pathlib import Path

from incident_assistant.domain.agent_models import HistoricalIncidentDocument
from incident_assistant.domain.models import Evidence, EvidenceSeverity


class SqliteRepository:
    """Expose bounded, parameterized coffee-machine investigation queries."""

    _stop_words = {
        "about",
        "after",
        "coffee",
        "from",
        "have",
        "incident",
        "machine",
        "that",
        "their",
        "there",
        "they",
        "this",
        "with",
    }

    def __init__(self, database_path: Path, timeout_seconds: float = 5.0) -> None:
        self._database_path = Path(database_path)
        self._timeout_seconds = timeout_seconds

    def _connect(self) -> sqlite3.Connection:
        if not self._database_path.is_file():
            raise FileNotFoundError(
                f"Database not found at {self._database_path}. Run seed_data.py first."
            )
        connection = sqlite3.connect(
            f"file:{self._database_path.as_posix()}?mode=ro",
            uri=True,
            timeout=self._timeout_seconds,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def check_brews(self, machine_id: str) -> tuple[Evidence, ...]:
        with closing(self._connect()) as connection:
            totals = connection.execute(
                """
                SELECT COUNT(*) AS total_brews,
                       COALESCE(SUM(CASE WHEN status != 'COMPLETED' THEN 1 ELSE 0 END), 0)
                           AS abnormal_brews
                FROM brew_cycles
                WHERE machine_id = ?
                """,
                (machine_id,),
            ).fetchone()
        total = int(totals["total_brews"])
        abnormal = int(totals["abnormal_brews"])
        return (
            Evidence(
                code="brews.abnormal",
                source="brew database",
                summary=f"{abnormal} of {total} brew cycles have a quality or safety warning.",
                severity=EvidenceSeverity.ERROR if abnormal else EvidenceSeverity.INFO,
                attributes={"count": abnormal, "total": total},
            ),
        )

    def check_machine_status(self, machine_id: str) -> tuple[Evidence, ...]:
        with closing(self._connect()) as connection:
            latest = connection.execute(
                """
                SELECT recorded_at, water_level_pct, bean_level_pct, milk_level_pct,
                       temperature_c, pressure_bar, cleaning_cycles_since, status,
                       error_code, error_message
                FROM sensor_readings
                WHERE machine_id = ?
                ORDER BY recorded_at DESC
                LIMIT 1
                """,
                (machine_id,),
            ).fetchone()
        if latest is None:
            return (
                Evidence(
                    code="machine.not_found",
                    source="sensor database",
                    summary=f"No telemetry was found for {machine_id}.",
                    severity=EvidenceSeverity.WARNING,
                ),
            )

        beans = float(latest["bean_level_pct"])
        milk = float(latest["milk_level_pct"])
        water = float(latest["water_level_pct"])
        temperature = float(latest["temperature_c"])
        cleaning_cycles = int(latest["cleaning_cycles_since"])
        error_code = str(latest["error_code"] or "")
        error_message = str(latest["error_message"] or "")
        supply_warning = beans <= 5 or milk <= 2 or water <= 5
        return (
            Evidence(
                code="machine.supplies",
                source="ingredient sensors",
                summary=(
                    f"Supply levels: beans {beans:.0f}%, milk {milk:.0f}%, water {water:.0f}%."
                ),
                severity=EvidenceSeverity.ERROR if supply_warning else EvidenceSeverity.INFO,
                attributes={"beans": beans, "milk": milk, "water": water},
            ),
            Evidence(
                code="machine.temperature",
                source="thermal sensors",
                summary=(
                    f"Boiler temperature is {temperature:.1f} C at "
                    f"{float(latest['pressure_bar']):.1f} bar."
                ),
                severity=EvidenceSeverity.ERROR if temperature >= 96 else EvidenceSeverity.INFO,
                attributes={
                    "temperature_c": temperature,
                    "pressure_bar": float(latest["pressure_bar"]),
                },
            ),
            Evidence(
                code="machine.cleaning",
                source="maintenance counter",
                summary=f"{cleaning_cycles} brew cycles have elapsed since cleaning.",
                severity=(
                    EvidenceSeverity.WARNING if cleaning_cycles >= 200 else EvidenceSeverity.INFO
                ),
                attributes={
                    "cycles_since_cleaning": cleaning_cycles,
                    "overdue": cleaning_cycles >= 200,
                },
            ),
            Evidence(
                code="machine.alert",
                source="sensor database",
                summary=(
                    f"Active alert: {error_code} — {error_message}"
                    if error_code
                    else "No active machine alert."
                ),
                severity=EvidenceSeverity.ERROR if error_code else EvidenceSeverity.INFO,
                attributes={"error_code": error_code, "status": str(latest["status"])},
            ),
        )

    def get_machine_status(self, machine_id: str) -> dict[str, object]:
        with closing(self._connect()) as connection:
            machine = connection.execute(
                """
                SELECT nickname, location, model, active
                FROM machines WHERE machine_id = ?
                """,
                (machine_id,),
            ).fetchone()
            latest = connection.execute(
                """
                SELECT recorded_at, status, error_code, error_message,
                       temperature_c, pressure_bar
                FROM sensor_readings WHERE machine_id = ?
                ORDER BY recorded_at DESC LIMIT 1
                """,
                (machine_id,),
            ).fetchone()
        return {
            "machine_id": machine_id,
            "found": machine is not None,
            "nickname": str(machine["nickname"]) if machine else None,
            "location": str(machine["location"]) if machine else None,
            "model": str(machine["model"]) if machine else None,
            "active": bool(machine["active"]) if machine else None,
            "latest_reading": dict(latest) if latest else None,
        }

    def get_recent_brews(self, machine_id: str, limit: int = 20) -> dict[str, object]:
        bounded_limit = max(1, min(limit, 100))
        with closing(self._connect()) as connection:
            abnormal_count = connection.execute(
                """
                SELECT COUNT(*) FROM brew_cycles
                WHERE machine_id = ? AND status != 'COMPLETED'
                """,
                (machine_id,),
            ).fetchone()[0]
            rows = connection.execute(
                """
                SELECT brew_id, brewed_at, drink_type, status, duration_seconds,
                       water_ml, beans_g, temperature_c
                FROM brew_cycles WHERE machine_id = ?
                ORDER BY brewed_at DESC LIMIT ?
                """,
                (machine_id, bounded_limit),
            ).fetchall()
        return {
            "machine_id": machine_id,
            "abnormal_brew_count": int(abnormal_count),
            "returned_count": len(rows),
            "brews": [dict(row) for row in rows],
        }

    def get_supply_levels(self, machine_id: str) -> dict[str, object]:
        latest = self._latest_sensor(machine_id)
        if latest is None:
            return {"machine_id": machine_id, "found": False}
        supplies = {
            "water_level_pct": float(latest["water_level_pct"]),
            "bean_level_pct": float(latest["bean_level_pct"]),
            "milk_level_pct": float(latest["milk_level_pct"]),
        }
        return {
            "machine_id": machine_id,
            "found": True,
            "recorded_at": latest["recorded_at"],
            **supplies,
            "low_supplies": [
                name.replace("_level_pct", "") for name, value in supplies.items() if value <= 5
            ],
        }

    def get_sensor_alerts(self, machine_id: str, limit: int = 10) -> dict[str, object]:
        bounded_limit = max(1, min(limit, 20))
        with closing(self._connect()) as connection:
            count = connection.execute(
                """
                SELECT COUNT(*) FROM sensor_readings
                WHERE machine_id = ? AND error_code IS NOT NULL
                """,
                (machine_id,),
            ).fetchone()[0]
            rows = connection.execute(
                """
                SELECT recorded_at, status, error_code, error_message
                FROM sensor_readings
                WHERE machine_id = ? AND error_code IS NOT NULL
                ORDER BY recorded_at DESC LIMIT ?
                """,
                (machine_id, bounded_limit),
            ).fetchall()
        return {
            "machine_id": machine_id,
            "alert_count": int(count),
            "returned_count": len(rows),
            "alerts": [dict(row) for row in rows],
        }

    def get_temperature_history(self, machine_id: str, limit: int = 10) -> dict[str, object]:
        bounded_limit = max(1, min(limit, 20))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT recorded_at, temperature_c, pressure_bar
                FROM sensor_readings WHERE machine_id = ?
                ORDER BY recorded_at DESC LIMIT ?
                """,
                (machine_id, bounded_limit),
            ).fetchall()
        temperatures = [float(row["temperature_c"]) for row in rows]
        return {
            "machine_id": machine_id,
            "reading_count": len(rows),
            "maximum_temperature_c": max(temperatures) if temperatures else None,
            "readings": [dict(row) for row in rows],
        }

    def get_cleaning_status(self, machine_id: str) -> dict[str, object]:
        latest = self._latest_sensor(machine_id)
        with closing(self._connect()) as connection:
            maintenance = connection.execute(
                """
                SELECT performed_at, event_type, technician_note, resolved
                FROM maintenance_events WHERE machine_id = ?
                ORDER BY performed_at DESC LIMIT 1
                """,
                (machine_id,),
            ).fetchone()
        cycles = int(latest["cleaning_cycles_since"]) if latest else None
        return {
            "machine_id": machine_id,
            "cycles_since_cleaning": cycles,
            "cleaning_due": cycles is not None and cycles >= 200,
            "latest_maintenance": dict(maintenance) if maintenance else None,
        }

    def get_machine_health(self, machine_id: str) -> dict[str, object]:
        status = self.get_machine_status(machine_id)
        if not status["found"]:
            return {"machine_id": machine_id, "health": "not_found", "indicators": []}
        supplies = self.get_supply_levels(machine_id)
        alerts = self.get_sensor_alerts(machine_id)
        cleaning = self.get_cleaning_status(machine_id)
        temperature = self.get_temperature_history(machine_id)
        indicators: list[str] = []
        if supplies.get("low_supplies"):
            indicators.append("low supplies: " + ", ".join(supplies["low_supplies"]))
        if alerts["alert_count"]:
            indicators.append(f"{alerts['alert_count']} sensor alerts")
        if cleaning["cleaning_due"]:
            indicators.append("cleaning overdue")
        maximum = temperature["maximum_temperature_c"]
        if maximum is not None and float(maximum) >= 96:
            indicators.append("unsafe temperature")
        return {
            "machine_id": machine_id,
            "health": "attention_required" if indicators else "healthy",
            "indicators": indicators,
            "nickname": status["nickname"],
            "location": status["location"],
        }

    def find_similar(
        self,
        incident_description: str,
        machine_id: str,
        limit: int = 3,
    ) -> tuple[Evidence, ...]:
        query_tokens = self._tokens(incident_description)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT incident_id, machine_id, title, description, root_cause,
                       resolution, error_codes, created_at
                FROM incidents
                ORDER BY created_at DESC
                """
            ).fetchall()
        ranked = sorted(
            (
                (
                    len(query_tokens & self._tokens(f"{row['title']} {row['description']}")),
                    row,
                )
                for row in rows
            ),
            key=lambda item: (item[0], str(item[1]["created_at"])),
            reverse=True,
        )
        matches = [item for item in ranked if item[0] > 0][: max(1, min(limit, 10))]
        if not matches:
            return (
                Evidence(
                    code="incidents.no_match",
                    source="incident history",
                    summary=f"No similar historical coffee incident was found for {machine_id}.",
                ),
            )
        return tuple(
            Evidence(
                code="incidents.similar",
                source="incident history",
                summary=(f"{row['incident_id']} on {row['machine_id']}: {row['root_cause']}"),
                severity=EvidenceSeverity.WARNING,
                details=(str(row["description"]), str(row["resolution"])),
                attributes={"score": score, "incident_id": row["incident_id"]},
            )
            for score, row in matches
        )

    def list_incident_documents(self) -> tuple[HistoricalIncidentDocument, ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT incident_id, machine_id, title, description, root_cause,
                       resolution, error_codes, service, severity
                FROM incidents ORDER BY incident_id
                """
            ).fetchall()
        return tuple(
            HistoricalIncidentDocument(
                incident_id=str(row["incident_id"]),
                machine_id=str(row["machine_id"]),
                title=str(row["title"]),
                description=str(row["description"]),
                root_cause=str(row["root_cause"]),
                resolution=str(row["resolution"]),
                error_codes=tuple(
                    code.strip() for code in str(row["error_codes"]).split(",") if code.strip()
                ),
                service=str(row["service"]),
                severity=str(row["severity"]),
            )
            for row in rows
        )

    def _latest_sensor(self, machine_id: str) -> sqlite3.Row | None:
        with closing(self._connect()) as connection:
            return connection.execute(
                """
                SELECT recorded_at, water_level_pct, bean_level_pct, milk_level_pct,
                       temperature_c, pressure_bar, cleaning_cycles_since,
                       status, error_code, error_message
                FROM sensor_readings WHERE machine_id = ?
                ORDER BY recorded_at DESC LIMIT 1
                """,
                (machine_id,),
            ).fetchone()

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9_]+", text.casefold())
            if len(token) > 2 and token not in cls._stop_words
        }

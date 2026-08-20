"""Create deterministic synthetic coffee-machine data for the demo application."""

from __future__ import annotations

import argparse
import csv
import random
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from incident_assistant.config import PROJECT_ROOT

MACHINES = (
    ("CM-1001", "Lobby Latte", "Main lobby", "BrewBot Mini", "low_beans"),
    ("CM-1002", "Breakroom Barista", "Second-floor breakroom", "FoamMaster", "milk_fault"),
    ("CM-1003", "Executive Espresso", "Executive lounge", "ThermoShot", "overheating"),
    ("CM-1004", "Lab Cappuccino", "Innovation lab", "CleanBrew X", "cleaning_overdue"),
    ("CM-1005", "Reception Roast", "Reception", "BrewBot Mini", "normal"),
    *(
        (
            f"CM-{number:04d}",
            f"Coffee Companion {number}",
            f"Office zone {number - 1000}",
            "BrewBot Mini",
            "normal",
        )
        for number in range(1006, 1021)
    ),
)

HISTORICAL_INCIDENTS = (
    (
        "CAF-0001",
        "CM-1001",
        "Espresso turned suspiciously watery",
        "Recent espresso shots were pale and fast while the bean sensor fell below five percent.",
        "Coffee bean hopper nearly empty",
        "Refilled the bean hopper and ran one calibration espresso.",
        "LOW_BEANS",
        "ingredient-sensors",
        "medium",
        "2026-07-12 09:30:00",
    ),
    (
        "CAF-0002",
        "CM-1002",
        "Cappuccino forgot how to foam",
        "Cappuccino cycles completed without foam after the milk line disconnected.",
        "Milk line disconnected or milk supply empty",
        "Reconnected and primed the milk line, then tested one cappuccino.",
        "MILK_LINE_DISCONNECTED",
        "milk-system",
        "medium",
        "2026-07-18 14:15:00",
    ),
    (
        "CAF-0003",
        "CM-1003",
        "Espresso machine running hotter than office gossip",
        "Boiler readings exceeded 99 C and several brew cycles aborted for safety.",
        "Brewing system overheating",
        "Powered down the heater, cleared the vent, and inspected the thermostat.",
        "OVERHEAT",
        "thermal-control",
        "high",
        "2026-07-24 11:45:00",
    ),
    (
        "CAF-0004",
        "CM-1004",
        "Coffee tastes like a science experiment",
        "The machine exceeded its cleaning interval and recent drinks carried bitter warnings.",
        "Cleaning cycle overdue",
        "Ran the approved cleaning cycle and replaced the brew-group tablet.",
        "CLEANING_OVERDUE",
        "maintenance",
        "medium",
        "2026-08-02 16:20:00",
    ),
    (
        "CAF-0005",
        "CM-1005",
        "Reception coffee false alarm",
        "Supplies, temperature, pressure, cleaning state, and recent brews were all healthy.",
        "No machine fault detected",
        "Confirmed the selected drink recipe and served another test coffee.",
        "HEALTHY",
        "machine-health",
        "low",
        "2026-08-07 10:00:00",
    ),
)

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE machines (
    machine_id TEXT PRIMARY KEY,
    nickname TEXT NOT NULL,
    location TEXT NOT NULL,
    model TEXT NOT NULL,
    active INTEGER NOT NULL CHECK (active IN (0, 1))
);

CREATE TABLE brew_cycles (
    brew_id TEXT PRIMARY KEY,
    machine_id TEXT NOT NULL REFERENCES machines(machine_id),
    brewed_at TEXT NOT NULL,
    drink_type TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_seconds REAL NOT NULL,
    water_ml REAL NOT NULL,
    beans_g REAL NOT NULL,
    temperature_c REAL NOT NULL
);

CREATE INDEX idx_brews_machine_time
ON brew_cycles(machine_id, brewed_at DESC);

CREATE TABLE sensor_readings (
    reading_id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id TEXT NOT NULL REFERENCES machines(machine_id),
    recorded_at TEXT NOT NULL,
    water_level_pct REAL NOT NULL,
    bean_level_pct REAL NOT NULL,
    milk_level_pct REAL NOT NULL,
    temperature_c REAL NOT NULL,
    pressure_bar REAL NOT NULL,
    cleaning_cycles_since INTEGER NOT NULL,
    status TEXT NOT NULL,
    error_code TEXT,
    error_message TEXT
);

CREATE INDEX idx_sensors_machine_time
ON sensor_readings(machine_id, recorded_at DESC);

CREATE TABLE maintenance_events (
    maintenance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id TEXT NOT NULL REFERENCES machines(machine_id),
    performed_at TEXT NOT NULL,
    event_type TEXT NOT NULL,
    technician_note TEXT NOT NULL,
    resolved INTEGER NOT NULL CHECK (resolved IN (0, 1))
);

CREATE TABLE incidents (
    incident_id TEXT PRIMARY KEY,
    machine_id TEXT NOT NULL REFERENCES machines(machine_id),
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    root_cause TEXT NOT NULL,
    resolution TEXT NOT NULL,
    error_codes TEXT NOT NULL,
    service TEXT NOT NULL,
    severity TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_incidents_machine_time
ON incidents(machine_id, created_at DESC);
"""


def seed(output_dir: Path, brews_per_machine: int = 500) -> tuple[Path, Path, Path]:
    """Create the SQLite database, machine log, and historical-incident CSV."""

    if brews_per_machine < 50:
        raise ValueError("brews_per_machine must be at least 50")

    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "support.db"
    temporary_database = output_dir / "support.db.tmp"
    log_path = output_dir / "app_logs.txt"
    incident_csv_path = output_dir / "incidents.csv"
    if temporary_database.exists():
        temporary_database.unlink()

    randomizer = random.Random(42)
    base_time = datetime(2026, 8, 19, 8, 0, tzinfo=UTC)
    with closing(sqlite3.connect(temporary_database)) as connection:
        connection.executescript(SCHEMA)
        connection.executemany(
            """
            INSERT INTO machines(machine_id, nickname, location, model, active)
            VALUES (?, ?, ?, ?, 1)
            """,
            (
                (identifier, nickname, location, model)
                for identifier, nickname, location, model, _ in MACHINES
            ),
        )
        connection.executemany(
            """
            INSERT INTO brew_cycles(
                brew_id, machine_id, brewed_at, drink_type, status,
                duration_seconds, water_ml, beans_g, temperature_c
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _brew_cycles(randomizer, base_time, brews_per_machine),
        )
        connection.executemany(
            """
            INSERT INTO sensor_readings(
                machine_id, recorded_at, water_level_pct, bean_level_pct,
                milk_level_pct, temperature_c, pressure_bar, cleaning_cycles_since,
                status, error_code, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _sensor_readings(base_time),
        )
        connection.executemany(
            """
            INSERT INTO maintenance_events(
                machine_id, performed_at, event_type, technician_note, resolved
            ) VALUES (?, ?, ?, ?, ?)
            """,
            _maintenance_events(base_time),
        )
        connection.executemany(
            """
            INSERT INTO incidents(
                incident_id, machine_id, title, description, root_cause,
                resolution, error_codes, service, severity, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'RESOLVED', ?)
            """,
            HISTORICAL_INCIDENTS,
        )
        connection.commit()

    temporary_database.replace(database_path)
    log_path.write_text("\n".join(_log_lines(base_time)) + "\n", encoding="utf-8")
    with incident_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            (
                "incident_id",
                "machine_id",
                "title",
                "description",
                "root_cause",
                "resolution",
                "error_codes",
                "service",
                "severity",
                "created_at",
            )
        )
        writer.writerows(HISTORICAL_INCIDENTS)
    return database_path, log_path, incident_csv_path


def _brew_cycles(
    randomizer: random.Random,
    base_time: datetime,
    brews_per_machine: int,
):
    drinks = ("ESPRESSO", "AMERICANO", "CAPPUCCINO", "LATTE")
    for machine_index, (machine_id, _, _, _, scenario) in enumerate(MACHINES):
        for index in range(brews_per_machine):
            recent_index = brews_per_machine - index
            drink_type = drinks[index % len(drinks)]
            status = "COMPLETED"
            duration_seconds = randomizer.uniform(24.0, 38.0)
            water_ml = randomizer.uniform(30.0, 220.0)
            beans_g = randomizer.uniform(7.0, 10.0)
            temperature_c = randomizer.uniform(89.0, 93.0)
            if scenario == "low_beans" and recent_index <= 34:
                drink_type = "ESPRESSO"
                status = "WATERY"
                duration_seconds = randomizer.uniform(14.0, 19.0)
                beans_g = randomizer.uniform(1.5, 3.0)
            elif scenario == "milk_fault" and recent_index <= 20:
                drink_type = "CAPPUCCINO"
                status = "NO_FOAM"
            elif scenario == "overheating" and recent_index <= 15:
                status = "ABORTED_OVERHEAT"
                temperature_c = randomizer.uniform(98.0, 102.0)
            elif scenario == "cleaning_overdue" and recent_index <= 40:
                status = "BITTER_WARNING"

            brewed_at = base_time - timedelta(minutes=recent_index + machine_index * 3)
            yield (
                f"BRW-{machine_index:02d}-{index:06d}",
                machine_id,
                brewed_at.strftime("%Y-%m-%d %H:%M:%S"),
                drink_type,
                status,
                round(duration_seconds, 1),
                round(water_ml, 1),
                round(beans_g, 1),
                round(temperature_c, 1),
            )


def _sensor_readings(base_time: datetime):
    for machine_id, _, _, _, scenario in MACHINES:
        for sample in range(6):
            water = 82 - sample * 3
            beans = 74 - sample * 4
            milk = 68 - sample * 3
            temperature = 91.0 + sample * 0.2
            pressure = 9.1
            cleaning_cycles = 80 + sample
            status = "OK"
            error_code = None
            error_message = None
            if scenario == "low_beans":
                beans = (18, 12, 8, 5, 3, 2)[sample]
                if beans <= 5:
                    status = "WARNING"
                    error_code = "LOW_BEANS"
                    error_message = "Bean hopper below five percent"
            elif scenario == "milk_fault":
                milk = (15, 10, 5, 2, 0, 0)[sample]
                if milk <= 2:
                    status = "ERROR"
                    error_code = "MILK_LINE_DISCONNECTED"
                    error_message = "Milk line pressure not detected"
            elif scenario == "overheating":
                temperature = (91.0, 92.0, 93.0, 96.0, 99.0, 101.0)[sample]
                if temperature >= 96.0:
                    status = "ERROR"
                    error_code = "OVERHEAT"
                    error_message = "Boiler temperature above safe threshold"
            elif scenario == "cleaning_overdue":
                cleaning_cycles = 196 + sample * 5
                if cleaning_cycles >= 200:
                    status = "WARNING"
                    error_code = "CLEANING_OVERDUE"
                    error_message = "Cleaning interval exceeded"
            yield (
                machine_id,
                (base_time + timedelta(minutes=sample * 10)).strftime("%Y-%m-%d %H:%M:%S"),
                float(water),
                float(beans),
                float(milk),
                float(temperature),
                pressure,
                cleaning_cycles,
                status,
                error_code,
                error_message,
            )


def _maintenance_events(base_time: datetime):
    for machine_id, _, _, _, scenario in MACHINES:
        days_ago = 75 if scenario == "cleaning_overdue" else 20
        note = (
            "Descaling postponed; cleaning tablet unavailable."
            if scenario == "cleaning_overdue"
            else "Routine cleaning and visual inspection completed."
        )
        yield (
            machine_id,
            (base_time - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S"),
            "CLEANING",
            note,
            0 if scenario == "cleaning_overdue" else 1,
        )


def _log_lines(base_time: datetime) -> list[str]:
    lines: list[str] = []
    for machine_id, _, _, _, scenario in MACHINES:
        timestamp = base_time.strftime("%Y-%m-%d %H:%M")
        lines.append(f"{timestamp} INFO  {machine_id} Machine heartbeat received")
        lines.append(f"{timestamp} INFO  {machine_id} Brew controller ready")
        if scenario == "low_beans":
            lines.extend(
                (
                    f"{timestamp} WARN  {machine_id} LOW_BEANS hopper at 2 percent",
                    f"{timestamp} WARN  {machine_id} Espresso extraction completed too quickly",
                    f"{timestamp} ERROR {machine_id} WATERY_DRINK quality threshold missed",
                )
            )
        elif scenario == "milk_fault":
            lines.extend(
                (
                    f"{timestamp} WARN  {machine_id} Milk level sensor reports empty",
                    f"{timestamp} ERROR {machine_id} MILK_LINE_DISCONNECTED",
                    f"{timestamp} ERROR {machine_id} Cappuccino foam cycle failed",
                )
            )
        elif scenario == "overheating":
            lines.extend(
                (
                    f"{timestamp} WARN  {machine_id} Boiler temperature rising",
                    f"{timestamp} ERROR {machine_id} OVERHEAT temperature 101 C",
                    f"{timestamp} ERROR {machine_id} Brew cycle aborted for safety",
                )
            )
        elif scenario == "cleaning_overdue":
            lines.extend(
                (
                    f"{timestamp} WARN  {machine_id} CLEANING_OVERDUE 221 cycles",
                    f"{timestamp} WARN  {machine_id} Brew-group residue threshold exceeded",
                    f"{timestamp} INFO  {machine_id} Cleaning reminder displayed",
                )
            )
        else:
            lines.extend(
                (
                    f"{timestamp} INFO  {machine_id} Supplies within normal range",
                    f"{timestamp} INFO  {machine_id} Temperature stable at 92 C",
                    f"{timestamp} INFO  {machine_id} Health check passed",
                )
            )
    return sorted(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory for generated data (default: ./data)",
    )
    parser.add_argument(
        "--brews-per-machine",
        type=int,
        default=500,
        help="Brew cycles generated for each of the 20 machines",
    )
    args = parser.parse_args()
    database, logs, incidents = seed(args.output_dir, args.brews_per_machine)
    total_brews = len(MACHINES) * args.brews_per_machine
    print(f"Generated {total_brews:,} brew cycles in {database}")
    print(f"Generated machine logs in {logs}")
    print(f"Generated historical incidents in {incidents}")


if __name__ == "__main__":
    main()

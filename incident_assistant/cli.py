"""Command-line presentation adapter."""

from __future__ import annotations

import argparse
import json

from incident_assistant.bootstrap import (
    build_agent_investigation_service,
    ensure_demo_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Investigate a machine incident")
    parser.add_argument("incident", help="Incident description containing a machine ID")
    args = parser.parse_args()

    ensure_demo_data()
    report = build_agent_investigation_service().investigate(args.incident)
    print(json.dumps(report.model_dump(mode="json"), indent=2))

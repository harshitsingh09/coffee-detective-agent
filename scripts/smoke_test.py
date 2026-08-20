"""Run one Coffee Machine Detective investigation from the command line."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from incident_assistant.bootstrap import build_agent_investigation_service, ensure_demo_data
from incident_assistant.config import Settings


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "incident",
        nargs="?",
        default="CM-1001 is suddenly making watery espresso.",
    )
    parser.add_argument(
        "--rules",
        action="store_true",
        help="Use the deterministic fallback without calling an LLM.",
    )
    args = parser.parse_args()

    settings = Settings.from_environment()
    if args.rules:
        settings = replace(settings, enable_ai_agent=False)
    ensure_demo_data(settings)
    report = build_agent_investigation_service(settings).investigate(args.incident)

    print(f"Provider: {report.llm_provider or 'rules'}")
    print(f"Mode: {report.execution_mode.value}")
    print(f"Machine: {report.machine_id}")
    print(f"Status: {report.diagnosis.status.value}")
    print(f"Root cause: {report.diagnosis.root_cause}")
    print(f"Confidence: {report.diagnosis.confidence:.0%}")
    print(f"Tools: {', '.join(result.name for result in report.tool_results) or 'none'}")


if __name__ == "__main__":
    main()

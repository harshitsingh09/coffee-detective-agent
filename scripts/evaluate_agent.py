"""Run the saved evaluation dataset and write JSON and CSV reports."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from incident_assistant.bootstrap import ensure_demo_data
from incident_assistant.config import PROJECT_ROOT, Settings
from incident_assistant.evaluation.runner import EvaluationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval_cases.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "evaluation",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--ai",
        action="store_true",
        help="Require the selected provider key and enable the live bounded agent.",
    )
    mode.add_argument(
        "--rules",
        action="store_true",
        help="Force deterministic rule-based evaluation without calling an LLM.",
    )
    args = parser.parse_args()
    settings = Settings.from_environment()
    if args.ai:
        settings = replace(settings, enable_ai_agent=True)
        if not settings.selected_api_key:
            provider_key = f"{settings.llm_provider.value.upper()}_API_KEY"
            parser.error(f"--ai requires {provider_key} for the selected provider.")
    elif args.rules:
        settings = replace(settings, enable_ai_agent=False)
    ensure_demo_data(settings)
    runner = EvaluationRunner(settings)
    report = runner.evaluate(runner.load_cases(args.cases))
    json_path, csv_path = runner.save(report, args.output_dir)
    print(f"Mode: {report.mode}")
    print(f"Cases: {report.metrics.total_cases}")
    print(f"Root-cause accuracy: {report.metrics.root_cause_accuracy:.1%}")
    print(f"Status accuracy: {report.metrics.status_accuracy:.1%}")
    print(f"Tool precision: {report.metrics.tool_selection_precision:.1%}")
    print(f"Tool recall: {report.metrics.tool_selection_recall:.1%}")
    print(f"JSON report: {json_path}")
    print(f"CSV results: {csv_path}")


if __name__ == "__main__":
    main()

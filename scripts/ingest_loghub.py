"""Download or benchmark the small official LogHub HDFS_2k sample."""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from incident_assistant.config import PROJECT_ROOT
from incident_assistant.evaluation.loghub import benchmark_hdfs_log

LOGHUB_SAMPLE_URL = "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "external" / "loghub" / "HDFS_2k.log"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "evaluation" / "loghub_benchmark.json"


def download_sample(destination: Path) -> None:
    """Download the bounded 2,000-line official sample to its isolated directory."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        LOGHUB_SAMPLE_URL,
        headers={"User-Agent": "incident-ai-assistant-loghub-benchmark/1.0"},
    )
    temporary = destination.with_suffix(".tmp")
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        payload = response.read(2_000_000)
    if len(payload) >= 2_000_000:
        raise ValueError("The LogHub sample exceeded the 2 MB safety limit.")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-lines", type=int, default=2_000)
    parser.add_argument(
        "--download-sample",
        action="store_true",
        help="Download the official 2,000-line sample before benchmarking it.",
    )
    args = parser.parse_args()

    if args.download_sample:
        download_sample(args.input)
    if not args.input.exists():
        parser.error(f"Input not found: {args.input}. Use --download-sample or provide --input.")

    report = benchmark_hdfs_log(args.input, max_lines=args.max_lines)
    try:
        report["source_file"] = args.input.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        report["source_file"] = args.input.name
    report["generated_at"] = datetime.now(UTC).isoformat()
    report["source_url"] = LOGHUB_SAMPLE_URL
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Parsed {report['parsed_lines']:,}/{report['total_lines']:,} lines")
    print(f"Parse rate: {report['parse_rate']:.1%}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()

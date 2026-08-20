"""Lightweight parsing benchmark for a separately stored LogHub HDFS sample."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

HDFS_LINE_PATTERN = re.compile(
    r"^(?P<date>\d{6})\s+(?P<time>\d{6})\s+(?P<pid>\d+)\s+"
    r"(?P<level>[A-Z]+)\s+(?P<component>[^:]+):\s?(?P<message>.*)$"
)
BLOCK_ID_PATTERN = re.compile(r"blk_-?\d+")


def benchmark_hdfs_log(path: Path, *, max_lines: int = 2_000) -> dict[str, Any]:
    """Parse a bounded LogHub sample and return transparent throughput statistics."""

    if max_lines < 1:
        raise ValueError("max_lines must be positive")
    started = perf_counter()
    total_lines = 0
    parsed_lines = 0
    character_count = 0
    severities: Counter[str] = Counter()
    components: Counter[str] = Counter()
    block_ids: set[str] = set()

    with path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            if total_lines >= max_lines:
                break
            total_lines += 1
            character_count += len(line)
            match = HDFS_LINE_PATTERN.match(line.rstrip("\r\n"))
            if match is None:
                continue
            parsed_lines += 1
            severities[match.group("level")] += 1
            components[match.group("component")] += 1
            block_ids.update(BLOCK_ID_PATTERN.findall(match.group("message")))

    elapsed_ms = (perf_counter() - started) * 1_000
    parse_rate = parsed_lines / total_lines if total_lines else 0.0
    lines_per_second = total_lines / (elapsed_ms / 1_000) if elapsed_ms else 0.0
    return {
        "dataset": "LogHub HDFS_2k",
        "source_file": str(path),
        "max_lines": max_lines,
        "total_lines": total_lines,
        "parsed_lines": parsed_lines,
        "parse_rate": parse_rate,
        "severity_counts": dict(sorted(severities.items())),
        "unique_components": len(components),
        "unique_block_ids": len(block_ids),
        "average_line_length": character_count / total_lines if total_lines else 0.0,
        "elapsed_ms": elapsed_ms,
        "lines_per_second": lines_per_second,
        "scope_note": (
            "External LogHub data is benchmark-only and is never queried by the "
            "synthetic machine investigation tools."
        ),
    }

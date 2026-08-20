"""Plain-text application-log adapter."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from pathlib import Path

from incident_assistant.domain.models import Evidence, EvidenceSeverity


class FileLogRepository:
    """Searches a text log while bounding memory use for large files."""

    def __init__(self, log_path: Path, encoding: str = "utf-8") -> None:
        self._log_path = Path(log_path)
        self._encoding = encoding

    def search(self, machine_id: str, limit: int = 50) -> tuple[Evidence, ...]:
        if not self._log_path.is_file():
            raise FileNotFoundError(
                f"Log file not found at {self._log_path}. Run seed_data.py first."
            )

        matches: deque[str] = deque(maxlen=max(1, min(limit, 500)))
        needle = machine_id.casefold()
        with self._log_path.open("r", encoding=self._encoding) as log_file:
            for line in log_file:
                clean_line = line.rstrip("\r\n")
                if needle in clean_line.casefold():
                    matches.append(clean_line)

        lines = tuple(matches)
        error_count = sum(" ERROR " in f" {line} " for line in lines)
        timeout_count = sum("timeout" in line.casefold() for line in lines)
        if not lines:
            return (
                Evidence(
                    code="logs.no_matches",
                    source="application logs",
                    summary=f"No log entries matched {machine_id}.",
                    severity=EvidenceSeverity.WARNING,
                ),
            )

        return (
            Evidence(
                code="logs.matches",
                source="application logs",
                summary=(
                    f"Found {len(lines)} recent log entries, including "
                    f"{error_count} errors and {timeout_count} timeout messages."
                ),
                severity=(EvidenceSeverity.ERROR if error_count else EvidenceSeverity.INFO),
                details=lines,
                attributes={
                    "match_count": len(lines),
                    "error_count": error_count,
                    "timeout_count": timeout_count,
                },
            ),
        )

    def search_structured(
        self,
        machine_id: str,
        keywords: Sequence[str] = (),
        limit: int = 20,
    ) -> dict[str, object]:
        """Return bounded JSON-compatible matches for a safe agent tool."""

        if not self._log_path.is_file():
            raise FileNotFoundError(
                f"Log file not found at {self._log_path}. Run seed_data.py first."
            )

        bounded_limit = max(1, min(limit, 100))
        normalized_keywords = tuple(
            dict.fromkeys(keyword.strip().casefold() for keyword in keywords if keyword.strip())
        )
        matches: deque[str] = deque(maxlen=bounded_limit)
        needle = machine_id.casefold()
        with self._log_path.open("r", encoding=self._encoding) as log_file:
            for line in log_file:
                clean_line = line.rstrip("\r\n")
                normalized_line = clean_line.casefold()
                if needle not in normalized_line:
                    continue
                if normalized_keywords and not any(
                    keyword in normalized_line for keyword in normalized_keywords
                ):
                    continue
                matches.append(clean_line)

        entries = tuple(matches)
        return {
            "machine_id": machine_id,
            "keywords": list(normalized_keywords),
            "match_count": len(entries),
            "error_count": sum(" error " in f" {line.casefold()} " for line in entries),
            "timeout_count": sum("timeout" in line.casefold() for line in entries),
            "entries": list(entries),
        }

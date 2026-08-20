"""Machine identifier extraction."""

import re


class RegexMachineIdExtractor:
    """Extract IDs such as CM-1001 without coupling extraction to an LLM."""

    _pattern = re.compile(r"\bCM-\d{4}\b", re.IGNORECASE)

    def extract(self, incident_description: str) -> str | None:
        match = self._pattern.search(incident_description)
        return match.group(0).upper() if match else None

"""Errors exposed by the application layer."""


class InvestigationError(Exception):
    """Base error for an investigation request."""


class InvalidIncidentError(InvestigationError):
    """The incident description cannot be investigated."""


class DataSourceError(InvestigationError):
    """A required evidence source is unavailable."""

"""Exception types raised across relphot."""

from __future__ import annotations

__all__ = ["ConfigError", "IngestError", "MatchError", "RelphotError", "TilingError"]


class RelphotError(Exception):
    """Base class for every relphot-specific error."""


class ConfigError(RelphotError):
    """Raised for an invalid or unrecognised settings file."""


class IngestError(RelphotError):
    """Raised when a catalogue file cannot be read into a FrameCatalog."""


class MatchError(RelphotError):
    """Raised when cross-matching frames into a common star list fails."""


class TilingError(RelphotError):
    """Raised when the adaptive tiling loop cannot meet hard_min_ref_candidates."""

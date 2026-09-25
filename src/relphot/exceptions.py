"""Exception types raised across relphot."""

from __future__ import annotations

__all__ = [
    "ComparisonError",
    "ConfigError",
    "IngestError",
    "MatchError",
    "ReferenceFrameError",
    "RelphotError",
    "TilingError",
]


class RelphotError(Exception):
    """Base class for every relphot-specific error."""


class ConfigError(RelphotError):
    """Raised for an invalid or unrecognised settings file."""


class IngestError(RelphotError):
    """Raised when a catalogue file cannot be read into a FrameCatalog."""


class MatchError(RelphotError):
    """Raised when cross-matching frames into a common star list fails."""


class ReferenceFrameError(RelphotError):
    """Raised when the fixed-set frame/star selection cannot satisfy min_ref_stars in every
    tile within max_dropped_frame_fraction."""


class TilingError(RelphotError):
    """Raised when the adaptive tiling loop cannot meet hard_min_ref_candidates."""


class ComparisonError(RelphotError):
    """Raised when a tile's comparison pool is smaller than min_comparison_stars."""

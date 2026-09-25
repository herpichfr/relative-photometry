"""Settings dataclasses for relphot, with defaults for every field.

An optional TOML file can override any subset of the defaults; a value not
present in the file simply keeps its dataclass default. An unknown top-level
or nested key in the TOML file is an error rather than a typo that silently
does nothing.
"""

from __future__ import annotations

import dataclasses
import tomllib
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from relphot.exceptions import ConfigError

__all__ = [
    "CatalogSettings",
    "ColumnMap",
    "Settings",
    "SiteSettings",
    "load_settings",
    "settings_from_dict",
    "settings_to_dict",
]


@dataclass(frozen=True, slots=True)
class ColumnMap:
    """Column-name mapping from a source catalogue to relphot's internal fields.

    Defaults match the robo43 SExtractor ``CATALOG``/``*_proc_catalog.csv``
    convention. A catalogue produced by another photometry method is read by
    pointing these names at its own columns; nothing else in
    :mod:`relphot.ingest` needs to change.
    """

    ra: str = "RA"
    dec: str = "DEC"
    x: str = "X_IMAGE"
    y: str = "Y_IMAGE"
    #: Name of the per-aperture flux vector column in a FITS BinTableHDU
    #: (e.g. SExtractor's ``FLUX_APER``, shape (n_sources, n_aper)).
    flux: str = "FLUX_APER"
    fluxerr: str = "FLUXERR_APER"
    flags: str = "FLAGS"
    snr: str = "SNR"
    fwhm: str = "FWHM"
    background: str = "BACKGROUND"
    #: The robo43 CSV convention splits the vector columns above into
    #: ``<prefix>1``, ``<prefix>2``, ... instead of one vector column. Used
    #: only by the CSV adapter, which probes ``f"{flux_prefix}{i}"`` for
    #: i = 1, 2, ... until a name is missing to discover n_aper.
    flux_prefix: str = "FLUX_APER_"
    fluxerr_prefix: str = "FLUXERR_APER_"


@dataclass(frozen=True, slots=True)
class CatalogSettings:
    """How a per-frame catalogue is read and which sources are trusted."""

    #: Name of the BinTableHDU holding the source catalogue in a FITS file.
    hdu_name: str = "CATALOG"
    #: Cross-match radius, in arcsec, used by relphot.match.
    match_radius_arcsec: float = 1.0
    #: Minimum fraction of frames a star must appear in to be kept.
    min_presence: float = 0.8
    #: FLAGS bit meaning "saturated" / "non-linear" (never set together with
    #: a trustworthy flux); OR'd against a source's FLAGS value.
    saturation_flag_bit: int = 4
    nonlinear_flag_bit: int = 64
    columns: ColumnMap = field(default_factory=ColumnMap)


@dataclass(frozen=True, slots=True)
class SiteSettings:
    """Fallback observatory geodetic position.

    Used only when a frame's primary header lacks ``LATITUDE``/``LONGITUD``
    (sexagesimal-degree strings) or ``ALTITUDE`` (metres) -- real ROBO43/T80S
    headers carry all three, so this is a last resort for other data.
    """

    latitude_deg: float | None = None
    longitude_deg: float | None = None
    elevation_m: float = 0.0


@dataclass(frozen=True, slots=True)
class Settings:
    """Top-level relphot settings for Phase 1a."""

    catalog: CatalogSettings = field(default_factory=CatalogSettings)
    site: SiteSettings = field(default_factory=SiteSettings)


def _build(cls: type, data: dict[str, Any], origin: str, path: str) -> Any:
    """Recursively build a dataclass ``cls`` from a plain nested dict.

    Every field not present in ``data`` keeps ``cls``'s own default (so a
    partial override dict works exactly like a complete one). Raises
    :class:`~relphot.exceptions.ConfigError` on any key in ``data`` that is
    not a field of ``cls``. Nested dataclass fields are resolved via
    :func:`typing.get_type_hints` because ``from __future__ import
    annotations`` turns every ``dataclasses.Field.type`` into a plain string.
    ``path`` is the dotted location for error messages, empty at the root.
    """
    hints = typing.get_type_hints(cls)
    field_names = {f.name for f in fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        bad = ", ".join(sorted(f"{path}.{k}" if path else k for k in unknown))
        msg = f"{origin}: unknown key(s) {bad}"
        raise ConfigError(msg)

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        hint = hints[name]
        sub_path = f"{path}.{name}" if path else name
        if is_dataclass(hint) and isinstance(value, dict):
            kwargs[name] = _build(hint, value, origin, sub_path)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def settings_to_dict(settings: Settings) -> dict[str, Any]:
    """``settings`` as a plain, JSON-serialisable nested dict."""
    return dataclasses.asdict(settings)


def settings_from_dict(data: dict[str, Any]) -> Settings:
    """Inverse of :func:`settings_to_dict`. Validates unknown keys like :func:`load_settings`."""
    return _build(Settings, data, "<dict>", "")


def load_settings(path: Path | str | None = None) -> Settings:
    """Load :class:`Settings` from an optional TOML file.

    ``path=None`` (the default) returns pure defaults. A relative path is
    resolved against the current working directory. Every field not present
    in the file keeps its dataclass default.
    """
    if path is None:
        return Settings()

    path = Path(path)
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        msg = f"cannot read config file {path}: {exc}"
        raise ConfigError(msg) from exc
    except tomllib.TOMLDecodeError as exc:
        msg = f"invalid TOML in {path}: {exc}"
        raise ConfigError(msg) from exc

    return _build(Settings, data, str(path), "")

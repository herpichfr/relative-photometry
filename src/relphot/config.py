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
    "ComparisonSettings",
    "DecorrelationSettings",
    "LightcurveSettings",
    "ReferenceSettings",
    "Settings",
    "SiteSettings",
    "TileSettings",
    "VariableSettings",
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
class TileSettings:
    """Adaptive rectangular tiling of the master frame's pixel projection.

    See :mod:`relphot.tiles`. ``tile_size_px`` is the nominal edge length of
    the regular starting grid; ``overlap_px`` extends each tile's candidate
    catchment beyond its core rectangle without moving any star's own light
    curve out of its core tile. The adaptive loop merges tiles short of
    ``min_ref_candidates`` clean reference candidates until every tile meets
    it (or only one tile is left); a final tile below
    ``hard_min_ref_candidates`` raises :class:`~relphot.exceptions.TilingError`.
    """

    tile_size_px: float = 1000.0
    overlap_px: float = 100.0
    min_ref_candidates: int = 50
    hard_min_ref_candidates: int = 20


@dataclass(frozen=True, slots=True)
class VariableSettings:
    """Known-variable-star cross-match, used to exclude reference candidates.

    ``catalogs`` are VizieR catalogue identifiers queried through
    ``astroquery.vizier`` (an optional dependency -- see the ``variables``
    extra); ``extra_catalogs`` appends further VizieR IDs (e.g. OGLE for
    Magellanic fields) without displacing the defaults. Results are cached
    per (catalogue, rounded field centre, radius) as ECSV under
    ``cache_dir``. Disabled, offline, or with astroquery missing, the
    cross-match logs a warning and flags nothing -- see
    :func:`relphot.variables.flag_known_variables`.
    """

    enabled: bool = True
    #: AAVSO VSX; Gaia DR3 variable-classification results; ASAS-SN variables
    #: (Jayasinghe et al. catv2021 table -- "II/366/catalog" does not exist).
    catalogs: tuple[str, ...] = ("B/vsx/vsx", "I/358/vclassre", "II/366/catv2021")
    extra_catalogs: tuple[str, ...] = ()
    match_radius_arcsec: float = 2.0
    cache_dir: str = "~/.cache/relphot/variables"


@dataclass(frozen=True, slots=True)
class ReferenceSettings:
    """Reference-candidate selection and per-tile reference construction.

    Per-tile reference is built from a fixed star set S_t valid in every kept frame
    of the night. A frame with too few reference stars is flagged and dropped from
    the whole night, not per-tile. See :func:`relphot.reference.select_reference_frames_and_stars`
    for frame-dropping logic. ``method`` selects ``"weighted_fixed_mean"`` (default,
    inverse-variance weighted mean) or ``"median_fixed"`` (plain median).
    """

    min_snr: float = 15.0
    isolation_radius_arcsec: float = 5.0
    baseline_iter: int = 2
    frame_outlier_sigma: float = 3.0
    #: Minimum number of reference stars in S_t for every tile in every kept frame.
    min_ref_stars: int = 20
    #: Maximum fraction of frames allowed to be dropped from the whole night.
    max_dropped_frame_fraction: float = 0.3
    #: Sigma threshold for per-star outlier rejection in reference star set selection.
    star_outlier_sigma: float = 5.0
    #: Maximum iterations for per-star outlier rejection loop.
    star_reject_iter: int = 5
    method: str = "weighted_fixed_mean"


@dataclass(frozen=True, slots=True)
class ComparisonSettings:
    """Stage-4 comparison-star selection and ensemble construction.

    Pool criteria mirror ReferenceSettings (presence, FLAGS==0, not a known
    variable) but relax the SNR floor and drop the isolation cut by default
    -- see comparison.py's module docstring.
    """

    min_snr: float = 5.0
    require_isolation: bool = False
    isolation_radius_arcsec: float = 5.0
    k_floor: float = 2.0
    k_floor_growth: float = 1.5
    max_k_floor: float = 8.0
    n_rounds: int = 3
    n_mag_bins: int = 20
    min_bin_stars: int = 5
    min_comparison_stars: int = 5
    clip_sigma: float = 3.0
    max_iter: int = 5
    ensemble_statistic: str = "weighted_clipped_mean"


@dataclass(frozen=True, slots=True)
class LightcurveSettings:
    """Stage-5 light curves, statistics, and best-aperture selection."""

    n_mag_bins: int = 20
    bad_flag_mask: int = 252
    keep_all_apertures_in_table: bool = False
    output_format: str = "auto"
    make_plot: bool = True


@dataclass(frozen=True, slots=True)
class DecorrelationSettings:
    """Tile-level seeing/airmass decorrelation for light curves.

    Fits per-star and then per-tile surface models to remove frame-level
    and stellar-property-dependent scatter before output.
    """

    enabled: bool = True
    min_epochs_per_star: int = 15
    clip_sigma_star: float = 4.0
    max_iter_star: int = 3
    mag_degree: int = 2
    use_crowding: bool = True
    crowding_degree: int = 1
    clip_sigma_surface: float = 4.0
    max_iter_surface: int = 2
    scope: str = "pooled"
    min_comparison_stars_per_tile: int = 30
    min_comparison_stars_total: int = 200
    use_airmass: bool = True
    max_missing_airmass_frac: float = 0.2


@dataclass(frozen=True, slots=True)
class Settings:
    """Top-level relphot settings."""

    catalog: CatalogSettings = field(default_factory=CatalogSettings)
    site: SiteSettings = field(default_factory=SiteSettings)
    tile: TileSettings = field(default_factory=TileSettings)
    variable: VariableSettings = field(default_factory=VariableSettings)
    reference: ReferenceSettings = field(default_factory=ReferenceSettings)
    comparison: ComparisonSettings = field(default_factory=ComparisonSettings)
    lightcurve: LightcurveSettings = field(default_factory=LightcurveSettings)
    decorrelation: DecorrelationSettings = field(default_factory=DecorrelationSettings)


def _build(cls: type, data: dict[str, Any], origin: str, path: str) -> Any:
    """Recursively build a dataclass ``cls`` from a plain nested dict.

    Every field not present in ``data`` keeps ``cls``'s own default (so a
    partial override dict works exactly like a complete one). Raises
    :class:`~relphot.exceptions.ConfigError` on any key in ``data`` that is
    not a field of ``cls``. Nested dataclass fields are resolved via
    :func:`typing.get_type_hints` because ``from __future__ import
    annotations`` turns every ``dataclasses.Field.type`` into a plain string.
    A tuple-typed field receives a TOML/JSON list as a tuple, since neither
    source format has a tuple literal. ``path`` is the dotted location for
    error messages, empty at the root.
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
        elif typing.get_origin(hint) is tuple and isinstance(value, list):
            kwargs[name] = tuple(value)
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

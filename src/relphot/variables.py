"""Known-variable-star cross-match, used to keep variables out of the reference.

:func:`fetch_known_variables` cone-searches the VizieR catalogues named in
:class:`~relphot.config.VariableSettings` (AAVSO VSX, Gaia DR3 variability,
ASAS-SN variables by default) through ``astroquery.vizier``, an optional
dependency imported lazily so relphot itself does not require it. Results are
cached per (catalogue, rounded field centre, radius) as ECSV files under
``cache_dir``, so a re-run of the same field never re-queries the network.

:func:`flag_known_variables` cross-matches the master star list against that
combined catalogue with a :class:`~scipy.spatial.cKDTree` on sky unit
vectors. Disabled in settings, offline, or with astroquery missing, it logs a
warning and flags nothing -- a known-variable check that fails open would
silently corrupt the reference, but a pipeline that hard-crashes without
network access is worse, so this one degrades instead.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from astropy.table import Table, vstack
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from relphot.config import Settings
    from relphot.match import MatchedNight

logger = logging.getLogger(__name__)

__all__ = ["VariableFetcher", "fetch_known_variables", "flag_known_variables"]

#: Standardised column names every catalogue's table is reduced to before
#: caching or cross-matching, regardless of that catalogue's own convention.
_RA_COL = "ra_deg"
_DEC_COL = "dec_deg"

#: (RA, Dec) column name pairs tried, in order, against a VizieR result table.
_RADEC_CANDIDATES = (
    ("RAJ2000", "DEJ2000"),
    ("RA_ICRS", "DE_ICRS"),
    ("_RAJ2000", "_DEJ2000"),
    ("RAdeg", "DEdeg"),
)


class VariableFetcher(Protocol):
    """Signature of :func:`fetch_known_variables`, for dependency injection in tests."""

    def __call__(
        self, ra_c: float, dec_c: float, radius_deg: float, settings: Settings
    ) -> Table: ...


def _unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """(N, 3) unit vectors on the sky sphere for an array of RA/Dec in degrees."""
    ra = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    cosd = np.cos(dec)
    return np.column_stack([cosd * np.cos(ra), cosd * np.sin(ra), np.sin(dec)])


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def _cache_path(
    cache_dir: Path, catalog: str, ra_c: float, dec_c: float, radius_deg: float
) -> Path:
    name = f"{_sanitize(catalog)}_{ra_c:.4f}_{dec_c:.4f}_{radius_deg:.4f}.ecsv"
    return cache_dir / name


def _standardise_radec(table: Table, catalog: str) -> Table | None:
    """Return ``table`` reduced to the standard ``ra_deg``/``dec_deg`` columns, or ``None``."""
    for ra_name, dec_name in _RADEC_CANDIDATES:
        if ra_name in table.colnames and dec_name in table.colnames:
            out = Table()
            out[_RA_COL] = np.asarray(table[ra_name], dtype=np.float64)
            out[_DEC_COL] = np.asarray(table[dec_name], dtype=np.float64)
            return out
    logger.warning("%s: no recognised RA/Dec columns in %s", catalog, table.colnames)
    return None


def _query_vizier(catalog: str, ra_c: float, dec_c: float, radius_deg: float) -> Table:
    """Cone-search one VizieR catalogue. Requires astroquery (lazy import)."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    vizier = Vizier(columns=["**"], row_limit=-1)
    centre = SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg)
    result = vizier.query_region(centre, radius=radius_deg * u.deg, catalog=catalog)

    empty = Table(names=[_RA_COL, _DEC_COL], dtype=[np.float64, np.float64])
    if result is None or len(result) == 0:
        return empty

    parts = [t for table in result for t in [_standardise_radec(table, catalog)] if t is not None]
    if not parts:
        return empty
    return vstack(parts, metadata_conflicts="silent")


def fetch_known_variables(
    ra_c: float, dec_c: float, radius_deg: float, settings: Settings
) -> Table:
    """All known variables within ``radius_deg`` of ``(ra_c, dec_c)``, across every
    configured catalogue.

    Cached per (catalogue, rounded centre, radius) as ECSV under
    ``settings.variable.cache_dir``. Requires ``astroquery`` -- raises
    whatever ``astroquery`` raises (network errors, ``ImportError`` if it is
    not installed) on a cache miss; callers wanting the "never crash"
    contract go through :func:`flag_known_variables` instead.
    """
    var_settings = settings.variable
    cache_dir = Path(var_settings.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    catalogs = tuple(var_settings.catalogs) + tuple(var_settings.extra_catalogs)
    tables: list[Table] = []
    for catalog in catalogs:
        cache_path = _cache_path(cache_dir, catalog, ra_c, dec_c, radius_deg)
        if cache_path.is_file():
            table = Table.read(cache_path, format="ascii.ecsv")
            logger.info("%s: %d known variables (cached)", catalog, len(table))
        else:
            table = _query_vizier(catalog, ra_c, dec_c, radius_deg)
            table.write(cache_path, format="ascii.ecsv", overwrite=True)
            logger.info("%s: %d known variables (queried)", catalog, len(table))
        if len(table):
            tables.append(table)

    if not tables:
        return Table(names=[_RA_COL, _DEC_COL], dtype=[np.float64, np.float64])
    return vstack(tables, metadata_conflicts="silent")


def flag_known_variables(
    night: MatchedNight, settings: Settings, fetcher: VariableFetcher | None = None
) -> np.ndarray:
    """Boolean mask, ``(n_stars,)``, of master stars matching a known variable.

    Disabled in settings, a fetch failure of any kind, or an empty result all
    log a warning and return an all-``False`` mask rather than raising.
    ``fetcher`` defaults to :func:`fetch_known_variables`; tests inject a
    fake one to avoid the network.
    """
    n = night.n_stars
    if not settings.variable.enabled:
        logger.warning("known-variable cross-match disabled by settings; flagging none")
        return np.zeros(n, dtype=bool)

    finite = np.isfinite(night.ra) & np.isfinite(night.dec)
    if not np.any(finite):
        logger.warning("no stars with finite RA/Dec; flagging none as known variables")
        return np.zeros(n, dtype=bool)

    ra_c = float(np.median(night.ra[finite]))
    dec_c = float(np.median(night.dec[finite]))
    field_vec = _unit_vectors(np.array([ra_c]), np.array([dec_c]))[0]
    star_vec = _unit_vectors(night.ra[finite], night.dec[finite])
    chord_to_centre = np.linalg.norm(star_vec - field_vec, axis=1)
    max_chord = float(np.max(chord_to_centre)) if chord_to_centre.size else 0.0
    max_sep_deg = np.degrees(2.0 * np.arcsin(np.clip(max_chord / 2.0, 0.0, 1.0)))
    radius_deg = max_sep_deg + settings.variable.match_radius_arcsec / 3600.0

    fetch = fetcher if fetcher is not None else fetch_known_variables
    try:
        table = fetch(ra_c, dec_c, radius_deg, settings)
    except Exception:
        # Any failure here (missing astroquery, network outage, a bad VizieR
        # response) must degrade to "no known variables", never crash the run.
        logger.warning("known-variable cross-match failed; flagging none", exc_info=True)
        return np.zeros(n, dtype=bool)

    if table is None or len(table) == 0:
        logger.info("known-variable cross-match: no variables found in the field")
        return np.zeros(n, dtype=bool)

    var_vec = _unit_vectors(np.asarray(table[_RA_COL]), np.asarray(table[_DEC_COL]))
    tree = cKDTree(var_vec)
    radius_rad = np.radians(settings.variable.match_radius_arcsec / 3600.0)
    chord_radius = 2.0 * np.sin(radius_rad / 2.0)
    dist, _ = tree.query(star_vec, k=1)
    matched = dist <= chord_radius

    mask = np.zeros(n, dtype=bool)
    idx_finite = np.nonzero(finite)[0]
    mask[idx_finite[matched]] = True
    logger.info("known-variable cross-match: %d/%d stars flagged", int(mask.sum()), n)
    return mask

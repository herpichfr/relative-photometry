"""Stage 6d: catalogue cross-match for the search outputs.

Three independent, network-optional lookups, each cached under
``settings.search.cache_dir`` and each degrading to "nothing found" (never
raising) on any failure -- offline, astroquery missing, a bad server
response, or a catalogue this astroquery version does not know about:

* :func:`fetch_known_variables_detailed`/:func:`match_known_variables` --
  name, type, and period from the same VizieR catalogues
  :mod:`relphot.variables` already cross-matches against (VSX, Gaia DR3
  variability, ASAS-SN), but keeping those columns instead of collapsing to
  a boolean mask.
* :func:`fetch_known_planets`/:func:`match_known_planets` -- confirmed
  exoplanets (NASA Exoplanet Archive ``pscomppars``) and TESS objects of
  interest (``toi``), via ``astroquery.ipac.nexsci.nasa_exoplanet_archive``.
* :func:`fetch_gaia_neighbours`/:func:`compute_neighbour_dilution` -- for
  each master star, its nearest Gaia DR3 neighbour's own nearest *other*
  Gaia source within ``neighbour_radius_arcsec``, and the maximum transit
  depth that neighbour could dilute to (used for the ``NEIGHBOUR_BLEND`` vetting
  flag in :mod:`relphot.transit_search`).

Every fetcher takes a ``fetcher`` (or a lower-level astroquery call) as an
injectable dependency, exactly like :mod:`relphot.variables`, so tests never
touch the network.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from astropy.table import Table, vstack
from scipy.spatial import cKDTree

from relphot.numeric import unit_vectors

if TYPE_CHECKING:
    from relphot.config import Settings
    from relphot.match import MatchedNight

logger = logging.getLogger(__name__)

__all__ = [
    "GaiaNeighbourFetcher",
    "KnownPlanetFetcher",
    "KnownVariableFetcher",
    "NeighbourDilutionResult",
    "PlanetMatchResult",
    "VariableMatchResult",
    "compute_neighbour_dilution",
    "fetch_gaia_neighbours",
    "fetch_known_planets",
    "fetch_known_variables_detailed",
    "is_disqualifying_variable_type",
    "match_known_planets",
    "match_known_variables",
]

_RA_COL = "ra_deg"
_DEC_COL = "dec_deg"

_RADEC_CANDIDATES = (
    ("RAJ2000", "DEJ2000"),
    ("RA_ICRS", "DE_ICRS"),
    ("_RAJ2000", "_DEJ2000"),
    ("RAdeg", "DEdeg"),
    ("ra", "dec"),
)

#: Per-catalogue (name, type, period) column candidates, tried in order.
_NAME_COLS = ("Name", "MainName", "OName", "ID", "asasn_name", "objID")
_TYPE_COLS = ("VarType", "Type", "Class", "var_type")
_PERIOD_COLS = ("Period", "period", "Per", "P")


class KnownVariableFetcher(Protocol):
    def __call__(
        self, ra_c: float, dec_c: float, radius_deg: float, settings: Settings
    ) -> Table: ...


class KnownPlanetFetcher(Protocol):
    def __call__(
        self, ra_c: float, dec_c: float, radius_deg: float, settings: Settings
    ) -> Table: ...


class GaiaNeighbourFetcher(Protocol):
    def __call__(
        self, ra_c: float, dec_c: float, radius_deg: float, settings: Settings
    ) -> Table: ...


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def _cache_path(cache_dir: Path, tag: str, ra_c: float, dec_c: float, radius_deg: float) -> Path:
    name = f"{_sanitize(tag)}_{ra_c:.4f}_{dec_c:.4f}_{radius_deg:.4f}.ecsv"
    return cache_dir / name


def _cached(
    cache_dir: Path, tag: str, ra_c: float, dec_c: float, radius_deg: float, fetch
) -> Table:
    """Read-through ECSV cache around ``fetch()`` (a zero-argument callable)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, tag, ra_c, dec_c, radius_deg)
    if path.is_file():
        return Table.read(path, format="ascii.ecsv")
    table = fetch()
    table.write(path, format="ascii.ecsv", overwrite=True)
    return table


def _first_present(table: Table, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in table.colnames:
            return name
    return None


def _standardise_variable_table(table: Table) -> Table | None:
    """Reduce a VizieR variable-star result to ra_deg/dec_deg/name/var_type/period_days."""
    radec = None
    for ra_name, dec_name in _RADEC_CANDIDATES:
        if ra_name in table.colnames and dec_name in table.colnames:
            radec = (ra_name, dec_name)
            break
    if radec is None:
        return None
    ra_name, dec_name = radec

    n = len(table)
    out = Table()
    out[_RA_COL] = np.asarray(table[ra_name], dtype=np.float64)
    out[_DEC_COL] = np.asarray(table[dec_name], dtype=np.float64)

    name_col = _first_present(table, _NAME_COLS)
    if name_col:
        out["name"] = np.asarray(table[name_col], dtype=str)
    elif "Source" in table.colnames:
        # Gaia DR3 variability tables carry only the numeric source id.
        out["name"] = np.array([f"Gaia DR3 {v}" for v in np.asarray(table["Source"])], dtype=str)
    else:
        out["name"] = np.array([""] * n, dtype=object)

    type_col = _first_present(table, _TYPE_COLS)
    out["var_type"] = np.asarray(table[type_col], dtype=str) if type_col else np.array([""] * n)

    period_col = _first_present(table, _PERIOD_COLS)
    if period_col:
        with np.errstate(invalid="ignore"):
            out["period_days"] = np.asarray(table[period_col], dtype=np.float64)
    else:
        out["period_days"] = np.full(n, np.nan)
    return out


def _query_vizier_detailed(catalog: str, ra_c: float, dec_c: float, radius_deg: float) -> Table:
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    vizier = Vizier(columns=["**"], row_limit=-1)
    centre = SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg)
    result = vizier.query_region(centre, radius=radius_deg * u.deg, catalog=catalog)

    empty = Table(
        names=[_RA_COL, _DEC_COL, "name", "var_type", "period_days"],
        dtype=[np.float64, np.float64, str, str, np.float64],
    )
    if result is None or len(result) == 0:
        return empty
    parts = [t for table in result for t in [_standardise_variable_table(table)] if t is not None]
    if not parts:
        return empty
    return vstack(parts, metadata_conflicts="silent")


def fetch_known_variables_detailed(
    ra_c: float, dec_c: float, radius_deg: float, settings: Settings
) -> Table:
    """Known variables within ``radius_deg``, keeping name/type/period (not just a mask).

    Cached per (catalogue, rounded centre, radius) as ECSV under
    ``settings.search.cache_dir``. Requires ``astroquery``.
    """
    cache_dir = Path(settings.search.cache_dir).expanduser()
    catalogs = tuple(settings.variable.catalogs) + tuple(settings.variable.extra_catalogs)
    tables = []
    for catalog in catalogs:
        table = _cached(
            cache_dir, f"varmeta_{catalog}", ra_c, dec_c, radius_deg,
            lambda catalog=catalog: _query_vizier_detailed(catalog, ra_c, dec_c, radius_deg),
        )
        if len(table):
            tables.append(table)
    if not tables:
        return Table(
            names=[_RA_COL, _DEC_COL, "name", "var_type", "period_days"],
            dtype=[np.float64, np.float64, str, str, np.float64],
        )
    return vstack(tables, metadata_conflicts="silent")


class VariableMatchResult:
    """Per-star known-variable cross-match, ``(n_stars,)`` arrays."""

    __slots__ = ("matched", "name", "period_days", "var_type")

    def __init__(self, matched: np.ndarray, name: np.ndarray, var_type: np.ndarray,
                 period_days: np.ndarray) -> None:
        self.matched = matched
        self.name = name
        self.var_type = var_type
        self.period_days = period_days


def _clean_name(value: object) -> str:
    """A catalogue string cell as ``str``, with masked/empty/placeholder cells as ``""``."""
    if value is None or value is np.ma.masked:
        return ""
    text = str(value).strip()
    return "" if text in ("", "0", "--", "nan", "None") else text


def match_known_variables(
    night: MatchedNight, settings: Settings, fetcher: KnownVariableFetcher | None = None
) -> VariableMatchResult:
    """Cross-match every master star against :func:`fetch_known_variables_detailed`."""
    n = night.n_stars
    matched = np.zeros(n, dtype=bool)
    name = np.full(n, "", dtype=object)
    var_type = np.full(n, "", dtype=object)
    period = np.full(n, np.nan)

    finite = np.isfinite(night.ra) & np.isfinite(night.dec)
    if not np.any(finite):
        return VariableMatchResult(matched, name, var_type, period)

    ra_c = float(np.median(night.ra[finite]))
    dec_c = float(np.median(night.dec[finite]))
    radius_deg = _field_radius_deg(night, ra_c, dec_c) + (
        settings.search.known_variable_match_radius_arcsec / 3600.0
    )

    fetch = fetcher if fetcher is not None else fetch_known_variables_detailed
    try:
        table = fetch(ra_c, dec_c, radius_deg, settings)
    except Exception:
        logger.warning("known-variable detailed cross-match failed", exc_info=True)
        return VariableMatchResult(matched, name, var_type, period)
    if table is None or len(table) == 0:
        return VariableMatchResult(matched, name, var_type, period)

    cat_vec = unit_vectors(np.asarray(table[_RA_COL]), np.asarray(table[_DEC_COL]))
    tree = cKDTree(cat_vec)
    star_vec = unit_vectors(night.ra[finite], night.dec[finite])
    radius_rad = np.radians(settings.search.known_variable_match_radius_arcsec / 3600.0)
    chord_radius = 2.0 * np.sin(radius_rad / 2.0)
    # Merge every catalogue entry within the radius, not just the nearest:
    # the same star usually appears in several catalogues (VSX, ASAS-SN,
    # Gaia DR3), and the nearest one alone may be the least informative
    # (e.g. a Gaia "SOLAR_LIKE" class hiding a VSX eclipsing-binary entry).
    # Names are taken in table order (catalogue order in settings), types are
    # joined, and the period is the first finite one.
    cat_name = [_clean_name(v) for v in np.asarray(table["name"])]
    cat_type = [_clean_name(v) for v in np.asarray(table["var_type"])]
    cat_period = np.asarray(table["period_days"], dtype=np.float64)
    hits = tree.query_ball_point(star_vec, r=chord_radius)

    idx_finite = np.nonzero(finite)[0]
    for i_star, rows in zip(idx_finite, hits, strict=True):
        if not rows:
            continue
        rows = sorted(rows)
        matched[i_star] = True
        name[i_star] = next((cat_name[r] for r in rows if cat_name[r]), "")
        types = []
        for r in rows:
            for part in cat_type[r].split("|"):
                if part and part not in types:
                    types.append(part)
        var_type[i_star] = "|".join(types)
        finite_p = [cat_period[r] for r in rows if np.isfinite(cat_period[r])]
        period[i_star] = finite_p[0] if finite_p else np.nan
    logger.info("known-variable detailed cross-match: %d/%d stars matched", int(matched.sum()), n)
    return VariableMatchResult(matched, name, var_type, period)


def _query_exoplanet_archive(
    table_name: str, ra_c: float, dec_c: float, radius_deg: float
) -> Table:
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

    result = NasaExoplanetArchive.query_region(
        table=table_name, coordinates=f"{ra_c} {dec_c}", radius=f"{radius_deg} deg"
    )
    out = Table()
    n = len(result) if result is not None else 0
    if n == 0:
        return Table(
            names=[_RA_COL, _DEC_COL, "name", "period_days", "depth", "is_toi"],
            dtype=[np.float64, np.float64, str, np.float64, np.float64, bool],
        )
    ra_name = "ra" if "ra" in result.colnames else "rastr"
    dec_name = "dec" if "dec" in result.colnames else "decstr"
    out[_RA_COL] = np.asarray(result[ra_name], dtype=np.float64)
    out[_DEC_COL] = np.asarray(result[dec_name], dtype=np.float64)
    if table_name == "toi":
        out["name"] = np.array([f"TOI-{v}" for v in np.asarray(result["toi"])])
        out["period_days"] = _safe_col(result, "pl_orbper", n)
        out["depth"] = _safe_col(result, "pl_trandep", n) / 1e6  # ppm -> fraction
        out["is_toi"] = np.ones(n, dtype=bool)
    else:
        out["name"] = np.asarray(result["pl_name"], dtype=str)
        out["period_days"] = _safe_col(result, "pl_orbper", n)
        rp = _safe_col(result, "pl_rade", n)
        rs = _safe_col(result, "st_rad", n)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["depth"] = (rp * 0.009157) ** 2 / rs**2  # Earth radii, Solar radii -> depth
        out["is_toi"] = np.zeros(n, dtype=bool)
    return out


def _safe_col(table: Table, name: str, n: int) -> np.ndarray:
    if name not in table.colnames:
        return np.full(n, np.nan)
    with np.errstate(invalid="ignore"):
        return np.asarray(table[name], dtype=np.float64)


def fetch_known_planets(ra_c: float, dec_c: float, radius_deg: float, settings: Settings) -> Table:
    """Confirmed exoplanets (``pscomppars``) and TOIs within ``radius_deg``.

    Cached under ``settings.search.cache_dir``. Requires ``astroquery`` with
    the ``astroquery.ipac.nexsci`` submodule; a table this astroquery
    version does not expose (older releases lack ``toi``) is skipped with a
    warning rather than failing the whole lookup.
    """
    cache_dir = Path(settings.search.cache_dir).expanduser()
    tables = []
    for table_name in ("pscomppars", "toi"):
        try:
            table = _cached(
                cache_dir, f"planets_{table_name}", ra_c, dec_c, radius_deg,
                lambda table_name=table_name: _query_exoplanet_archive(
                    table_name, ra_c, dec_c, radius_deg
                ),
            )
        except Exception:
            logger.warning("exoplanet archive table %s unavailable", table_name, exc_info=True)
            continue
        if len(table):
            tables.append(table)
    if not tables:
        return Table(
            names=[_RA_COL, _DEC_COL, "name", "period_days", "depth", "is_toi"],
            dtype=[np.float64, np.float64, str, np.float64, np.float64, bool],
        )
    return vstack(tables, metadata_conflicts="silent")


class PlanetMatchResult:
    """Per-star known-exoplanet cross-match, ``(n_stars,)`` arrays."""

    __slots__ = ("depth", "is_toi", "matched", "name", "period_days")

    def __init__(self, matched, name, period_days, depth, is_toi) -> None:
        self.matched = matched
        self.name = name
        self.period_days = period_days
        self.depth = depth
        self.is_toi = is_toi


def match_known_planets(
    night: MatchedNight, settings: Settings, fetcher: KnownPlanetFetcher | None = None
) -> PlanetMatchResult:
    """Cross-match every master star against :func:`fetch_known_planets`."""
    n = night.n_stars
    matched = np.zeros(n, dtype=bool)
    name = np.full(n, "", dtype=object)
    period = np.full(n, np.nan)
    depth = np.full(n, np.nan)
    is_toi = np.zeros(n, dtype=bool)

    finite = np.isfinite(night.ra) & np.isfinite(night.dec)
    if not np.any(finite):
        return PlanetMatchResult(matched, name, period, depth, is_toi)

    ra_c = float(np.median(night.ra[finite]))
    dec_c = float(np.median(night.dec[finite]))
    radius_deg = _field_radius_deg(night, ra_c, dec_c) + (
        settings.search.known_planet_match_radius_arcsec / 3600.0
    )

    fetch = fetcher if fetcher is not None else fetch_known_planets
    try:
        table = fetch(ra_c, dec_c, radius_deg, settings)
    except Exception:
        logger.warning("known-planet cross-match failed", exc_info=True)
        return PlanetMatchResult(matched, name, period, depth, is_toi)
    if table is None or len(table) == 0:
        return PlanetMatchResult(matched, name, period, depth, is_toi)

    cat_vec = unit_vectors(np.asarray(table[_RA_COL]), np.asarray(table[_DEC_COL]))
    tree = cKDTree(cat_vec)
    star_vec = unit_vectors(night.ra[finite], night.dec[finite])
    radius_rad = np.radians(settings.search.known_planet_match_radius_arcsec / 3600.0)
    chord_radius = 2.0 * np.sin(radius_rad / 2.0)
    dist, idx = tree.query(star_vec, k=1)
    ok = dist <= chord_radius

    idx_finite = np.nonzero(finite)[0]
    matched[idx_finite[ok]] = True
    name[idx_finite[ok]] = np.asarray(table["name"])[idx[ok]]
    period[idx_finite[ok]] = np.asarray(table["period_days"])[idx[ok]]
    depth[idx_finite[ok]] = np.asarray(table["depth"])[idx[ok]]
    is_toi[idx_finite[ok]] = np.asarray(table["is_toi"])[idx[ok]]
    logger.info("known-planet cross-match: %d/%d stars matched", int(matched.sum()), n)
    return PlanetMatchResult(matched, name, period, depth, is_toi)


def fetch_gaia_neighbours(
    ra_c: float, dec_c: float, radius_deg: float, settings: Settings
) -> Table:
    """Gaia DR3 sources (``ra_deg``, ``dec_deg``, ``gaia_id``, ``gmag``) within ``radius_deg``.

    Queried from ``settings.search.gaia_catalog`` via VizieR. Cached under
    ``settings.search.cache_dir``. Requires ``astroquery``.
    """
    cache_dir = Path(settings.search.cache_dir).expanduser()

    def _fetch() -> Table:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astroquery.vizier import Vizier

        vizier = Vizier(columns=["RA_ICRS", "DE_ICRS", "Source", "Gmag"], row_limit=-1)
        centre = SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg)
        result = vizier.query_region(
            centre, radius=radius_deg * u.deg, catalog=settings.search.gaia_catalog
        )
        empty = Table(
            names=[_RA_COL, _DEC_COL, "gaia_id", "gmag"],
            dtype=[np.float64, np.float64, str, np.float64],
        )
        if result is None or len(result) == 0:
            return empty
        table = result[0]
        out = Table()
        out[_RA_COL] = np.asarray(table["RA_ICRS"], dtype=np.float64)
        out[_DEC_COL] = np.asarray(table["DE_ICRS"], dtype=np.float64)
        out["gaia_id"] = np.asarray(table["Source"], dtype=str)
        out["gmag"] = np.asarray(table["Gmag"], dtype=np.float64)
        return out

    return _cached(cache_dir, "gaia_neighbours", ra_c, dec_c, radius_deg, _fetch)


class NeighbourDilutionResult:
    """Per-star nearest-neighbour blend dilution, ``(n_stars,)`` arrays."""

    __slots__ = ("gaia_id", "has_neighbour", "max_dilutable_depth", "neighbour_sep_arcsec")

    def __init__(self, gaia_id, has_neighbour, neighbour_sep_arcsec, max_dilutable_depth) -> None:
        self.gaia_id = gaia_id
        self.has_neighbour = has_neighbour
        self.neighbour_sep_arcsec = neighbour_sep_arcsec
        self.max_dilutable_depth = max_dilutable_depth


def compute_neighbour_dilution(
    night: MatchedNight, settings: Settings, fetcher: GaiaNeighbourFetcher | None = None
) -> NeighbourDilutionResult:
    """For each master star, the maximum transit depth a blended neighbour could dilute to.

    Matches each star to its nearest Gaia DR3 source (within
    ``settings.search.gaia_match_radius_arcsec``), then finds that source's
    own nearest *other* Gaia source within
    ``settings.search.neighbour_radius_arcsec``. ``max_dilutable_depth`` is
    ``r / (1 + r)`` where ``r`` is the neighbour-to-star flux ratio from
    their Gmag difference -- an observed depth below this is achievable by a
    100%-deep eclipse on the neighbour alone, diluted by the star's own flux
    (see ``NEIGHBOUR_BLEND`` in :mod:`relphot.transit_search`).
    """
    n = night.n_stars
    gaia_id = np.full(n, "", dtype=object)
    has_neighbour = np.zeros(n, dtype=bool)
    sep = np.full(n, np.nan)
    max_depth = np.full(n, np.nan)

    finite = np.isfinite(night.ra) & np.isfinite(night.dec)
    if not np.any(finite):
        return NeighbourDilutionResult(gaia_id, has_neighbour, sep, max_depth)

    ra_c = float(np.median(night.ra[finite]))
    dec_c = float(np.median(night.dec[finite]))
    radius_deg = _field_radius_deg(night, ra_c, dec_c) + (
        settings.search.neighbour_radius_arcsec / 3600.0
    )

    fetch = fetcher if fetcher is not None else fetch_gaia_neighbours
    try:
        table = fetch(ra_c, dec_c, radius_deg, settings)
    except Exception:
        logger.warning("Gaia neighbour cross-match failed", exc_info=True)
        return NeighbourDilutionResult(gaia_id, has_neighbour, sep, max_depth)
    if table is None or len(table) < 2:
        return NeighbourDilutionResult(gaia_id, has_neighbour, sep, max_depth)

    gaia_vec = unit_vectors(np.asarray(table[_RA_COL]), np.asarray(table[_DEC_COL]))
    gaia_gmag = np.asarray(table["gmag"], dtype=np.float64)
    gaia_tree = cKDTree(gaia_vec)

    # Match master stars to their own Gaia counterpart.
    star_vec = unit_vectors(night.ra[finite], night.dec[finite])
    match_radius_rad = np.radians(settings.search.gaia_match_radius_arcsec / 3600.0)
    match_chord = 2.0 * np.sin(match_radius_rad / 2.0)
    dist_own, idx_own = gaia_tree.query(star_vec, k=1)
    own_ok = dist_own <= match_chord

    # Each Gaia source's own nearest *other* Gaia source.
    neigh_radius_rad = np.radians(settings.search.neighbour_radius_arcsec / 3600.0)
    neigh_chord = 2.0 * np.sin(neigh_radius_rad / 2.0)
    dist2, idx2 = gaia_tree.query(gaia_vec, k=2)
    neigh_dist = dist2[:, 1]
    neigh_idx = idx2[:, 1]
    neigh_ok = neigh_dist <= neigh_chord

    idx_finite = np.nonzero(finite)[0]
    for local_i, star_i in enumerate(idx_finite):
        if not own_ok[local_i]:
            continue
        g = idx_own[local_i]
        gaia_id[star_i] = str(table["gaia_id"][g])
        if not neigh_ok[g]:
            continue
        neighbour = neigh_idx[g]
        delta_gmag = gaia_gmag[neighbour] - gaia_gmag[g]
        if not np.isfinite(delta_gmag):
            continue
        ratio = 10.0 ** (-0.4 * delta_gmag)  # neighbour_flux / star_flux
        has_neighbour[star_i] = True
        theta = 2.0 * np.arcsin(np.clip(neigh_dist[g] / 2.0, 0.0, 1.0))
        sep[star_i] = np.degrees(theta) * 3600.0
        max_depth[star_i] = ratio / (1.0 + ratio)

    logger.info(
        "neighbour dilution: %d/%d stars have a Gaia neighbour", int(has_neighbour.sum()), n
    )
    return NeighbourDilutionResult(gaia_id, has_neighbour, sep, max_depth)


def _field_radius_deg(night: MatchedNight, ra_c: float, dec_c: float) -> float:
    """Angular radius from ``(ra_c, dec_c)`` covering every finite-position star."""
    finite = np.isfinite(night.ra) & np.isfinite(night.dec)
    if not np.any(finite):
        return 0.0
    field_vec = unit_vectors(np.array([ra_c]), np.array([dec_c]))[0]
    star_vec = unit_vectors(night.ra[finite], night.dec[finite])
    chord = np.linalg.norm(star_vec - field_vec, axis=1)
    max_chord = float(np.max(chord)) if chord.size else 0.0
    return float(np.degrees(2.0 * np.arcsin(np.clip(max_chord / 2.0, 0.0, 1.0))))


def is_disqualifying_variable_type(var_type: str, settings: Settings) -> bool:
    """Whether a catalogue variability-type string rules out transit candidacy.

    ``var_type`` may combine several ``|``-separated tokens (VSX/Gaia
    convention); disqualifying when *any* token contains (case-insensitive)
    one of ``settings.search.disqualifying_variable_types``. An empty or
    unrecognised type string is never disqualifying.
    """
    if not var_type:
        return False
    tokens = [tok.strip().upper() for tok in var_type.split("|") if tok.strip()]
    keywords = [kw.upper() for kw in settings.search.disqualifying_variable_types]
    return any(kw in tok for tok in tokens for kw in keywords)

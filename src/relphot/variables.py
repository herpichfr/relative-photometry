"""Known-variable-star cross-match, and per-star intrinsic variability search.

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

The above is Stage-3 machinery. :func:`compute_star_variability` is Stage
6c: a per-star search for genuine intrinsic variability (amplitude,
periodicity, noise excess) on CBV-regressed light curves. Unlike
:mod:`relphot.transit_search`, a per-star *free* CBV regression is used here
(:func:`relphot.transit_search.fit_nuisance_model` fit directly to each
star's own light curve) -- variability characterisation has no
transit-safety constraint to protect. A frame flagged by
:func:`relphot.cotrend.detect_systematic_frames` is excluded before scoring,
so a shared instrumental glitch is never credited as genuine variability.
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from astropy.table import Table, vstack
from scipy.spatial import cKDTree

from relphot.numeric import (
    fit_noise_floor,
    mad_sigma,
    nanmedian_quiet,
    robust_clip_series,
    unit_vectors,
)
from relphot.transit_search import fit_nuisance_model

if TYPE_CHECKING:
    from relphot.comparison import ComparisonResult
    from relphot.config import Settings
    from relphot.cotrend import CotrendResult
    from relphot.match import MatchedNight
    from relphot.tiles import TileMap

logger = logging.getLogger(__name__)

__all__ = [
    "StarVariabilityResult",
    "VariableFetcher",
    "classify_variability",
    "compute_star_variability",
    "fetch_known_variables",
    "fit_star_variability_metrics",
    "flag_known_variables",
    "plot_variable_star",
]

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
    field_vec = unit_vectors(np.array([ra_c]), np.array([dec_c]))[0]
    star_vec = unit_vectors(night.ra[finite], night.dec[finite])
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

    var_vec = unit_vectors(np.asarray(table[_RA_COL]), np.asarray(table[_DEC_COL]))
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


@dataclass(slots=True)
class StarVariabilityResult:
    """Per-star variability characterisation, ``(n_stars,)`` unless noted.

    Computed at each star's best aperture from a per-star *free*
    constant+CBV regression (no polynomial -- see
    :func:`fit_star_variability_metrics`'s docstring for why).
    ``rms_robust`` (MAD) and ``rms_std`` (plain std) describe that fit's
    residual; ``noise_floor`` is fit from ``rms_robust`` across every
    searched star (a MAD stays meaningful even for a star with a real,
    partial-coverage dip), while ``excess = rms_std / noise_floor`` -- a MAD
    would be blind to an eclipse covering less than about half the points.
    ``von_neumann`` is the von Neumann ratio (~2 for white noise, below 2 for
    correlated/periodic variability); ``von_neumann_significance`` is how far
    below 2, in the white-noise-expected standard errors.
    ``ls_period_days``/``ls_power``/``ls_fap`` are the best Lomb-Scargle
    period, power, and false-alarm probability (NaN if too few points).
    ``amplitude`` is the robust 2-98 percentile flux range. ``trend_slope``/
    ``trend_significance`` come from a *separate* constant+CBV+linear fit
    (used only for the trend-only class, so a real trend never leaks into
    the primary residual above). ``systematic_excluded`` marks a star that
    only passed the variable-candidate rule because of
    :func:`relphot.cotrend.detect_systematic_frames`-flagged epochs -- with
    those epochs dropped it no longer passes, so it is not credited as
    variable. ``variable_candidate``/``variable_class`` are the final call.
    """

    searched: np.ndarray
    aper: np.ndarray
    n_good: np.ndarray
    rms_robust: np.ndarray
    rms_std: np.ndarray
    chi2_reduced: np.ndarray
    noise_floor: np.ndarray
    excess: np.ndarray
    von_neumann: np.ndarray
    von_neumann_significance: np.ndarray
    ls_period_days: np.ndarray
    ls_power: np.ndarray
    ls_fap: np.ndarray
    amplitude: np.ndarray
    trend_slope: np.ndarray
    trend_significance: np.ndarray
    systematic_excluded: np.ndarray
    variable_candidate: np.ndarray
    variable_class: list[str]


def classify_variability(
    candidate: bool,
    ls_fap: float,
    eclipse_fraction: float,
    trend_significance: float,
    von_neumann_significance: float,
    settings: Settings,
) -> str:
    """One of ``"periodic"``, ``"eclipse-like"``, ``"trend-only"``, ``"irregular"``, ``"none"``.

    Evaluated in that priority order for a star already marked a variable
    ``candidate``; a non-candidate is always ``"none"``.
    """
    search = settings.search
    if not candidate:
        return "none"
    if np.isfinite(ls_fap) and ls_fap < search.ls_fap_threshold:
        return "periodic"
    if np.isfinite(eclipse_fraction) and eclipse_fraction >= search.eclipse_min_fraction:
        return "eclipse-like"
    if (
        np.isfinite(trend_significance)
        and trend_significance >= search.von_neumann_sigma_threshold
        and not (
            np.isfinite(von_neumann_significance)
            and von_neumann_significance >= search.von_neumann_sigma_threshold
        )
    ):
        return "trend-only"
    return "irregular"


def fit_star_variability_metrics(
    t_g: np.ndarray,
    y_norm: np.ndarray,
    err_norm: np.ndarray,
    cbv_rows: np.ndarray,
    settings: Settings,
) -> dict | None:
    """One star's variability metrics from its (already epoch-selected) light curve.

    The primary fit removed here is *constant + CBVs only* -- no polynomial
    -- deliberately: a poly(deg >= 1) fit over a single night's baseline can
    absorb genuine smooth variability (an eclipse's ingress/egress slope, a
    slowly-varying rotational modulation), exactly the signal this path
    exists to find. ``rms_robust``/``chi2_reduced``/``von_neumann``/the
    Lomb-Scargle search/``eclipse_fraction`` are all computed on this
    constant+CBV residual. ``rms_std`` is the plain standard deviation of
    that same residual (not a MAD): a MAD is insensitive to an eclipse or
    dip covering less than ~50% of points, which is the common case, so it
    must not be what ``excess`` is judged against -- only the noise *floor*
    (fit across the star population from ``rms_robust``, which does need to
    be insensitive to any one star's own variability) uses the robust
    version. The linear-trend significance is deliberately from a *separate*
    constant+CBV+linear fit, used only for the trend-only classification --
    it must not leak a smooth trend into the primary (trend-free) residual.
    """
    search = settings.search
    w = 1.0 / err_norm**2
    base = fit_nuisance_model(t_g, y_norm, w, cbv_rows, 0)
    if base is None:
        return None
    _x_base, _beta, resid, chi2, dof = base
    n = resid.size
    rms_robust = float(mad_sigma(resid))
    rms_std = float(np.std(resid))
    chi2_reduced = chi2 / dof if dof > 0 else np.nan

    von_neumann = np.nan
    von_neumann_sig = np.nan
    if n >= 4:
        denom_vn = float(np.sum((resid - resid.mean()) ** 2))
        if denom_vn > 0:
            von_neumann = float(np.sum(np.diff(resid) ** 2) / denom_vn)
            se_eta = np.sqrt(4.0 / (n + 1))
            von_neumann_sig = (2.0 - von_neumann) / se_eta

    trend_slope = np.nan
    trend_sig = np.nan
    base_trend = fit_nuisance_model(t_g, y_norm, w, cbv_rows, 1)
    if base_trend is not None:
        x_trend, beta_trend, _resid_trend, chi2_trend, dof_trend = base_trend
        if x_trend.shape[1] > 1:
            try:
                cov_trend = np.linalg.inv((x_trend * w[:, None]).T @ x_trend)
                chi2_reduced_trend = chi2_trend / dof_trend if dof_trend > 0 else np.nan
                se_trend = np.sqrt(max(chi2_reduced_trend, 1e-12) * cov_trend[1, 1])
                trend_slope = float(beta_trend[1])
                if np.isfinite(se_trend) and se_trend > 0:
                    trend_sig = abs(trend_slope) / se_trend
            except np.linalg.LinAlgError:
                pass

    ls_period = np.nan
    ls_power = np.nan
    ls_fap = np.nan
    baseline = t_g[-1] - t_g[0]
    min_period_d = search.ls_min_period_minutes / 1440.0
    max_period_d = baseline * search.ls_max_period_factor
    if n >= 8 and baseline > 0 and min_period_d < max_period_d:
        try:
            from astropy.timeseries import LombScargle

            with warnings.catch_warnings():
                # astropy's false_alarm_probability (Baluev approximation) can
                # take a sqrt of a tiny negative number for an edge-case
                # baseline/frequency combination; that is a known astropy
                # wrinkle, not a symptom of anything wrong with our data.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ls = LombScargle(t_g, resid, err_norm)
                freq, power = ls.autopower(
                    minimum_frequency=1.0 / max_period_d, maximum_frequency=1.0 / min_period_d
                )
                if power.size:
                    best = int(np.argmax(power))
                    ls_period = float(1.0 / freq[best])
                    ls_power = float(power[best])
                    ls_fap = float(ls.false_alarm_probability(power[best]))
        except Exception:
            logger.debug("Lomb-Scargle failed", exc_info=True)

    eclipse_fraction = np.nan
    if np.isfinite(rms_robust) and rms_robust > 0:
        eclipse_fraction = float(
            np.count_nonzero(resid < -search.eclipse_depth_sigma * rms_robust) / n
        )

    return {
        "rms_robust": rms_robust,
        "rms_std": rms_std,
        "chi2_reduced": chi2_reduced,
        "von_neumann": von_neumann,
        "von_neumann_significance": von_neumann_sig,
        "ls_period_days": ls_period,
        "ls_power": ls_power,
        "ls_fap": ls_fap,
        "trend_slope": trend_slope,
        "trend_significance": trend_sig,
        "eclipse_fraction": eclipse_fraction,
    }


def compute_star_variability(
    night: MatchedNight,
    tilemap: TileMap,
    comparison_result: ComparisonResult,
    cotrend_result: CotrendResult,
    lc: np.ndarray,
    lc_err: np.ndarray,
    epoch_ok: np.ndarray,
    frame_kept: np.ndarray,
    star_best_aper: np.ndarray,
    systematic_frames: np.ndarray,
    settings: Settings,
) -> StarVariabilityResult:
    """Per-star variability search at each star's best aperture.

    ``lc``/``lc_err``/``epoch_ok`` are the decorrelated light curves (see
    :class:`~relphot.lightcurve.LightCurveResult`); ``systematic_frames`` is
    :func:`relphot.cotrend.detect_systematic_frames`'s output.
    """
    search = settings.search
    n_kept = int(np.count_nonzero(frame_kept))
    search = replace(search, min_epochs=search.effective_min_epochs(n_kept))
    logger.info("variability: min epochs %d (%d kept frames)", search.min_epochs, n_kept)
    n_stars = night.n_stars
    n_aper = night.n_aper
    bjd = np.array([m.bjd_tdb for m in night.frame_meta], dtype=np.float64)
    core_tile = tilemap.core_tile

    searched = np.zeros(n_stars, dtype=bool)
    aper_arr = np.full(n_stars, -1, dtype=np.int64)
    n_good_arr = np.zeros(n_stars, dtype=np.int64)
    rms_robust = np.full(n_stars, np.nan)
    rms_std = np.full(n_stars, np.nan)
    chi2_reduced = np.full(n_stars, np.nan)
    von_neumann = np.full(n_stars, np.nan)
    von_neumann_sig = np.full(n_stars, np.nan)
    ls_period = np.full(n_stars, np.nan)
    ls_power = np.full(n_stars, np.nan)
    ls_fap = np.full(n_stars, np.nan)
    amplitude = np.full(n_stars, np.nan)
    trend_slope = np.full(n_stars, np.nan)
    trend_sig = np.full(n_stars, np.nan)
    eclipse_fraction = np.full(n_stars, np.nan)
    mag_at_aper = np.full(n_stars, np.nan)

    per_star_epochs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for i in range(n_stars):
        t_tile = core_tile[i]
        a = int(star_best_aper[i])
        if t_tile < 0 or a < 0 or not (0 <= a < n_aper):
            continue
        good = epoch_ok[i] & frame_kept & np.isfinite(lc[i, :, a]) & np.isfinite(lc_err[i, :, a])
        n_good_arr[i] = int(np.count_nonzero(good))
        if n_good_arr[i] < search.min_epochs:
            continue

        idx = np.nonzero(good)[0]
        order = np.argsort(bjd[idx])
        idx = idx[order]
        t_g = bjd[idx]
        y = lc[i, idx, a].astype(np.float64)
        err = lc_err[i, idx, a].astype(np.float64)
        med = nanmedian_quiet(y)
        if not np.isfinite(med) or med == 0:
            continue
        y_norm = y / med
        err_norm = err / med

        # Remove isolated single-epoch outliers before scoring variability,
        # for the same reason relphot.transit_search does: a robust, global
        # clip this far out cannot remove real stellar variability.
        keep = robust_clip_series(y_norm, search.lc_clip_sigma, search.lc_clip_window)
        if not np.all(keep):
            idx = idx[keep]
            t_g = t_g[keep]
            y_norm = y_norm[keep]
            err_norm = err_norm[keep]
            if t_g.shape[0] < search.min_epochs:
                continue

        n_cbv = int(np.count_nonzero(np.isfinite(cotrend_result.basis[t_tile, a, :, 0])))
        cbv_rows = cotrend_result.basis[t_tile, a, :n_cbv, :][:, idx]

        searched[i] = True
        aper_arr[i] = a
        mag_at_aper[i] = comparison_result.mag[i, a]
        per_star_epochs[i] = (idx, t_g, y_norm, err_norm)

        metrics = fit_star_variability_metrics(t_g, y_norm, err_norm, cbv_rows, settings)
        if metrics is None:
            continue
        rms_robust[i] = metrics["rms_robust"]
        rms_std[i] = metrics["rms_std"]
        chi2_reduced[i] = metrics["chi2_reduced"]
        von_neumann[i] = metrics["von_neumann"]
        von_neumann_sig[i] = metrics["von_neumann_significance"]
        ls_period[i] = metrics["ls_period_days"]
        ls_power[i] = metrics["ls_power"]
        ls_fap[i] = metrics["ls_fap"]
        trend_slope[i] = metrics["trend_slope"]
        trend_sig[i] = metrics["trend_significance"]
        eclipse_fraction[i] = metrics["eclipse_fraction"]
        amplitude[i] = float(np.percentile(y_norm, 98) - np.percentile(y_norm, 2))

    floor_func = fit_noise_floor(
        mag_at_aper, rms_robust, searched & np.isfinite(rms_robust), search.variability_n_mag_bins,
        search.variability_min_bin_stars,
    )
    with np.errstate(invalid="ignore"):
        noise_floor = floor_func(mag_at_aper)
        excess = np.where(noise_floor > 0, rms_std / noise_floor, np.nan)

    def _passes(exc: float, vn_sig: float, fap: float) -> bool:
        if not np.isfinite(exc) or exc < search.excess_rms_threshold:
            return False
        vn_ok = np.isfinite(vn_sig) and vn_sig >= search.von_neumann_sigma_threshold
        ls_ok = np.isfinite(fap) and fap < search.ls_fap_threshold
        return bool(vn_ok or ls_ok)

    variable_candidate = np.array(
        [searched[i] and _passes(excess[i], von_neumann_sig[i], ls_fap[i]) for i in range(n_stars)]
    )

    systematic_excluded = np.zeros(n_stars, dtype=bool)
    for i in np.nonzero(variable_candidate)[0]:
        idx, t_g, y_norm, err_norm = per_star_epochs[i]
        sys_here = systematic_frames[idx]
        if not np.any(sys_here):
            continue
        keep = ~sys_here
        if np.count_nonzero(keep) < search.min_epochs:
            continue
        a = aper_arr[i]
        t_tile = core_tile[i]
        n_cbv = int(np.count_nonzero(np.isfinite(cotrend_result.basis[t_tile, a, :, 0])))
        cbv_rows = cotrend_result.basis[t_tile, a, :n_cbv, :][:, idx][:, keep]
        metrics2 = fit_star_variability_metrics(
            t_g[keep], y_norm[keep], err_norm[keep], cbv_rows, settings
        )
        if metrics2 is None:
            systematic_excluded[i] = True
            variable_candidate[i] = False
            continue
        rms_std2 = metrics2["rms_std"]
        excess2 = rms_std2 / noise_floor[i] if noise_floor[i] > 0 else np.nan
        if not _passes(excess2, metrics2["von_neumann_significance"], metrics2["ls_fap"]):
            systematic_excluded[i] = True
            variable_candidate[i] = False

    variable_class = [
        classify_variability(
            bool(variable_candidate[i]),
            ls_fap[i],
            eclipse_fraction[i],
            trend_sig[i],
            von_neumann_sig[i],
            settings,
        )
        for i in range(n_stars)
    ]

    if n_stars > 0 and not searched.any():
        logger.warning(
            "variability: no star has >= %d good epochs (%d kept frames); nothing searched",
            search.min_epochs, n_kept,
        )

    logger.info(
        "variability: %d/%d searched, %d candidates",
        int(searched.sum()), n_stars, int(variable_candidate.sum()),
    )

    return StarVariabilityResult(
        searched=searched,
        aper=aper_arr,
        n_good=n_good_arr,
        rms_robust=rms_robust,
        rms_std=rms_std,
        chi2_reduced=chi2_reduced,
        noise_floor=noise_floor,
        excess=excess,
        von_neumann=von_neumann,
        von_neumann_significance=von_neumann_sig,
        ls_period_days=ls_period,
        ls_power=ls_power,
        ls_fap=ls_fap,
        amplitude=amplitude,
        trend_slope=trend_slope,
        trend_significance=trend_sig,
        systematic_excluded=systematic_excluded,
        variable_candidate=variable_candidate,
        variable_class=variable_class,
    )


def plot_variable_star(
    star_id: int,
    t: np.ndarray,
    y_norm: np.ndarray,
    resid: np.ndarray,
    metrics: dict,
    path: str,
) -> None:
    """Three-panel diagnostic PNG: light curve, Lomb-Scargle periodogram, phase fold.

    ``metrics`` is one row of the dict :func:`fit_star_variability_metrics`
    returns. The phase-fold panel is only drawn when ``ls_period_days`` is
    finite.

    Raises
    ------
    RelphotError
        If matplotlib is not installed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        from relphot.exceptions import RelphotError

        msg = "install the 'lightcurve' extra: pip install 'relphot[lightcurve]'"
        raise RelphotError(msg) from None

    has_period = np.isfinite(metrics.get("ls_period_days", np.nan))
    fig, axes = plt.subplots(1, 3 if has_period else 2, figsize=(15 if has_period else 10, 4))
    ax_lc, ax_ls = axes[0], axes[1]

    t0 = t[0]
    ax_lc.plot((t - t0) * 24.0, y_norm, ".", color="0.4", ms=3)
    ax_lc.set_xlabel("hours from start")
    ax_lc.set_ylabel("normalised flux")
    ax_lc.set_title(f"star {star_id}: rms={metrics.get('rms_robust', np.nan) * 1000:.2f} mmag")

    period = metrics.get("ls_period_days", np.nan)
    fap = metrics.get("ls_fap", np.nan)
    if np.isfinite(period):
        from astropy.timeseries import LombScargle

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ls = LombScargle(t, resid)
            baseline = t[-1] - t[0]
            freq, pw = ls.autopower(
                minimum_frequency=1.0 / max(baseline, period * 2),
                maximum_frequency=1.0 / (period / 4),
            )
        ax_ls.plot(1.0 / freq * 24.0, pw, "-", color="C0", lw=1)
        ax_ls.axvline(period * 24.0, color="C3", ls="--", lw=1)
        ax_ls.set_xlabel("period (hours)")
        ax_ls.set_ylabel("LS power")
        ax_ls.set_title(f"P={period * 24:.3f} h, FAP={fap:.1e}" if np.isfinite(fap) else "")

        ax_fold = axes[2]
        phase = ((t - t0) / period) % 1.0
        ax_fold.plot(phase, y_norm, ".", color="0.4", ms=3)
        ax_fold.plot(phase + 1, y_norm, ".", color="0.8", ms=3)
        ax_fold.set_xlabel("phase")
        ax_fold.set_ylabel("normalised flux")
        ax_fold.set_title("phase-folded")
    else:
        ax_ls.set_title("no periodogram (too few points)")

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    logger.info("wrote %s", path)

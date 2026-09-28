"""Phase 2, Unit B: cross-night variability and transit search on tied multi-night products.

Consumes only what :func:`relphot.multinight.load_multinight` returns --
:class:`~relphot.multinight.NightCrossMatch`, :class:`~relphot.multinight.NightTie`
(for its per-(night, aperture, magnitude) calibration floor,
:meth:`~relphot.multinight.NightTie.floor_at`), :class:`~relphot.multinight.MultiNightLightCurves`
and the plain ``night_info`` metadata list. Four largely independent pieces:

- :func:`compute_internight_variability` (B2) -- a chi2 test of each star's
  nightly means against their (weighted) grand mean, catching a star whose
  level shifted between nights (an eclipsing binary caught only once, a
  slow trend, ...).
- :func:`cross_reference_nights` (B3) -- pulls each night's own
  ``lc/night_lc_search_metrics.parquet`` (written by ``relphot search``) back
  in via the cross-match, so a star's per-night verdicts and single-night
  transit events are visible together.
- :func:`combined_periodogram` (B4) and :func:`bls_search` (B6) -- astropy
  Lomb-Scargle/Box-Least-Squares over the full multi-night calibrated epoch
  series, looped over stars (astropy's own period-grid evaluation is
  internally vectorised; the outer loop here is over stars, not periods).
- :func:`period_compatibility` (B5) -- the key new capability: a single
  per-night transit event does not by itself determine a period, but every
  *other* night's flat (or matching) data at the predicted in-transit times
  constrains it. Vectorised over the period grid via cumulative sums and
  `~numpy.searchsorted` over each other night's sorted epochs (a Python loop
  over the -- few -- other nights is fine; the period axis never is).

:func:`relphot.cli` wires these into ``relphot multisearch``.
"""

from __future__ import annotations

import csv
import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

from relphot.config import MultiNightSettings
from relphot.exceptions import MultiNightError
from relphot.multinight import MultiNightLightCurves, NightCrossMatch, NightTie
from relphot.numeric import mad_sigma

logger = logging.getLogger(__name__)

__all__ = [
    "CrossReference",
    "PeriodCompatibility",
    "bls_search",
    "build_search_metrics_columns",
    "combined_periodogram",
    "commensurate_periods",
    "compute_internight_variability",
    "cross_reference_nights",
    "period_compatibility",
    "plot_candidate_star",
    "run_multisearch",
    "save_period_compat_csvs",
    "save_search_metrics_table",
    "save_transits_csv",
    "save_variables_csv",
    "select_periodogram_stars",
]


def compute_internight_variability(
    mlc: MultiNightLightCurves, settings: MultiNightSettings
) -> dict[str, np.ndarray]:
    """B2: inter-night (long-term) variability chi2 test on nightly means.

    For every global star with ``n_nights >= settings.min_nights``:
    ``chi2 = sum_n ((night_mean_mag - M_w) / night_mean_err)**2`` with
    ``M_w`` the inverse-variance weighted mean over its nights, ``dof =
    n_nights - 1``, ``p = scipy.stats.chi2.sf(chi2, dof)``. ``amplitude`` is
    the max-min spread of its nightly means. ``candidate`` requires ``p <
    internight_p_threshold`` and ``amplitude >= internight_min_amplitude_mag``.

    Returns a dict of ``(n_global,)`` arrays: ``chi2, p, amplitude,
    weighted_mean, eligible, candidate``.
    """
    night_mean_mag = mlc.night_mean_mag
    night_mean_err = mlc.night_mean_err
    n_global = night_mean_mag.shape[1]

    valid = np.isfinite(night_mean_mag) & np.isfinite(night_mean_err) & (night_mean_err > 0)
    w = np.where(valid, 1.0 / night_mean_err**2, 0.0)
    wtot = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        weighted_mean = np.where(
            wtot > 0, (w * np.where(valid, night_mean_mag, 0.0)).sum(axis=0) / wtot, np.nan
        )
        chi2 = np.nansum(
            np.where(valid, ((night_mean_mag - weighted_mean[None, :]) / night_mean_err) ** 2, 0.0),
            axis=0,
        )

    any_valid = valid.any(axis=0)
    amplitude = np.full(n_global, np.nan)
    if np.any(any_valid):
        sub = np.where(valid[:, any_valid], night_mean_mag[:, any_valid], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            amplitude[any_valid] = np.nanmax(sub, axis=0) - np.nanmin(sub, axis=0)

    dof = np.maximum(mlc.n_nights - 1, 0)
    p = np.full(n_global, np.nan)
    pos_dof = dof > 0
    if np.any(pos_dof):
        with np.errstate(invalid="ignore"):
            p[pos_dof] = scipy_stats.chi2.sf(chi2[pos_dof], dof[pos_dof])

    eligible = mlc.n_nights >= settings.min_nights
    candidate = (
        eligible
        & np.isfinite(p)
        & (p < settings.internight_p_threshold)
        & np.isfinite(amplitude)
        & (amplitude >= settings.internight_min_amplitude_mag)
    )
    return {
        "chi2": chi2,
        "p": p,
        "amplitude": amplitude,
        "weighted_mean": weighted_mean,
        "eligible": eligible,
        "candidate": candidate,
    }


@dataclass(slots=True)
class CrossReference:
    """B3: per-night ``relphot search`` results, cross-referenced onto the global star list.

    ``excess_by_night``/``ls_period_by_night`` map night label -> ``(n_global,)``
    array (NaN where that night has no per-night search or does not contain
    the star). ``n_nights_var_candidate``/``recurrent_variable`` count/flag
    the variability verdict across nights. ``events`` maps a global id with
    >= 1 per-night transit event to a list of event dicts (``night``,
    ``night_index``, ``tc``, ``depth``, ``sigma_depth``, ``duration_hours``,
    ``snr``, ``tier``, ``flags``). ``night_available`` is False for a night
    whose ``night_lc_search_metrics.parquet`` was missing or unreadable.
    """

    n_nights_var_candidate: np.ndarray
    recurrent_variable: np.ndarray
    excess_by_night: dict[str, np.ndarray]
    ls_period_by_night: dict[str, np.ndarray]
    events: dict[int, list[dict]]
    known_variable_name: np.ndarray
    known_planet_name: np.ndarray
    gaia_id: np.ndarray
    night_available: np.ndarray


def _local_column(
    table, star_id: np.ndarray, size: int, colname: str, fill=np.nan, dtype=np.float64,
):
    """``table[colname]`` scattered onto a dense ``(size,)`` array indexed by ``star_id``."""
    arr = np.full(size, fill, dtype=dtype)
    if colname in table.colnames and size:
        arr[star_id] = np.asarray(table[colname], dtype=dtype)
    return arr


def cross_reference_nights(
    night_info: list[dict], xmatch: NightCrossMatch,
) -> CrossReference:
    """Read each night's per-night search metrics (if present) back onto the global list.

    Reads ``lc/night_lc_search_metrics.parquet`` under each night's directory.

    A missing or unreadable file logs a warning and leaves that night out
    (its columns stay NaN/empty everywhere) -- the rest still runs.
    """
    from astropy.table import Table

    n_global = xmatch.ra.shape[0]
    n_nights = len(night_info)

    var_candidate_count = np.zeros(n_global, dtype=np.int64)
    excess_by_night: dict[str, np.ndarray] = {}
    ls_period_by_night: dict[str, np.ndarray] = {}
    known_variable_name = np.full(n_global, "", dtype=object)
    known_planet_name = np.full(n_global, "", dtype=object)
    gaia_id = np.full(n_global, "", dtype=object)
    night_available = np.zeros(n_nights, dtype=bool)
    events: dict[int, list[dict]] = {}

    for n, info in enumerate(night_info):
        label = info["label"]
        excess_by_night[label] = np.full(n_global, np.nan)
        ls_period_by_night[label] = np.full(n_global, np.nan)

        path = Path(info["directory"]) / "lc" / "night_lc_search_metrics.parquet"
        if not path.is_file():
            logger.warning(
                "night %s: %s not found; treating as no per-night search", label, path
            )
            continue
        try:
            table = Table.read(path)
        except Exception:
            logger.warning(
                "night %s: failed to read %s; treating as no per-night search",
                label, path, exc_info=True,
            )
            continue
        night_available[n] = True

        star_id = np.asarray(table["star_id"], dtype=np.int64)
        size = int(star_id.max()) + 1 if star_id.size else 0

        excess_local = _local_column(table, star_id, size, "variability_excess")
        ls_period_local = _local_column(table, star_id, size, "variability_ls_period_days")
        var_cand_local = _local_column(
            table, star_id, size, "variability_candidate", fill=False, dtype=bool
        )
        transit_cand_local = _local_column(
            table, star_id, size, "transit_candidate", fill=False, dtype=bool
        )
        tc_local = _local_column(table, star_id, size, "transit_tc_bjd_tdb")
        depth_local = _local_column(table, star_id, size, "transit_depth")
        snr_local = _local_column(table, star_id, size, "transit_snr")
        dur_local = _local_column(table, star_id, size, "transit_duration_hours")
        tier_local = _local_column(table, star_id, size, "transit_tier", fill=-1, dtype=np.int64)
        flags_local = _local_column(
            table, star_id, size, "transit_flags_str", fill="", dtype=object
        )
        known_var_local = _local_column(
            table, star_id, size, "known_variable_name", fill="", dtype=object
        )
        known_planet_local = _local_column(
            table, star_id, size, "known_planet_name", fill="", dtype=object
        )
        gaia_local = _local_column(table, star_id, size, "gaia_id", fill="", dtype=object)

        idx_n = xmatch.index[n]
        present = (idx_n >= 0) & (idx_n < size)
        local_present = idx_n[present]

        excess_by_night[label][present] = excess_local[local_present]
        ls_period_by_night[label][present] = ls_period_local[local_present]

        var_here = np.zeros(n_global, dtype=bool)
        var_here[present] = var_cand_local[local_present]
        var_candidate_count += var_here.astype(np.int64)

        for g in np.nonzero(present)[0]:
            local = idx_n[g]
            if known_var_local[local]:
                known_variable_name[g] = known_variable_name[g] or known_var_local[local]
            if known_planet_local[local]:
                known_planet_name[g] = known_planet_name[g] or known_planet_local[local]
            if gaia_local[local]:
                gaia_id[g] = gaia_id[g] or gaia_local[local]

        transit_here = np.zeros(n_global, dtype=bool)
        transit_here[present] = transit_cand_local[local_present]
        for g in np.nonzero(transit_here)[0]:
            local = idx_n[g]
            depth = float(depth_local[local])
            snr = float(snr_local[local])
            has_snr = np.isfinite(depth) and np.isfinite(snr) and snr > 0
            sigma_depth = depth / snr if has_snr else np.nan
            events.setdefault(int(g), []).append({
                "night": label,
                "night_index": n,
                "tc": float(tc_local[local]),
                "depth": depth,
                "sigma_depth": sigma_depth,
                "duration_hours": float(dur_local[local]),
                "snr": snr,
                "tier": int(tier_local[local]),
                "flags": str(flags_local[local]),
            })

    recurrent_variable = var_candidate_count >= 2
    return CrossReference(
        n_nights_var_candidate=var_candidate_count,
        recurrent_variable=recurrent_variable,
        excess_by_night=excess_by_night,
        ls_period_by_night=ls_period_by_night,
        events=events,
        known_variable_name=known_variable_name,
        known_planet_name=known_planet_name,
        gaia_id=gaia_id,
        night_available=night_available,
    )


def select_periodogram_stars(
    mlc: MultiNightLightCurves, internight: dict[str, np.ndarray], cross_ref: CrossReference,
    settings: MultiNightSettings,
) -> np.ndarray:
    """B4/B6 star selection: ``"candidates"`` (union of every flagged source) or ``"all"``.

    ``"all"`` is every star with ``n_nights >= settings.min_nights`` and
    ``>= 20`` finite calibrated epoch magnitudes.
    """
    n_global = mlc.mag.shape[0]
    if settings.periodogram_stars == "all":
        n_finite = np.count_nonzero(np.isfinite(mlc.mag), axis=1)
        return (mlc.n_nights >= settings.min_nights) & (n_finite >= 20)

    # "any per-night variability candidate" is exactly n_nights_var_candidate >= 1.
    any_var_candidate = cross_ref.n_nights_var_candidate >= 1
    transit_event_star = np.zeros(n_global, dtype=bool)
    if cross_ref.events:
        transit_event_star[np.array(list(cross_ref.events.keys()), dtype=np.int64)] = True

    return (
        internight["candidate"]
        | cross_ref.recurrent_variable
        | any_var_candidate
        | transit_event_star
    )


def combined_periodogram(
    tie: NightTie, mlc: MultiNightLightCurves, star_ids: np.ndarray, settings: MultiNightSettings,
) -> dict[str, np.ndarray]:
    """B4: astropy Lomb-Scargle on calibrated epoch magnitudes, per star in ``star_ids``.

    Errors are ``hypot(mag_err, floor_at(night, aperture, mean_mag))`` --
    the statistical epoch error combined in quadrature with the tied
    night's calibration floor at that star's magnitude. Frequency grid:
    ``[1/baseline, periodogram_max_freq]`` at ``periodogram_samples_per_peak``.
    Looped over stars (astropy's own period-grid evaluation is vectorised
    internally); logs progress and the elapsed time.
    """
    from astropy.timeseries import LombScargle

    n = len(star_ids)
    best_period = np.full(n, np.nan)
    best_power = np.full(n, np.nan)
    best_fap = np.full(n, np.nan)
    second_period = np.full(n, np.nan)
    candidate = np.zeros(n, dtype=bool)

    eligible_nights = mlc.n_nights[star_ids] >= settings.min_nights_periodic
    n_skipped = int(np.count_nonzero(~eligible_nights))
    if n_skipped:
        logger.info(
            "combined Lomb-Scargle: skipping %d/%d candidate stars with < %d nights of data",
            n_skipped, n, settings.min_nights_periodic,
        )

    t0 = time.monotonic()
    for i, g in enumerate(star_ids):
        if not eligible_nights[i]:
            continue
        aper = int(mlc.aperture[g])
        if aper < 0:
            continue
        mag_g = float(mlc.mean_mag[g])
        floor_per_night = np.array([
            tie.floor_at(int(n_idx), aper, np.array([mag_g]))[0] for n_idx in range(len(tie.labels))
        ])
        floor_epoch = floor_per_night[mlc.night_of_frame]
        y = mlc.mag[g].astype(np.float64)
        err = np.hypot(mlc.mag_err[g].astype(np.float64), floor_epoch)
        good = np.isfinite(y) & np.isfinite(err) & (err > 0)
        if np.count_nonzero(good) < 20:
            continue
        tt, yy, ee = mlc.bjd_tdb[good], y[good], err[good]
        baseline = float(tt.max() - tt.min())
        min_freq = 1.0 / baseline if baseline > 0 else np.nan
        max_freq = settings.periodogram_max_freq
        if not np.isfinite(min_freq) or min_freq >= max_freq:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ls = LombScargle(tt, yy, ee)
            freq, power = ls.autopower(
                minimum_frequency=min_freq, maximum_frequency=max_freq,
                samples_per_peak=settings.periodogram_samples_per_peak,
            )
            if power.size == 0:
                continue
            best = int(np.argmax(power))
            best_period[i] = 1.0 / freq[best]
            best_power[i] = power[best]
            best_fap[i] = float(ls.false_alarm_probability(power[best]))

            window = max(round(freq.size * 0.01), 3)
            lo, hi = max(best - window, 0), min(best + window + 1, freq.size)
            mask = np.ones(freq.size, dtype=bool)
            mask[lo:hi] = False
            if np.any(mask):
                second = int(np.argmax(np.where(mask, power, -np.inf)))
                second_period[i] = 1.0 / freq[second]
        candidate[i] = np.isfinite(best_fap[i]) and best_fap[i] < settings.ls_fap_threshold

        if (i + 1) % 500 == 0:
            logger.info(
                "combined Lomb-Scargle: %d/%d stars (%.1f s elapsed)",
                i + 1, n, time.monotonic() - t0,
            )
    logger.info("combined Lomb-Scargle: %d stars in %.1f s", n, time.monotonic() - t0)

    return {
        "period": best_period, "power": best_power, "fap": best_fap,
        "second_period": second_period, "candidate": candidate,
    }


@dataclass(slots=True)
class PeriodCompatibility:
    """B5: per-period exclusion/support verdict for one star's reference transit event.

    ``status`` is int8 per period: -1 excluded, 0 unconstrained, 1
    supported. ``depth_obs``/``sigma_obs`` are the most-constraining
    covered box's values per period (the box, across every other night and
    every commensurate epoch there, with the largest
    ``|depth_obs - depth| / sigma_total``); NaN where no other night has a
    box meeting ``event_min_box_coverage`` at that period.
    ``allowed_intervals`` merges contiguous non-excluded periods into a
    compact string, e.g. ``"0.51-0.63;1.20-1.34;>7.9"`` (an interval
    reaching the grid's own upper end is shown open-ended). ``best_period``/
    ``best_snr`` are the highest combined-depth-SNR *supported* period, NaN
    if none.
    """

    periods: np.ndarray
    status: np.ndarray
    depth_obs: np.ndarray
    sigma_obs: np.ndarray
    frac_excluded: float
    frac_supported: float
    frac_unconstrained: float
    allowed_intervals: str
    best_period: float
    best_snr: float


def _format_intervals(periods: np.ndarray, allowed_mask: np.ndarray) -> str:
    """Merged contiguous runs of ``allowed_mask`` as a compact string.

    See :class:`PeriodCompatibility`.
    """
    if not np.any(allowed_mask):
        return ""
    idx = np.nonzero(allowed_mask)[0]
    breaks = np.nonzero(np.diff(idx) > 1)[0]
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    parts = []
    for s, e in zip(starts, ends, strict=True):
        if e == periods.size - 1:
            parts.append(f">{periods[s]:.3g}")
        else:
            parts.append(f"{periods[s]:.3g}-{periods[e]:.3g}")
    return ";".join(parts)


def period_compatibility(
    t: np.ndarray, flux: np.ndarray, err: np.ndarray, night_id: np.ndarray,
    t0: float, depth: float, sigma_depth: float, duration_days: float,
    periods: np.ndarray, settings: MultiNightSettings,
) -> PeriodCompatibility:
    """B5: which candidate periods are excluded/supported by every *other* night's data.

    ``t``/``flux``/``err``/``night_id`` are a star's full multi-night epoch
    arrays (e.g. ``mlc.bjd_tdb``, ``mlc.flux_norm[g]``, ``mlc.flux_norm_err[g]``,
    ``mlc.night_of_frame``); ``t0``/``depth``/``sigma_depth``/``duration_days``
    describe its single reference transit event (its highest-SNR per-night
    event). The event's own night is inferred as the night of the epoch
    nearest ``t0`` and excluded from the "other nights" loop.

    A period-grid point is excluded if any covered predicted box (coverage
    >= ``event_min_box_coverage``) observes a depth more than
    ``event_exclusion_sigma`` *below* the expectation; supported if some
    covered box is itself a >= ``event_exclusion_sigma`` detection whose
    depth is within ``depth_consistency_sigma`` of the expectation;
    otherwise unconstrained. "Out-of-transit level" is approximated by the
    night's own overall median flux (already ~1 by construction: `flux` is
    per-night median-normalised), since transits/dips occupy only a small
    fraction of any night -- this keeps the whole box-statistics
    computation a `~numpy.searchsorted` over cumulative sums, vectorised
    over the full period grid, with no per-period Python loop (the loop
    here is over the -- few -- other nights).
    """
    periods = np.asarray(periods, dtype=np.float64)
    n_periods = periods.size
    status = np.zeros(n_periods, dtype=np.int8)
    depth_obs = np.full(n_periods, np.nan)
    sigma_obs = np.full(n_periods, np.nan)
    best_absz = np.full(n_periods, -1.0)
    covered_any = np.zeros(n_periods, dtype=bool)

    t = np.asarray(t, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    night_id = np.asarray(night_id)
    duration_days = float(duration_days)

    finite_all = np.isfinite(t) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    if np.any(finite_all):
        ref_night = night_id[finite_all][np.argmin(np.abs(t[finite_all] - t0))]
    else:
        ref_night = None
    other_nights = [n for n in np.unique(night_id) if n != ref_night]

    for on in other_nights:
        mask = finite_all & (night_id == on)
        if np.count_nonzero(mask) < 3:
            continue
        t_n, flux_n, err_n = t[mask], flux[mask], err[mask]
        order = np.argsort(t_n)
        t_s, flux_s, err_s = t_n[order], flux_n[order], err_n[order]
        w_s = 1.0 / err_s**2
        span_start, span_end = float(t_s[0]), float(t_s[-1])
        cadence = float(np.median(np.diff(t_s))) if t_s.size > 1 else duration_days
        baseline_level = float(np.median(flux_s))
        scatter = float(mad_sigma(flux_s))
        n_total = t_s.size

        cw = np.concatenate([[0.0], np.cumsum(w_s)])
        cwf = np.concatenate([[0.0], np.cumsum(w_s * flux_s)])

        k_min = np.ceil((span_start - duration_days / 2.0 - t0) / periods).astype(np.int64)
        k_max = np.floor((span_end + duration_days / 2.0 - t0) / periods).astype(np.int64)
        n_k = np.clip(k_max - k_min + 1, 0, None)
        k_width = int(n_k.max()) if n_k.size else 0
        if k_width <= 0:
            continue

        j = np.arange(k_width)
        k_grid = k_min[:, None] + j[None, :]
        valid_k = (j[None, :] < n_k[:, None])
        tc = t0 + k_grid * periods[:, None]

        left = tc - duration_days / 2.0
        right = tc + duration_days / 2.0
        li = np.searchsorted(t_s, left)
        ri = np.searchsorted(t_s, right)
        n_in = ri - li
        sumw = cw[ri] - cw[li]
        sumwf = cwf[ri] - cwf[li]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_in = np.where(sumw > 0, sumwf / sumw, np.nan)
            sigma_in = np.where(sumw > 0, np.sqrt(1.0 / sumw), np.nan)
            n_out = n_total - n_in
            sigma_out = np.where(n_out > 0, scatter / np.sqrt(np.maximum(n_out, 1)), np.nan)
        sigma_box = np.hypot(sigma_in, sigma_out)
        depth_box = baseline_level - mean_in
        coverage = np.clip(n_in * cadence / max(duration_days, 1e-9), 0.0, 1.0)

        box_valid = valid_k & (coverage >= settings.event_min_box_coverage) & np.isfinite(depth_box)
        if not np.any(box_valid):
            continue
        depth_box_f = np.where(box_valid, depth_box, np.nan)
        sigma_box_f = np.where(box_valid, sigma_box, np.nan)
        sigma_total = np.hypot(sigma_box_f, sigma_depth)

        excluded_box = box_valid & (
            depth_box_f < depth - settings.event_exclusion_sigma * sigma_total
        )
        supported_box = (
            box_valid
            & (depth_box_f > settings.event_exclusion_sigma * sigma_box_f)
            & (np.abs(depth_box_f - depth) < settings.depth_consistency_sigma * sigma_total)
        )

        row_has_valid = np.any(box_valid, axis=1)
        covered_any |= row_has_valid
        excl_p = np.any(excluded_box, axis=1)
        supp_p = np.any(supported_box, axis=1)
        status = np.where(excl_p, np.int8(-1), status)
        status = np.where((status != -1) & supp_p, np.int8(1), status)

        with np.errstate(invalid="ignore", divide="ignore"):
            z_box = np.where(box_valid, np.abs(depth_box_f - depth) / sigma_total, -1.0)
        best_j = np.argmax(z_box, axis=1)
        cand_absz = z_box[np.arange(n_periods), best_j]
        take = row_has_valid & (cand_absz > best_absz)
        depth_obs = np.where(take, depth_box_f[np.arange(n_periods), best_j], depth_obs)
        sigma_obs = np.where(take, sigma_box_f[np.arange(n_periods), best_j], sigma_obs)
        best_absz = np.where(take, cand_absz, best_absz)

    frac_excluded = float(np.mean(status == -1))
    frac_supported = float(np.mean(status == 1))
    frac_unconstrained = float(np.mean(status == 0))
    allowed_intervals = _format_intervals(periods, status != -1)

    best_period = float("nan")
    best_snr = float("nan")
    supported = status == 1
    if np.any(supported):
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = depth_obs / np.where(sigma_obs > 0, sigma_obs, np.inf)
        snr_supported = np.where(supported, snr, -np.inf)
        j = int(np.argmax(snr_supported))
        best_period = float(periods[j])
        best_snr = float(snr[j])

    return PeriodCompatibility(
        periods=periods, status=status, depth_obs=depth_obs, sigma_obs=sigma_obs,
        frac_excluded=frac_excluded, frac_supported=frac_supported,
        frac_unconstrained=frac_unconstrained, allowed_intervals=allowed_intervals,
        best_period=best_period, best_snr=best_snr,
    )


def commensurate_periods(
    events: list[dict], period_min_days: float, depth_consistency_sigma: float,
) -> list[dict]:
    """If >= 2 per-night events exist: candidate periods |tc_i - tc_j| / k, and their consistency.

    Returns a list of dicts, one per event pair: ``dt_days, depth_consistent
    (bool), depth_sigma, candidate_periods`` (a list of ``ΔT/k`` while >=
    ``period_min_days``). Marking which candidate periods fall in a star's
    allowed intervals is left to the caller (it has the grid/status; this
    function only enumerates the commensurate values).
    """
    out = []
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            ei, ej = events[i], events[j]
            dt = abs(ej["tc"] - ei["tc"])
            if dt <= 0:
                continue
            sigma_d = float(np.hypot(ei["sigma_depth"], ej["sigma_depth"]))
            depth_consistent = (
                np.isfinite(sigma_d) and sigma_d > 0
                and abs(ei["depth"] - ej["depth"]) / sigma_d < depth_consistency_sigma
            )
            candidates = []
            k = 1
            while dt / k >= period_min_days:
                candidates.append(dt / k)
                k += 1
            out.append({
                "night_i": ei["night"], "night_j": ej["night"], "dt_days": dt,
                "depth_consistent": bool(depth_consistent), "depth_sigma": sigma_d,
                "candidate_periods": candidates,
            })
    return out


def bls_search(
    mlc: MultiNightLightCurves, star_ids: np.ndarray, settings: MultiNightSettings,
) -> dict[str, np.ndarray]:
    """B6: astropy BoxLeastSquares on ``flux_norm`` for each star in ``star_ids``.

    No per-star free detrending before the search (this project's
    transit-safe rule): only the per-night median normalisation already in
    ``flux_norm``. Looped over stars; each call's own period grid
    (``BoxLeastSquares.autoperiod``) and power evaluation
    (``BoxLeastSquares.power``) are astropy's own vectorised implementation,
    not a per-period Python loop here.
    """
    from astropy.timeseries import BoxLeastSquares

    n = len(star_ids)
    period = np.full(n, np.nan)
    t0 = np.full(n, np.nan)
    depth = np.full(n, np.nan)
    depth_snr = np.full(n, np.nan)
    duration = np.full(n, np.nan)
    nights_in_transit = np.zeros(n, dtype=np.int64)
    candidate = np.zeros(n, dtype=bool)

    # A trial duration close to or longer than a night's own span is a
    # night-to-night step, not a bounded transit -- cap it at a fraction of
    # the median night span (over every night in mlc, not just this star's).
    night_ids_all = np.unique(mlc.night_of_frame)
    night_spans = []
    for nid in night_ids_all:
        t_n = mlc.bjd_tdb[mlc.night_of_frame == nid]
        if t_n.size >= 2:
            night_spans.append(float(t_n.max() - t_n.min()))
    median_night_span = float(np.median(night_spans)) if night_spans else np.nan

    durations_days = np.asarray([h / 24.0 for h in settings.bls_durations_hours], dtype=np.float64)
    if np.isfinite(median_night_span) and median_night_span > 0:
        max_duration_days = settings.bls_max_duration_fraction * median_night_span
        durations_days = durations_days[durations_days <= max_duration_days]
    logger.info(
        "BLS: median night span %.3f d; using trial durations (d): %s",
        median_night_span, np.round(durations_days, 4).tolist(),
    )
    if durations_days.size == 0:
        logger.warning(
            "BLS: no trial duration survives bls_max_duration_fraction; skipping all stars"
        )

    eligible_nights = mlc.n_nights[star_ids] >= settings.min_nights_periodic
    n_skipped = int(np.count_nonzero(~eligible_nights))
    if n_skipped:
        logger.info(
            "BLS: skipping %d/%d candidate stars with < %d nights of data",
            n_skipped, n, settings.min_nights_periodic,
        )

    start_t = time.monotonic()
    for i, g in enumerate(star_ids):
        if not eligible_nights[i] or durations_days.size == 0:
            continue
        y = mlc.flux_norm[g].astype(np.float64)
        e = mlc.flux_norm_err[g].astype(np.float64)
        t_arr = mlc.bjd_tdb
        good = np.isfinite(t_arr) & np.isfinite(y) & np.isfinite(e) & (e > 0)
        if np.count_nonzero(good) < 20:
            continue
        tt, yy, ee = t_arr[good], y[good], e[good]
        baseline = float(tt.max() - tt.min())
        if baseline <= 0:
            continue
        max_p = min(settings.period_max_days, baseline / 1.0)
        min_p = settings.period_min_days
        if min_p >= max_p:
            continue
        durs = durations_days[durations_days < min_p]
        if durs.size == 0:
            continue
        try:
            bls = BoxLeastSquares(tt, yy, ee)
            periods_grid = bls.autoperiod(durs, minimum_period=min_p, maximum_period=max_p)
            result = bls.power(periods_grid, durs, oversample=settings.period_grid_oversample)
        except Exception:
            logger.debug("BLS failed for star %d", g, exc_info=True)
            continue
        if result.period.size == 0:
            continue

        best = int(np.argmax(result.power))
        p_best = float(result.period[best])
        d_best = float(result.duration[best])
        t0_best = float(result.transit_time[best])
        depth_best = float(result.depth[best])
        snr_best = float(result.depth_snr[best]) if np.isfinite(result.depth_snr[best]) else np.nan

        phase = ((tt - t0_best + p_best / 2.0) % p_best) - p_best / 2.0
        in_transit = np.abs(phase) < d_best / 2.0
        night_of_good = mlc.night_of_frame[good]
        n_nights_in = 0
        for nid in np.unique(night_of_good):
            night_mask = night_of_good == nid
            n_in_night = int(np.count_nonzero(in_transit & night_mask))
            n_out_night = int(np.count_nonzero(~in_transit & night_mask))
            if n_in_night > 0 and n_out_night >= settings.bls_min_out_of_transit:
                n_nights_in += 1

        period[i], t0[i], depth[i] = p_best, t0_best, depth_best
        depth_snr[i], duration[i], nights_in_transit[i] = snr_best, d_best, n_nights_in
        candidate[i] = (
            np.isfinite(snr_best)
            and snr_best >= settings.bls_snr_threshold
            and n_nights_in >= settings.bls_min_nights_in_transit
        )

        if (i + 1) % 500 == 0:
            logger.info("BLS: %d/%d stars (%.1f s elapsed)", i + 1, n, time.monotonic() - start_t)
    logger.info("BLS: %d stars in %.1f s", n, time.monotonic() - start_t)

    return {
        "period": period, "t0": t0, "depth": depth, "depth_snr": depth_snr,
        "duration": duration, "nights_in_transit": nights_in_transit, "candidate": candidate,
    }


def _resolve_table_format(fmt: str) -> str:
    """``fmt`` as given, or (for ``"auto"``) ``"parquet"`` if pyarrow is importable, else
    ``"fits"``.
    """
    import importlib.util

    if fmt != "auto":
        return fmt
    return "parquet" if importlib.util.find_spec("pyarrow") is not None else "fits"


def _write_table(columns: dict, path: Path, fmt: str) -> Path:
    """``columns`` (a plain dict of equal-length 1-D arrays) written to ``path`` as parquet
    or FITS.
    """
    import importlib.util

    from astropy.table import Table

    actual_fmt = _resolve_table_format(fmt)
    if actual_fmt == "parquet" and importlib.util.find_spec("pyarrow") is None:
        msg = "pyarrow is required for parquet output"
        raise MultiNightError(msg)

    columns = {
        key: (np.asarray(value, dtype=str) if np.asarray(value).dtype == object else value)
        for key, value in columns.items()
    }
    table = Table(columns)
    if actual_fmt == "parquet":
        out_path = path.with_suffix(".parquet")
        table.write(out_path, format="parquet", overwrite=True)
    else:
        out_path = path.with_suffix(".fits")
        table.write(out_path, format="fits", overwrite=True)
    logger.info("wrote %s (%d rows)", out_path, len(table))
    return out_path


def build_search_metrics_columns(
    xmatch: NightCrossMatch, mlc: MultiNightLightCurves, internight: dict,
    cross_ref: CrossReference, search_result: dict,
) -> dict:
    """B7: one row per global star for ``multinight_search_metrics.(parquet|fits)``."""
    n_global = xmatch.ra.shape[0]
    columns: dict = {
        "global_id": np.arange(n_global, dtype=np.int64),
        "ra": xmatch.ra,
        "dec": xmatch.dec,
        "mean_mag": mlc.mean_mag,
        "n_nights": mlc.n_nights,
        "internight_chi2": internight["chi2"],
        "internight_p": internight["p"],
        "internight_amplitude": internight["amplitude"],
        "internight_candidate": internight["candidate"],
        "n_nights_var_candidate": cross_ref.n_nights_var_candidate,
        "recurrent_variable": cross_ref.recurrent_variable,
        "known_variable_name": cross_ref.known_variable_name,
        "known_planet_name": cross_ref.known_planet_name,
        "gaia_id": cross_ref.gaia_id,
        "ls_period_days": search_result["ls"]["period"],
        "ls_power": search_result["ls"]["power"],
        "ls_fap": search_result["ls"]["fap"],
        "ls_second_period_days": search_result["ls"]["second_period"],
        "periodic_candidate": search_result["ls"]["candidate"],
        "bls_period_days": search_result["bls"]["period"],
        "bls_t0": search_result["bls"]["t0"],
        "bls_depth": search_result["bls"]["depth"],
        "bls_depth_snr": search_result["bls"]["depth_snr"],
        "bls_duration_days": search_result["bls"]["duration"],
        "bls_nights_in_transit": search_result["bls"]["nights_in_transit"],
        "bls_candidate": search_result["bls"]["candidate"],
    }

    is_variable = (
        internight["candidate"]
        | cross_ref.recurrent_variable
        | (cross_ref.n_nights_var_candidate >= 1)
    )
    columns["bls_flags"] = np.where(is_variable, "VARIABLE", "").astype(object)
    for label, arr in cross_ref.excess_by_night.items():
        columns[f"excess_{label}"] = arr
    for label, arr in cross_ref.ls_period_by_night.items():
        columns[f"ls_period_{label}"] = arr

    n_events = np.zeros(n_global, dtype=np.int64)
    for g, events in cross_ref.events.items():
        n_events[g] = len(events)

    frac_excluded = np.full(n_global, np.nan)
    frac_supported = np.full(n_global, np.nan)
    frac_unconstrained = np.full(n_global, np.nan)
    allowed_intervals = np.full(n_global, "", dtype=object)
    pc_best_period = np.full(n_global, np.nan)
    pc_best_snr = np.full(n_global, np.nan)
    for g, pc in search_result["period_compat"].items():
        frac_excluded[g] = pc.frac_excluded
        frac_supported[g] = pc.frac_supported
        frac_unconstrained[g] = pc.frac_unconstrained
        allowed_intervals[g] = pc.allowed_intervals
        pc_best_period[g] = pc.best_period
        pc_best_snr[g] = pc.best_snr

    columns["n_transit_events"] = n_events
    columns["period_compat_frac_excluded"] = frac_excluded
    columns["period_compat_frac_supported"] = frac_supported
    columns["period_compat_frac_unconstrained"] = frac_unconstrained
    columns["period_compat_allowed_intervals"] = allowed_intervals
    columns["period_compat_best_period_days"] = pc_best_period
    columns["period_compat_best_snr"] = pc_best_snr
    return columns


def save_search_metrics_table(columns: dict, path: Path | str, fmt: str = "auto") -> Path:
    """Write :func:`build_search_metrics_columns`' output as ``{path}.(parquet|fits)``."""
    return _write_table(columns, Path(path), fmt)


def save_variables_csv(
    path_csv: Path | str, xmatch: NightCrossMatch, mlc: MultiNightLightCurves, internight: dict,
    cross_ref: CrossReference, ls_result: dict,
) -> int:
    """``{...}_variables.csv``: inter-night, periodic, or recurrent candidates.

    Returns the row count.
    """
    mask = internight["candidate"] | ls_result["candidate"] | cross_ref.recurrent_variable
    idx = np.nonzero(mask)[0]
    fieldnames = [
        "global_id", "ra", "dec", "mean_mag", "n_nights", "internight_candidate",
        "internight_chi2", "internight_p", "internight_amplitude",
        "recurrent_variable", "n_nights_var_candidate", "periodic_candidate",
        "ls_period_days", "ls_fap", "known_variable_name",
    ]
    path_csv = Path(path_csv)
    with path_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for g in idx:
            writer.writerow({
                "global_id": int(g), "ra": float(xmatch.ra[g]), "dec": float(xmatch.dec[g]),
                "mean_mag": float(mlc.mean_mag[g]), "n_nights": int(mlc.n_nights[g]),
                "internight_candidate": bool(internight["candidate"][g]),
                "internight_chi2": float(internight["chi2"][g]),
                "internight_p": float(internight["p"][g]),
                "internight_amplitude": float(internight["amplitude"][g]),
                "recurrent_variable": bool(cross_ref.recurrent_variable[g]),
                "n_nights_var_candidate": int(cross_ref.n_nights_var_candidate[g]),
                "periodic_candidate": bool(ls_result["candidate"][g]),
                "ls_period_days": float(ls_result["period"][g]),
                "ls_fap": float(ls_result["fap"][g]),
                "known_variable_name": str(cross_ref.known_variable_name[g]),
            })
    logger.info("wrote %s (%d rows)", path_csv, idx.size)
    return int(idx.size)


def save_transits_csv(
    path_csv: Path | str, xmatch: NightCrossMatch, mlc: MultiNightLightCurves,
    cross_ref: CrossReference, bls_result: dict, period_compat: dict[int, PeriodCompatibility],
) -> int:
    """``{...}_transits.csv``: stars with a per-night event or ``bls_candidate``.

    Returns the row count.
    """
    n_global = xmatch.ra.shape[0]
    has_events = np.zeros(n_global, dtype=bool)
    if cross_ref.events:
        has_events[np.array(list(cross_ref.events.keys()), dtype=np.int64)] = True
    mask = has_events | bls_result["candidate"]
    idx = np.nonzero(mask)[0]
    fieldnames = [
        "global_id", "ra", "dec", "mean_mag", "n_nights", "n_transit_events",
        "bls_candidate", "bls_period_days", "bls_depth_snr", "bls_nights_in_transit",
        "period_compat_frac_excluded", "period_compat_frac_supported",
        "period_compat_allowed_intervals", "period_compat_best_period_days",
        "period_compat_best_snr", "known_planet_name",
    ]
    path_csv = Path(path_csv)
    with path_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for g in idx:
            pc = period_compat.get(int(g))
            writer.writerow({
                "global_id": int(g), "ra": float(xmatch.ra[g]), "dec": float(xmatch.dec[g]),
                "mean_mag": float(mlc.mean_mag[g]), "n_nights": int(mlc.n_nights[g]),
                "n_transit_events": len(cross_ref.events.get(int(g), [])),
                "bls_candidate": bool(bls_result["candidate"][g]),
                "bls_period_days": float(bls_result["period"][g]),
                "bls_depth_snr": float(bls_result["depth_snr"][g]),
                "bls_nights_in_transit": int(bls_result["nights_in_transit"][g]),
                "period_compat_frac_excluded": pc.frac_excluded if pc else np.nan,
                "period_compat_frac_supported": pc.frac_supported if pc else np.nan,
                "period_compat_allowed_intervals": pc.allowed_intervals if pc else "",
                "period_compat_best_period_days": pc.best_period if pc else np.nan,
                "period_compat_best_snr": pc.best_snr if pc else np.nan,
                "known_planet_name": str(cross_ref.known_planet_name[g]),
            })
    logger.info("wrote %s (%d rows)", path_csv, idx.size)
    return int(idx.size)


def save_period_compat_csvs(
    out_dir: Path | str, period_compat: dict[int, PeriodCompatibility]
) -> list[Path]:
    """``{out_dir}/g<global_id>.csv`` (period, status, depth_obs, sigma_obs) for each entry."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for g, pc in period_compat.items():
        path = out_dir / f"g{g}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["period", "status", "depth_obs", "sigma_obs"])
            for p, s, d, sg in zip(pc.periods, pc.status, pc.depth_obs, pc.sigma_obs, strict=True):
                writer.writerow([p, int(s), d, sg])
        paths.append(path)
    logger.info("wrote %d period-compatibility CSVs to %s", len(paths), out_dir)
    return paths


def plot_candidate_star(
    path_png: Path | str, g: int, tie: NightTie, mlc: MultiNightLightCurves,
    ls_result: dict, bls_result: dict, period_compat: dict[int, PeriodCompatibility],
) -> None:
    """One panel per night of calibrated magnitude vs time, a phase-fold panel when a
    periodic/BLS period exists, and the period-compatibility strip for a transit-event star.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping candidate plot for star %d", g)
        return

    path_png = Path(path_png)
    n_nights = len(tie.labels)

    # The phase-fold panel is only meaningful -- and only shown -- for a
    # star this session actually called periodic or transiting; "some
    # period value happens to be finite" is not enough (B4/B6 leave a
    # period on stars below their own candidate threshold too).
    is_periodic_candidate = bool(ls_result["candidate"][g])
    is_bls_candidate = bool(bls_result["candidate"][g])
    show_phase = is_periodic_candidate or is_bls_candidate
    period = np.nan
    if is_periodic_candidate and np.isfinite(ls_result["period"][g]):
        period = float(ls_result["period"][g])
    elif is_bls_candidate and np.isfinite(bls_result["period"][g]):
        period = float(bls_result["period"][g])
    show_phase = show_phase and np.isfinite(period)
    pc = period_compat.get(int(g))

    n_panels = n_nights + int(show_phase) + int(pc is not None)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8.0, 2.2 * n_panels), squeeze=False)
    axes = axes[:, 0]

    mag_g = mlc.mag[g]
    mag_err_g = mlc.mag_err[g]
    finite = np.isfinite(mag_g)
    if np.any(finite):
        y_lo, y_hi = np.nanpercentile(mag_g[finite], [1.0, 99.0])
        span = y_hi - y_lo
        pad = 0.1 * span if span > 0 else 0.1
        y_lo -= pad
        y_hi += pad
    else:
        y_lo, y_hi = 0.0, 1.0

    for n in range(n_nights):
        ax = axes[n]
        in_night = mlc.night_of_frame == n
        ax.errorbar(
            mlc.bjd_tdb[in_night], mag_g[in_night], yerr=mag_err_g[in_night],
            fmt=".", ms=3, alpha=0.6,
        )
        ax.set_ylim(y_hi, y_lo)  # bottom = faint (large mag), top = bright
        ax.set_ylabel(tie.labels[n])
    axes[0].set_title(f"global star {g}")

    row = n_nights
    if show_phase:
        phase = ((mlc.bjd_tdb - mlc.bjd_tdb[0]) / period) % 1.0
        ax = axes[row]
        ax.errorbar(
            phase[finite], mag_g[finite], yerr=mag_err_g[finite], fmt=".", ms=4, alpha=0.5,
        )
        ax.set_ylim(y_hi, y_lo)
        ax.set_xlabel("phase")
        ax.set_ylabel(f"P={period:.4f} d")
        row += 1

    if pc is not None:
        ax = axes[row]
        colors = np.where(
            pc.status == -1, "tab:red", np.where(pc.status == 1, "tab:green", "lightgrey")
        )
        ax.scatter(pc.periods, np.ones_like(pc.periods), c=colors, s=8, marker="|")
        ax.set_xscale("log")
        ax.set_yticks([])
        ax.set_xlabel("period (days) -- red excluded, green supported, grey unconstrained")

    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=110)
    plt.close(fig)
    logger.info("wrote %s", path_png)


def _period_grid_for_event(
    duration_days: float, baseline_days: float, settings: MultiNightSettings,
) -> np.ndarray:
    """B5's period grid: uniform in frequency, capped at 2e5 points (logs if capped)."""
    f_min = 1.0 / settings.period_max_days
    f_max = 1.0 / settings.period_min_days
    if np.isfinite(baseline_days) and baseline_days > 0:
        t_baseline = baseline_days
    else:
        t_baseline = settings.period_max_days
    d_nu = duration_days / (settings.period_grid_oversample * t_baseline**2)
    n_grid = int(np.ceil((f_max - f_min) / d_nu)) + 1 if d_nu > 0 else 2
    if n_grid > 200_000:
        logger.info("period grid capped at 200000 points (uncapped would be %d)", n_grid)
        n_grid = 200_000
    freq_grid = np.linspace(f_min, f_max, max(n_grid, 2))
    return np.sort(1.0 / freq_grid)


def run_multisearch(
    xmatch: NightCrossMatch, tie: NightTie, mlc: MultiNightLightCurves, night_info: list[dict],
    settings: MultiNightSettings,
) -> dict:
    """B2-B6, assembled: the whole cross-night search pipeline on one already-tied ``.npz``."""
    internight = compute_internight_variability(mlc, settings)
    cross_ref = cross_reference_nights(night_info, xmatch)

    star_mask = select_periodogram_stars(mlc, internight, cross_ref, settings)
    star_ids = np.nonzero(star_mask)[0]
    logger.info(
        "periodogram/BLS candidate set (%s): %d/%d stars",
        settings.periodogram_stars, star_ids.size, xmatch.ra.shape[0],
    )

    ls_result = combined_periodogram(tie, mlc, star_ids, settings)
    bls_result = bls_search(mlc, star_ids, settings)

    n_global = xmatch.ra.shape[0]

    def _scatter(values, fill=np.nan, dtype=np.float64) -> np.ndarray:
        out = np.full(n_global, fill, dtype=dtype)
        out[star_ids] = np.asarray(values, dtype=dtype)
        return out

    ls_full = {
        "period": _scatter(ls_result["period"]),
        "power": _scatter(ls_result["power"]),
        "fap": _scatter(ls_result["fap"]),
        "second_period": _scatter(ls_result["second_period"]),
        "candidate": _scatter(ls_result["candidate"], fill=False, dtype=bool),
    }
    bls_full = {
        "period": _scatter(bls_result["period"]),
        "t0": _scatter(bls_result["t0"]),
        "depth": _scatter(bls_result["depth"]),
        "depth_snr": _scatter(bls_result["depth_snr"]),
        "duration": _scatter(bls_result["duration"]),
        "nights_in_transit": _scatter(bls_result["nights_in_transit"], fill=0, dtype=np.int64),
        "candidate": _scatter(bls_result["candidate"], fill=False, dtype=bool),
    }

    finite_bjd = mlc.bjd_tdb[np.isfinite(mlc.bjd_tdb)]
    baseline_days = float(finite_bjd.max() - finite_bjd.min()) if finite_bjd.size else np.nan

    period_compat: dict[int, PeriodCompatibility] = {}
    commensurate: dict[int, list[dict]] = {}
    t_pc0 = time.monotonic()
    for g, events in cross_ref.events.items():
        best_event = max(
            events, key=lambda e: e["snr"] if np.isfinite(e["snr"]) else -np.inf
        )
        t0_ref = best_event["tc"]
        depth_ref = best_event["depth"]
        sigma_depth_ref = best_event["sigma_depth"]
        duration_ref_days = best_event["duration_hours"] / 24.0
        if not (
            np.isfinite(t0_ref) and np.isfinite(depth_ref) and np.isfinite(sigma_depth_ref)
            and np.isfinite(duration_ref_days) and duration_ref_days > 0
        ):
            continue

        periods = _period_grid_for_event(duration_ref_days, baseline_days, settings)
        flux_g = mlc.flux_norm[g].astype(np.float64)
        err_g = mlc.flux_norm_err[g].astype(np.float64)
        pc = period_compatibility(
            mlc.bjd_tdb, flux_g, err_g, mlc.night_of_frame,
            t0_ref, depth_ref, sigma_depth_ref, duration_ref_days, periods, settings,
        )
        period_compat[int(g)] = pc
        if len(events) >= 2:
            commensurate[int(g)] = commensurate_periods(
                events, settings.period_min_days, settings.depth_consistency_sigma
            )
    logger.info(
        "period compatibility: %d transit-event stars in %.1f s",
        len(cross_ref.events), time.monotonic() - t_pc0,
    )

    return {
        "internight": internight,
        "cross_ref": cross_ref,
        "star_ids": star_ids,
        "ls": ls_full,
        "bls": bls_full,
        "period_compat": period_compat,
        "commensurate": commensurate,
    }

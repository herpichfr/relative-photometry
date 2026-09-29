"""Periodogram analysis for the relphot results database (``relphot db analyze``).

See docs/DB_PLAN.md ("analyze") for the schema this fills in
(``relphot.periodogram``) and the CLASS/PERIOD rules
:mod:`relphot.db.refresh` applies afterwards.

For each object needing analysis, :func:`analyze` deletes its existing
``relphot.periodogram`` rows and recomputes:

- a Lomb-Scargle periodogram per night (``scope='night:<night_id>'``) on
  that night's own flux, skipped when fewer than 10 points survive;
- a combined Lomb-Scargle periodogram (``scope='combined'``) when the
  object has data in >= 2 nights, on tied magnitudes when a multi-night
  run ties every one of its nights, otherwise on concatenated
  per-night-normalised flux (``periodogram.input`` records which);
- a combined BLS periodogram (``method='BLS'``, ``scope='combined'``) for
  an object with class ``'EXOP'`` or a transit/BLS detection, on
  concatenated per-night-normalised flux, when the combined time span
  allows a non-empty period range;
- for every per-night ``'transit'`` detection, a trapezoid fit
  (``relphot.transit_shape``: tc, depth, T14, ingress fraction, one constant
  baseline, on tied flux when a multi-night tie covers the night), and, for
  every pair of converged fits of one object, a "matching transits"
  probability from depth and shape (``relphot.transit_match``). Events are
  never merged and no status is changed;
- for a variable (``is_var``) or an object with a literature variable period,
  a combined Lomb-Scargle period refined by a 2-harmonic Fourier fit with one
  offset per night (``relphot.period_estimate``, one row per set of nights,
  kept as history). With a literature period it is verified in windows around
  ``lit_period * h`` for h in (0.5, 1, 2), so an eclipsing binary found at half
  the catalogued period is still compared with it; an object with a literature
  period gets a row even when it cannot be verified, with a ``verify_status``
  (``verified``, ``no_literature``, ``lit_period_outside_grid``,
  ``no_peak_in_window``, ``insufficient_data``) and a ``verify_note`` saying why.
  No free trend is fitted to a star's own light curve.

Every periodogram's frequency grid is uniform (``f_k = fmin + k*df``,
cycles/day); a grid that would exceed ``max_periodogram_points`` has its
``df`` increased to fit instead, and ``periodogram.coarsened`` records it.
Work is split into chunks of objects, each fetched, computed, written, and
committed as one transaction, so an interruption keeps every chunk already
committed. The pure per-object computation (:func:`_compute_object`) has no
database access, so it also runs under a :class:`~concurrent.futures.
ProcessPoolExecutor` across chunks when more than one worker is requested.
"""

from __future__ import annotations

import logging
import math
import os
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import psycopg
from psycopg.types.json import Jsonb

from relphot.config import DbSettings, Settings
from relphot.db.refresh import refresh_objects

logger = logging.getLogger(__name__)

__all__ = ["AnalyzeReport", "analyze"]

_DEFAULT_CHUNK_SIZE = 200
_MIN_NIGHT_POINTS = 10
_MAG_PER_LN10 = 2.5 / math.log(10.0)


@dataclass(slots=True)
class AnalyzeReport:
    """Counts and elapsed time from one :func:`analyze` call."""

    n_objects: int
    n_ls_night: int
    n_ls_combined: int
    n_bls: int
    n_coarsened: int
    elapsed_s: float
    n_transit_shapes: int = 0
    n_transit_matches: int = 0
    n_period_estimates: int = 0


@dataclass(slots=True)
class _NightData:
    night_id: int
    night_date: date
    bjd: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    telescope: str | None = None


@dataclass(slots=True)
class _TransitDet:
    det_id: int
    night_id: int
    tc: float
    depth: float | None
    duration_h: float
    flags: str | None = None


@dataclass(slots=True)
class _ObjTask:
    obj_id: int
    is_exop: bool
    bls_eligible: bool
    nights: list[_NightData]
    tie: dict[int, tuple[float, float]] | None
    settings: DbSettings
    is_var: bool = False
    lit_period: float | None = None
    lit_period_err: float | None = None
    lit_catalog: str | None = None
    night_ties: dict[int, tuple[float, float]] = field(default_factory=dict)
    transits: list[_TransitDet] = field(default_factory=list)


@dataclass(slots=True)
class _ObjResult:
    """Everything :func:`_compute_object` returns for one object (all plain Python data)."""

    periodograms: list[tuple]
    shapes: list[dict]
    matches: list[dict]
    estimate: dict | None


def _select_target_obj_ids(
    conn: psycopg.Connection, *, all_candidates: bool, obj_ids: Sequence[int] | None
) -> list[int]:
    """Object ids to analyse: ``obj_ids`` verbatim, else every candidate (``all_candidates``),
    else only dirty candidates (no periodogram row, or touched since the oldest one)."""
    if obj_ids is not None:
        return list(obj_ids)

    candidate_filter = (
        "(o.is_exop OR o.is_var OR EXISTS "
        "(SELECT 1 FROM relphot.detection d WHERE d.obj_id = o.obj_id))"
    )
    if all_candidates:
        sql = f"SELECT o.obj_id FROM relphot.object o WHERE {candidate_filter} ORDER BY o.obj_id"
    else:
        sql = f"""
            SELECT o.obj_id FROM relphot.object o
            WHERE {candidate_filter}
              AND (
                  NOT EXISTS (SELECT 1 FROM relphot.periodogram p WHERE p.obj_id = o.obj_id)
                  OR o.data_updated_at > (
                      SELECT MIN(p.computed_at) FROM relphot.periodogram p
                      WHERE p.obj_id = o.obj_id
                  )
              )
            ORDER BY o.obj_id
        """
    with conn.cursor() as cur:
        cur.execute(sql)
        return [row[0] for row in cur.fetchall()]


def _fetch_chunk_data(
    conn: psycopg.Connection, obj_ids: list[int], settings: DbSettings
) -> dict[int, _ObjTask]:
    """Bulk-fetch every input :func:`_compute_object` needs for ``obj_ids`` in five queries."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT o.obj_id, o.is_exop, o.is_var,
                   EXISTS (
                       SELECT 1 FROM relphot.detection d
                       WHERE d.obj_id = o.obj_id AND d.kind IN ('transit', 'bls')
                   ) AS has_transit_like,
                   lv.period, lv.period_err, lv.catalog
            FROM relphot.object o
            LEFT JOIN LATERAL (
                SELECT cm.period, cm.period_err, cm.catalog
                FROM relphot.catalog_match cm
                WHERE cm.obj_id = o.obj_id
                  AND cm.catalog NOT IN ('NASA Exoplanet Archive', 'TOI')
                ORDER BY (cm.type IS NULL), cm.catalog
                LIMIT 1
            ) lv ON true
            WHERE o.obj_id = ANY(%(obj_ids)s)
            """,
            {"obj_ids": obj_ids},
        )
        meta_rows = cur.fetchall()

        cur.execute(
            """
            SELECT d.obj_id, d.det_id, d.night_id, d.tc_bjd_tdb, d.depth, d.duration_h,
                   d.flags
            FROM relphot.detection d
            WHERE d.obj_id = ANY(%(obj_ids)s) AND d.kind = 'transit'
              AND d.night_id IS NOT NULL AND d.tc_bjd_tdb IS NOT NULL
              AND d.duration_h IS NOT NULL
            ORDER BY d.det_id
            """,
            {"obj_ids": obj_ids},
        )
        transit_rows = cur.fetchall()

        cur.execute(
            """
            SELECT sn.obj_id, sn.night_id, n.night_date, lc.bjd_tdb, lc.flux, lc.flux_err,
                   n.telescope
            FROM relphot.lightcurve lc
            JOIN relphot.star_night sn ON sn.obj_id = lc.obj_id AND sn.night_id = lc.night_id
            JOIN relphot.night n ON n.night_id = sn.night_id
            WHERE lc.obj_id = ANY(%(obj_ids)s)
            ORDER BY sn.obj_id, n.night_date
            """,
            {"obj_ids": obj_ids},
        )
        lc_rows = cur.fetchall()

        cur.execute(
            """
            SELECT t.obj_id, t.night_id, t.mn_run_id, r.loaded_at, t.mag, t.mag_err
            FROM relphot.tie t
            JOIN relphot.mn_run r ON r.mn_run_id = t.mn_run_id
            WHERE t.obj_id = ANY(%(obj_ids)s)
            """,
            {"obj_ids": obj_ids},
        )
        tie_rows = cur.fetchall()

    nights_by_obj: dict[int, list[_NightData]] = {}
    for obj_id, night_id, night_date, bjd, flux, flux_err, telescope in lc_rows:
        nights_by_obj.setdefault(obj_id, []).append(
            _NightData(
                night_id=night_id,
                night_date=night_date,
                bjd=np.asarray(bjd, dtype=np.float64),
                flux=np.asarray(flux, dtype=np.float64),
                flux_err=np.asarray(flux_err, dtype=np.float64),
                telescope=telescope,
            )
        )

    transits_by_obj: dict[int, list[_TransitDet]] = {}
    for obj_id, det_id, night_id, tc, depth, duration_h, det_flags in transit_rows:
        transits_by_obj.setdefault(obj_id, []).append(
            _TransitDet(
                det_id=det_id, night_id=night_id, tc=float(tc),
                depth=None if depth is None else float(depth), duration_h=float(duration_h),
                flags=det_flags,
            )
        )

    # per (obj_id, mn_run_id): the set of this object's night_ids that run ties,
    # and the run's own loaded_at
    runs_by_obj: dict[int, dict[int, tuple[set[int], object]]] = {}
    tie_by_obj: dict[int, dict[int, dict[int, tuple[float, float]]]] = {}
    for obj_id, night_id, mn_run_id, loaded_at, mag, mag_err in tie_rows:
        runs = runs_by_obj.setdefault(obj_id, {})
        nights_set, _ = runs.get(mn_run_id, (set(), loaded_at))
        nights_set.add(night_id)
        runs[mn_run_id] = (nights_set, loaded_at)
        tie_by_obj.setdefault(obj_id, {}).setdefault(mn_run_id, {})[night_id] = (mag, mag_err)

    tasks: dict[int, _ObjTask] = {}
    for obj_id, is_exop, is_var, has_transit_like, lit_period, lit_err, lit_cat in meta_rows:
        nights = sorted(nights_by_obj.get(obj_id, []), key=lambda nd: nd.night_date)
        obj_night_ids = {nd.night_id for nd in nights}

        tie_map: dict[int, tuple[float, float]] | None = None
        night_ties: dict[int, tuple[float, float]] = {}
        runs = runs_by_obj.get(obj_id)
        if runs and obj_night_ids:
            best_run_id = max(
                runs, key=lambda rid: (len(runs[rid][0] & obj_night_ids), runs[rid][1])
            )
            covered, _ = runs[best_run_id]
            night_ties = {
                nid: entry for nid, entry in tie_by_obj[obj_id][best_run_id].items()
                if entry[0] is not None
            }
            if obj_night_ids <= covered:
                tie_map = tie_by_obj[obj_id][best_run_id]

        tasks[obj_id] = _ObjTask(
            obj_id=obj_id,
            is_exop=bool(is_exop),
            bls_eligible=bool(is_exop) or bool(has_transit_like),
            nights=nights,
            tie=tie_map,
            settings=settings,
            is_var=bool(is_var),
            lit_period=None if lit_period is None else float(lit_period),
            lit_period_err=None if lit_err is None else float(lit_err),
            lit_catalog=lit_cat,
            night_ties=night_ties,
            transits=transits_by_obj.get(obj_id, []),
        )
    return tasks


def _build_freq_grid(
    fmin: float, fmax: float, df: float, max_points: int
) -> tuple[np.ndarray, float, bool]:
    """A uniform frequency grid ``f_k = fmin + k*df`` over ``[fmin, fmax]``.

    ``df`` is increased (and ``coarsened`` set) just enough to keep the grid
    at ``max_points`` when it would otherwise be larger. Returns an empty
    array when ``fmax <= fmin`` or ``df`` is non-positive.
    """
    if df <= 0 or fmax <= fmin:
        return np.asarray([], dtype=np.float64), df, False
    n = math.floor((fmax - fmin) / df) + 1
    coarsened = False
    if n > max_points:
        n = max_points
        df = (fmax - fmin) / (n - 1)
        coarsened = True
    freq = fmin + df * np.arange(n, dtype=np.float64)
    return freq, df, coarsened


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


#: PostgreSQL's ``real`` (float4) underflows on a nonzero magnitude below this
#: (its own hardware minimum is far smaller, but a Lomb-Scargle power or
#: false-alarm probability below this floor is already practically zero, and
#: an Baluev FAP on very clean synthetic data can compute far below it).
_REAL_UNDERFLOW_FLOOR = 1e-30


def _real_safe(value: float) -> float | None:
    """``value`` as a value ``relphot.periodogram``'s ``real``/``real[]`` columns can store.

    Non-finite maps to ``None``; a nonzero magnitude below
    ``_REAL_UNDERFLOW_FLOOR`` is clamped up to that floor (keeping its sign)
    instead of being passed through to Postgres, which raises
    ``NumericValueOutOfRange`` on an underflowing ``real``.
    """
    value = float(value)
    if not math.isfinite(value):
        return None
    if 0.0 < abs(value) < _REAL_UNDERFLOW_FLOOR:
        return math.copysign(_REAL_UNDERFLOW_FLOOR, value)
    return value


def _good_night_flux(nd: _NightData) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """This night's (t, flux/median, flux_err/median), finite and positive-error only."""
    t, f, e = nd.bjd, nd.flux, nd.flux_err
    good = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (e > 0) & (f > 0)
    if np.count_nonzero(good) == 0:
        return None
    tt, ff, ee = t[good], f[good], e[good]
    med = np.median(ff)
    if not np.isfinite(med) or med <= 0:
        return None
    return tt, ff / med, ee / med


def _concat_night_normalised_flux(
    task: _ObjTask,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray] | None:
    """Every night's own-median-normalised flux, concatenated and time-sorted.

    The last element is each point's ``night_id``.
    """
    ts: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    dys: list[np.ndarray] = []
    ns: list[np.ndarray] = []
    for nd in task.nights:
        got = _good_night_flux(nd)
        if got is None:
            continue
        tt, yy, ee = got
        ts.append(tt)
        ys.append(yy)
        dys.append(ee)
        ns.append(np.full(tt.size, nd.night_id, dtype=np.int64))
    if not ts:
        return None
    t_all = np.concatenate(ts)
    y_all = np.concatenate(ys)
    dy_all = np.concatenate(dys)
    n_all = np.concatenate(ns)
    order = np.argsort(t_all)
    t_all, y_all, dy_all, n_all = t_all[order], y_all[order], dy_all[order], n_all[order]
    if t_all.size < _MIN_NIGHT_POINTS:
        return None
    span = float(t_all.max() - t_all.min())
    if span <= 0:
        return None
    return t_all, y_all, dy_all, span, n_all


def _combined_series(
    task: _ObjTask,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float, np.ndarray] | None:
    """The combined-LS input: tied magnitudes when a run ties every night, else flux.

    The last element is each point's ``night_id``.
    """
    if task.tie is not None:
        ts: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        dys: list[np.ndarray] = []
        ns: list[np.ndarray] = []
        for nd in task.nights:
            tie_entry = task.tie.get(nd.night_id)
            if tie_entry is None:
                continue
            mag_night, mag_err_night = tie_entry
            got = _good_night_flux(nd)
            if got is None:
                continue
            tt, yy, _ = got
            # the fractional flux error, from the same "good" epochs _good_night_flux used
            t_raw, f_raw, e_raw = nd.bjd, nd.flux, nd.flux_err
            good = (
                np.isfinite(t_raw) & np.isfinite(f_raw) & np.isfinite(e_raw)
                & (e_raw > 0) & (f_raw > 0)
            )
            frac_err = e_raw[good] / f_raw[good]
            mag_epoch = mag_night - 2.5 * np.log10(yy)
            mag_err_epoch = np.hypot(float(mag_err_night), _MAG_PER_LN10 * frac_err)
            ts.append(tt)
            ys.append(mag_epoch)
            dys.append(mag_err_epoch)
            ns.append(np.full(tt.size, nd.night_id, dtype=np.int64))
        if ts:
            t_all = np.concatenate(ts)
            y_all = np.concatenate(ys)
            dy_all = np.concatenate(dys)
            n_all = np.concatenate(ns)
            order = np.argsort(t_all)
            t_all, y_all, dy_all, n_all = t_all[order], y_all[order], dy_all[order], n_all[order]
            if t_all.size >= _MIN_NIGHT_POINTS:
                span = float(t_all.max() - t_all.min())
                if span > 0:
                    return t_all, y_all, dy_all, "tied-mag", span, n_all

    fallback = _concat_night_normalised_flux(task)
    if fallback is None:
        return None
    t_all, y_all, dy_all, span, n_all = fallback
    return t_all, y_all, dy_all, "night-normalised", span, n_all


def _ls_periodogram(
    t: np.ndarray, y: np.ndarray, dy: np.ndarray, span: float, s: DbSettings
) -> tuple | None:
    """``(ls, freq, power, df, coarsened)`` of a Lomb-Scargle on the standard grid, or ``None``."""
    from astropy.timeseries import LombScargle

    fmin = 1.0 / min(s.ls_max_period_days, span)
    fmax = 1.0 / s.ls_min_period_days
    if not (fmax > fmin):
        return None
    df = 1.0 / (s.ls_samples_per_peak * span)
    freq, df, coarsened = _build_freq_grid(fmin, fmax, df, s.max_periodogram_points)
    if freq.size < 2:
        return None

    ls = LombScargle(t, y, dy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        power = ls.power(freq)
    return ls, freq, power, df, coarsened


def _ls_fap(ls, peak_power: float, freq: np.ndarray, s: DbSettings) -> float:
    """Baluev false-alarm probability of ``peak_power`` over the whole ``freq`` grid."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(
            ls.false_alarm_probability(
                peak_power, method="baluev",
                minimum_frequency=float(freq[0]), maximum_frequency=float(freq[-1]),
                samples_per_peak=s.ls_samples_per_peak,
            )
        )


def _run_ls(
    t: np.ndarray, y: np.ndarray, dy: np.ndarray, span: float, s: DbSettings,
    *, scope: str, input_label: str | None,
) -> tuple | None:
    grid = _ls_periodogram(t, y, dy, span, s)
    if grid is None:
        return None
    ls, freq, power, df, coarsened = grid
    best = int(np.argmax(power))
    peak_period = float(1.0 / freq[best])
    peak_power = float(power[best])
    fap = _ls_fap(ls, peak_power, freq, s)
    return (
        "LS", scope, float(freq[0]), float(df), int(freq.size),
        [_real_safe(p) for p in power], peak_period, _real_safe(peak_power), _real_safe(fap),
        bool(coarsened), input_label, None,
    )


def _run_bls(
    t: np.ndarray, y: np.ndarray, dy: np.ndarray, span: float, s: DbSettings
) -> tuple | None:
    from astropy.timeseries import BoxLeastSquares

    period_max = min(s.bls_max_period_days, span / 1.5)
    period_min = s.bls_min_period_days
    if not (period_max > period_min):
        return None
    durations_days = [h / 24.0 for h in s.bls_durations_hours if h / 24.0 < period_min]
    if not durations_days:
        return None

    fmin = 1.0 / period_max
    fmax = 1.0 / period_min
    df = 1.0 / (s.ls_samples_per_peak * span)
    freq, df, coarsened = _build_freq_grid(fmin, fmax, df, s.max_periodogram_points)
    if freq.size < 2:
        return None
    periods = 1.0 / freq

    bls = BoxLeastSquares(t, y, dy)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = bls.power(periods, durations_days, objective="snr")
    except Exception:
        logger.debug("BLS failed", exc_info=True)
        return None

    power = np.asarray(result.power, dtype=np.float64)
    if power.size == 0 or not np.any(np.isfinite(power)):
        return None
    best = int(np.nanargmax(power))
    peak_period = float(result.period[best])
    peak_power = float(power[best])
    extra = {
        "depth": _finite_or_none(result.depth[best]),
        "depth_snr": _finite_or_none(result.depth_snr[best]),
        "t0": _finite_or_none(result.transit_time[best]),
        "duration": _finite_or_none(result.duration[best]),
    }
    return (
        "BLS", "combined", float(freq[0]), float(df), int(freq.size),
        [_real_safe(p) for p in power], peak_period, _real_safe(peak_power), None,
        bool(coarsened), None, extra,
    )


_SHAPE_N_PARAMS = 5
_SHAPE_MIN_POINTS = 8
_HARMONICS = (0.5, 1.0, 2.0)


def _param_errors(jac: np.ndarray, chi2_red: float) -> np.ndarray | None:
    """1-sigma errors from ``(J^T J)^-1`` (columns rescaled first), variance x ``max(1, chi2_red)``.

    ``None`` when the Jacobian is not finite; a parameter the data do not
    constrain (zero column, or non-positive variance) gets ``NaN``.
    """
    norms = np.linalg.norm(jac, axis=0)
    if not np.all(np.isfinite(norms)):
        return None
    norms = np.where(norms > 0, norms, 1.0)
    scaled = jac / norms
    cov = np.linalg.pinv(scaled.T @ scaled) / np.outer(norms, norms) * max(1.0, chi2_red)
    diag = np.diag(cov)
    return np.where(diag > 0, np.sqrt(np.abs(diag)), np.nan)


def _trapezoid_shape(t: np.ndarray, tc: float, t14: float, ingress_frac: float) -> np.ndarray:
    """Unit-depth trapezoid: 1 on the flat bottom, 0 outside ``t14``, linear ingress/egress.

    ``ingress_frac`` is T12/T14 in [0, 0.5]; a floor of ``1e-3 * t14`` on the
    ingress duration keeps the box limit finite.
    """
    half = 0.5 * t14
    tau = max(ingress_frac * t14, 1e-3 * t14)
    return np.clip((half - np.abs(t - tc)) / tau, 0.0, 1.0)


def _incompleteness(
    t_night: np.ndarray, tc: float, t14: float, det: _TransitDet
) -> tuple[bool, str | None, float, bool]:
    """Whether the event is incomplete, why, its observed in-transit span, and ingress+egress seen.

    Incomplete (R6: the duration is then only a minimum) when the detection's flags
    include EDGE or PARTIAL, when the predicted ingress ``tc - t14/2`` is before the
    night's first epoch or the predicted egress ``tc + t14/2`` after its last, or when a
    gap longer than twice the median cadence contains the predicted ingress or egress
    time. A gap in the middle of the transit (dropped frames) does not shorten it and
    is not a reason. Only the current flags and data decide (a previously stored verdict
    is never an input, so a stale one is cleared). Returns
    ``(lower_limit, reason, span, complete_edges)``; ``span`` is the observed part of
    ``[tc - t14/2, tc + t14/2]`` (the lower limit on T14: clipped to the first/last
    epoch, and to the first epoch after / last epoch before a gap covering ingress /
    egress), ``complete_edges`` whether both ingress and egress were observed.
    ``t_night`` is the night's good epochs (any order).
    """
    reasons: list[str] = []
    tokens = {tok for tok in (det.flags or "").split("|") if tok}
    for name in ("EDGE", "PARTIAL"):
        if name in tokens:
            reasons.append(f"flag {name}")

    ingress, egress = tc - 0.5 * t14, tc + 0.5 * t14
    span = t14
    ingress_seen = egress_seen = True
    if t_night.size:
        t_sorted = np.sort(t_night)
        t_first, t_last = float(t_sorted[0]), float(t_sorted[-1])
        ingress_seen = ingress >= t_first
        egress_seen = egress <= t_last
        if not ingress_seen:
            reasons.append("truncated: predicted ingress before the first epoch")
        if not egress_seen:
            reasons.append("truncated: predicted egress after the last epoch")
        lo, hi = max(ingress, t_first), min(egress, t_last)
        if t_sorted.size > 1:
            cadence = float(np.median(np.diff(t_sorted)))

            def gap_around(x: float) -> tuple[float, float] | None:
                """(last epoch before, first epoch after) ``x`` if a >2x-cadence gap holds it."""
                k = int(np.searchsorted(t_sorted, x))
                if 0 < k < t_sorted.size and t_sorted[k - 1] < x < t_sorted[k] and (
                    t_sorted[k] - t_sorted[k - 1] > 2.0 * cadence
                ):
                    return float(t_sorted[k - 1]), float(t_sorted[k])
                return None

            if cadence > 0 and ingress_seen:
                gap = gap_around(ingress)
                if gap is not None:
                    reasons.append("gap covers ingress")
                    ingress_seen = False
                    lo = max(lo, gap[1])
            if cadence > 0 and egress_seen:
                gap = gap_around(egress)
                if gap is not None:
                    reasons.append("gap covers egress")
                    egress_seen = False
                    hi = min(hi, gap[0])
        span = max(hi - lo, 0.0)
    edges_seen = bool(ingress_seen and egress_seen)
    return bool(reasons), "; ".join(reasons) if reasons else None, span, edges_seen


def _fit_transit_shape(
    nd: _NightData, det: _TransitDet, tie_entry: tuple[float, float] | None,
    tie_ref: float | None,
) -> dict:
    """Trapezoid + constant-baseline fit to one night's light curve around ``det``.

    The flux is normalised to the night median; when a multi-night tie covers
    the night it is scaled to the object's mean tied flux
    (``10**(-0.4 (mag_night - tie_ref))``), which changes the baseline's units
    only -- the depth is the fraction of the baseline. Parameters
    ``(tc, depth, t14, ingress_frac, baseline)`` start from the detection's
    values and are bounded to depth > 0, ``t14`` in [0.3, 3] x the detection
    duration, ingress fraction in [0, 0.5], ``tc`` within half a duration of the
    detection. Errors come from ``(J^T J)^-1`` scaled by ``max(1, chi2_red)``.
    No trend is fitted: the baseline is one constant.

    Duration is reported for every event, but for an incomplete one (see
    :func:`_incompleteness`) ``t14_h`` is the observed in-transit span, a lower limit
    (``t14_lower_limit`` true, ``incomplete_reason`` says why): ``t14_err`` is NULL
    and ``ingress_frac``/``ingress_err`` are NULL unless both ingress and egress were
    observed. A failed fit keeps the detection's own values (and its flag-based verdict).
    """
    from scipy.optimize import least_squares

    d0 = det.duration_h / 24.0
    out: dict = {
        "det_id": det.det_id, "tc": det.tc, "tc_err": None,
        "depth": _real_safe(det.depth) if det.depth is not None else None,
        "depth_err": None, "t14_h": det.duration_h, "t14_err": None,
        "ingress_frac": None, "ingress_err": None, "chi2_red": None, "n_points": 0,
        "input": "night", "converged": False, "t14_lower_limit": False,
        "incomplete_reason": None,
    }

    def apply_incompleteness(t_night: np.ndarray, tc: float, t14: float) -> bool:
        """Record the R6 verdict for ``(tc, t14)``; return whether both edges were observed."""
        lower, reason, span, edges_seen = _incompleteness(t_night, tc, t14, det)
        out["t14_lower_limit"] = lower
        out["incomplete_reason"] = reason
        if lower:
            out["t14_h"] = float(span * 24.0)
            out["t14_err"] = None
        return edges_seen

    got = _good_night_flux(nd)
    if got is None or not (d0 > 0):
        apply_incompleteness(np.empty(0), det.tc, d0)
        return out
    t, y, e = got
    t_night = t
    if tie_entry is not None and tie_ref is not None:
        scale = 10.0 ** (-0.4 * (tie_entry[0] - tie_ref))
        y, e = y * scale, e * scale
        out["input"] = "tied"
    # time relative to the detection's tc, so finite-difference steps are not scaled by a BJD
    t_rel = t - det.tc
    sel = np.abs(t_rel) <= max(1.5 * d0, d0 + 1.0 / 24.0)
    t_rel, y, e = t_rel[sel], y[sel], e[sel]
    out["n_points"] = int(t_rel.size)
    in_transit = np.abs(t_rel) <= 0.5 * d0
    if t_rel.size < _SHAPE_MIN_POINTS or np.count_nonzero(in_transit) < 2:
        apply_incompleteness(t_night, det.tc, d0)
        return out

    b0 = float(np.median(y[~in_transit])) if np.any(~in_transit) else float(np.median(y))
    if not (b0 > 0):
        apply_incompleteness(t_night, det.tc, d0)
        return out
    depth0 = det.depth if det.depth is not None and det.depth > 0 else 1.0 - float(
        np.median(y[in_transit])
    ) / b0
    depth0 = float(np.clip(depth0, 1e-4, 0.9))
    x0 = np.array([0.0, depth0, d0, 0.2, b0])  # (tc - det.tc, depth, t14, ingress_frac, baseline)
    lower = np.array([-0.5 * d0, 1e-6, 0.3 * d0, 0.0, 0.1 * b0])
    upper = np.array([0.5 * d0, 1.0, 3.0 * d0, 0.5, 10.0 * b0])

    def resid(x: np.ndarray) -> np.ndarray:
        return (y - x[4] * (1.0 - x[1] * _trapezoid_shape(t_rel, x[0], x[2], x[3]))) / e

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            res = least_squares(
                resid, x0, bounds=(lower, upper),
                x_scale=np.array([0.05 * d0, depth0, d0, 0.25, b0]),
            )
    except (ValueError, np.linalg.LinAlgError):
        logger.debug("transit shape fit failed", exc_info=True)
        apply_incompleteness(t_night, det.tc, d0)
        return out

    x = res.x
    dof = t_rel.size - _SHAPE_N_PARAMS
    chi2_red = float(np.sum(res.fun**2)) / dof
    err = _param_errors(res.jac, chi2_red)
    tc_fit = float(det.tc + x[0])
    out.update(
        tc=tc_fit, depth=_real_safe(x[1]), t14_h=float(x[2] * 24.0),
        ingress_frac=_real_safe(x[3]), chi2_red=_real_safe(chi2_red),
    )
    edges_seen = apply_incompleteness(t_night, tc_fit, float(x[2]))
    if err is not None and bool(res.success) and np.all(np.isfinite(err[:3])):
        out.update(
            tc_err=_finite_or_none(err[0]), depth_err=_real_safe(err[1]),
            t14_err=None if out["t14_lower_limit"] else _finite_or_none(err[2] * 24.0),
            ingress_err=_real_safe(err[3]), converged=True,
        )
    if out["t14_lower_limit"] and not edges_seen:
        # the unobserved side of the trapezoid is not constrained by the data
        out["ingress_frac"] = None
        out["ingress_err"] = None
    return out


def _commensurate_periods(dt: float, period_min: float, cap: int) -> list[float]:
    """``dt / k`` for k = 1, 2, ... while ``>= period_min``, at most ``cap`` values."""
    if not (dt >= period_min > 0):
        return []
    k = np.arange(1, min(cap, int(dt // period_min)) + 1)
    return (dt / k).tolist()


def _match_pairs(
    shapes: list[dict], telescope_of: dict[int, str | None], s: DbSettings
) -> list[dict]:
    """Matching-transit probability for every pair of converged shapes of one object.

    ``z_k = (x_a - x_b) / sqrt(err_a**2 + err_b**2 + sys_k**2)`` for the depth,
    T14 and ingress fraction (see :class:`~relphot.config.DbSettings` for the
    systematic floors); ``chi2 = sum z**2`` over the finite terms, ``dof`` their
    number, ``p_match = chi2.sf(chi2, dof)``. Pairs are ordered by detection id
    (``det_a < det_b``). Nothing here merges events or touches a status.

    Duration (R6): an incomplete event's T14 is only a lower limit ``L``. When exactly
    one event of a pair is a lower limit and the other has a measured ``T``, the T14
    term is one-sided, ``z = max(0, L - T) / sqrt(err_T**2 + sys**2)``: no penalty
    when ``T >= L`` (the term is 0 and still counts in ``dof``), a penalty otherwise.
    When both are lower limits there is no T14 term (dropped from ``dof``). The
    ingress term is dropped whenever either event has no ingress fraction (an event
    whose ingress or egress was not observed stores none).
    """
    from scipy.stats import chi2 as chi2_dist

    ok = sorted((sh for sh in shapes if sh["converged"]), key=lambda sh: sh["det_id"])
    if len(ok) < 2:
        return []

    def col(name: str) -> np.ndarray:
        return np.array([np.nan if sh[name] is None else sh[name] for sh in ok], dtype=np.float64)

    tc, depth, depth_e = col("tc"), col("depth"), col("depth_err")
    t14, t14_e = col("t14_h"), col("t14_err")
    t14_low = np.array([bool(sh.get("t14_lower_limit")) for sh in ok], dtype=bool)
    ing, ing_e = col("ingress_frac"), col("ingress_err")
    tele = np.array([telescope_of.get(sh["det_id"]) for sh in ok], dtype=object)
    ia, ib = np.triu_indices(len(ok), k=1)
    same = tele[ia] == tele[ib]
    dt = np.abs(tc[ia] - tc[ib])

    with np.errstate(all="ignore"):
        cross = np.where(same, 0.0, s.match_depth_sys_frac_cross_telescope)
        sys_depth = (s.match_depth_sys_frac + cross) * 0.5 * (depth[ia] + depth[ib])
        sys_t14 = s.match_t14_sys_frac * 0.5 * (t14[ia] + t14[ib])

        def zscore(x: np.ndarray, x_err: np.ndarray, sys: np.ndarray | float) -> np.ndarray:
            return (x[ia] - x[ib]) / np.sqrt(x_err[ia] ** 2 + x_err[ib] ** 2 + sys**2)

        low_a, low_b = t14_low[ia], t14_low[ib]
        # a lower limit carries no error of its own; the measured event's error sets the scale
        e_a = np.where(low_a, 0.0, t14_e[ia])
        e_b = np.where(low_b, 0.0, t14_e[ib])
        diff = t14[ia] - t14[ib]
        sigma = np.sqrt(e_a**2 + e_b**2 + sys_t14**2)
        z_t14 = np.where(
            low_a & low_b, np.nan,
            np.where(
                low_a, np.maximum(diff, 0.0),
                np.where(low_b, np.maximum(-diff, 0.0), diff),
            ) / sigma,
        )
        z = np.stack(
            [
                zscore(depth, depth_e, sys_depth),
                z_t14,
                zscore(ing, ing_e, s.match_ingress_sys),
            ],
            axis=1,
        )
        finite = np.isfinite(z)
        chi2 = np.where(finite, z * z, 0.0).sum(axis=1)
        dof = finite.sum(axis=1)
        p_match = np.where(dof > 0, chi2_dist.sf(chi2, np.maximum(dof, 1)), np.nan)

    det_ids = np.array([sh["det_id"] for sh in ok], dtype=np.int64)
    rows = []
    for i in range(ia.size):
        rows.append({
            "det_a": int(det_ids[ia[i]]), "det_b": int(det_ids[ib[i]]),
            "dt_days": float(dt[i]),
            "depth_z": _real_safe(z[i, 0]), "t14_z": _real_safe(z[i, 1]),
            "ingress_z": _real_safe(z[i, 2]),
            "chi2": _real_safe(chi2[i]) if dof[i] > 0 else None,
            "dof": int(dof[i]),
            "p_match": _real_safe(p_match[i]),
            "same_telescope": bool(same[i]),
            "commensurate_periods": _commensurate_periods(
                float(dt[i]), s.match_period_min_days, s.match_max_commensurate
            ),
        })
    return rows


def _transit_shapes(task: _ObjTask) -> list[dict]:
    """One :func:`_fit_transit_shape` per per-night transit detection of the object."""
    night_by_id = {nd.night_id: nd for nd in task.nights}
    tie_mags = [entry[0] for entry in task.night_ties.values()]
    tie_ref = float(np.mean(tie_mags)) if tie_mags else None
    shapes = []
    for det in task.transits:
        nd = night_by_id.get(det.night_id)
        if nd is None:
            continue
        shapes.append(_fit_transit_shape(nd, det, task.night_ties.get(det.night_id), tie_ref))
    return shapes


def _refine_fourier(
    t: np.ndarray, y: np.ndarray, dy: np.ndarray, night_idx: np.ndarray, f0: float, span: float
) -> tuple[float, float] | None:
    """Refine frequency ``f0`` with a 2-harmonic Fourier series + one offset per night.

    Nonlinear least squares in ``(f, harmonic amplitudes, offsets)`` with
    ``f`` bounded to ``f0 +/- 0.5/span``. Returns ``(f, sigma_f)``, ``sigma_f``
    from ``(J^T J)^-1`` scaled by ``max(1, sqrt(chi2_red))``; ``None`` when the
    fit is under-determined or fails.
    """
    from scipy.optimize import least_squares

    _, inv = np.unique(night_idx, return_inverse=True)
    n_nights = int(inv.max()) + 1
    n = t.size
    n_par = 5 + n_nights
    if n - n_par < 1:
        return None
    tt = t - float(np.mean(t))
    w = 1.0 / dy
    offsets = np.zeros((n, n_nights))
    offsets[np.arange(n), inv] = 1.0

    def design(f: float) -> np.ndarray:
        phi = 2.0 * np.pi * f * tt
        return np.column_stack(
            [np.cos(phi), np.sin(phi), np.cos(2.0 * phi), np.sin(2.0 * phi), offsets]
        )

    def resid(x: np.ndarray) -> np.ndarray:
        return (y - design(x[0]) @ x[1:]) * w

    def jac(x: np.ndarray) -> np.ndarray:
        phi = 2.0 * np.pi * x[0] * tt
        a1, b1, a2, b2 = x[1:5]
        d_model = 2.0 * np.pi * tt * (
            (-a1 * np.sin(phi) + b1 * np.cos(phi))
            + 2.0 * (-a2 * np.sin(2.0 * phi) + b2 * np.cos(2.0 * phi))
        )
        return np.column_stack([-w * d_model, -design(x[0]) * w[:, None]])

    beta0, *_ = np.linalg.lstsq(design(f0) * w[:, None], y * w, rcond=None)
    x0 = np.concatenate(([f0], beta0))
    half_width = 0.5 / span
    lower = np.concatenate(([max(f0 - half_width, 1e-12)], np.full(n_par - 1, -np.inf)))
    upper = np.concatenate(([f0 + half_width], np.full(n_par - 1, np.inf)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            res = least_squares(resid, x0, jac=jac, bounds=(lower, upper), x_scale="jac")
    except (ValueError, np.linalg.LinAlgError):
        logger.debug("Fourier refinement failed", exc_info=True)
        return None
    chi2_red = float(np.sum(res.fun**2)) / (n - n_par)
    err = _param_errors(res.jac, chi2_red)
    if err is None or not math.isfinite(float(err[0])) or not (res.x[0] > 0):
        return None
    return float(res.x[0]), float(err[0])


#: ``period_estimate.verify_status`` values.
_VERIFIED = "verified"
_NO_LITERATURE = "no_literature"
_OUTSIDE_GRID = "lit_period_outside_grid"
_NO_PEAK = "no_peak_in_window"
_INSUFFICIENT = "insufficient_data"


def _verification_windows(
    freq: np.ndarray, power: np.ndarray, lit: float, window_frac: float
) -> tuple[tuple[float, int, float] | None, str, str | None]:
    """Best LS peak among the literature-period windows ``lit * h * (1 +/- window_frac)``.

    Returns ``(best, status, note)``. ``best`` is ``(power, grid index, h)`` of the
    highest genuine peak (a grid local maximum inside its window) over h in 0.5, 1, 2,
    or ``None``. ``status`` is ``'verified'`` with a best peak, ``'lit_period_outside_grid'``
    when no window overlaps the LS frequency grid (the note says how the literature
    period compares with the grid's period range), else ``'no_peak_in_window'``.
    """
    best: tuple[float, int, float] | None = None
    any_in_grid = False
    for h in _HARMONICS:
        centre = lit * h
        in_window = (freq >= 1.0 / (centre * (1.0 + window_frac))) & (
            freq <= 1.0 / (centre * (1.0 - window_frac))
        )
        if not np.any(in_window):
            continue
        any_in_grid = True
        idx = np.flatnonzero(in_window)
        kk = int(idx[np.nanargmax(power[idx])]) if np.any(np.isfinite(power[idx])) else -1
        if kk < 0 or not np.isfinite(power[kk]):
            continue
        left = power[kk - 1] if kk > 0 else -np.inf
        right = power[kk + 1] if kk + 1 < power.size else -np.inf
        if power[kk] >= left and power[kk] >= right and (best is None or power[kk] > best[0]):
            best = (float(power[kk]), kk, h)
    if best is not None:
        return best, _VERIFIED, None
    if not any_in_grid:
        p_max, p_min = 1.0 / float(freq[0]), 1.0 / float(freq[-1])
        if lit * (1.0 - window_frac) > p_max or lit * 0.5 * (1.0 + window_frac) > p_max:
            note = f"P_lit {lit:g} d > LS max period {p_max:.3g} d"
        elif lit * 2.0 * (1.0 - window_frac) < p_min or lit * (1.0 + window_frac) < p_min:
            note = f"P_lit {lit:g} d < LS min period {p_min:.3g} d"
        else:
            note = f"P_lit {lit:g} d windows fall between LS grid points"
        return None, _OUTSIDE_GRID, note
    pct = window_frac * 100.0
    return None, _NO_PEAK, (
        f"no LS peak within +/-{pct:g}% of P_lit {lit:g} d x 0.5, 1 or 2"
    )


def _period_estimate(task: _ObjTask) -> dict | None:
    """Combined-LS period of a variable, refined, and verified against its literature period.

    Runs for a variable (``is_var``) or an object with a literature variable
    period, on every night the object has (one night is stored too). The period is
    the global LS peak; with a literature period it is instead the highest peak among
    the windows ``lit_period * h * (1 +/- lit_period_window_frac)`` for h in (0.5, 1, 2).
    ``delta = period / h - lit_period``. An object with a literature period gets a row
    for every set of nights even when it cannot be verified; ``verify_status`` says
    which case (``verified``, ``no_literature`` for a variable without a literature
    period, ``lit_period_outside_grid``, ``no_peak_in_window``, ``insufficient_data``
    when there is too little data for any LS) and ``verify_note`` why. Without
    a verified peak the period is the global LS peak and harmonic/delta are NULL.
    """
    s = task.settings
    lit = task.lit_period
    if lit is not None and not (math.isfinite(lit) and lit > 0):
        lit = None
    if not (task.is_var or lit is not None):
        return None

    date_of = {nd.night_id: nd.night_date for nd in task.nights}

    def row(
        night_ids: list[int], span: float | None, input_label: str, status: str,
        note: str | None, **fit: object,
    ) -> dict:
        out = {
            "method": "LS", "input": input_label, "night_ids": night_ids,
            "n_nights": len(night_ids), "last_night": max(date_of[n] for n in night_ids),
            "baseline_days": span, "period": None, "period_err": None, "power": None,
            "fap": None, "lit_period": lit,
            "lit_period_err": task.lit_period_err if lit is not None else None,
            "lit_catalog": task.lit_catalog if lit is not None else None,
            "harmonic": None, "delta": None, "delta_err": None, "delta_z": None,
            "verify_status": status, "verify_note": note,
        }
        out.update(fit)
        return out

    series = _combined_series(task)
    grid = None
    if series is not None:
        grid = _ls_periodogram(series[0], series[1], series[2], series[4], s)
    if series is None or grid is None or not np.any(np.isfinite(grid[2])):
        if lit is None:
            return None
        usable = [nd for nd in task.nights if _good_night_flux(nd) is not None]
        if not usable:
            return None
        times = np.concatenate([nd.bjd[np.isfinite(nd.bjd)] for nd in usable])
        span = float(times.max() - times.min()) if times.size else None
        return row(
            sorted(nd.night_id for nd in usable), span, "night", _INSUFFICIENT,
            "too few usable epochs or no LS frequency grid for this baseline",
        )
    t, y, dy, label, span, night_idx = series
    ls, freq, power, _df, _coarsened = grid
    input_label = "tied" if label == "tied-mag" else "night"
    night_ids = sorted(int(n) for n in np.unique(night_idx))

    k = int(np.nanargmax(power))
    harmonic: float | None = None
    if lit is None:
        status, note = _NO_LITERATURE, None
    else:
        best, status, note = _verification_windows(freq, power, lit, s.lit_period_window_frac)
        if best is not None:
            k, harmonic = best[1], best[2]

    peak_power = float(power[k])
    fap = _ls_fap(ls, peak_power, freq, s)
    period = float(1.0 / freq[k])
    period_err: float | None = None
    refined = _refine_fourier(t, y, dy, night_idx, float(freq[k]), span)
    if refined is not None:
        f_fit, f_err = refined
        period = 1.0 / f_fit
        period_err = f_err / f_fit**2

    delta = delta_err = delta_z = None
    if lit is not None and harmonic is not None:
        delta = period / harmonic - lit
        if period_err is not None:
            delta_err = math.hypot(period_err / harmonic, task.lit_period_err or 0.0)
            if delta_err > 0:
                delta_z = delta / delta_err

    return row(
        night_ids, float(span), input_label, status, note,
        period=period, period_err=period_err, power=_real_safe(peak_power),
        fap=_real_safe(fap), harmonic=harmonic, delta=delta, delta_err=delta_err,
        delta_z=_real_safe(delta_z) if delta_z is not None else None,
    )


def _compute_object(task: _ObjTask) -> _ObjResult:
    """Every analysis result for one object: periodograms, transit shapes/matches, period estimate.

    No database access -- runs standalone in a worker process. Each element
    of ``periodograms`` (LS per night, LS combined, BLS combined) is
    ``(method, scope, fmin, df, n, power, peak_period, peak_power, fap, coarsened, input, extra)``.
    """
    s = task.settings
    rows: list[tuple] = []

    for nd in task.nights:
        got = _good_night_flux(nd)
        if got is None:
            continue
        tt, yy, ee = got
        if tt.size < _MIN_NIGHT_POINTS:
            continue
        span = float(tt.max() - tt.min())
        if span <= 0:
            continue
        row = _run_ls(tt, yy, ee, span, s, scope=f"night:{nd.night_id}", input_label=None)
        if row is not None:
            rows.append(row)

    if len(task.nights) >= 2:
        combined = _combined_series(task)
        if combined is not None:
            t_c, y_c, dy_c, input_label, span_c, _night_c = combined
            row = _run_ls(t_c, y_c, dy_c, span_c, s, scope="combined", input_label=input_label)
            if row is not None:
                rows.append(row)

        if task.bls_eligible:
            flux_combined = _concat_night_normalised_flux(task)
            if flux_combined is not None:
                t_f, y_f, dy_f, span_f, _night_f = flux_combined
                row = _run_bls(t_f, y_f, dy_f, span_f, s)
                if row is not None:
                    rows.append(row)

    shapes = _transit_shapes(task)
    telescope_of = {
        det.det_id: nd.telescope
        for det in task.transits for nd in task.nights if nd.night_id == det.night_id
    }
    return _ObjResult(
        periodograms=rows, shapes=shapes, matches=_match_pairs(shapes, telescope_of, s),
        estimate=_period_estimate(task),
    )


def analyze(
    conn: psycopg.Connection,
    *,
    all_candidates: bool = False,
    obj_ids: Sequence[int] | None = None,
    settings: Settings | None = None,
    workers: int | None = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> AnalyzeReport:
    """Recompute periodograms, transit shapes/matches, period estimates and (via
    :func:`~relphot.db.refresh.refresh_objects`) PERIOD for objects needing analysis.

    ``obj_ids`` (given verbatim) takes priority over ``all_candidates``
    (every planet-host/variable/detected object) over the default (only
    those with no periodogram row, or touched since the oldest one --
    see :mod:`relphot.db.analyze`'s module docstring). ``workers`` is the
    number of :class:`~concurrent.futures.ProcessPoolExecutor` worker
    processes (default: ``os.cpu_count()``); ``1`` runs everything in this
    process. Work is committed one chunk of ``chunk_size`` objects at a
    time, so an interrupted run keeps every chunk already committed.
    """
    t0 = time.monotonic()
    settings = settings if settings is not None else Settings()
    db_settings = settings.db

    target_ids = _select_target_obj_ids(conn, all_candidates=all_candidates, obj_ids=obj_ids)
    if not target_ids:
        return AnalyzeReport(0, 0, 0, 0, 0, time.monotonic() - t0)

    max_workers = workers if workers is not None else (os.cpu_count() or 1)
    chunks = [target_ids[i : i + chunk_size] for i in range(0, len(target_ids), chunk_size)]

    n_ls_night = n_ls_combined = n_bls = n_coarsened = 0
    n_shapes = n_matches = n_estimates = 0
    executor = ProcessPoolExecutor(max_workers=max_workers) if max_workers > 1 else None
    try:
        for chunk in chunks:
            try:
                # Flags first: is_var / lit period (set by loads and catalogue matches)
                # decide which objects get a period estimate.
                refresh_objects(
                    conn, chunk,
                    bls_min_snr=db_settings.bls_min_snr,
                    ls_fap_threshold=db_settings.ls_fap_threshold,
                    class_multinight_kinds=db_settings.class_multinight_kinds,
                )
                tasks = _fetch_chunk_data(conn, chunk, db_settings)
                if executor is not None and len(tasks) > 1:
                    results = list(executor.map(_compute_object, tasks.values()))
                else:
                    results = [_compute_object(task) for task in tasks.values()]

                with conn.cursor() as cur:
                    obj_id_list = list(tasks.keys())
                    cur.execute(
                        "DELETE FROM relphot.periodogram WHERE obj_id = ANY(%(obj_ids)s)",
                        {"obj_ids": obj_id_list},
                    )
                    cur.execute(
                        "DELETE FROM relphot.transit_match WHERE obj_id = ANY(%(obj_ids)s)",
                        {"obj_ids": obj_id_list},
                    )
                    cur.execute(
                        "DELETE FROM relphot.transit_shape WHERE obj_id = ANY(%(obj_ids)s)",
                        {"obj_ids": obj_id_list},
                    )
                    insert_rows = []
                    shape_rows = []
                    match_rows = []
                    estimate_rows = []
                    lower_limit_rows: list[tuple[bool, int]] = []
                    for obj_id, result in zip(tasks.keys(), results, strict=True):
                        shape_rows.extend(
                            (
                                sh["det_id"], obj_id, sh["tc"], sh["tc_err"], sh["depth"],
                                sh["depth_err"], sh["t14_h"], sh["t14_err"],
                                sh["t14_lower_limit"], sh["incomplete_reason"],
                                sh["ingress_frac"], sh["ingress_err"], sh["chi2_red"],
                                sh["n_points"], sh["input"], sh["converged"],
                            )
                            for sh in result.shapes
                        )
                        lower_limit_rows.extend(
                            (bool(sh["t14_lower_limit"]), sh["det_id"]) for sh in result.shapes
                        )
                        match_rows.extend(
                            (
                                m["det_a"], m["det_b"], obj_id, m["dt_days"], m["depth_z"],
                                m["t14_z"], m["ingress_z"], m["chi2"], m["dof"], m["p_match"],
                                m["same_telescope"], m["commensurate_periods"],
                            )
                            for m in result.matches
                        )
                        est = result.estimate
                        if est is not None:
                            estimate_rows.append((
                                obj_id, est["method"], est["input"], est["night_ids"],
                                est["n_nights"], est["last_night"], est["baseline_days"],
                                est["period"], est["period_err"], est["power"], est["fap"],
                                est["lit_period"], est["lit_period_err"], est["lit_catalog"],
                                est["harmonic"], est["delta"], est["delta_err"],
                                est["delta_z"], est["verify_status"], est["verify_note"],
                            ))
                        for (
                            method, scope, fmin, df, n, power, peak_period, peak_power,
                            fap, coarsened, input_label, extra,
                        ) in result.periodograms:
                            insert_rows.append((
                                obj_id, scope, method, fmin, df, n, power, peak_period,
                                peak_power, fap, coarsened, input_label,
                                Jsonb(extra) if extra is not None else None,
                            ))
                            if method == "LS" and scope.startswith("night:"):
                                n_ls_night += 1
                            elif method == "LS" and scope == "combined":
                                n_ls_combined += 1
                            elif method == "BLS":
                                n_bls += 1
                            if coarsened:
                                n_coarsened += 1
                    if insert_rows:
                        cur.executemany(
                            """
                            INSERT INTO relphot.periodogram
                                (obj_id, scope, method, fmin, df, n, power, peak_period,
                                 peak_power, fap, coarsened, input, extra, computed_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                            """,
                            insert_rows,
                        )
                    if shape_rows:
                        cur.executemany(
                            """
                            INSERT INTO relphot.transit_shape
                                (det_id, obj_id, tc, tc_err, depth, depth_err, t14_h, t14_err,
                                 t14_lower_limit, incomplete_reason, ingress_frac, ingress_err,
                                 chi2_red, n_points, input, converged, computed_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                    %s, now())
                            """,
                            shape_rows,
                        )
                        n_shapes += len(shape_rows)
                    if lower_limit_rows:
                        # the recomputed verdict (search flags OR what the fit window shows)
                        # replaces the stored one, so an earlier over-eager true is cleared
                        cur.executemany(
                            "UPDATE relphot.detection SET duration_lower_limit = %s "
                            "WHERE det_id = %s",
                            lower_limit_rows,
                        )
                    if match_rows:
                        cur.executemany(
                            """
                            INSERT INTO relphot.transit_match
                                (det_a, det_b, obj_id, dt_days, depth_z, t14_z, ingress_z, chi2,
                                 dof, p_match, same_telescope, commensurate_periods, computed_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                            """,
                            match_rows,
                        )
                        n_matches += len(match_rows)
                    if estimate_rows:
                        cur.executemany(
                            """
                            INSERT INTO relphot.period_estimate
                                (obj_id, method, input, night_ids, n_nights, last_night,
                                 baseline_days, period, period_err, power, fap, lit_period,
                                 lit_period_err, lit_catalog, harmonic, delta, delta_err,
                                 delta_z, verify_status, verify_note, computed_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                    %s, %s, %s, %s, %s, now())
                            ON CONFLICT (obj_id, method, night_ids) DO UPDATE SET
                                computed_at = now(), input = EXCLUDED.input,
                                n_nights = EXCLUDED.n_nights, last_night = EXCLUDED.last_night,
                                baseline_days = EXCLUDED.baseline_days,
                                period = EXCLUDED.period, period_err = EXCLUDED.period_err,
                                power = EXCLUDED.power, fap = EXCLUDED.fap,
                                lit_period = EXCLUDED.lit_period,
                                lit_period_err = EXCLUDED.lit_period_err,
                                lit_catalog = EXCLUDED.lit_catalog,
                                harmonic = EXCLUDED.harmonic, delta = EXCLUDED.delta,
                                delta_err = EXCLUDED.delta_err, delta_z = EXCLUDED.delta_z,
                                verify_status = EXCLUDED.verify_status,
                                verify_note = EXCLUDED.verify_note
                            """,
                            estimate_rows,
                        )
                        n_estimates += len(estimate_rows)
                    refresh_objects(
                        conn, obj_id_list,
                        bls_min_snr=db_settings.bls_min_snr,
                        ls_fap_threshold=db_settings.ls_fap_threshold,
                        class_multinight_kinds=db_settings.class_multinight_kinds,
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    return AnalyzeReport(
        n_objects=len(target_ids), n_ls_night=n_ls_night, n_ls_combined=n_ls_combined,
        n_bls=n_bls, n_coarsened=n_coarsened, elapsed_s=time.monotonic() - t0,
        n_transit_shapes=n_shapes, n_transit_matches=n_matches, n_period_estimates=n_estimates,
    )

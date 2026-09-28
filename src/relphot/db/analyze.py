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
  allows a non-empty period range.

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
from dataclasses import dataclass
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


@dataclass(slots=True)
class _NightData:
    night_id: int
    night_date: date
    bjd: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray


@dataclass(slots=True)
class _ObjTask:
    obj_id: int
    klass: str | None
    bls_eligible: bool
    nights: list[_NightData]
    tie: dict[int, tuple[float, float]] | None
    settings: DbSettings


def _select_target_obj_ids(
    conn: psycopg.Connection, *, all_candidates: bool, obj_ids: Sequence[int] | None
) -> list[int]:
    """Object ids to analyse: ``obj_ids`` verbatim, else every candidate (``all_candidates``),
    else only dirty candidates (no periodogram row, or touched since the oldest one)."""
    if obj_ids is not None:
        return list(obj_ids)

    candidate_filter = (
        "(o.class IN ('EXOP', 'VAR') OR EXISTS "
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
    """Bulk-fetch every input :func:`_compute_object` needs for ``obj_ids`` in three queries."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT o.obj_id, o.class,
                   EXISTS (
                       SELECT 1 FROM relphot.detection d
                       WHERE d.obj_id = o.obj_id AND d.kind IN ('transit', 'bls')
                   ) AS has_transit_like
            FROM relphot.object o
            WHERE o.obj_id = ANY(%(obj_ids)s)
            """,
            {"obj_ids": obj_ids},
        )
        meta_rows = cur.fetchall()

        cur.execute(
            """
            SELECT sn.obj_id, sn.night_id, n.night_date, lc.bjd_tdb, lc.flux, lc.flux_err
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
    for obj_id, night_id, night_date, bjd, flux, flux_err in lc_rows:
        nights_by_obj.setdefault(obj_id, []).append(
            _NightData(
                night_id=night_id,
                night_date=night_date,
                bjd=np.asarray(bjd, dtype=np.float64),
                flux=np.asarray(flux, dtype=np.float64),
                flux_err=np.asarray(flux_err, dtype=np.float64),
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
    for obj_id, klass, has_transit_like in meta_rows:
        nights = sorted(nights_by_obj.get(obj_id, []), key=lambda nd: nd.night_date)
        obj_night_ids = {nd.night_id for nd in nights}

        tie_map: dict[int, tuple[float, float]] | None = None
        runs = runs_by_obj.get(obj_id)
        if runs and obj_night_ids:
            best_run_id = max(
                runs, key=lambda rid: (len(runs[rid][0] & obj_night_ids), runs[rid][1])
            )
            covered, _ = runs[best_run_id]
            if obj_night_ids <= covered:
                tie_map = tie_by_obj[obj_id][best_run_id]

        tasks[obj_id] = _ObjTask(
            obj_id=obj_id,
            klass=klass,
            bls_eligible=(klass == "EXOP") or bool(has_transit_like),
            nights=nights,
            tie=tie_map,
            settings=settings,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Every night's own-median-normalised flux, concatenated and time-sorted."""
    ts: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    dys: list[np.ndarray] = []
    for nd in task.nights:
        got = _good_night_flux(nd)
        if got is None:
            continue
        tt, yy, ee = got
        ts.append(tt)
        ys.append(yy)
        dys.append(ee)
    if not ts:
        return None
    t_all = np.concatenate(ts)
    y_all = np.concatenate(ys)
    dy_all = np.concatenate(dys)
    order = np.argsort(t_all)
    t_all, y_all, dy_all = t_all[order], y_all[order], dy_all[order]
    if t_all.size < _MIN_NIGHT_POINTS:
        return None
    span = float(t_all.max() - t_all.min())
    if span <= 0:
        return None
    return t_all, y_all, dy_all, span


def _combined_series(
    task: _ObjTask,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float] | None:
    """The combined-LS input: tied magnitudes when a run ties every night, else flux."""
    if task.tie is not None:
        ts: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        dys: list[np.ndarray] = []
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
        if ts:
            t_all = np.concatenate(ts)
            y_all = np.concatenate(ys)
            dy_all = np.concatenate(dys)
            order = np.argsort(t_all)
            t_all, y_all, dy_all = t_all[order], y_all[order], dy_all[order]
            if t_all.size >= _MIN_NIGHT_POINTS:
                span = float(t_all.max() - t_all.min())
                if span > 0:
                    return t_all, y_all, dy_all, "tied-mag", span

    fallback = _concat_night_normalised_flux(task)
    if fallback is None:
        return None
    t_all, y_all, dy_all, span = fallback
    return t_all, y_all, dy_all, "night-normalised", span


def _run_ls(
    t: np.ndarray, y: np.ndarray, dy: np.ndarray, span: float, s: DbSettings,
    *, scope: str, input_label: str | None,
) -> tuple | None:
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
        best = int(np.argmax(power))
        peak_period = float(1.0 / freq[best])
        peak_power = float(power[best])
        fap = float(
            ls.false_alarm_probability(
                peak_power, method="baluev",
                minimum_frequency=float(freq[0]), maximum_frequency=float(freq[-1]),
                samples_per_peak=s.ls_samples_per_peak,
            )
        )
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


def _compute_object(task: _ObjTask) -> list[tuple]:
    """Every periodogram row for one object: LS per night, LS combined, BLS combined.

    No database access -- runs standalone in a worker process. Each element
    of the returned list is
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
            t_c, y_c, dy_c, input_label, span_c = combined
            row = _run_ls(t_c, y_c, dy_c, span_c, s, scope="combined", input_label=input_label)
            if row is not None:
                rows.append(row)

        if task.bls_eligible:
            flux_combined = _concat_night_normalised_flux(task)
            if flux_combined is not None:
                t_f, y_f, dy_f, span_f = flux_combined
                row = _run_bls(t_f, y_f, dy_f, span_f, s)
                if row is not None:
                    rows.append(row)

    return rows


def analyze(
    conn: psycopg.Connection,
    *,
    all_candidates: bool = False,
    obj_ids: Sequence[int] | None = None,
    settings: Settings | None = None,
    workers: int | None = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> AnalyzeReport:
    """Recompute periodograms (and, via :func:`~relphot.db.refresh.refresh_objects`, PERIOD)
    for objects needing analysis.

    ``obj_ids`` (given verbatim) takes priority over ``all_candidates``
    (every ``'EXOP'``/``'VAR'``/detected object) over the default (only
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
    executor = ProcessPoolExecutor(max_workers=max_workers) if max_workers > 1 else None
    try:
        for chunk in chunks:
            tasks = _fetch_chunk_data(conn, chunk, db_settings)
            if executor is not None and len(tasks) > 1:
                results = list(executor.map(_compute_object, tasks.values()))
            else:
                results = [_compute_object(task) for task in tasks.values()]

            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM relphot.periodogram WHERE obj_id = ANY(%(obj_ids)s)",
                        {"obj_ids": list(tasks.keys())},
                    )
                    insert_rows = []
                    for obj_id, obj_rows in zip(tasks.keys(), results, strict=True):
                        for (
                            method, scope, fmin, df, n, power, peak_period, peak_power,
                            fap, coarsened, input_label, extra,
                        ) in obj_rows:
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
                    refresh_objects(
                        conn, list(tasks.keys()),
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
    )

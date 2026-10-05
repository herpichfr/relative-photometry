"""Automatic verdicts on one night's transit events (``relphot db analyze``).

Three kinds of rule write ``detection.auto_status = 'REJECTED'`` and ``auto_reason`` in ONE
per-night pass, :func:`update_auto_verdicts`, which unions the reasons of every rule that fires:
the cross-candidate (coincidence) veto below, the shape rules of :func:`shape_reasons`
(``EDGE_OUTLIER``, ``NO_DIP``, ``NO_BASELINE``, read from the ``transit_shape`` row) and the
catalogue-period rule of :func:`variability_reasons` (``VARIABILITY``, read from the object's
catalogue period and light curves). A person's ``status`` always wins
(:func:`relphot.web.app._effective_status`).

The coincidence veto, see docs/DB_PLAN.md ("Coincident events"). One planet cannot transit two
stars at once, so an event whose trapezoid fit (``relphot.transit_shape``) has many look-alikes
on the SAME night -- other objects' events with the same centre time and a similar T14 and
depth -- is a systematic (on the live data these coincide with seeing excursions and are spread
uniformly over the detector). Such an event stays a detection but is marked
``detection.auto_status = 'REJECTED'`` with ``auto_reason``; ``detection.status`` is the PERSON's
verdict and is never set here.

:func:`coincidence` is the pure numpy/scipy computation, no database. For every event *i*, over
the other events *j* of the night:

- window ``win_ij = max(coincidence_tc_frac * min(T14_i, T14_j),
  coincidence_tc_nsigma * hypot(tc_err_i, tc_err_j), cadence)``; a missing ``tc_err`` counts as
  0 and ``cadence`` is the median spacing of the night's frames;
- *shape-similar*: ``|ln(T14_i / T14_j)| < ln(coincidence_t14_ratio)`` (a lower-limit T14, one
  that is only a minimum, is also similar to any duration up to that ratio shorter) and
  ``|ln(|depth_i| / |depth_j|)| < ln(coincidence_depth_ratio)``;
- ``n_i`` = number of shape-similar *j* with ``|tc_i - tc_j| <= win_ij``;
- chance baseline: for a centre time uniform over ``[lo, hi]`` (the night's first and last
  frame, widened to hold every event) the probability of falling within ``win_ij`` of ``tc_i``
  is ``pw_ij`` (the window truncated at the span's edges); with ``M_i`` shape-similar events and
  ``pbar_i`` the mean ``pw_ij`` over them, ``n_expected_i = M_i * pbar_i`` and
  ``p_chance_i = P(Binomial(M_i, pbar_i) >= n_i)`` (1 when ``n_i = 0``);
- rejected iff ``n_i >= coincidence_min_similar`` and ``p_chance_i < coincidence_max_p``.

:func:`update_auto_verdicts` (alias ``update_coincidence``) runs it per night on the stored
shapes: it rewrites the night's ``relphot.transit_coincidence`` rows, evaluates the shape
rules, writes the unioned verdict of the night's transit detections (only rows whose verdict
differs, clearing a reason exactly when its rule no longer fires) and reports the objects whose
automatic verdict changed (their flags need a refresh). Only converged shapes of the search's
own per-night transit detections (``origin = 'search'``) are judged or counted by the coincidence
rule, and only those detections are judged by the shape rules and ``VARIABILITY``; a
user-origin detection is none of these. With ``skip_det_ids`` (the events a person has vetted,
:mod:`relphot.db.vetted`) those events are still counted as look-alikes of the others (and as
other events of their object for ``VARIABILITY``) but their own verdict and
``transit_coincidence`` row are left exactly as stored. It does not commit.

``VARIABILITY`` (:mod:`relphot.dip_variability`) rejects a search event whose star's CATALOGUE
period explains the dip: the object has a ``catalog_match`` row that is not a planet catalogue
(:data:`relphot.objflags.PLANET_CATALOGS`) with ``period > 0`` and no planet-catalogue row at all
(a known host's period is a transit period, so a repeat would confirm it). The period is that of
the eligible row nearest on the sky, used as catalogued. The event's night is predicted from the
object's other nights folded at that period (``auto_var_phase_*``), or other search events of the
object repeat at multiples of it (``auto_var_repeat_p_max``); either fires the rule.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

import numpy as np
import psycopg
from scipy.stats import binom

from relphot.config import DbSettings
from relphot.dip_variability import VariabilityVerdict, variability_verdict
from relphot.objflags import PLANET_CATALOGS

logger = logging.getLogger(__name__)

__all__ = [
    "CoincidenceReport",
    "CoincidenceResult",
    "coincidence",
    "shape_reasons",
    "update_auto_verdicts",
    "update_coincidence",
    "variability_reasons",
]

#: Rows of the pairwise matrices computed at a time (bounds the memory of a large night).
_BLOCK = 512


@dataclass(slots=True)
class CoincidenceResult:
    """Per-event output of :func:`coincidence`, in the order of the input arrays.

    ``similar[i]`` holds the input indices of the events counted in ``n_similar[i]``, nearest in
    time first; ``window_days[i]`` is the mean window over them. An event without a usable
    ``tc`` / T14 / depth is not ``evaluated``: it has no similar events, ``p_chance = 1`` and is
    never rejected.
    """

    n_similar: np.ndarray
    n_expected: np.ndarray
    p_chance: np.ndarray
    rejected: np.ndarray
    window_days: np.ndarray
    evaluated: np.ndarray
    similar: list[np.ndarray]


def coincidence(
    tc: Sequence[float] | np.ndarray,
    t14_h: Sequence[float] | np.ndarray,
    depth: Sequence[float] | np.ndarray,
    tc_err: Sequence[float] | np.ndarray | None = None,
    t14_lower_limit: Sequence[bool] | np.ndarray | None = None,
    *,
    t_first: float | None = None,
    t_last: float | None = None,
    cadence: float = 0.0,
    settings: DbSettings | None = None,
) -> CoincidenceResult:
    """Count, for every event of one night, the look-alikes among the others (module docstring).

    ``tc`` is in days (BJD_TDB), ``t14_h`` in hours, ``depth`` any unit (its sign is ignored),
    ``tc_err`` in days (NaN or ``None``: 0), ``t14_lower_limit`` flags a T14 that is only a
    minimum. ``t_first`` / ``t_last`` are the night's first and last frame time (the chance
    span is widened to hold every event) and ``cadence`` its median frame spacing, in days.
    """
    s = settings if settings is not None else DbSettings()
    tc = np.asarray(tc, dtype=float)
    n = tc.size
    w = np.asarray(t14_h, dtype=float) / 24.0
    d = np.abs(np.asarray(depth, dtype=float))
    if tc_err is None:
        te = np.zeros(n)
    else:
        te = np.nan_to_num(np.asarray(tc_err, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if t14_lower_limit is None:
        ll = np.zeros(n, dtype=bool)
    else:
        ll = np.asarray(t14_lower_limit, dtype=bool)

    result = CoincidenceResult(
        n_similar=np.zeros(n, dtype=np.int64),
        n_expected=np.zeros(n),
        p_chance=np.ones(n),
        rejected=np.zeros(n, dtype=bool),
        window_days=np.zeros(n),
        evaluated=np.isfinite(tc) & np.isfinite(w) & (w > 0) & np.isfinite(d) & (d > 0),
        similar=[np.zeros(0, dtype=np.int64) for _ in range(n)],
    )
    idx = np.flatnonzero(result.evaluated)
    m = idx.size
    if m < 2:
        return result

    tcv, wv, dv, tev, llv = tc[idx], w[idx], d[idx], te[idx], ll[idx]
    lo, hi = float(tcv.min()), float(tcv.max())
    if t_first is not None and math.isfinite(t_first):
        lo = min(lo, float(t_first))
    if t_last is not None and math.isfinite(t_last):
        hi = max(hi, float(t_last))
    span = hi - lo  # 0 only if every event is at one time and no frame times widen it
    cad = float(cadence) if math.isfinite(cadence) and cadence > 0 else 0.0

    ln_w = np.log(wv)
    ln_d = np.log(dv)
    ln_t14_ratio = math.log(s.coincidence_t14_ratio)
    ln_depth_ratio = math.log(s.coincidence_depth_ratio)

    counts = np.zeros(m, dtype=np.int64)
    n_shape = np.zeros(m, dtype=np.int64)
    pbar = np.zeros(m)
    win_sum = np.zeros(m)
    similar_local: list[np.ndarray] = [np.zeros(0, dtype=np.int64)] * m

    for start in range(0, m, _BLOCK):
        stop = min(start + _BLOCK, m)
        rows = np.arange(start, stop)
        # ln(T14_i / T14_j): a lower-limit duration is also similar to a shorter one
        dlw = ln_w[start:stop, None] - ln_w[None, :]
        similar_t14 = np.abs(dlw) < ln_t14_ratio
        similar_t14 |= llv[start:stop, None] & (dlw <= ln_t14_ratio)
        similar_t14 |= llv[None, :] & (-dlw <= ln_t14_ratio)
        shape = similar_t14 & (np.abs(ln_d[start:stop, None] - ln_d[None, :]) < ln_depth_ratio)
        shape[rows - start, rows] = False

        win = np.maximum(
            np.maximum(
                s.coincidence_tc_frac * np.minimum(wv[start:stop, None], wv[None, :]),
                s.coincidence_tc_nsigma * np.hypot(tev[start:stop, None], tev[None, :]),
            ),
            cad,
        )
        dt = np.abs(tcv[start:stop, None] - tcv[None, :])
        close = (dt <= win) & shape
        if span > 0:
            pw = (
                np.minimum(tcv[start:stop, None] + win, hi)
                - np.maximum(tcv[start:stop, None] - win, lo)
            ) / span
            pw = np.clip(pw, 0.0, 1.0)
        else:
            pw = np.ones_like(win)  # no span to spread over: every coincidence is certain

        n_shape[start:stop] = shape.sum(axis=1)
        counts[start:stop] = close.sum(axis=1)
        pbar[start:stop] = np.where(
            n_shape[start:stop] > 0,
            (pw * shape).sum(axis=1) / np.maximum(n_shape[start:stop], 1),
            0.0,
        )
        win_sum[start:stop] = (win * close).sum(axis=1)
        for k in np.flatnonzero(counts[start:stop]):
            js = np.flatnonzero(close[k])
            similar_local[start + k] = js[np.argsort(dt[k, js], kind="stable")]

    p = np.ones(m)
    hit = counts > 0
    if hit.any():
        p[hit] = binom.sf(counts[hit] - 1, n_shape[hit], np.clip(pbar[hit], 0.0, 1.0))
    rejected = (counts >= s.coincidence_min_similar) & (p < s.coincidence_max_p)

    result.n_similar[idx] = counts
    result.n_expected[idx] = n_shape * pbar
    result.p_chance[idx] = p
    result.rejected[idx] = rejected
    result.window_days[idx] = np.where(hit, win_sum / np.maximum(counts, 1), 0.0)
    for k in np.flatnonzero(hit):
        result.similar[int(idx[k])] = idx[similar_local[k]]
    return result


@dataclass(slots=True)
class CoincidenceReport:
    """What one :func:`update_auto_verdicts` call did.

    ``n_rejected`` counts the events auto-rejected after the pass (by any rule), the ``n_*``
    rule counts the events each rule fired on (one event can be counted under several).
    """

    n_nights: int = 0
    n_events: int = 0
    n_rejected: int = 0
    n_coincidence: int = 0
    n_edge_outlier: int = 0
    n_no_dip: int = 0
    n_no_baseline: int = 0
    n_variability: int = 0
    #: objects with an event whose automatic verdict (rejected or not) changed
    changed_obj_ids: list[int] = field(default_factory=list)


_EVENTS_SQL = """
    SELECT d.det_id, d.obj_id, ts.tc, ts.tc_err, ts.depth, ts.t14_h, ts.t14_lower_limit
    FROM relphot.detection d JOIN relphot.transit_shape ts ON ts.det_id = d.det_id
    WHERE d.night_id = %s AND d.kind = 'transit' AND d.origin = 'search' AND ts.converged
    ORDER BY d.det_id
"""

#: the stored shapes the shape rules (:func:`shape_reasons`) judge: every search event with a row
_SHAPE_EVENTS_SQL = """
    SELECT d.det_id, ts.depth, ts.converged, ts.edge_clip_bjd, ts.edge_adjacent, ts.n_outside
    FROM relphot.detection d JOIN relphot.transit_shape ts ON ts.det_id = d.det_id
    WHERE d.night_id = %s AND d.kind = 'transit' AND d.origin = 'search'
    ORDER BY d.det_id
"""

_INSERT_SQL = """
    INSERT INTO relphot.transit_coincidence
        (det_id, night_id, n_similar, n_expected, p_chance, similar_det_ids, rejected,
         computed_at)
    VALUES (%s, %s, %s, %s, %s, %s::bigint[], %s, now())
"""

#: the rules of the verdict pass, in the order their reasons are written
_RULE_ORDER = ("COINCIDENCE", "EDGE_OUTLIER", "NO_DIP", "NO_BASELINE", "VARIABILITY")

#: the search events of one night that ``VARIABILITY`` may judge (the detection's own tc and
#: duration, not the trapezoid fit)
_VARIABILITY_EVENTS_SQL = """
    SELECT det_id, obj_id, tc_bjd_tdb, duration_h FROM relphot.detection
    WHERE night_id = %s AND kind = 'transit' AND origin = 'search'
      AND tc_bjd_tdb IS NOT NULL AND duration_h IS NOT NULL
    ORDER BY det_id
"""

_CATALOG_SQL = """
    SELECT obj_id, catalog, name, type, period, period_err, sep_arcsec
    FROM relphot.catalog_match WHERE obj_id = ANY(%s)
"""

#: every night's light curve of the objects, one row per (object, night)
_LC_SQL = """
    SELECT obj_id, night_id, bjd_tdb, flux, flux_err FROM relphot.lightcurve
    WHERE obj_id = ANY(%s) ORDER BY obj_id, night_id
"""

#: every search event of the objects, on any night (whatever its status or supersede link)
_OTHER_EVENTS_SQL = """
    SELECT det_id, obj_id, night_id, tc_bjd_tdb, duration_h FROM relphot.detection
    WHERE obj_id = ANY(%s) AND kind = 'transit' AND origin = 'search'
      AND tc_bjd_tdb IS NOT NULL AND duration_h IS NOT NULL
    ORDER BY det_id
"""

#: the catalogued type of an exoplanet transit (VSX): its period is a transit period
_PLANET_TYPE = "EP"


def _short_p(p: float) -> str:
    """``p`` with one significant digit and no exponent padding: 3e-9."""
    if p <= 0:
        return "<1e-300"
    return re.sub(r"e([+-])0*(\d)", r"e\1\2", f"{p:.0e}")


def _reason(n_similar: int, window_days: float, n_expected: float, p_chance: float) -> str:
    return (
        f"too many similar events: {n_similar} other events on this night within "
        f"±{window_days * 1440.0:.0f} min with similar depth and T14 "
        f"(expected {n_expected:.1f} by chance, p={_short_p(p_chance)})"
    )


def _variability_period(rows: Sequence[tuple]) -> tuple[float, float | None, str] | None:
    """``(period, period_err, catalog)`` of one object's catalogue period, or ``None``.

    ``rows`` are its ``catalog_match`` rows ``(catalog, name, type, period, period_err,
    sep_arcsec)``. An object with a planet-catalogue row (:data:`PLANET_CATALOGS`) has none; else
    the eligible row (``period > 0``, not a type marking an exoplanet transit: VSX ``EP``, matched
    as a token of the ``|`` / ``,``-separated type) nearest on the sky (NULL separation last)
    gives it.
    """
    if any(row[0] in PLANET_CATALOGS for row in rows):
        return None
    eligible = [
        row for row in rows
        if row[3] is not None and math.isfinite(row[3]) and row[3] > 0
        and _PLANET_TYPE not in {tok.strip().upper() for tok in re.split(r"[|,]", row[2] or "")}
    ]
    if not eligible:
        return None
    catalog, _name, _type, period, period_err, _sep = min(
        eligible, key=lambda row: (row[5] is None, row[5] or 0.0, row[0], row[1])
    )
    return float(period), (float(period_err) if period_err else None), catalog


def _variability_reason(period: float, catalog: str, verdict: VariabilityVerdict) -> str:
    """The reason of a fired ``VARIABILITY``, naming the test(s) that fired with their numbers."""
    which = f"P={period:.4f} d ({catalog})"
    phase = (
        f"predicts this dip from {verdict.n_nights} other night"
        f"{'s' if verdict.n_nights != 1 else ''} folded (predicted/observed depth "
        f"{verdict.ratio:.2f}, phase coverage {verdict.cov * 100.0:.0f} %)"
    )
    repeat = (
        f"{verdict.n_match} other event{'s' if verdict.n_match != 1 else ''} at n*P "
        f"(or odd n*P/2), chance p={_short_p(verdict.p_tail)}"
    )
    if verdict.phase and verdict.repeat:
        return (
            f"variability: the catalogue period {which} {phase} and the dip repeats at "
            f"multiples of it: {repeat}"
        )
    if verdict.phase:
        return f"variability: the catalogue period {which} {phase}"
    return f"variability: the dip repeats at multiples of the catalogue period {which}: {repeat}"


def variability_reasons(
    cur: psycopg.Cursor,
    night_id: int,
    skip: Collection[int] = (),
    settings: DbSettings | None = None,
) -> dict[int, str]:
    """The ``VARIABILITY`` rule on one night: ``{det_id: reason}`` of the events it rejects.

    Reads only. Judges every search transit detection of the night (not in ``skip``, with a
    ``tc_bjd_tdb`` and ``duration_h``) of an object with a catalogue period
    (:func:`_variability_period`) and a light curve on the night, with
    :func:`relphot.dip_variability.variability_verdict`: the other nights are the object's other
    light curves, the other events its search events on other nights. One query each for the
    catalogue rows, light curves and events of all the night's eligible objects.
    """
    s = settings if settings is not None else DbSettings()
    skip_ids = {int(i) for i in skip}
    cur.execute(_VARIABILITY_EVENTS_SQL, (night_id,))
    events = [row for row in cur.fetchall() if row[0] not in skip_ids]
    if not events:
        return {}

    cur.execute(_CATALOG_SQL, ([int(o) for o in {e[1] for e in events}],))
    rows: dict[int, list[tuple]] = {}
    for obj_id, catalog, name, type_, period, period_err, sep in cur.fetchall():
        rows.setdefault(obj_id, []).append((catalog, name, type_, period, period_err, sep))
    periods = {obj_id: _variability_period(r) for obj_id, r in rows.items()}
    events = [e for e in events if periods.get(e[1]) is not None]
    if not events:
        return {}

    obj_ids = sorted({int(e[1]) for e in events})
    cur.execute(_LC_SQL, (obj_ids,))
    lcs: dict[int, dict[int, tuple]] = {}
    for obj_id, lc_night, bjd, flux, flux_err in cur.fetchall():
        lcs.setdefault(obj_id, {})[lc_night] = (bjd, flux, flux_err)
    cur.execute(_OTHER_EVENTS_SQL, (obj_ids,))
    object_events: dict[int, list[tuple]] = {}
    for det_id, obj_id, ev_night, tc, duration in cur.fetchall():
        object_events.setdefault(obj_id, []).append((det_id, ev_night, tc, duration))

    reasons: dict[int, str] = {}
    for det_id, obj_id, tc, duration in events:
        night_lcs = lcs.get(obj_id, {})
        if night_id not in night_lcs:
            continue
        period, period_err, catalog = periods[obj_id]
        others = [ev for ev in object_events.get(obj_id, []) if ev[1] != night_id]
        verdict = variability_verdict(
            night_lcs[night_id],
            [lc for other_night, lc in sorted(night_lcs.items()) if other_night != night_id],
            tc, duration, [ev[2] for ev in others], [ev[3] for ev in others],
            period, period_err, s,
        )
        if verdict.fired:
            reasons[det_id] = _variability_reason(period, catalog, verdict)
    return reasons


def shape_reasons(
    depth: float | None,
    converged: bool | None,
    edge_clip_bjd: Sequence[float] | None,
    edge_adjacent: bool | None,
    n_outside: int | None,
    settings: DbSettings | None = None,
) -> dict[str, str]:
    """The shape rules that reject one stored per-night transit event: ``{rule: reason}``.

    From its ``transit_shape`` row (``relphot.db.analyze._fit_transit_shape``); a NULL input
    never fires a rule:

    - ``EDGE_OUTLIER``: the fit excluded isolated edge epoch(s) and they were the only epoch(s)
      beyond the detection's own search box on that side (``edge_adjacent``): the box starts or
      ends right next to the artefact that made the dip;
    - ``NO_DIP``: a converged fit with a depth below ``auto_nodip_depth``;
    - ``NO_BASELINE``: a converged fit with fewer than ``auto_nobaseline_min_epochs`` epochs
      outside the fitted trapezoid (``n_outside``) and a depth above ``auto_nobaseline_min_depth``:
      the baseline is extrapolated from almost nothing, not measured.
    """
    s = settings if settings is not None else DbSettings()
    reasons: dict[str, str] = {}
    if edge_adjacent and edge_clip_bjd:
        k = len(edge_clip_bjd)
        what = "isolated edge epochs were" if k > 1 else "isolated edge epoch was"
        reasons["EDGE_OUTLIER"] = (
            f"edge outlier: {k} {what} excluded from the fit and the only data beyond the "
            "search box on its side"
        )
    if converged and depth is not None and math.isfinite(depth):
        if depth < s.auto_nodip_depth:
            reasons["NO_DIP"] = (
                f"no dip: the fitted trapezoid depth is {depth:.1e} (< {s.auto_nodip_depth:.0e})"
            )
        if (
            n_outside is not None and n_outside < s.auto_nobaseline_min_epochs
            and depth > s.auto_nobaseline_min_depth
        ):
            reasons["NO_BASELINE"] = (
                f"no baseline: only {n_outside} epoch{'s' if n_outside != 1 else ''} outside the "
                f"fitted trapezoid (< {s.auto_nobaseline_min_epochs}) for a depth of {depth:.2f} "
                f"(> {s.auto_nobaseline_min_depth:g})"
            )
    return reasons


def update_auto_verdicts(
    conn: psycopg.Connection,
    night_ids: Sequence[int],
    settings: DbSettings | None = None,
    skip_det_ids: Collection[int] = (),
) -> CoincidenceReport:
    """Re-evaluate every night in ``night_ids`` from its stored transit shapes (module docstring).

    Per night: the ``transit_coincidence`` rows are deleted and inserted afresh and every rule
    (coincidence, then :func:`shape_reasons`, then :func:`variability_reasons`) is evaluated; an
    event rejected by one or more rules gets ``auto_status = 'REJECTED'`` and ``auto_reason`` =
    the reasons of all the rules that fire, joined by ``"; "``. The verdict is computed from
    scratch from the stored shapes (and light curves) each time and only the rows whose
    ``auto_status`` / ``auto_reason`` differ are written, so a re-run is idempotent and a reason
    disappears exactly when its rule no longer fires (a transit detection no rule rejects is
    cleared). A person's ``status`` is never touched. Events in ``skip_det_ids`` keep their stored
    verdict and coincidence row (module docstring). The whole night is judged, whichever objects
    were just analysed. Returns the report with ``changed_obj_ids`` (sorted): the objects that
    have an event whose automatic verdict (rejected or not) changed, whose flags the caller must
    refresh. Does not commit.
    """
    s = settings if settings is not None else DbSettings()
    report = CoincidenceReport()
    changed: set[int] = set()
    skip = {int(i) for i in skip_det_ids}

    with conn.cursor() as cur:
        for night_id in sorted({int(n) for n in night_ids}):
            cur.execute(
                "SELECT det_id, obj_id, auto_status, auto_reason FROM relphot.detection "
                "WHERE night_id = %s AND kind = 'transit'",
                (night_id,),
            )
            before = {
                det_id: (obj_id, auto, why) for det_id, obj_id, auto, why in cur.fetchall()
                if det_id not in skip
            }

            cur.execute(_EVENTS_SQL, (night_id,))
            events = cur.fetchall()
            cur.execute(
                "SELECT bjd_tdb FROM relphot.frame "
                "WHERE night_id = %s AND bjd_tdb IS NOT NULL ORDER BY bjd_tdb",
                (night_id,),
            )
            frames = np.array([row[0] for row in cur.fetchall()], dtype=float)
            t_first = float(frames[0]) if frames.size else None
            t_last = float(frames[-1]) if frames.size else None
            cadence = float(np.median(np.diff(frames))) if frames.size >= 2 else 0.0

            det_ids = [e[0] for e in events]
            if events:
                res = coincidence(
                    [e[2] for e in events], [e[5] for e in events], [e[4] for e in events],
                    [e[3] for e in events], [e[6] for e in events],
                    t_first=t_first, t_last=t_last, cadence=cadence, settings=s,
                )
            else:
                res = None

            cur.execute(
                "DELETE FROM relphot.transit_coincidence "
                "WHERE night_id = %s AND det_id <> ALL(%s)",
                (night_id, sorted(skip)),
            )

            # rule -> {det_id: reason}
            fired: dict[str, dict[int, str]] = {rule: {} for rule in _RULE_ORDER}
            if res is not None:
                rows = []
                for i, det_id in enumerate(det_ids):
                    if not res.evaluated[i] or det_id in skip:
                        continue
                    rejected = bool(res.rejected[i])
                    rows.append((
                        det_id, night_id, int(res.n_similar[i]), float(res.n_expected[i]),
                        float(res.p_chance[i]), [det_ids[j] for j in res.similar[i]], rejected,
                    ))
                    if rejected:
                        fired["COINCIDENCE"][det_id] = _reason(
                            int(res.n_similar[i]), float(res.window_days[i]),
                            float(res.n_expected[i]), float(res.p_chance[i]),
                        )
                if rows:
                    cur.executemany(_INSERT_SQL, rows)
                report.n_events += len(rows)

            cur.execute(_SHAPE_EVENTS_SQL, (night_id,))
            for det_id, depth, converged, clip_bjd, adjacent, n_outside in cur.fetchall():
                if det_id in skip:
                    continue
                for rule, why in shape_reasons(
                    depth, converged, clip_bjd, adjacent, n_outside, s
                ).items():
                    fired[rule][det_id] = why

            fired["VARIABILITY"] = variability_reasons(cur, night_id, skip, s)

            desired: dict[int, str] = {}
            for rule in _RULE_ORDER:
                for det_id, why in fired[rule].items():
                    desired[det_id] = f"{desired[det_id]}; {why}" if det_id in desired else why

            updates_set = []
            updates_clear = []
            for det_id, (obj_id, auto, why) in before.items():
                new = desired.get(det_id)
                if (auto == "REJECTED") != (new is not None):
                    changed.add(obj_id)
                if new is None:
                    if auto is not None or why is not None:
                        updates_clear.append((det_id,))
                elif auto != "REJECTED" or why != new:
                    updates_set.append((new, det_id))
            if updates_clear:
                cur.executemany(
                    "UPDATE relphot.detection SET auto_status = NULL, auto_reason = NULL "
                    "WHERE det_id = %s",
                    updates_clear,
                )
            if updates_set:
                cur.executemany(
                    "UPDATE relphot.detection SET auto_status = 'REJECTED', auto_reason = %s "
                    "WHERE det_id = %s",
                    updates_set,
                )

            report.n_nights += 1
            report.n_rejected += sum(1 for det_id in desired if det_id in before)
            report.n_coincidence += len(fired["COINCIDENCE"])
            report.n_edge_outlier += len(fired["EDGE_OUTLIER"])
            report.n_no_dip += len(fired["NO_DIP"])
            report.n_no_baseline += len(fired["NO_BASELINE"])
            report.n_variability += len(fired["VARIABILITY"])

    report.changed_obj_ids = sorted(changed)
    logger.info(
        "auto verdicts: %d nights, %d events judged, %d auto-rejected (coincidence %d, "
        "edge outlier %d, no dip %d, no baseline %d, variability %d), %d objects changed",
        report.n_nights, report.n_events, report.n_rejected, report.n_coincidence,
        report.n_edge_outlier, report.n_no_dip, report.n_no_baseline, report.n_variability,
        len(report.changed_obj_ids),
    )
    return report


#: the coincidence veto's original name; the pass now also applies the shape rules
update_coincidence = update_auto_verdicts

"""Recompute derived summary fields on ``relphot.object`` rows.

Set-based SQL only -- no per-object Python loop. See docs/DB_PLAN.md ("CLASS
precedence", ``analyze``) for the rules this implements. Only the auto-derived
fields (n_nights, mean_mag, first/last night, detection/known summaries) and
the derived ``period``/``period_source`` are computed by the main UPDATE here.
The two independent flags ``is_exop`` / ``is_var``, their sources, the derived
``class`` / ``class_source`` labels and the review counters are set first, in the
same transaction, by :func:`relphot.objflags.refresh_flags`; the period branches
below read the flags it just wrote. Neither flag depends on the other, so a star can
be a planet host and a variable at once.

There is no manual pin on a flag. A person's per-night verdicts
(``relphot.user_night_review``: CONFIRMED / REJECTED / NULL = automatic, for EXOP and
VAR separately) override only the automatic evidence of the night they are set on;
literature matches and gated multi-night detections (``class_multinight_kinds``,
default only ``'recurrent'``) always count; a night loaded later is evaluated
automatically. See :mod:`relphot.objflags`. ``period``/``period_source`` have their own independent
manual guard: a row whose ``period_source`` is ``'manual'`` (set by the web when a
person enters a period by hand) keeps its period untouched. ``object.status`` is
never set here. A detection a person created by reprocessing (``detection.origin
= 'user'``) never sets a flag and never enters the best-transit summary: those
come from the search's own detections only (``n_detections`` still counts them all). A search
transit that is auto-rejected (``detection.auto_status = 'REJECTED'``, too many similar events
on its night, see :mod:`relphot.db.coincidence`) is likewise no evidence and not the best
transit, unless a person CONFIRMED it.

``n_detections``, ``best_snr``, ``depth``, ``duration_h``, and ``amplitude``
keep counting every detection regardless of origin. ``duration_h`` comes from
the best (highest-SNR) transit and ``duration_lower_limit`` says whether that
transit is incomplete, i.e. ``duration_h`` is only a minimum.

For an auto (non-manual) ``period_source``, the priority is: literature
``known_period`` -> ``'catalog'`` first (a period measured from a few nights
cannot beat a catalogued one: on T80S 20251104-06, 384 of 386 significant
combined-LS peaks of catalogued variables disagreed with the literature
period; ``period_err`` is the catalogue's error, possibly NULL); else the
object's best ``relphot.period_estimate`` (largest ``n_nights``, then latest)
when its fap <= ``ls_fap_threshold`` -> ``'LS'`` (``period_err`` and
``period_n_nights`` from it; the FAP gate tests significance, not
precision; a ``method = 'LS'`` estimate only -- a user-guided ``'LS-guided'`` one never sets
PERIOD -- and not one flagged ``long_period_needs_tie``); else the combined BLS peak
(``relphot.periodogram`` scope ``'combined'``, method ``'BLS'``, extra ``'depth_snr'`` >=
``bls_min_snr``,
for a planet host) -> ``'BLS'``; else the combined LS peak (scope
``'combined'``, method ``'LS'``, fap <= ``ls_fap_threshold``, for a variable)
-> ``'LS'``; else the median per-night LS period -> ``'night-LS'``; else
``NULL``.
"""

from __future__ import annotations

from collections.abc import Sequence

import psycopg

from relphot.config import DbSettings
from relphot.objflags import refresh_flags

__all__ = ["refresh_objects"]

_REFRESH_SQL = """
WITH target AS (
    SELECT obj_id FROM relphot.object{target_filter}
),
sn_agg AS (
    SELECT sn.obj_id, COUNT(DISTINCT sn.night_id) AS n_nights, AVG(sn.mag) AS mean_mag
    FROM relphot.star_night sn JOIN target t ON t.obj_id = sn.obj_id
    GROUP BY sn.obj_id
),
night_range AS (
    SELECT sn.obj_id, MIN(n.night_date) AS first_night, MAX(n.night_date) AS last_night
    FROM relphot.star_night sn
    JOIN relphot.night n ON n.night_id = sn.night_id
    JOIN target t ON t.obj_id = sn.obj_id
    GROUP BY sn.obj_id
),
det_agg AS (
    SELECT d.obj_id, COUNT(*) AS n_detections
    FROM relphot.detection d JOIN target t ON t.obj_id = d.obj_id
    GROUP BY d.obj_id
),
best_transit AS (
    SELECT DISTINCT ON (d.obj_id) d.obj_id, d.snr AS best_snr, d.depth, d.duration_h,
           d.duration_lower_limit
    FROM relphot.detection d JOIN target t ON t.obj_id = d.obj_id
    WHERE d.kind = 'transit' AND d.snr IS NOT NULL AND d.origin = 'search'
          AND (d.auto_status IS DISTINCT FROM 'REJECTED' OR d.status = 'CONFIRMED')
    ORDER BY d.obj_id, d.snr DESC
),
var_amp AS (
    SELECT d.obj_id, MAX(d.amplitude) AS amplitude
    FROM relphot.detection d JOIN target t ON t.obj_id = d.obj_id
    WHERE d.kind = 'variable'
    GROUP BY d.obj_id
),
var_period_med AS (
    SELECT d.obj_id, percentile_cont(0.5) WITHIN GROUP (ORDER BY d.period) AS median_period
    FROM relphot.detection d JOIN target t ON t.obj_id = d.obj_id
    WHERE d.kind = 'variable' AND d.period IS NOT NULL
    GROUP BY d.obj_id
),
cm_agg AS (
    SELECT cm.obj_id, TRUE AS known,
           string_agg(DISTINCT cm.catalog, ', ' ORDER BY cm.catalog) AS source_db
    FROM relphot.catalog_match cm JOIN target t ON t.obj_id = cm.obj_id
    GROUP BY cm.obj_id
),
planet_cm AS (
    SELECT DISTINCT ON (cm.obj_id) cm.obj_id, cm.name, cm.period, cm.period_err
    FROM relphot.catalog_match cm JOIN target t ON t.obj_id = cm.obj_id
    WHERE cm.catalog IN ('NASA Exoplanet Archive', 'TOI')
    ORDER BY cm.obj_id, cm.catalog
),
variable_cm AS (
    SELECT DISTINCT ON (cm.obj_id) cm.obj_id, cm.name, cm.type, cm.period, cm.period_err
    FROM relphot.catalog_match cm JOIN target t ON t.obj_id = cm.obj_id
    WHERE cm.catalog NOT IN ('NASA Exoplanet Archive', 'TOI')
    ORDER BY cm.obj_id, (cm.type IS NULL), cm.catalog
),
best_pe AS (
    SELECT DISTINCT ON (pe.obj_id) pe.obj_id, pe.period, pe.period_err, pe.n_nights, pe.fap
    FROM relphot.period_estimate pe JOIN target t ON t.obj_id = pe.obj_id
    WHERE pe.method = 'LS' AND pe.period IS NOT NULL
        AND pe.verify_status IS DISTINCT FROM 'long_period_needs_tie'
    ORDER BY pe.obj_id, pe.n_nights DESC, pe.computed_at DESC
),
bls_combined AS (
    SELECT p.obj_id, p.peak_period, (p.extra ->> 'depth_snr')::double precision AS depth_snr
    FROM relphot.periodogram p JOIN target t ON t.obj_id = p.obj_id
    WHERE p.scope = 'combined' AND p.method = 'BLS'
),
ls_combined AS (
    SELECT p.obj_id, p.peak_period, p.fap
    FROM relphot.periodogram p JOIN target t ON t.obj_id = p.obj_id
    WHERE p.scope = 'combined' AND p.method = 'LS'
)
UPDATE relphot.object o
SET
    n_nights = COALESCE(sa.n_nights, 0),
    mean_mag = sa.mean_mag,
    first_night = nr.first_night,
    last_night = nr.last_night,
    n_detections = COALESCE(da.n_detections, 0),
    best_snr = bt.best_snr,
    depth = bt.depth,
    duration_h = bt.duration_h,
    duration_lower_limit = bt.duration_lower_limit,
    amplitude = va.amplitude,
    known = COALESCE(cm.known, false),
    source_db = cm.source_db,
    known_name = COALESCE(pc.name, vc.name),
    known_type = vc.type,
    known_period = COALESCE(pc.period, vc.period),
    period = CASE WHEN o.period_source IS DISTINCT FROM 'manual' THEN
                COALESCE(
                    pc.period, vc.period,
                    CASE WHEN pe.fap <= %(ls_fap_threshold)s THEN pe.period END,
                    CASE WHEN o.is_exop AND bc.depth_snr >= %(bls_min_snr)s
                         THEN bc.peak_period END,
                    CASE WHEN o.is_var AND lc.fap <= %(ls_fap_threshold)s
                         THEN lc.peak_period END,
                    vpm.median_period
                )
             ELSE o.period END,
    period_source = CASE WHEN o.period_source IS DISTINCT FROM 'manual' THEN
                CASE
                    WHEN COALESCE(pc.period, vc.period) IS NOT NULL THEN 'catalog'
                    WHEN pe.fap <= %(ls_fap_threshold)s THEN 'LS'
                    WHEN o.is_exop AND bc.depth_snr >= %(bls_min_snr)s THEN 'BLS'
                    WHEN o.is_var AND lc.fap <= %(ls_fap_threshold)s THEN 'LS'
                    WHEN vpm.median_period IS NOT NULL THEN 'night-LS'
                    ELSE NULL
                END
             ELSE o.period_source END,
    period_err = CASE WHEN o.period_source IS DISTINCT FROM 'manual' THEN
                CASE
                    WHEN pc.period IS NOT NULL THEN pc.period_err
                    WHEN vc.period IS NOT NULL THEN vc.period_err
                    WHEN pe.fap <= %(ls_fap_threshold)s THEN pe.period_err
                    ELSE NULL
                END
             ELSE NULL END,
    period_n_nights = CASE WHEN o.period_source IS DISTINCT FROM 'manual'
                        AND COALESCE(pc.period, vc.period) IS NULL
                        AND pe.fap <= %(ls_fap_threshold)s
                    THEN pe.n_nights ELSE NULL END,
    updated_at = now()
FROM target tgt
LEFT JOIN sn_agg sa ON sa.obj_id = tgt.obj_id
LEFT JOIN night_range nr ON nr.obj_id = tgt.obj_id
LEFT JOIN det_agg da ON da.obj_id = tgt.obj_id
LEFT JOIN best_transit bt ON bt.obj_id = tgt.obj_id
LEFT JOIN var_amp va ON va.obj_id = tgt.obj_id
LEFT JOIN var_period_med vpm ON vpm.obj_id = tgt.obj_id
LEFT JOIN cm_agg cm ON cm.obj_id = tgt.obj_id
LEFT JOIN planet_cm pc ON pc.obj_id = tgt.obj_id
LEFT JOIN variable_cm vc ON vc.obj_id = tgt.obj_id
LEFT JOIN best_pe pe ON pe.obj_id = tgt.obj_id
LEFT JOIN bls_combined bc ON bc.obj_id = tgt.obj_id
LEFT JOIN ls_combined lc ON lc.obj_id = tgt.obj_id
WHERE o.obj_id = tgt.obj_id
"""


def refresh_objects(
    conn: psycopg.Connection,
    obj_ids: Sequence[int] | None = None,
    *,
    bls_min_snr: float | None = None,
    ls_fap_threshold: float | None = None,
    class_multinight_kinds: Sequence[str] | None = None,
) -> None:
    """Recompute every derived ``relphot.object`` field, in one set-based UPDATE.

    ``obj_ids`` restricts the refresh to those objects; ``None`` (the
    default) refreshes every object in the table. ``bls_min_snr`` and
    ``ls_fap_threshold`` gate the BLS/LS PERIOD priority (see the module
    docstring); ``class_multinight_kinds`` gates which multi-night detection
    kinds count towards CLASS (see the module docstring); each defaults to
    :class:`~relphot.config.DbSettings`'s own default when not given. Does
    not commit -- callers that want this visible outside their own
    transaction must call :meth:`psycopg.Connection.commit` themselves.
    """
    if bls_min_snr is None:
        bls_min_snr = DbSettings().bls_min_snr
    if ls_fap_threshold is None:
        ls_fap_threshold = DbSettings().ls_fap_threshold
    if class_multinight_kinds is None:
        class_multinight_kinds = DbSettings().class_multinight_kinds

    params: dict[str, object] = {
        "bls_min_snr": bls_min_snr, "ls_fap_threshold": ls_fap_threshold,
    }
    if obj_ids is None:
        sql = _REFRESH_SQL.format(target_filter="")
    else:
        obj_ids = list(obj_ids)
        if not obj_ids:
            return
        sql = _REFRESH_SQL.format(target_filter=" WHERE obj_id = ANY(%(obj_ids)s)")
        params["obj_ids"] = obj_ids
    # flags first (same transaction): the PERIOD branches below read o.is_exop / o.is_var
    refresh_flags(conn, obj_ids, class_multinight_kinds=list(class_multinight_kinds))
    with conn.cursor() as cur:
        cur.execute(sql, params)

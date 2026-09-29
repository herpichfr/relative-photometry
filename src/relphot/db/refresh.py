"""Recompute derived summary fields on ``relphot.object`` rows.

Set-based SQL only -- no per-object Python loop. See docs/DB_PLAN.md ("CLASS
precedence", ``analyze``) for the rules this implements. Only the auto-derived
fields (n_nights, mean_mag, first/last night, detection/known summaries),
the two independent flags ``is_exop`` / ``is_var`` (each guarded by its own
``exop_source`` / ``var_source``), and the derived ``class`` /
``class_source`` labels are touched by the class logic here. A flag whose
source is ``'manual'`` (set by the web) is never changed; neither flag
depends on the other, so a star can be a planet host and a variable at once.
``class`` is the derived label (``'EXOP+VAR'`` if both flags, else ``'EXOP'``,
``'VAR'`` or ``'UNC'``) and ``class_source`` is ``'manual'`` iff either flag
source is. ``period``/``period_source`` have their own independent manual
guard: a row whose ``period_source`` is ``'manual'`` (set by the web when a
person enters a period by hand) keeps its period untouched by this function.
``object.status`` is never set here.

``is_exop`` is set by a known planet match, a per-night ``'transit'``
detection (``night_id`` set), or a multi-night ``'bls'`` detection whose kind
is in ``class_multinight_kinds``; ``is_var`` by a known variable match, a
per-night ``'variable'`` detection, or a multi-night ``'internight'`` /
``'ls_periodic'`` / ``'recurrent'`` detection whose kind is in
``class_multinight_kinds`` (default: only ``'recurrent'`` --
``'bls'``/``'ls_periodic'``/``'internight'`` thresholds are uncalibrated and
dominated by night-step artefacts, see :class:`~relphot.config.DbSettings`).
``n_detections``, ``best_snr``, ``depth``, ``duration_h``, and ``amplitude``
keep counting every detection regardless of this gate. ``duration_h`` comes from
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
precision); else the combined BLS peak (``relphot.periodogram`` scope
``'combined'``, method ``'BLS'``, extra ``'depth_snr'`` >= ``bls_min_snr``,
for a planet host) -> ``'BLS'``; else the combined LS peak (scope
``'combined'``, method ``'LS'``, fap <= ``ls_fap_threshold``, for a variable)
-> ``'LS'``; else the median per-night LS period -> ``'night-LS'``; else
``NULL``.
"""

from __future__ import annotations

from collections.abc import Sequence

import psycopg

from relphot.config import DbSettings

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
    WHERE d.kind = 'transit' AND d.snr IS NOT NULL
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
has_transit_or_bls AS (
    SELECT DISTINCT d.obj_id FROM relphot.detection d JOIN target t ON t.obj_id = d.obj_id
    WHERE d.kind IN ('transit', 'bls')
        AND (d.night_id IS NOT NULL
             OR (d.mn_run_id IS NOT NULL AND d.kind = ANY(%(class_multinight_kinds)s)))
),
has_var_like AS (
    SELECT DISTINCT d.obj_id FROM relphot.detection d JOIN target t ON t.obj_id = d.obj_id
    WHERE d.kind IN ('variable', 'internight', 'ls_periodic', 'recurrent')
        AND (d.night_id IS NOT NULL
             OR (d.mn_run_id IS NOT NULL AND d.kind = ANY(%(class_multinight_kinds)s)))
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
flag_auto AS (
    SELECT t.obj_id,
           (pc.obj_id IS NOT NULL OR hb.obj_id IS NOT NULL) AS auto_exop,
           (vc.obj_id IS NOT NULL OR hv.obj_id IS NOT NULL) AS auto_var
    FROM target t
    LEFT JOIN planet_cm pc ON pc.obj_id = t.obj_id
    LEFT JOIN variable_cm vc ON vc.obj_id = t.obj_id
    LEFT JOIN has_transit_or_bls hb ON hb.obj_id = t.obj_id
    LEFT JOIN has_var_like hv ON hv.obj_id = t.obj_id
),
flag_calc AS (
    SELECT o.obj_id,
           COALESCE(o.exop_source = 'manual', false) AS exop_manual,
           COALESCE(o.var_source = 'manual', false) AS var_manual,
           CASE WHEN o.exop_source = 'manual' THEN o.is_exop ELSE fa.auto_exop END AS is_exop,
           CASE WHEN o.var_source = 'manual' THEN o.is_var ELSE fa.auto_var END AS is_var
    FROM relphot.object o JOIN flag_auto fa ON fa.obj_id = o.obj_id
),
best_pe AS (
    SELECT DISTINCT ON (pe.obj_id) pe.obj_id, pe.period, pe.period_err, pe.n_nights, pe.fap
    FROM relphot.period_estimate pe JOIN target t ON t.obj_id = pe.obj_id
    WHERE pe.method = 'LS' AND pe.period IS NOT NULL
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
    is_exop = fc.is_exop,
    is_var = fc.is_var,
    exop_source = CASE WHEN fc.exop_manual THEN 'manual' ELSE 'auto' END,
    var_source = CASE WHEN fc.var_manual THEN 'manual' ELSE 'auto' END,
    class = CASE
                WHEN fc.is_exop AND fc.is_var THEN 'EXOP+VAR'
                WHEN fc.is_exop THEN 'EXOP'
                WHEN fc.is_var THEN 'VAR'
                ELSE 'UNC'
            END,
    class_source = CASE WHEN fc.exop_manual OR fc.var_manual THEN 'manual' ELSE 'auto' END,
    period = CASE WHEN o.period_source IS DISTINCT FROM 'manual' THEN
                COALESCE(
                    pc.period, vc.period,
                    CASE WHEN pe.fap <= %(ls_fap_threshold)s THEN pe.period END,
                    CASE WHEN fc.is_exop AND bc.depth_snr >= %(bls_min_snr)s
                         THEN bc.peak_period END,
                    CASE WHEN fc.is_var AND lc.fap <= %(ls_fap_threshold)s
                         THEN lc.peak_period END,
                    vpm.median_period
                )
             ELSE o.period END,
    period_source = CASE WHEN o.period_source IS DISTINCT FROM 'manual' THEN
                CASE
                    WHEN COALESCE(pc.period, vc.period) IS NOT NULL THEN 'catalog'
                    WHEN pe.fap <= %(ls_fap_threshold)s THEN 'LS'
                    WHEN fc.is_exop AND bc.depth_snr >= %(bls_min_snr)s THEN 'BLS'
                    WHEN fc.is_var AND lc.fap <= %(ls_fap_threshold)s THEN 'LS'
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
LEFT JOIN has_transit_or_bls hb ON hb.obj_id = tgt.obj_id
LEFT JOIN has_var_like hv ON hv.obj_id = tgt.obj_id
LEFT JOIN cm_agg cm ON cm.obj_id = tgt.obj_id
LEFT JOIN planet_cm pc ON pc.obj_id = tgt.obj_id
LEFT JOIN variable_cm vc ON vc.obj_id = tgt.obj_id
LEFT JOIN flag_calc fc ON fc.obj_id = tgt.obj_id
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

    if obj_ids is not None:
        obj_ids = list(obj_ids)
        if not obj_ids:
            return
        sql = _REFRESH_SQL.format(target_filter=" WHERE obj_id = ANY(%(obj_ids)s)")
        params: dict[str, object] = {
            "obj_ids": obj_ids, "bls_min_snr": bls_min_snr, "ls_fap_threshold": ls_fap_threshold,
            "class_multinight_kinds": list(class_multinight_kinds),
        }
    else:
        sql = _REFRESH_SQL.format(target_filter="")
        params = {
            "bls_min_snr": bls_min_snr, "ls_fap_threshold": ls_fap_threshold,
            "class_multinight_kinds": list(class_multinight_kinds),
        }
    with conn.cursor() as cur:
        cur.execute(sql, params)

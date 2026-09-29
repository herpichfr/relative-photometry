"""Per-night user review and automatic object classification.

Constants and functions for deriving exoplanet/variable flags from per-night
user verdicts, literature matches, and automatic detections. Literature (known
planets/variables) and gated multi-night evidence add; a per-night verdict
(``relphot.user_night_review``) overrides only that night's automatic evidence and
never removes a literature match.

``refresh_flags`` is called first in the refresh pipeline (before the main
:func:`relphot.db.refresh.refresh_objects` UPDATE) and sets is_exop, is_var,
exop_source, var_source, class, class_source, n_review_pending, n_nights_reviewed.
This module imports only the standard library and psycopg (plus, lazily,
:mod:`relphot.config`) so the web app can use it without importing ``relphot.db``.
"""

from __future__ import annotations

import psycopg

__all__ = [
    "NIGHT_EXOP_KINDS",
    "NIGHT_VAR_KINDS",
    "PLANET_CATALOGS",
    "night_state",
    "refresh_flags",
]

#: Per-night detection kinds that set exoplanet flag automatically.
NIGHT_EXOP_KINDS = ("transit", "bls")

#: Per-night detection kinds that set variable flag automatically.
NIGHT_VAR_KINDS = ("variable", "internight", "ls_periodic", "recurrent")

#: Catalog names that indicate a known exoplanet host.
PLANET_CATALOGS = ("NASA Exoplanet Archive", "TOI")

_REFRESH_FLAGS_SQL = """
WITH target AS (
    SELECT obj_id FROM relphot.object{target_filter}
),
night_auto AS (
    SELECT d.obj_id, d.night_id,
           bool_or(d.kind = ANY(%(night_exop_kinds)s)) AS exop_ev,
           bool_or(d.kind = ANY(%(night_var_kinds)s)) AS var_ev,
           bool_or(d.kind = ANY(%(night_exop_kinds)s)
                   AND COALESCE(d.status, 'UNCONFIRMED') = 'UNCONFIRMED') AS exop_open,
           bool_or(d.kind = ANY(%(night_var_kinds)s)
                   AND COALESCE(d.status, 'UNCONFIRMED') = 'UNCONFIRMED') AS var_open
    FROM relphot.detection d JOIN target t ON t.obj_id = d.obj_id
    WHERE d.night_id IS NOT NULL AND d.origin = 'search'
          AND COALESCE(d.status, 'UNCONFIRMED') <> 'REJECTED'
    GROUP BY d.obj_id, d.night_id
),
night_rev AS (
    SELECT * FROM relphot.user_night_review
    WHERE obj_id IN (SELECT obj_id FROM target)
),
night_state AS (
    SELECT COALESCE(na.obj_id, nr.obj_id) AS obj_id,
           COALESCE(na.night_id, nr.night_id) AS night_id,
           CASE nr.exop_verdict WHEN 'CONFIRMED' THEN true WHEN 'REJECTED' THEN false
                ELSE COALESCE(na.exop_ev, false) END AS exop_on,
           CASE nr.var_verdict WHEN 'CONFIRMED' THEN true WHEN 'REJECTED' THEN false
                ELSE COALESCE(na.var_ev, false) END AS var_on,
           (nr.exop_verdict IS NULL AND COALESCE(na.exop_open, false))
           OR (nr.var_verdict IS NULL AND COALESCE(na.var_open, false)) AS pending
    FROM night_auto na FULL JOIN night_rev nr
         ON na.obj_id = nr.obj_id AND na.night_id = nr.night_id
),
ns_agg AS (
    SELECT obj_id,
           bool_or(exop_on) AS exop_any,
           bool_or(var_on) AS var_any,
           count(*) FILTER (WHERE pending) AS n_pending
    FROM night_state
    GROUP BY obj_id
),
rev_agg AS (
    SELECT obj_id,
           bool_or(exop_verdict IS NOT NULL) AS has_exop_rev,
           bool_or(var_verdict IS NOT NULL) AS has_var_rev
    FROM night_rev
    GROUP BY obj_id
),
reviewed_nights AS (
    SELECT obj_id, count(*) AS n_reviewed
    FROM (
        SELECT obj_id, night_id FROM night_rev
        WHERE exop_verdict IS NOT NULL OR var_verdict IS NOT NULL OR note IS NOT NULL
        UNION
        SELECT d.obj_id, d.night_id
        FROM relphot.detection d JOIN target t ON t.obj_id = d.obj_id
        WHERE d.night_id IS NOT NULL AND d.origin = 'search'
              AND d.status IN ('CONFIRMED', 'REJECTED')
    ) reviewed
    GROUP BY obj_id
),
mn AS (
    SELECT d.obj_id,
           bool_or(d.kind = ANY(%(night_exop_kinds)s)) AS exop_ev,
           bool_or(d.kind = ANY(%(night_var_kinds)s)) AS var_ev,
           count(*) FILTER (WHERE COALESCE(d.status, 'UNCONFIRMED') = 'UNCONFIRMED') AS n_open
    FROM relphot.detection d JOIN target t ON t.obj_id = d.obj_id
    WHERE d.mn_run_id IS NOT NULL AND d.origin = 'search'
          AND COALESCE(d.status, 'UNCONFIRMED') <> 'REJECTED'
          AND d.kind = ANY(%(class_multinight_kinds)s)
    GROUP BY d.obj_id
),
lit AS (
    SELECT cm.obj_id,
           bool_or(cm.catalog = ANY(%(planet_catalogs)s)) AS planet,
           bool_or(cm.catalog <> ALL(%(planet_catalogs)s)) AS var
    FROM relphot.catalog_match cm JOIN target t ON t.obj_id = cm.obj_id
    GROUP BY cm.obj_id
),
flags AS (
    SELECT t.obj_id,
           COALESCE(lit.planet OR mn.exop_ev OR ns.exop_any, false) AS is_exop,
           COALESCE(lit.var OR mn.var_ev OR ns.var_any, false) AS is_var,
           COALESCE(ra.has_exop_rev, false) AS exop_manual,
           COALESCE(ra.has_var_rev, false) AS var_manual,
           COALESCE(ns.n_pending, 0) + COALESCE(mn.n_open, 0) AS n_pending,
           COALESCE(rn.n_reviewed, 0) AS n_reviewed
    FROM target t
    LEFT JOIN ns_agg ns ON ns.obj_id = t.obj_id
    LEFT JOIN rev_agg ra ON ra.obj_id = t.obj_id
    LEFT JOIN reviewed_nights rn ON rn.obj_id = t.obj_id
    LEFT JOIN mn ON mn.obj_id = t.obj_id
    LEFT JOIN lit ON lit.obj_id = t.obj_id
)
UPDATE relphot.object o
SET is_exop = f.is_exop,
    is_var = f.is_var,
    exop_source = CASE WHEN f.exop_manual THEN 'manual' ELSE 'auto' END,
    var_source = CASE WHEN f.var_manual THEN 'manual' ELSE 'auto' END,
    class = CASE
        WHEN f.is_exop AND f.is_var THEN 'EXOP+VAR'
        WHEN f.is_exop THEN 'EXOP'
        WHEN f.is_var THEN 'VAR'
        ELSE 'UNC'
    END,
    class_source = CASE WHEN f.exop_manual OR f.var_manual THEN 'manual' ELSE 'auto' END,
    n_review_pending = f.n_pending,
    n_nights_reviewed = f.n_reviewed,
    updated_at = now()
FROM flags f
WHERE o.obj_id = f.obj_id
"""


def refresh_flags(
    conn: psycopg.Connection,
    obj_ids: list[int] | None = None,
    *,
    class_multinight_kinds: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Recompute per-night flags and object classification from reviews and detections.

    Runs BEFORE the main :func:`relphot.db.refresh.refresh_objects` UPDATE in the
    same transaction. Sets is_exop, is_var, exop_source, var_source, class,
    class_source, n_review_pending, n_nights_reviewed for target objects.

    Literature (known planets/variables) and gated multi-night detections add; a
    per-night verdict overrides only that night's automatic evidence and never
    removes a literature match. A night's search detections with status REJECTED
    are ignored (one event's evidence only). ``exop_source`` / ``var_source`` are
    ``'manual'`` iff the object has a verdict of that kind on some night. Every
    target object is updated (an object with no evidence gets false / ``'UNC'``).

    ``obj_ids`` restricts the update to those objects; ``None`` (the default)
    updates every object in the table. ``class_multinight_kinds`` gates which
    multi-night detection kinds count towards CLASS; defaults to
    :class:`relphot.config.DbSettings` when not given. Does not commit.
    """
    if class_multinight_kinds is None:
        from relphot.config import DbSettings

        class_multinight_kinds = DbSettings().class_multinight_kinds

    params: dict[str, object] = {
        "night_exop_kinds": list(NIGHT_EXOP_KINDS),
        "night_var_kinds": list(NIGHT_VAR_KINDS),
        "planet_catalogs": list(PLANET_CATALOGS),
        "class_multinight_kinds": list(class_multinight_kinds),
    }
    if obj_ids is None:
        target_filter = ""
    else:
        obj_ids = list(obj_ids)
        if not obj_ids:
            return
        target_filter = " WHERE obj_id = ANY(%(obj_ids)s)"
        params["obj_ids"] = obj_ids

    with conn.cursor() as cur:
        cur.execute(_REFRESH_FLAGS_SQL.format(target_filter=target_filter), params)


def night_state(
    detections: list[dict],
    exop_verdict: str | None,
    var_verdict: str | None,
) -> dict:
    """Compute the effective flags and pending status for one night's detections.

    Mirrors the ``night_auto`` / ``night_state`` CTEs of :func:`refresh_flags` for a
    single night.

    ``detections`` is a list of dicts with keys: kind (str), status (str or None),
    origin (str).

    Returns a dict with keys:
    - auto_exop, auto_var: whether automatic evidence is present
    - exop_open, var_open: whether automatic evidence is unconfirmed (open)
    - exop_effective, var_effective: the effective flag value (verdict overrides auto)
    - pending: whether this night awaits review
    """
    # Only the search's own, not rejected, events are automatic evidence.
    search_dets = [
        d for d in detections
        if d.get("origin") == "search" and (d.get("status") or "UNCONFIRMED") != "REJECTED"
    ]

    exop_ev = any(d["kind"] in NIGHT_EXOP_KINDS for d in search_dets)
    var_ev = any(d["kind"] in NIGHT_VAR_KINDS for d in search_dets)

    exop_open = any(
        d["kind"] in NIGHT_EXOP_KINDS and (d.get("status") or "UNCONFIRMED") == "UNCONFIRMED"
        for d in search_dets
    )
    var_open = any(
        d["kind"] in NIGHT_VAR_KINDS and (d.get("status") or "UNCONFIRMED") == "UNCONFIRMED"
        for d in search_dets
    )

    if exop_verdict == "CONFIRMED":
        exop_effective = True
    elif exop_verdict == "REJECTED":
        exop_effective = False
    else:
        exop_effective = exop_ev

    if var_verdict == "CONFIRMED":
        var_effective = True
    elif var_verdict == "REJECTED":
        var_effective = False
    else:
        var_effective = var_ev

    pending = (exop_verdict is None and exop_open) or (var_verdict is None and var_open)

    return {
        "auto_exop": exop_ev,
        "auto_var": var_ev,
        "exop_open": exop_open,
        "var_open": var_open,
        "exop_effective": exop_effective,
        "var_effective": var_effective,
        "pending": pending,
    }

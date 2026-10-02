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

It also holds the rules for superseded transit events (``relphot.detection.superseded_by``):
the light curve of ONE object on ONE night may carry several transit events -- the search's and a
person's RERUNs -- that fit the same transit; one is the active event and the others are
superseded by it (:func:`plan_keep`, :func:`keep_transit_event`).
"""

from __future__ import annotations

import psycopg

__all__ = [
    "NIGHT_EXOP_KINDS",
    "NIGHT_VAR_KINDS",
    "OVERLAP_FRAC",
    "PLANET_CATALOGS",
    "competing_events",
    "events_overlap",
    "keep_transit_event",
    "night_state",
    "plan_keep",
    "plan_night_exop_to_events",
    "refresh_flags",
    "sync_night_exop_from_events",
    "transit_night_verdict",
]

#: Per-night detection kinds that set exoplanet flag automatically.
NIGHT_EXOP_KINDS = ("transit", "bls")

#: Per-night detection kinds that set variable flag automatically.
NIGHT_VAR_KINDS = ("variable", "internight", "ls_periodic", "recurrent")

#: Catalog names that indicate a known exoplanet host.
PLANET_CATALOGS = ("NASA Exoplanet Archive", "TOI")

#: Two transit events of one light curve are the same event (one supersedes the other) when their
#: centre times differ by at most this fraction of the longer of their durations: the convention
#: of the night reload (a saved review goes to the event within half its duration) and of the
#: RERUN worker (the search event within half the guessed width).
OVERLAP_FRAC = 0.5

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
    WHERE d.night_id IS NOT NULL AND d.superseded_by IS NULL
          AND (d.origin = 'search' OR EXISTS (
                   SELECT 1 FROM relphot.detection s
                   WHERE s.superseded_by = d.det_id AND s.origin = 'search'
                         AND (s.auto_status IS DISTINCT FROM 'REJECTED' OR s.status = 'CONFIRMED')
          ))
          AND COALESCE(d.status, 'UNCONFIRMED') <> 'REJECTED'
          AND (d.auto_status IS DISTINCT FROM 'REJECTED' OR d.status = 'CONFIRMED')
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
    are ignored (one event's evidence only), and so are those with ``auto_status =
    'REJECTED'`` (too many similar events on the night, :mod:`relphot.db.coincidence`)
    unless a person CONFIRMED them: neither evidence nor open for review.
    A superseded event (``superseded_by`` set) is neither: the active event of its light curve
    stands for it. A person's own (``origin = 'user'``) active event counts as the night's
    automatic evidence only when it supersedes a search event that itself would count, so a
    RERUN of an automatic candidate keeps the night's evidence and its place in the review queue,
    and a RERUN of nothing the search found never sets a flag.
    ``exop_source`` / ``var_source`` are
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
    origin (str), and optionally auto_status (str or None: ``'REJECTED'`` is the automatic
    "too many similar events" verdict of :mod:`relphot.db.coincidence`), det_id and
    superseded_by (int or None: the event this one was superseded by; see
    :func:`refresh_flags` for how superseded and RERUN events count).

    Returns a dict with keys:
    - auto_exop, auto_var: whether automatic evidence is present
    - exop_open, var_open: whether automatic evidence is unconfirmed (open)
    - exop_effective, var_effective: the effective flag value (verdict overrides auto)
    - pending: whether this night awaits review
    """
    def _auto_ok(d: dict) -> bool:
        """Not rejected by a person, nor automatically unless a person CONFIRMED it."""
        return (d.get("status") or "UNCONFIRMED") != "REJECTED" and not (
            d.get("auto_status") == "REJECTED"
            and (d.get("status") or "UNCONFIRMED") != "CONFIRMED"
        )

    def _stands_for_search(d: dict) -> bool:
        """A person's event that supersedes a search event whose own status would count."""
        det_id = d.get("det_id")
        return det_id is not None and any(
            s.get("superseded_by") == det_id and s.get("origin") == "search"
            and (
                s.get("auto_status") != "REJECTED"
                or (s.get("status") or "UNCONFIRMED") == "CONFIRMED"
            )
            for s in detections
        )

    # Only the active search events that are not rejected -- by a person, or automatically
    # unless a person CONFIRMED them -- are automatic evidence; so is a person's active event
    # that supersedes such a search event.
    search_dets = [
        d for d in detections
        if d.get("superseded_by") is None
        and (d.get("origin") == "search" or _stands_for_search(d))
        and _auto_ok(d)
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


def transit_night_verdict(statuses: list[str | None]) -> str | None:
    """The night EXOP verdict that goes with the person's verdicts on one night's transit events.

    ``statuses`` are the ``detection.status`` values of the active transit events (not superseded
    by another) of one object on one night (``None`` reads as UNCONFIRMED). One CONFIRMED event
    makes the night CONFIRMED; otherwise the night is REJECTED when every event is REJECTED;
    anything else (no events, or some still open) is no verdict, i.e. ``None`` = automatic. An
    event's automatic rejection (``auto_status``) is not a person's verdict and is not looked at
    here.
    """
    states = [s or "UNCONFIRMED" for s in statuses]
    if "CONFIRMED" in states:
        return "CONFIRMED"
    if states and all(s == "REJECTED" for s in states):
        return "REJECTED"
    return None


def plan_night_exop_to_events(
    events: list[tuple[int, str | None]], previous: str | None, new: str | None
) -> tuple[list[int], str]:
    """The ``det_id`` s of one night's transit events to set, and to what, when that night's
    EXOP verdict goes from ``previous`` to ``new`` (the inverse of :func:`transit_night_verdict`).

    ``events`` are ``(det_id, status)`` of the active transit events (not superseded by another)
    of the object on the night.

    - ``new`` REJECTED: every event not yet REJECTED becomes REJECTED.
    - ``new`` CONFIRMED: every event not REJECTED becomes CONFIRMED; an event a person REJECTED
      stays so, unless all of them are REJECTED, then all become CONFIRMED.
    - ``new`` ``None`` (back to automatic): the events that carry the verdict being cleared
      become UNCONFIRMED; the others are left alone.

    Nothing to do (``([], "")``) when the verdict does not change.
    """
    if new == previous:
        return [], ""
    states = {det_id: status or "UNCONFIRMED" for det_id, status in events}
    if new == "REJECTED":
        return [d for d, s in states.items() if s != "REJECTED"], "REJECTED"
    if new == "CONFIRMED":
        keep_rejected = any(s != "REJECTED" for s in states.values())
        return [
            d for d, s in states.items()
            if s != "CONFIRMED" and not (s == "REJECTED" and keep_rejected)
        ], "CONFIRMED"
    return [d for d, s in states.items() if s == previous], "UNCONFIRMED"


def sync_night_exop_from_events(cur, pairs: set[tuple[int, int | None]]) -> list[dict]:
    """Events -> night: bring the EXOP verdict of each ``(obj_id, night_id)`` to what its active
    transit events (not superseded by another) say (:func:`transit_night_verdict`); a row left
    with no verdict and no note is deleted. A superseded event's own status is kept but is not
    looked at. ``cur`` is a cursor of an open transaction, which the caller commits.

    Returns ``[{obj_id, night_id, exop}]`` for the pairs, ``exop`` being the verdict or ``None``.
    """
    pairs = {(o, n) for o, n in pairs if n is not None}
    if not pairs:
        return []
    obj_ids, night_ids = zip(*sorted(pairs), strict=True)
    cur.execute(
        "SELECT d.obj_id, d.night_id, array_agg(COALESCE(d.status, 'UNCONFIRMED')) "
        "FROM relphot.detection d "
        "JOIN unnest(%s::bigint[], %s::integer[]) AS p(obj_id, night_id) "
        "ON p.obj_id = d.obj_id AND p.night_id = d.night_id "
        "WHERE d.kind = 'transit' AND d.superseded_by IS NULL GROUP BY d.obj_id, d.night_id",
        (list(obj_ids), list(night_ids)),
    )
    states = {(o, n): statuses for o, n, statuses in cur.fetchall()}
    result = []
    for obj_id, night_id in sorted(pairs):
        verdict = transit_night_verdict(states.get((obj_id, night_id), []))
        params = {"obj_id": obj_id, "night_id": night_id, "verdict": verdict}
        if verdict is not None:
            cur.execute(
                "INSERT INTO relphot.user_night_review (obj_id, night_id, exop_verdict) "
                "VALUES (%(obj_id)s, %(night_id)s, %(verdict)s) "
                "ON CONFLICT (obj_id, night_id) DO UPDATE SET "
                "exop_verdict = EXCLUDED.exop_verdict, updated_at = now() "
                "WHERE relphot.user_night_review.exop_verdict IS DISTINCT FROM "
                "EXCLUDED.exop_verdict",
                params,
            )
        else:
            cur.execute(
                "UPDATE relphot.user_night_review SET exop_verdict = NULL, updated_at = now() "
                "WHERE obj_id = %(obj_id)s AND night_id = %(night_id)s "
                "AND exop_verdict IS NOT NULL",
                params,
            )
            cur.execute(
                "DELETE FROM relphot.user_night_review WHERE obj_id = %(obj_id)s "
                "AND night_id = %(night_id)s AND exop_verdict IS NULL AND var_verdict IS NULL "
                "AND note IS NULL",
                params,
            )
        result.append({"obj_id": obj_id, "night_id": night_id, "exop": verdict})
    return result


# --------------------------------------------------------------------------
# superseded transit events: one active event per transit of one light curve
#
# The events of ONE object on ONE night (one light curve; never two stars of a night) that fit
# the same transit are a group: the search's event and the person's RERUNs. The group has one
# active event (``superseded_by`` NULL); the others point at it. ``events`` below are dicts with
# ``det_id``, ``tc`` (``detection.tc_bjd_tdb``, BJD), ``duration_h`` and ``superseded_by`` of the
# transit events of that one object and night.
# --------------------------------------------------------------------------


def events_overlap(
    tc_a: float | None, duration_a_h: float | None, tc_b: float | None, duration_b_h: float | None
) -> bool:
    """Whether two transit events of one light curve are the same transit: their centre times
    (BJD) differ by at most :data:`OVERLAP_FRAC` of the longer duration (hours). An event with no
    centre time overlaps nothing; a missing duration counts as zero."""
    if tc_a is None or tc_b is None:
        return False
    longest = max(d for d in (duration_a_h, duration_b_h, 0.0) if d is not None)
    return abs(tc_a - tc_b) <= OVERLAP_FRAC * longest / 24.0


def _group_of(keep_id: int, events: list[dict]) -> set[int]:
    """The events ``keep_id`` competes with (and itself): those that overlap it in time, and
    everything already linked to them or to it (the active event they were superseded by, and the
    events superseded by that one)."""
    by_id = {e["det_id"]: e for e in events}
    keep = by_id[keep_id]
    base = {keep_id} | {
        e["det_id"] for e in events
        if e["det_id"] != keep_id
        and events_overlap(keep["tc"], keep["duration_h"], e["tc"], e["duration_h"])
    }

    def root(det_id: int) -> int:
        parent = by_id[det_id].get("superseded_by")
        return parent if parent in by_id else det_id

    roots = {root(i) for i in base}
    return base | roots | {i for i in by_id if root(i) in roots}


def plan_keep(keep_id: int, events: list[dict]) -> dict[int, int | None]:
    """The ``superseded_by`` changes that make ``keep_id`` the active event of its group.

    ``{det_id: new superseded_by}`` for each event whose link must change: ``keep_id`` itself
    becomes active (``None``) and every other event of its group (:func:`_group_of`) is
    superseded by it, whatever it was superseded by before, so links stay flat. An event of the
    night outside the group (another transit) is left alone. ``{}`` when ``keep_id`` is not among
    ``events`` or already stands alone as the active event.
    """
    by_id = {e["det_id"]: e for e in events}
    if keep_id not in by_id:
        return {}
    changes: dict[int, int | None] = {}
    for det_id in _group_of(keep_id, events):
        wanted = None if det_id == keep_id else keep_id
        if by_id[det_id].get("superseded_by") != wanted:
            changes[det_id] = wanted
    return changes


def competing_events(events: list[dict]) -> dict[int, list[int]]:
    """For each event, the ``det_id`` s of the other events of the light curve it competes with
    (its group without itself, sorted): the events a "keep this" on it would supersede, or that
    superseded it. Empty for an event that is the only one of its transit."""
    return {
        e["det_id"]: sorted(_group_of(e["det_id"], events) - {e["det_id"]}) for e in events
    }


_NIGHT_TRANSITS_SQL = (
    "SELECT det_id, tc_bjd_tdb, duration_h, superseded_by FROM relphot.detection "
    "WHERE obj_id = %s AND night_id = %s AND kind = 'transit' ORDER BY det_id"
)


def keep_transit_event(cur, det_id: int) -> tuple[int, int, dict[int, int | None]]:
    """Make the per-night transit event ``det_id`` the active event of its group
    (:func:`plan_keep`) and write the changed ``superseded_by`` links. Used by the RERUN worker
    for the event it just stored (the older events it overlaps are superseded) and by the web's
    "keep this" button (which swaps the links back and forth). Only the events of that one
    object and night are read or written; the database refuses a link across objects or nights
    anyway.

    ``cur`` is a cursor of an open transaction, which the caller commits. Returns
    ``(obj_id, night_id, changes)``, ``changes`` as in :func:`plan_keep` (``{}``: nothing to do).
    Raises :class:`LookupError` when ``det_id`` is not a per-night transit event.
    """
    cur.execute(
        "SELECT obj_id, night_id FROM relphot.detection "
        "WHERE det_id = %s AND kind = 'transit' AND night_id IS NOT NULL",
        (det_id,),
    )
    found = cur.fetchone()
    if found is None:
        msg = f"per-night transit event {det_id} not found"
        raise LookupError(msg)
    obj_id, night_id = found
    cur.execute(_NIGHT_TRANSITS_SQL, (obj_id, night_id))
    events = [
        {"det_id": i, "tc": tc, "duration_h": dur, "superseded_by": sup}
        for i, tc, dur, sup in cur.fetchall()
    ]
    changes = plan_keep(det_id, events)
    for changed_id, superseded_by in changes.items():
        cur.execute(
            "UPDATE relphot.detection SET superseded_by = %s WHERE det_id = %s",
            (superseded_by, changed_id),
        )
    return obj_id, night_id, changes

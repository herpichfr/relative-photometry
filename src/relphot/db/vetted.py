"""Which transit events a person has already vetted (and that automatic re-runs must not touch).

An event is *vetted* when any of these holds:

- its ``status`` is ``CONFIRMED`` or ``REJECTED`` (a person's verdict), or it has ``notes``;
- it is a person's own event (``origin = 'user'``, a RERUN);
- its object has a ``user_night_review`` row for its night (a per-night verdict or note);
- it takes part in a supersede link (``superseded_by``, either end): a person kept one of two
  events.

``UNCONFIRMED`` / NULL status with no notes, no review and no link is *unvetted*. Used by
:func:`relphot.db.reload_search.reload_search_detections` (replaces only unvetted search events),
``relphot db analyze --keep-vetted`` (does not refit, re-match or re-judge a vetted event).
"""

from __future__ import annotations

from collections.abc import Sequence

import psycopg

__all__ = ["VETTED_CONDITION", "vetted_det_ids"]

#: SQL condition on ``relphot.detection d``: the transit event is vetted (module docstring).
VETTED_CONDITION = """(
    d.status IN ('CONFIRMED', 'REJECTED') OR d.notes IS NOT NULL OR d.origin = 'user'
    OR d.superseded_by IS NOT NULL
    OR EXISTS (SELECT 1 FROM relphot.detection s WHERE s.superseded_by = d.det_id)
    OR EXISTS (SELECT 1 FROM relphot.user_night_review r
               WHERE r.obj_id = d.obj_id AND r.night_id = d.night_id)
)"""


def vetted_det_ids(
    cur: psycopg.Cursor,
    *,
    night_ids: Sequence[int] | None = None,
    obj_ids: Sequence[int] | None = None,
) -> set[int]:
    """``det_id`` of the vetted per-night transit events of ``night_ids`` and/or ``obj_ids``.

    Both filters given: the events satisfying both. Neither: every night's.
    """
    cur.execute(
        f"""
        SELECT d.det_id FROM relphot.detection d
        WHERE d.kind = 'transit' AND d.night_id IS NOT NULL
          AND (%(night_ids)s::int[] IS NULL OR d.night_id = ANY(%(night_ids)s::int[]))
          AND (%(obj_ids)s::bigint[] IS NULL OR d.obj_id = ANY(%(obj_ids)s::bigint[]))
          AND {VETTED_CONDITION}
        """,
        {
            "night_ids": None if night_ids is None else [int(n) for n in night_ids],
            "obj_ids": None if obj_ids is None else [int(o) for o in obj_ids],
        },
    )
    return {row[0] for row in cur.fetchall()}

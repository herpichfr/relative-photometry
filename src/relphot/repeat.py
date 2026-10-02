"""Web-safe parts of the repeated-event work: loading, prediction and decision matching.

:mod:`relphot.db.families` computes the families of repeated transit events and needs the whole
``relphot.db`` package (pandas); the web API and the CLI only read the stored rows and predict
windows, so the pieces they share live here and import nothing but psycopg, astropy and
:mod:`relphot.config`. :mod:`relphot.db.families` re-exports them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import psycopg

from relphot.config import DbSettings

__all__ = [
    "Window",
    "decision_tol_days",
    "load_families",
    "match_decision",
    "parse_when",
    "predict_windows",
]


@dataclass(slots=True)
class Window:
    """One predicted transit window of a family (BJD_TDB), merged over its allowed aliases."""

    obj_id: int
    fam_id: int | None
    start: float
    end: float
    n_aliases: int
    n_aliases_total: int
    depth: float | None
    t14_h: float | None
    t14_lower_limit: bool
    #: a member of the family was rejected (or a decision made) since it was computed
    stale: bool = False


def parse_when(text: str, *, end: bool) -> float:
    """A BJD (a number >= 2.4e6) or a UTC date / date-time as a BJD_TDB-like Julian date.

    A date without a time is its start (``end`` false) or the end of that day (``end`` true).
    The barycentric correction (a few minutes at most) is ignored: the predicted windows are
    wider than that. Raises ``ValueError`` for text that is neither.
    """
    from astropy.time import Time

    try:
        value = float(text)
    except ValueError:
        value = None
    if value is not None and value >= 2.4e6:
        return value
    jd = float(Time(text, scale="utc").tdb.jd)
    return jd + 1.0 if end and "T" not in text and " " not in text.strip() else jd


def decision_tol_days(t14_a_h: float, t14_b_h: float) -> float:
    """How far (days) a stored decision's centre times may sit from a pair's own: half of the
    longer T14, at least 0.01 d."""
    return max(0.5 * max(t14_a_h, t14_b_h) / 24.0, 0.01)



def match_decision(a: dict, b: dict, decisions: list[dict]) -> str | None:
    """The person's SAME / DIFFERENT for the pair ``(a, b)``, matched by night and centre time.

    A stored decision applies when its nights are the pair's nights and both its centre times are
    within half of the longer T14 (at least 0.01 d) of the events' own -- so it is re-attached
    after a reload moved the events a little. The nearest wins.
    """
    first, second = sorted((a, b), key=lambda e: (e["night_id"], e["tc"]))
    tol = decision_tol_days(a["t14_h"], b["t14_h"])
    best: tuple[float, str] | None = None
    for d in decisions:
        if d["night_a"] != first["night_id"] or d["night_b"] != second["night_id"]:
            continue
        miss = max(abs(d["tc_a"] - first["tc"]), abs(d["tc_b"] - second["tc"]))
        if miss <= tol and (best is None or miss < best[0]):
            best = (miss, d["decision"])
    return None if best is None else best[1]


def load_families(
    conn: psycopg.Connection,
    obj_ids: Sequence[int] | None = None,
    *,
    telescope: str | None = None,
    accepted_only: bool = False,
) -> list[dict]:
    """The stored families with their aliases (SELECT only), as :func:`predict_windows` takes them.

    ``telescope``: only objects with a night of that telescope. Each family dict has the
    ``repeat_family`` columns, ``obj_name``, ``stale`` (a member was rejected or superseded, or the
    person decided on a pair, since the family was computed) and ``aliases`` (``repeat_ephemeris``
    rows).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.fam_id, f.obj_id, o.name, f.family_key, f.n_members, f.member_night_ids,
                   f.involves_loose, f.depth, f.t14_h, f.t14_lower_limit, f.score, f.accepted,
                   EXISTS (
                       SELECT 1 FROM relphot.repeat_family_member m
                       JOIN relphot.detection d ON d.det_id = m.det_id
                       WHERE m.fam_id = f.fam_id
                         AND (d.status = 'REJECTED' OR d.superseded_by IS NOT NULL
                              OR (d.auto_status = 'REJECTED'
                                  AND d.status IS DISTINCT FROM 'CONFIRMED'))
                   ) OR EXISTS (
                       SELECT 1 FROM relphot.repeat_decision r
                       WHERE r.obj_id = f.obj_id AND r.updated_at > f.computed_at
                   ) AS stale
            FROM relphot.repeat_family f JOIN relphot.object o ON o.obj_id = f.obj_id
            WHERE (%(obj_ids)s::bigint[] IS NULL OR f.obj_id = ANY(%(obj_ids)s))
              AND (NOT %(accepted)s OR f.accepted)
              AND (%(tele)s::text IS NULL OR EXISTS (
                  SELECT 1 FROM relphot.star_night sn JOIN relphot.night n
                      ON n.night_id = sn.night_id
                  WHERE sn.obj_id = f.obj_id AND n.telescope = %(tele)s))
            ORDER BY f.obj_id, f.fam_id
            """,
            {
                "obj_ids": None if obj_ids is None else list(obj_ids), "accepted": accepted_only,
                "tele": telescope,
            },
        )
        cols = (
            "fam_id", "obj_id", "obj_name", "family_key", "n_members", "member_night_ids",
            "involves_loose", "depth", "t14_h", "t14_lower_limit", "score", "accepted", "stale",
        )
        families = [dict(zip(cols, row, strict=True), aliases=[]) for row in cur.fetchall()]
        by_id = {f["fam_id"]: f for f in families}
        if by_id:
            cur.execute(
                "SELECT fam_id, alias_k, period, period_err, tc0, tc0_err, status "
                "FROM relphot.repeat_ephemeris WHERE fam_id = ANY(%s) ORDER BY fam_id, alias_k",
                (list(by_id),),
            )
            for fam_id, k, period, period_err, tc0, tc0_err, status in cur.fetchall():
                by_id[fam_id]["aliases"].append({
                    "k": k, "period": period, "period_err": period_err, "tc0": tc0,
                    "tc0_err": tc0_err, "status": status,
                })
    return families


def predict_windows(
    families: Sequence[dict],
    start: float,
    end: float,
    settings: DbSettings | None = None,
    *,
    statuses: Sequence[str] = ("allowed",),
) -> list[Window]:
    """Time windows (BJD_TDB) in ``[start, end]`` where a family may transit, merged over aliases.

    ``families`` are :func:`load_families` dicts, or the families of :func:`compute_families`
    (``obj_id``, optional ``fam_id``, ``depth``, ``t14_h``, ``t14_lower_limit``, ``aliases``).
    An alias with one of ``statuses`` predicts, for every epoch ``m`` with its centre ``tc0 + m P``
    within reach of the range, the window ``centre +- (n sigma + T14 / 2)`` with ``sigma =
    hypot(tc0_err, |m| period_err)`` and ``n = db.repeat_n_sigma_window``. A family's windows are
    merged when they overlap; ``n_aliases`` counts the aliases predicting a transit in the merged
    window and ``n_aliases_total`` the aliases considered: a window only some aliases predict is the
    one that tells them apart. Sorted by start.
    """
    s = settings if settings is not None else DbSettings()
    out: list[Window] = []
    for fam in families:
        aliases = [a for a in fam["aliases"] if a["status"] in statuses]
        if not aliases:
            continue
        half_t14 = 0.5 * fam["t14_h"] / 24.0
        raw: list[tuple[float, float, int]] = []
        for idx, al in enumerate(aliases):
            period = al["period"]
            sig_p = al["period_err"] or 0.0
            sig_0 = al["tc0_err"] or 0.0
            reach = half_t14 + s.repeat_n_sigma_window * (sig_0 + 10.0 * sig_p)
            m_lo = math.floor((start - reach - al["tc0"]) / period)
            m_hi = math.ceil((end + reach - al["tc0"]) / period)
            for m in range(m_lo, m_hi + 1):
                centre = al["tc0"] + m * period
                half = half_t14 + s.repeat_n_sigma_window * math.hypot(sig_0, abs(m) * sig_p)
                if centre + half >= start and centre - half <= end:
                    raw.append((centre - half, centre + half, idx))
        raw.sort()
        merged: list[list] = []
        for lo, hi, idx in raw:
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
                merged[-1][2].add(idx)
            else:
                merged.append([lo, hi, {idx}])
        for lo, hi, idxs in merged:
            out.append(Window(
                obj_id=fam["obj_id"], fam_id=fam.get("fam_id"), start=lo, end=hi,
                n_aliases=len(idxs), n_aliases_total=len(aliases), depth=fam.get("depth"),
                t14_h=fam["t14_h"], t14_lower_limit=bool(fam.get("t14_lower_limit")),
                stale=bool(fam.get("stale")),
            ))
    out.sort(key=lambda w: (w.start, w.obj_id))
    return out
